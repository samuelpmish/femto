// Sparse matrix integration experiments (CUDA backend).
//
// Explores, for poisson (NC=1) and elasticity (NC=3) stiffness assembly:
//   1. parallel work mappings   : pair-per-thread, scalar-entry-per-thread,
//                                 multiple elements per block, element-per-thread,
//                                 warp-per-CSR-row (atomic-free)
//   2. qdata layouts            : q-major (current), SoA over qpts, element-blocked,
//                                 plus callback functors instead of stored C(i,k,j,m)
//   3. shape function storage   : shared / global / on-the-fly, test x trial
//   4. two-pass assembly        : element matrices to global memory, then a
//                                 gather pass driven by a dof-to-element map
//                                 (block/BSR-aware for elasticity)
//   5. access-pattern skeletons : the pair kernel with its arithmetic removed
//                                 (each thread only increments its entries),
//                                 bounding what the data access pattern alone
//                                 can sustain
//
// All variants consume the same inputs as the library's integrate_sparse_matrix:
// a CSR pattern, per-qpt constitutive data C(q,i,k,j,m), and mesh geometry.
// Every variant is verified against the library's own CSR fill.
//
// usage: cuda_spmat_tuning [size(0=smoke,1=full)] [warmup] [reps]

#include "fusion_common.hpp"

#include "forall.hpp"
#include "misc/macros.hpp"

#include <cstring>
#include <map>
#include <vector>

using namespace femto;
using namespace fusion;

static constexpr memory::space gpu_mem = memory::space::gpu;
static constexpr memory::space cpu_mem = memory::space::cpu;

static __device__ __forceinline__ int find_column_in_sparse_row(
  nd::view<const int, 1, gpu_mem> col_ind, int row_start, int row_end, int col) {
  int lo = row_start;
  int hi = row_end;
  while (lo < hi) {
    int mid = lo + (hi - lo) / 2;
    if (col_ind[mid] < col) { lo = mid + 1; } else { hi = mid; }
  }
  return (lo < row_end && col_ind[lo] == col) ? lo : -1;
}

constexpr uint32_t round_up_warp(uint32_t x) { return ((x + 31) / 32) * 32; }

////////////////////////////////////////////////////////////////////////////////
// materials (poisson + neohookean come from fusion_common.hpp)
////////////////////////////////////////////////////////////////////////////////

// linear isotropic elasticity: a du_dX-independent tangent, used by the
// constant-coefficient functor experiment
struct LinearIsotropicJacobianModel {
  __host__ __device__ mat<3,3,mat<3,3>> operator()(mat3 /*du_dX*/) const {
    mat<3,3,mat<3,3>> D{};
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
          for (int l = 0; l < 3; l++) {
            D[i][j][k][l] = lambda * (i == j) * (k == l)
                          + mu * ((i == k) * (j == l) + (i == l) * (j == k));
          }
        }
      }
    }
    return D;
  }
  double lambda, mu;
};

////////////////////////////////////////////////////////////////////////////////
// common kernel arguments
////////////////////////////////////////////////////////////////////////////////

// qdata layouts (DSZ = NC*3*NC*3 doubles per quadrature point):
//   QMajor : qdata(e*NQ+q, c)        -- the library's current layout
//   SoA    : qdata(c, e*NQ+q)        -- component-major (coalesced for qpt-parallel threads)
//   EBlock : qdata(e, c*NQ+q)        -- element-blocked SoA
//   ESoA   : qdata(c*NQ+q, e)        -- element-minor (coalesced for element-per-thread)
enum class QL { QMajor, SoA, EBlock, ESoA };

// where a shape function gradient comes from at its point of use
enum class SM {
  Shared,  // physical gradients precomputed into shared memory during setup
  Global,  // reference gradient table read from global memory, mapped on the fly
  Fly      // reference gradient evaluated from the qpt coordinate on demand
};

struct KArgs {
  nd::view<double, 1, gpu_mem> values;
  nd::view<const int, 1, gpu_mem> row_ptr;
  nd::view<const int, 1, gpu_mem> col_ind;
  nd::view<const uint32_t, 2, gpu_mem> ids;     // (nelem, NN) solution dof-node ids
  nd::view<const uint32_t, 2, gpu_mem> X_ids;   // (nelem, NNX) coordinate node ids
  nd::view<const double, 2, gpu_mem> X;         // node coordinates
  nd::view<const double, 2, gpu_mem> qdata;     // interpretation given by a QL
  nd::view<const double, 3, gpu_mem> G;         // (NQ, NN, 3) reference gradients
  nd::view<const double, 3, gpu_mem> GX;        // (NQ, NNX, 3)
  nd::view<const double, 1, gpu_mem> W;         // (NQ)
  nd::view<const double, 2, gpu_mem> xi_pts;    // (NQ, 3) expanded qpt coordinates
  nd::view<const uint16_t, 2, gpu_mem> pair_k;  // (nelem, NN*NN) block-column index per pair (optional)
  uint32_t nelem;
  uint32_t nq_total;
};

// CSR position of entry (i,j) for node pair (I,J), from the precomputed
// block-column index k: position = row_ptr[row_node*NC + i] + k*NC + j
template < uint32_t NC >
__device__ __forceinline__ int csr_position(const KArgs & a, uint32_t e, uint32_t pair,
                                            uint32_t row_node, uint32_t i, uint32_t j) {
  return a.row_ptr(int(row_node * NC + i)) + int(a.pair_k(e, pair) * NC + j);
}

template < QL L, uint32_t NQ, uint32_t DSZ >
__device__ __forceinline__ double qread(const KArgs & a, uint32_t e, uint32_t q, uint32_t c) {
  if constexpr (L == QL::QMajor) { return a.qdata(e * NQ + q, c); }
  if constexpr (L == QL::SoA)    { return a.qdata(c, e * NQ + q); }
  if constexpr (L == QL::EBlock) { return a.qdata(e, c * NQ + q); }
  if constexpr (L == QL::ESoA)   { return a.qdata(c * NQ + q, e); }
  return 0.0;
}

// dN_I[k] * C[i][k][j][m] * dN_J[m], C flat in the library's (i,k,j,m) order
template < uint32_t NC >
__device__ __forceinline__ double contract(const double * C, uint32_t i, uint32_t j,
                                           const vec3 & gI, const vec3 & gJ) {
  double sum = 0.0;
  for (uint32_t k = 0; k < 3; k++) {
    for (uint32_t m = 0; m < 3; m++) {
      if constexpr (NC == 1) {
        sum += gI[k] * C[k * 3 + m] * gJ[m];
      } else {
        sum += gI[k] * C[((i * 3 + k) * 3 + j) * 3 + m] * gJ[m];
      }
    }
  }
  return sum;
}

// same contraction, C read straight from global memory in layout L
template < uint32_t NC, QL L, uint32_t NQ, uint32_t DSZ >
__device__ __forceinline__ double contract_global(const KArgs & a, uint32_t e, uint32_t q,
                                                  uint32_t i, uint32_t j,
                                                  const vec3 & gI, const vec3 & gJ) {
  double sum = 0.0;
  for (uint32_t k = 0; k < 3; k++) {
    for (uint32_t m = 0; m < 3; m++) {
      uint32_t c = (NC == 1) ? (k * 3 + m) : (((i * 3 + k) * 3 + j) * 3 + m);
      sum += gI[k] * qread<L, NQ, DSZ>(a, e, q, c) * gJ[m];
    }
  }
  return sum;
}

////////////////////////////////////////////////////////////////////////////////
// element-block shared memory workspace + cooperative setup
////////////////////////////////////////////////////////////////////////////////

template < Geometry geom, uint32_t NC, uint32_t P, bool Phys >
struct BlockSmem {
  static constexpr uint32_t NN = nodes_per_element<geom, P>();
  static constexpr uint32_t NNX = nodes_per_element<geom, 1>();
  static constexpr uint32_t NQ = qpts_per_element<geom, P>();
  static constexpr uint32_t DSZ = NC * 3 * NC * 3;

  uint32_t ids[NN];
  double wJ[NQ];
  mat3 dxi[NQ];                    // dxi/dX per qpt
  vec3 xi[NQ];                     // qpt coordinates (for on-the-fly evaluation)
  vec3 dNdX[Phys ? NQ * NN : 1];   // physical gradients (Shared mode only)
  double D[NQ * DSZ];              // staged constitutive data
};

// stage_qfast controls the index order of the D staging loop:
//   false: thread strides c fastest (contiguous under QMajor)
//   true : thread strides q fastest (contiguous under SoA / EBlock)
template < Geometry geom, uint32_t NC, uint32_t P, bool Phys, QL L, bool stage_qfast, bool stage_D >
__device__ void block_setup(BlockSmem<geom, NC, P, Phys> & sm, const KArgs & a, uint32_t e) {
  constexpr uint32_t NN = sm.NN;
  constexpr uint32_t NNX = sm.NNX;
  constexpr uint32_t NQ = sm.NQ;
  constexpr uint32_t DSZ = sm.DSZ;

  const uint32_t tid = threadIdx.x;
  const uint32_t bs = blockDim.x;

  for (uint32_t i = tid; i < NN; i += bs) { sm.ids[i] = a.ids(e, i); }

  if constexpr (stage_D) {
    for (uint32_t idx = tid; idx < NQ * DSZ; idx += bs) {
      uint32_t q = stage_qfast ? (idx % NQ) : (idx / DSZ);
      uint32_t c = stage_qfast ? (idx / NQ) : (idx % DSZ);
      sm.D[q * DSZ + c] = qread<L, NQ, DSZ>(a, e, q, c);
    }
  }

  for (uint32_t q = tid; q < NQ; q += bs) {
    mat3 dX{};
    for (uint32_t i = 0; i < NNX; i++) {
      vec3 g = {a.GX(q, i, 0), a.GX(q, i, 1), a.GX(q, i, 2)};
      uint32_t xid = a.X_ids(e, i);
      for (uint32_t d = 0; d < 3; d++) { dX[d] += a.X(xid, d) * g; }
    }
    sm.dxi[q] = inv(dX);
    sm.wJ[q] = a.W(q) * det(dX);
    sm.xi[q] = {a.xi_pts(q, 0), a.xi_pts(q, 1), a.xi_pts(q, 2)};
    if constexpr (Phys) {
      for (uint32_t i = 0; i < NN; i++) {
        vec3 g = {a.G(q, i, 0), a.G(q, i, 1), a.G(q, i, 2)};
        sm.dNdX[q * NN + i] = dot(g, sm.dxi[q]);
      }
    }
  }
  __syncthreads();
}

template < SM M, Geometry geom, uint32_t NC, uint32_t P, bool Phys >
__device__ __forceinline__ vec3 get_dN(const BlockSmem<geom, NC, P, Phys> & sm, const KArgs & a,
                                       const FiniteElement<geom, Family::H1> & el,
                                       uint32_t q, uint32_t I) {
  constexpr uint32_t NN = sm.NN;
  if constexpr (M == SM::Shared) {
    return sm.dNdX[q * NN + I];
  } else if constexpr (M == SM::Global) {
    vec3 g = {a.G(q, I, 0), a.G(q, I, 1), a.G(q, I, 2)};
    return dot(g, sm.dxi[q]);
  } else {
    return dot(el.shape_function_gradient(sm.xi[q], I), sm.dxi[q]);
  }
}

////////////////////////////////////////////////////////////////////////////////
// experiment 1+2+3: the pair-per-thread kernel family
////////////////////////////////////////////////////////////////////////////////

// one element per blockDim.z slice, one (I,J) node pair per thread, an NCxNC
// block accumulated in registers, scattered with binary search + atomicAdd
template < Geometry geom, uint32_t NC, uint32_t P, QL L, bool stage_qfast, bool stage_D,
           SM TestM, SM TrialM, uint32_t E, bool UsePos = false >
__global__ void k_pair(KArgs a, FiniteElement<geom, Family::H1> el) {
  constexpr bool Phys = (TestM == SM::Shared) || (TrialM == SM::Shared);
  using Smem = BlockSmem<geom, NC, P, Phys>;
  constexpr uint32_t NN = Smem::NN;
  constexpr uint32_t NQ = Smem::NQ;
  constexpr uint32_t DSZ = Smem::DSZ;

  __shared__ Smem sm[E];
  const uint32_t z = threadIdx.z;
  uint32_t e = blockIdx.x * E + z;
  bool active = e < a.nelem;
  block_setup<geom, NC, P, Phys, L, stage_qfast, stage_D>(sm[z], a, active ? e : 0);
  if (!active) { return; }

  for (uint32_t pair = threadIdx.x; pair < NN * NN; pair += blockDim.x) {
    uint32_t I = pair / NN;
    uint32_t J = pair % NN;

    double B[NC][NC] = {};
    for (uint32_t q = 0; q < NQ; q++) {
      vec3 gI = get_dN<TestM>(sm[z], a, el, q, I);
      vec3 gJ = get_dN<TrialM>(sm[z], a, el, q, J);
      for (uint32_t i = 0; i < NC; i++) {
        for (uint32_t j = 0; j < NC; j++) {
          double c;
          if constexpr (stage_D) {
            c = contract<NC>(&sm[z].D[q * DSZ], i, j, gI, gJ);
          } else {
            c = contract_global<NC, L, NQ, DSZ>(a, e, q, i, j, gI, gJ);
          }
          B[i][j] += sm[z].wJ[q] * c;
        }
      }
    }

    for (uint32_t i = 0; i < NC; i++) {
      if constexpr (UsePos) {
        for (uint32_t j = 0; j < NC; j++) {
          atomicAdd(&a.values(csr_position<NC>(a, e, pair, sm[z].ids[I], i, j)), B[i][j]);
        }
      } else {
        int row = int(sm[z].ids[I] * NC + i);
        int row_start = a.row_ptr(row);
        int row_end = a.row_ptr(row + 1);
        for (uint32_t j = 0; j < NC; j++) {
          int col = int(sm[z].ids[J] * NC + j);
          int position = find_column_in_sparse_row(a.col_ind, row_start, row_end, col);
          atomicAdd(&a.values(position), B[i][j]);
        }
      }
    }
  }
}

// affine (p1 simplex) collapse: on a linear tet the jacobian and the basis
// gradients are constant over the element, so the quadrature loop moves into
// the qdata: A_e(I,i,J,j) = detJ * dN_I . (sum_q w_q C_q)(i,:,j,:) . dN_J.
// One contraction per pair instead of NQ -- the FFCx "geometry tensor" trick
// generalized to an arbitrary per-qpt tangent.
template < Geometry geom, uint32_t NC, uint32_t P, uint32_t E, bool UsePos >
__global__ void k_pair_affine(KArgs a) {
  static_assert(geom == Geometry::Tetrahedron && P == 1, "affine collapse assumes constant gradients");
  constexpr uint32_t NN = 4;
  constexpr uint32_t NQ = qpts_per_element<geom, P>();
  constexpr uint32_t DSZ = NC * 3 * NC * 3;

  __shared__ uint32_t s_ids[E][NN];
  __shared__ double s_dN[E][NN][3];
  __shared__ double s_detJ[E];
  __shared__ double s_D[E][DSZ];

  const uint32_t z = threadIdx.z;
  const uint32_t tid = threadIdx.x;
  const uint32_t tpe = blockDim.x;
  uint32_t e = blockIdx.x * E + z;
  bool active = e < a.nelem;
  uint32_t es = active ? e : 0;

  for (uint32_t i = tid; i < NN; i += tpe) { s_ids[z][i] = a.ids(es, i); }

  if (tid == 0) {
    mat3 dX{};
    for (uint32_t i = 0; i < NN; i++) {
      vec3 g = {a.GX(0, i, 0), a.GX(0, i, 1), a.GX(0, i, 2)};
      uint32_t xid = a.X_ids(es, i);
      for (uint32_t d = 0; d < 3; d++) { dX[d] += a.X(xid, d) * g; }
    }
    mat3 Ji = inv(dX);
    s_detJ[z] = det(dX);
    for (uint32_t i = 0; i < NN; i++) {
      vec3 g = {a.GX(0, i, 0), a.GX(0, i, 1), a.GX(0, i, 2)};
      vec3 dN = dot(g, Ji);
      for (uint32_t d = 0; d < 3; d++) { s_dN[z][i][d] = dN[d]; }
    }
  }

  for (uint32_t c = tid; c < DSZ; c += tpe) {
    double s = 0.0;
    for (uint32_t q = 0; q < NQ; q++) { s += a.W(q) * qread<QL::QMajor, NQ, DSZ>(a, es, q, c); }
    s_D[z][c] = s;
  }
  __syncthreads();

  if (!active) { return; }

  double detJ = s_detJ[z];
  for (uint32_t pair = tid; pair < NN * NN; pair += tpe) {
    uint32_t I = pair / NN;
    uint32_t J = pair % NN;
    vec3 gI = {s_dN[z][I][0], s_dN[z][I][1], s_dN[z][I][2]};
    vec3 gJ = {s_dN[z][J][0], s_dN[z][J][1], s_dN[z][J][2]};

    for (uint32_t i = 0; i < NC; i++) {
      int row = int(s_ids[z][I] * NC + i);
      int row_start = 0, row_end = 0;
      if constexpr (!UsePos) { row_start = a.row_ptr(row); row_end = a.row_ptr(row + 1); }
      for (uint32_t j = 0; j < NC; j++) {
        double v = detJ * contract<NC>(s_D[z], i, j, gI, gJ);
        int position;
        if constexpr (UsePos) {
          position = csr_position<NC>(a, e, pair, s_ids[z][I], i, j);
        } else {
          position = find_column_in_sparse_row(a.col_ind, row_start, row_end, int(s_ids[z][J] * NC + j));
        }
        atomicAdd(&a.values(position), v);
      }
    }
  }
}

// one scalar matrix entry (I*NC+i, J*NC+j) per thread
template < Geometry geom, uint32_t NC, uint32_t P >
__global__ void k_scalar(KArgs a, FiniteElement<geom, Family::H1> el) {
  using Smem = BlockSmem<geom, NC, P, true>;
  constexpr uint32_t NN = Smem::NN;
  constexpr uint32_t NQ = Smem::NQ;
  constexpr uint32_t DSZ = Smem::DSZ;

  __shared__ Smem sm;
  uint32_t e = blockIdx.x;
  if (e >= a.nelem) { return; }
  block_setup<geom, NC, P, true, QL::QMajor, false, true>(sm, a, e);

  for (uint32_t ent = threadIdx.x; ent < NN * NC * NN * NC; ent += blockDim.x) {
    uint32_t j = ent % NC;
    uint32_t J = (ent / NC) % NN;
    uint32_t i = (ent / (NC * NN)) % NC;
    uint32_t I = ent / (NC * NN * NC);

    double sum = 0.0;
    for (uint32_t q = 0; q < NQ; q++) {
      sum += sm.wJ[q] * contract<NC>(&sm.D[q * DSZ], i, j, sm.dNdX[q * NN + I], sm.dNdX[q * NN + J]);
    }

    int row = int(sm.ids[I] * NC + i);
    int col = int(sm.ids[J] * NC + j);
    int position = find_column_in_sparse_row(a.col_ind, a.row_ptr(row), a.row_ptr(row + 1), col);
    atomicAdd(&a.values(position), sum);
  }
}

// a whole element per thread, everything read from global memory
template < Geometry geom, uint32_t NC, uint32_t P, QL L >
__global__ void k_elem_thread(KArgs a) {
  constexpr uint32_t NN = nodes_per_element<geom, P>();
  constexpr uint32_t NNX = nodes_per_element<geom, 1>();
  constexpr uint32_t NQ = qpts_per_element<geom, P>();
  constexpr uint32_t DSZ = NC * 3 * NC * 3;

  uint32_t e = threadIdx.x + blockIdx.x * blockDim.x;
  if (e >= a.nelem) { return; }

  double wJ[NQ];
  mat3 dxi[NQ];
  for (uint32_t q = 0; q < NQ; q++) {
    mat3 dX{};
    for (uint32_t i = 0; i < NNX; i++) {
      vec3 g = {a.GX(q, i, 0), a.GX(q, i, 1), a.GX(q, i, 2)};
      uint32_t xid = a.X_ids(e, i);
      for (uint32_t d = 0; d < 3; d++) { dX[d] += a.X(xid, d) * g; }
    }
    dxi[q] = inv(dX);
    wJ[q] = a.W(q) * det(dX);
  }

  for (uint32_t I = 0; I < NN; I++) {
    int rows[NC];
    int row_starts[NC];
    int row_ends[NC];
    for (uint32_t i = 0; i < NC; i++) {
      rows[i] = int(a.ids(e, I) * NC + i);
      row_starts[i] = a.row_ptr(rows[i]);
      row_ends[i] = a.row_ptr(rows[i] + 1);
    }
    for (uint32_t J = 0; J < NN; J++) {
      double B[NC][NC] = {};
      for (uint32_t q = 0; q < NQ; q++) {
        vec3 gI = {a.G(q, I, 0), a.G(q, I, 1), a.G(q, I, 2)};
        vec3 gJ = {a.G(q, J, 0), a.G(q, J, 1), a.G(q, J, 2)};
        gI = dot(gI, dxi[q]);
        gJ = dot(gJ, dxi[q]);
        for (uint32_t i = 0; i < NC; i++) {
          for (uint32_t j = 0; j < NC; j++) {
            B[i][j] += wJ[q] * contract_global<NC, L, NQ, DSZ>(a, e, q, i, j, gI, gJ);
          }
        }
      }
      for (uint32_t i = 0; i < NC; i++) {
        for (uint32_t j = 0; j < NC; j++) {
          int col = int(a.ids(e, J) * NC + j);
          int position = find_column_in_sparse_row(a.col_ind, row_starts[i], row_ends[i], col);
          atomicAdd(&a.values(position), B[i][j]);
        }
      }
    }
  }
}

// affine residual (p1 tet): r_I(c) += detJ * dN_I . (sum_q w_q f_q(c,:)),
// one element per thread
template < uint32_t NC, uint32_t NQ >
__global__ void k_res_affine(KArgs a,
                             nd::view<const double, 3, gpu_mem> f,
                             nd::view<double, 2, gpu_mem> r) {
  constexpr uint32_t NN = 4;
  uint32_t e = threadIdx.x + blockIdx.x * blockDim.x;
  if (e >= a.nelem) { return; }

  mat3 dX{};
  for (uint32_t i = 0; i < NN; i++) {
    vec3 g = {a.GX(0, i, 0), a.GX(0, i, 1), a.GX(0, i, 2)};
    uint32_t xid = a.X_ids(e, i);
    for (uint32_t d = 0; d < 3; d++) { dX[d] += a.X(xid, d) * g; }
  }
  mat3 Ji = inv(dX);
  double detJ = det(dX);

  double fh[NC][3] = {};
  for (uint32_t q = 0; q < NQ; q++) {
    double w = a.W(q);
    for (uint32_t c = 0; c < NC; c++) {
      for (uint32_t d = 0; d < 3; d++) { fh[c][d] += w * f(e * NQ + q, c, d); }
    }
  }

  for (uint32_t I = 0; I < NN; I++) {
    vec3 g = {a.G(0, I, 0), a.G(0, I, 1), a.G(0, I, 2)};
    vec3 dN = dot(g, Ji);
    for (uint32_t c = 0; c < NC; c++) {
      double sum = dN[0] * fh[c][0] + dN[1] * fh[c][1] + dN[2] * fh[c][2];
      atomicAdd(&r(a.ids(e, I), c), detJ * sum);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// experiment 2b: constitutive data via callback functor
////////////////////////////////////////////////////////////////////////////////

// reads a compact per-qpt state (du_dX) and evaluates the tangent on the fly
template < uint32_t NC, typename Model >
struct StateFunctor {
  const double * state;   // (nq, NC*3) flattened
  Model model;
  __device__ void eval(uint32_t qid, double * out) const {
    grad_t<NC> g;
    const double * src = state + qid * NC * 3;
    double * gp = reinterpret_cast<double *>(&g);
    for (uint32_t c = 0; c < NC * 3; c++) { gp[c] = src[c]; }
    jac_t<NC> D = model(g);
    const double * Dp = reinterpret_cast<const double *>(&D);
    for (uint32_t c = 0; c < NC * 3 * NC * 3; c++) { out[c] = Dp[c]; }
  }
};

// evaluates a state-independent tangent from material constants alone
template < uint32_t NC, typename Model >
struct ConstFunctor {
  Model model;
  __device__ void eval(uint32_t /*qid*/, double * out) const {
    jac_t<NC> D = model(grad_t<NC>{});
    const double * Dp = reinterpret_cast<const double *>(&D);
    for (uint32_t c = 0; c < NC * 3 * NC * 3; c++) { out[c] = Dp[c]; }
  }
};

template < Geometry geom, uint32_t NC, uint32_t P, typename QF >
__global__ void k_pair_functor(KArgs a, QF qf) {
  using Smem = BlockSmem<geom, NC, P, true>;
  constexpr uint32_t NN = Smem::NN;
  constexpr uint32_t NQ = Smem::NQ;
  constexpr uint32_t DSZ = Smem::DSZ;

  __shared__ Smem sm;
  uint32_t e = blockIdx.x;
  if (e >= a.nelem) { return; }
  block_setup<geom, NC, P, true, QL::QMajor, false, false>(sm, a, e);

  for (uint32_t q = threadIdx.x; q < NQ; q += blockDim.x) {
    qf.eval(e * NQ + q, &sm.D[q * DSZ]);
  }
  __syncthreads();

  for (uint32_t pair = threadIdx.x; pair < NN * NN; pair += blockDim.x) {
    uint32_t I = pair / NN;
    uint32_t J = pair % NN;
    double B[NC][NC] = {};
    for (uint32_t q = 0; q < NQ; q++) {
      for (uint32_t i = 0; i < NC; i++) {
        for (uint32_t j = 0; j < NC; j++) {
          B[i][j] += sm.wJ[q] * contract<NC>(&sm.D[q * DSZ], i, j, sm.dNdX[q * NN + I], sm.dNdX[q * NN + J]);
        }
      }
    }
    for (uint32_t i = 0; i < NC; i++) {
      int row = int(sm.ids[I] * NC + i);
      int row_start = a.row_ptr(row);
      int row_end = a.row_ptr(row + 1);
      for (uint32_t j = 0; j < NC; j++) {
        int col = int(sm.ids[J] * NC + j);
        int position = find_column_in_sparse_row(a.col_ind, row_start, row_end, col);
        atomicAdd(&a.values(position), B[i][j]);
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// experiment 1e: warp-per-row assembly (atomic-free)
////////////////////////////////////////////////////////////////////////////////

// pass A: per-(element,qpt) geometry to global memory
template < Geometry geom, uint32_t P >
__global__ void k_geometry(KArgs a,
                           nd::view<double, 1, gpu_mem> gWJ,
                           nd::view<double, 2, gpu_mem> gJi) {
  constexpr uint32_t NNX = nodes_per_element<geom, 1>();
  constexpr uint32_t NQ = qpts_per_element<geom, P>();

  uint32_t t = threadIdx.x + blockIdx.x * blockDim.x;
  if (t >= a.nelem * NQ) { return; }
  uint32_t e = t / NQ;
  uint32_t q = t % NQ;

  mat3 dX{};
  for (uint32_t i = 0; i < NNX; i++) {
    vec3 g = {a.GX(q, i, 0), a.GX(q, i, 1), a.GX(q, i, 2)};
    uint32_t xid = a.X_ids(e, i);
    for (uint32_t d = 0; d < 3; d++) { dX[d] += a.X(xid, d) * g; }
  }
  mat3 Ji = inv(dX);
  gWJ(t) = a.W(q) * det(dX);
  for (uint32_t d = 0; d < 9; d++) { gJi(t, d) = Ji[d / 3][d % 3]; }
}

// pass B: one warp per node row; lanes parallelize over trial nodes; row
// accumulators live in shared memory and are written back without atomics.
// For NC=3 the CSR is addressed through its block (BSR) structure:
// position(i,j) = row_ptr[3r+i] + 3k + j, k = block-column index in the row.
template < Geometry geom, uint32_t NC, uint32_t P >
__global__ void k_row_warp(KArgs a,
                           nd::view<const int, 1, gpu_mem> node_rowptr,
                           nd::view<const int, 1, gpu_mem> node_cols,
                           nd::view<const int, 1, gpu_mem> adj_off,
                           nd::view<const uint32_t, 1, gpu_mem> adj_ent,
                           nd::view<const double, 1, gpu_mem> gWJ,
                           nd::view<const double, 2, gpu_mem> gJi,
                           uint32_t num_node_rows,
                           uint32_t maxlen) {
  constexpr uint32_t NN = nodes_per_element<geom, P>();
  constexpr uint32_t NQ = qpts_per_element<geom, P>();
  constexpr uint32_t DSZ = NC * 3 * NC * 3;
  static_assert(NN <= 32, "warp-per-row kernel assumes NN <= warp size");

  extern __shared__ double row_acc[];
  const uint32_t warp = threadIdx.y;
  const uint32_t lane = threadIdx.x;
  double * acc = row_acc + warp * maxlen * NC * NC;

  uint32_t r = blockIdx.x * blockDim.y + warp;
  if (r >= num_node_rows) { return; }

  int nrow_start = node_rowptr(r);
  int nrow_end = node_rowptr(r + 1);
  uint32_t len = nrow_end - nrow_start;

  for (uint32_t idx = lane; idx < len * NC * NC; idx += 32) { acc[idx] = 0.0; }
  __syncwarp();

  for (int adj = adj_off(r); adj < adj_off(r + 1); adj++) {
    uint32_t ent = adj_ent(adj);
    uint32_t e = ent >> 5;
    uint32_t I = ent & 31;

    for (uint32_t J = lane; J < NN; J += 32) {
      double B[NC][NC] = {};
      for (uint32_t q = 0; q < NQ; q++) {
        uint32_t t = e * NQ + q;
        mat3 Ji;
        for (uint32_t d = 0; d < 9; d++) { Ji[d / 3][d % 3] = gJi(t, d); }
        vec3 gI = {a.G(q, I, 0), a.G(q, I, 1), a.G(q, I, 2)};
        vec3 gJ = {a.G(q, J, 0), a.G(q, J, 1), a.G(q, J, 2)};
        gI = dot(gI, Ji);
        gJ = dot(gJ, Ji);
        double wJ = gWJ(t);
        for (uint32_t i = 0; i < NC; i++) {
          for (uint32_t j = 0; j < NC; j++) {
            B[i][j] += wJ * contract_global<NC, QL::QMajor, NQ, DSZ>(a, e, q, i, j, gI, gJ);
          }
        }
      }

      // block-column index of trial node J within this node row
      int c = int(a.ids(e, J));
      int lo = nrow_start, hi = nrow_end;
      while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (node_cols(mid) < c) { lo = mid + 1; } else { hi = mid; }
      }
      uint32_t k = lo - nrow_start;
      for (uint32_t i = 0; i < NC; i++) {
        for (uint32_t j = 0; j < NC; j++) {
          acc[(k * NC + i) * NC + j] += B[i][j];
        }
      }
    }
    __syncwarp();
  }

  for (uint32_t idx = lane; idx < len * NC * NC; idx += 32) {
    uint32_t k = idx / (NC * NC);
    uint32_t i = (idx / NC) % NC;
    uint32_t j = idx % NC;
    int position = a.row_ptr(r * NC + i) + int(k * NC + j);
    a.values(position) = acc[(k * NC + i) * NC + j];
  }
}

////////////////////////////////////////////////////////////////////////////////
// experiment 4: two-pass assembly
////////////////////////////////////////////////////////////////////////////////

// pass 1: element matrices to global memory
//   ij_major : A_e[(e*NCNC + ij)*NN*NN + pair]  (coalesced stores across pairs)
//   pair_major: A_e[(e*NN*NN + pair)*NCNC + ij] (contiguous per pair)
template < Geometry geom, uint32_t NC, uint32_t P, bool ij_major >
__global__ void k_pass1(KArgs a, nd::view<double, 1, gpu_mem> A_e) {
  using Smem = BlockSmem<geom, NC, P, true>;
  constexpr uint32_t NN = Smem::NN;
  constexpr uint32_t NQ = Smem::NQ;
  constexpr uint32_t DSZ = Smem::DSZ;
  constexpr uint32_t NCNC = NC * NC;

  __shared__ Smem sm;
  uint32_t e = blockIdx.x;
  if (e >= a.nelem) { return; }
  block_setup<geom, NC, P, true, QL::QMajor, false, true>(sm, a, e);

  for (uint32_t pair = threadIdx.x; pair < NN * NN; pair += blockDim.x) {
    uint32_t I = pair / NN;
    uint32_t J = pair % NN;
    double B[NC][NC] = {};
    for (uint32_t q = 0; q < NQ; q++) {
      for (uint32_t i = 0; i < NC; i++) {
        for (uint32_t j = 0; j < NC; j++) {
          B[i][j] += sm.wJ[q] * contract<NC>(&sm.D[q * DSZ], i, j, sm.dNdX[q * NN + I], sm.dNdX[q * NN + J]);
        }
      }
    }
    for (uint32_t ij = 0; ij < NCNC; ij++) {
      uint32_t idx = ij_major ? (e * NCNC + ij) * NN * NN + pair
                              : (e * NN * NN + pair) * NCNC + ij;
      A_e(idx) = B[ij / NC][ij % NC];
    }
  }
}

// experiment 8: applying element_matrix.cu's write-bound design to pass 1.
// The toy (repo root element_matrix.cu) shows that pair-per-thread blocks
// with component-major / pair-minor (ij-major) stores write K_e at ~82% of
// copy bandwidth when the integrand is trivial. k_pass1 already has exactly
// that mapping and layout, so these variants strip it down to locate where
// the general kernel falls off that ceiling:
//   k_pass1_writes : the ij-major stores alone (the write floor at this
//                    launch shape)
//   k_pass1_staged : block_setup + stores, quadrature contraction dropped
//   k_pass1_affine : the p1-simplex collapse (one contraction per pair, no
//                    quadrature loop, no dof ids) with dense ij-major stores
//                    -- real physics restructured into the toy's regime
template < Geometry geom, uint32_t NC, uint32_t P >
__global__ void k_pass1_writes(nd::view<double, 1, gpu_mem> A_e) {
  constexpr uint32_t NN = nodes_per_element<geom, P>();
  const uint32_t e = blockIdx.x;
  for (uint32_t pair = threadIdx.x; pair < NN * NN; pair += blockDim.x) {
    for (uint32_t ij = 0; ij < NC * NC; ij++) {
      A_e((e * NC * NC + ij) * NN * NN + pair) = 1.0;
    }
  }
}

template < Geometry geom, uint32_t NC, uint32_t P >
__global__ void k_pass1_staged(KArgs a, nd::view<double, 1, gpu_mem> A_e) {
  using Smem = BlockSmem<geom, NC, P, true>;
  constexpr uint32_t NN = Smem::NN;
  constexpr uint32_t NQ = Smem::NQ;
  constexpr uint32_t DSZ = Smem::DSZ;

  __shared__ Smem sm;
  const uint32_t e = blockIdx.x;
  if (e >= a.nelem) { return; }
  block_setup<geom, NC, P, true, QL::QMajor, false, true>(sm, a, e);

  for (uint32_t pair = threadIdx.x; pair < NN * NN; pair += blockDim.x) {
    double keep = sm.wJ[pair % NQ] + sm.D[pair % (NQ * DSZ)] + sm.dNdX[pair % (NQ * NN)][0];
    double B = 1.0 + 0.0 * keep;
    for (uint32_t ij = 0; ij < NC * NC; ij++) {
      A_e((e * NC * NC + ij) * NN * NN + pair) = B;
    }
  }
}

template < uint32_t NC, uint32_t E >
__global__ void k_pass1_affine(KArgs a, nd::view<double, 1, gpu_mem> A_e) {
  constexpr uint32_t NN = 4;
  constexpr uint32_t NQ = qpts_per_element<Geometry::Tetrahedron, 1>();
  constexpr uint32_t DSZ = NC * 3 * NC * 3;
  constexpr uint32_t NCNC = NC * NC;

  __shared__ double s_dN[E][NN][3];
  __shared__ double s_detJ[E];
  __shared__ double s_D[E][DSZ];

  const uint32_t z = threadIdx.z;
  const uint32_t tid = threadIdx.x;
  const uint32_t tpe = blockDim.x;
  uint32_t e = blockIdx.x * E + z;
  bool active = e < a.nelem;
  uint32_t es = active ? e : 0;

  if (tid == 0) {
    mat3 dX{};
    for (uint32_t i = 0; i < NN; i++) {
      vec3 g = {a.GX(0, i, 0), a.GX(0, i, 1), a.GX(0, i, 2)};
      uint32_t xid = a.X_ids(es, i);
      for (uint32_t d = 0; d < 3; d++) { dX[d] += a.X(xid, d) * g; }
    }
    mat3 Ji = inv(dX);
    s_detJ[z] = det(dX);
    for (uint32_t i = 0; i < NN; i++) {
      vec3 g = {a.GX(0, i, 0), a.GX(0, i, 1), a.GX(0, i, 2)};
      vec3 dN = dot(g, Ji);
      for (uint32_t d = 0; d < 3; d++) { s_dN[z][i][d] = dN[d]; }
    }
  }

  for (uint32_t c = tid; c < DSZ; c += tpe) {
    double s = 0.0;
    for (uint32_t q = 0; q < NQ; q++) { s += a.W(q) * qread<QL::QMajor, NQ, DSZ>(a, es, q, c); }
    s_D[z][c] = s;
  }
  __syncthreads();

  if (!active) { return; }

  double detJ = s_detJ[z];
  for (uint32_t pair = tid; pair < NN * NN; pair += tpe) {
    uint32_t I = pair / NN;
    uint32_t J = pair % NN;
    vec3 gI = {s_dN[z][I][0], s_dN[z][I][1], s_dN[z][I][2]};
    vec3 gJ = {s_dN[z][J][0], s_dN[z][J][1], s_dN[z][J][2]};
    for (uint32_t i = 0; i < NC; i++) {
      for (uint32_t j = 0; j < NC; j++) {
        A_e((e * NCNC + i * NC + j) * NN * NN + pair) = detJ * contract<NC>(s_D[z], i, j, gI, gJ);
      }
    }
  }
}

// pass 2, NC=1: one thread per CSR entry, summing its element contributions
__global__ void k_gather_scalar(nd::view<double, 1, gpu_mem> values,
                                nd::view<const int, 1, gpu_mem> gmap_off,
                                nd::view<const uint32_t, 1, gpu_mem> gmap_ent,
                                nd::view<const double, 1, gpu_mem> A_e,
                                uint32_t nnz, uint32_t NN2) {
  uint32_t t = threadIdx.x + blockIdx.x * blockDim.x;
  if (t >= nnz) { return; }
  double sum = 0.0;
  for (int g = gmap_off(t); g < gmap_off(t + 1); g++) {
    uint32_t ent = gmap_ent(g);
    sum += A_e((ent >> 10) * NN2 + (ent & 1023));
  }
  values(t) = sum;
}

// pass 2, NC=3, BSR-aware: 9 threads share one block's gather list (broadcast),
// each handling one (i,j) component; dof-to-element info is looked up once per
// 3x3 block instead of once per scalar entry
template < bool ij_major >
__global__ void k_gather_block9(nd::view<double, 1, gpu_mem> values,
                                nd::view<const int, 1, gpu_mem> row_ptr,
                                nd::view<const int, 1, gpu_mem> gmap_off,
                                nd::view<const uint32_t, 1, gpu_mem> gmap_ent,
                                nd::view<const uint32_t, 1, gpu_mem> blk_row,
                                nd::view<const uint32_t, 1, gpu_mem> blk_k,
                                nd::view<const double, 1, gpu_mem> A_e,
                                uint32_t nblocks, uint32_t NN2) {
  uint32_t t = threadIdx.x + blockIdx.x * blockDim.x;
  if (t >= nblocks * 9) { return; }
  uint32_t b = t / 9;
  uint32_t ij = t % 9;
  double sum = 0.0;
  for (int g = gmap_off(b); g < gmap_off(b + 1); g++) {
    uint32_t ent = gmap_ent(g);
    uint32_t e = ent >> 10;
    uint32_t pair = ent & 1023;
    uint32_t idx = ij_major ? (e * 9 + ij) * NN2 + pair : (e * NN2 + pair) * 9 + ij;
    sum += A_e(idx);
  }
  uint32_t r = blk_row(b);
  uint32_t k = blk_k(b);
  values(row_ptr(r * 3 + ij / 3) + int(3 * k + ij % 3)) = sum;
}

// pass 2, NC=3: one thread per block, 9 accumulators
template < bool ij_major >
__global__ void k_gather_block1(nd::view<double, 1, gpu_mem> values,
                                nd::view<const int, 1, gpu_mem> row_ptr,
                                nd::view<const int, 1, gpu_mem> gmap_off,
                                nd::view<const uint32_t, 1, gpu_mem> gmap_ent,
                                nd::view<const uint32_t, 1, gpu_mem> blk_row,
                                nd::view<const uint32_t, 1, gpu_mem> blk_k,
                                nd::view<const double, 1, gpu_mem> A_e,
                                uint32_t nblocks, uint32_t NN2) {
  uint32_t b = threadIdx.x + blockIdx.x * blockDim.x;
  if (b >= nblocks) { return; }
  double sum[9] = {};
  for (int g = gmap_off(b); g < gmap_off(b + 1); g++) {
    uint32_t ent = gmap_ent(g);
    uint32_t e = ent >> 10;
    uint32_t pair = ent & 1023;
    for (uint32_t ij = 0; ij < 9; ij++) {
      uint32_t idx = ij_major ? (e * 9 + ij) * NN2 + pair : (e * NN2 + pair) * 9 + ij;
      sum[ij] += A_e(idx);
    }
  }
  uint32_t r = blk_row(b);
  uint32_t k = blk_k(b);
  for (uint32_t ij = 0; ij < 9; ij++) {
    values(row_ptr(r * 3 + ij / 3) + int(3 * k + ij % 3)) = sum[ij];
  }
}

////////////////////////////////////////////////////////////////////////////////
// experiment 7: access-pattern skeletons
////////////////////////////////////////////////////////////////////////////////

// the pair kernel with its arithmetic removed, leaving only the data motion,
// to establish whether the assembly's access pattern can go fast at all:
//
//   k_pair_skel_traffic : block_setup unchanged (ids, qdata, geometry and the
//                         physical gradients staged exactly as in k_pair), the
//                         quadrature contraction dropped -- every global load
//                         and store of the real kernel, none of the pair flops
//   k_pair_skel_incr    : ids only, then the same per-entry scatter -- the CSR
//                         output pattern in isolation (UsePos = true replaces
//                         the binary search with the precomputed block-column
//                         index, separating search cost from atomic conflicts)
//
// both increment by exactly 1.0, so each entry of the fill ends up holding its
// own element multiplicity, which the driver verifies from the host gather map

template < Geometry geom, uint32_t NC, uint32_t P >
__global__ void k_pair_skel_traffic(KArgs a) {
  using Smem = BlockSmem<geom, NC, P, true>;
  constexpr uint32_t NN = Smem::NN;
  constexpr uint32_t NQ = Smem::NQ;
  constexpr uint32_t DSZ = Smem::DSZ;

  __shared__ Smem sm;
  const uint32_t e = blockIdx.x;
  block_setup<geom, NC, P, true, QL::QMajor, false, true>(sm, a, e);

  for (uint32_t pair = threadIdx.x; pair < NN * NN; pair += blockDim.x) {
    uint32_t I = pair / NN;
    uint32_t J = pair % NN;

    // touch every staged array through a data-dependent index so the staging
    // loads survive; 0.0 * keep is not foldable under strict IEEE arithmetic,
    // so the increment stays exactly 1.0 without the reads being elided
    double keep = sm.wJ[pair % NQ] + sm.D[pair % (NQ * DSZ)] + sm.dNdX[pair % (NQ * NN)][0];
    double B = 1.0 + 0.0 * keep;

    for (uint32_t i = 0; i < NC; i++) {
      int row = int(sm.ids[I] * NC + i);
      int row_start = a.row_ptr(row);
      int row_end = a.row_ptr(row + 1);
      for (uint32_t j = 0; j < NC; j++) {
        int col = int(sm.ids[J] * NC + j);
        int position = find_column_in_sparse_row(a.col_ind, row_start, row_end, col);
        atomicAdd(&a.values(position), B);
      }
    }
  }
}

// the floor: one (element, pair) per thread over a flat grid, no shared
// memory, no sync, no block structure -- each thread loads only its two node
// ids, locates its NC x NC block in the CSR, and adds 1
template < Geometry geom, uint32_t NC, uint32_t P, bool UsePos >
__global__ void k_skel_flat(KArgs a) {
  constexpr uint32_t NN = nodes_per_element<geom, P>();

  uint32_t t = threadIdx.x + blockIdx.x * blockDim.x;
  if (t >= a.nelem * NN * NN) { return; }
  uint32_t e = t / (NN * NN);
  uint32_t pair = t % (NN * NN);
  uint32_t row_node = a.ids(e, pair / NN);
  uint32_t col_node = a.ids(e, pair % NN);

  for (uint32_t i = 0; i < NC; i++) {
    if constexpr (UsePos) {
      for (uint32_t j = 0; j < NC; j++) {
        atomicAdd(&a.values(csr_position<NC>(a, e, pair, row_node, i, j)), 1.0);
      }
    } else {
      int row = int(row_node * NC + i);
      int row_start = a.row_ptr(row);
      int row_end = a.row_ptr(row + 1);
      for (uint32_t j = 0; j < NC; j++) {
        int col = int(col_node * NC + j);
        int position = find_column_in_sparse_row(a.col_ind, row_start, row_end, col);
        atomicAdd(&a.values(position), 1.0);
      }
    }
  }
}

// which input is the problem? the traffic skeleton with each staged input
// individually removable, while the full BlockSmem footprint stays declared
// (anchored by ids) so every variant runs at skel:traffic's occupancy:
//   StageD    : the qdata staging loop
//   StageGeom : the per-qpt geometry (X gather, GX table, inv/det, dNdX)
// (0,0) isolates the occupancy cost of the footprint itself relative to
// k_pair_skel_incr; adding one flag at a time prices each input stream.
template < Geometry geom, uint32_t NC, uint32_t P, bool StageD, bool StageGeom >
__global__ void k_pair_skel_stage(KArgs a) {
  using Smem = BlockSmem<geom, NC, P, true>;
  constexpr uint32_t NN = Smem::NN;
  constexpr uint32_t NNX = Smem::NNX;
  constexpr uint32_t NQ = Smem::NQ;
  constexpr uint32_t DSZ = Smem::DSZ;

  __shared__ Smem sm;
  const uint32_t e = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  const uint32_t bs = blockDim.x;

  for (uint32_t i = tid; i < NN; i += bs) { sm.ids[i] = a.ids(e, i); }

  if constexpr (StageD) {
    for (uint32_t idx = tid; idx < NQ * DSZ; idx += bs) {
      sm.D[idx] = qread<QL::QMajor, NQ, DSZ>(a, e, idx / DSZ, idx % DSZ);
    }
  }

  if constexpr (StageGeom) {
    for (uint32_t q = tid; q < NQ; q += bs) {
      mat3 dX{};
      for (uint32_t i = 0; i < NNX; i++) {
        vec3 g = {a.GX(q, i, 0), a.GX(q, i, 1), a.GX(q, i, 2)};
        uint32_t xid = a.X_ids(e, i);
        for (uint32_t d = 0; d < 3; d++) { dX[d] += a.X(xid, d) * g; }
      }
      sm.dxi[q] = inv(dX);
      sm.wJ[q] = a.W(q) * det(dX);
      sm.xi[q] = {a.xi_pts(q, 0), a.xi_pts(q, 1), a.xi_pts(q, 2)};
      for (uint32_t i = 0; i < NN; i++) {
        vec3 g = {a.G(q, i, 0), a.G(q, i, 1), a.G(q, i, 2)};
        sm.dNdX[q * NN + i] = dot(g, sm.dxi[q]);
      }
    }
  }
  __syncthreads();

  for (uint32_t pair = threadIdx.x; pair < NN * NN; pair += blockDim.x) {
    uint32_t I = pair / NN;
    uint32_t J = pair % NN;

    double keep = 0.0;
    if constexpr (StageD) { keep += sm.D[pair % (NQ * DSZ)]; }
    if constexpr (StageGeom) { keep += sm.wJ[pair % NQ] + sm.dNdX[pair % (NQ * NN)][0]; }
    double B = 1.0 + 0.0 * keep;

    for (uint32_t i = 0; i < NC; i++) {
      int row = int(sm.ids[I] * NC + i);
      int row_start = a.row_ptr(row);
      int row_end = a.row_ptr(row + 1);
      for (uint32_t j = 0; j < NC; j++) {
        int col = int(sm.ids[J] * NC + j);
        int position = find_column_in_sparse_row(a.col_ind, row_start, row_end, col);
        atomicAdd(&a.values(position), B);
      }
    }
  }
}

template < Geometry geom, uint32_t NC, uint32_t P, bool UsePos >
__global__ void k_pair_skel_incr(KArgs a) {
  constexpr uint32_t NN = nodes_per_element<geom, P>();

  __shared__ uint32_t s_ids[NN];
  const uint32_t e = blockIdx.x;
  for (uint32_t i = threadIdx.x; i < NN; i += blockDim.x) { s_ids[i] = a.ids(e, i); }
  __syncthreads();

  for (uint32_t pair = threadIdx.x; pair < NN * NN; pair += blockDim.x) {
    uint32_t I = pair / NN;
    uint32_t J = pair % NN;
    for (uint32_t i = 0; i < NC; i++) {
      if constexpr (UsePos) {
        for (uint32_t j = 0; j < NC; j++) {
          atomicAdd(&a.values(csr_position<NC>(a, e, pair, s_ids[I], i, j)), 1.0);
        }
      } else {
        int row = int(s_ids[I] * NC + i);
        int row_start = a.row_ptr(row);
        int row_end = a.row_ptr(row + 1);
        for (uint32_t j = 0; j < NC; j++) {
          int col = int(s_ids[J] * NC + j);
          int position = find_column_in_sparse_row(a.col_ind, row_start, row_end, col);
          atomicAdd(&a.values(position), 1.0);
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// layout conversion kernels (untimed setup)
////////////////////////////////////////////////////////////////////////////////

__global__ void k_to_soa(uint32_t nq, uint32_t dsz,
                         nd::view<const double, 2, gpu_mem> src, nd::view<double, 2, gpu_mem> dst) {
  uint32_t t = threadIdx.x + blockIdx.x * blockDim.x;
  if (t >= nq * dsz) { return; }
  uint32_t c = t / nq;
  uint32_t q = t % nq;
  dst(c, q) = src(q, c);
}

__global__ void k_to_eblock(uint32_t nelem, uint32_t NQ, uint32_t dsz,
                            nd::view<const double, 2, gpu_mem> src, nd::view<double, 2, gpu_mem> dst) {
  uint32_t t = threadIdx.x + blockIdx.x * blockDim.x;
  if (t >= nelem * NQ * dsz) { return; }
  uint32_t e = t / (NQ * dsz);
  uint32_t rem = t % (NQ * dsz);
  uint32_t c = rem / NQ;
  uint32_t q = rem % NQ;
  dst(e, c * NQ + q) = src(e * NQ + q, c);
}

__global__ void k_to_esoa(uint32_t nelem, uint32_t NQ, uint32_t dsz,
                          nd::view<const double, 2, gpu_mem> src, nd::view<double, 2, gpu_mem> dst) {
  uint32_t t = threadIdx.x + blockIdx.x * blockDim.x;
  if (t >= nelem * NQ * dsz) { return; }
  uint32_t e = t / (NQ * dsz);
  uint32_t rem = t % (NQ * dsz);
  uint32_t c = rem / NQ;
  uint32_t q = rem % NQ;
  dst(c * NQ + q, e) = src(e * NQ + q, c);
}

////////////////////////////////////////////////////////////////////////////////
// timing
////////////////////////////////////////////////////////////////////////////////

template < typename callable >
static double bench_gpu(callable && f, int warmup, int reps) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  for (int i = 0; i < warmup; i++) { f(); }
  CUDA_CHECK(cudaDeviceSynchronize());
  double best = 1.0e30;
  for (int i = 0; i < reps; i++) {
    CUDA_CHECK(cudaEventRecord(start));
    f();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    best = std::min(best, double(ms) * 1.0e-3);
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return best;
}

////////////////////////////////////////////////////////////////////////////////
// host-side maps (dof ids, node graph, adjacency, gather map)
////////////////////////////////////////////////////////////////////////////////

template < typename T >
nd::array<T, 1, cpu_mem> to_array(const std::vector<T> & v) {
  nd::array<T, 1, cpu_mem> a({uint32_t(v.size())});
  for (uint32_t i = 0; i < v.size(); i++) { a(i) = v[i]; }
  return a;
}

struct HostMaps {
  nd::array<uint32_t, 2, cpu_mem> ids, X_ids;
  std::vector<int> node_rowptr;        // node-graph CSR (== matrix CSR when NC=1)
  std::vector<int> node_cols;
  std::vector<int> adj_off;            // node -> (element, local node) adjacency
  std::vector<uint32_t> adj_ent;       // packed e<<5 | I
  std::vector<int> gmap_off;           // node-pair block -> (element, pair) gather map
  std::vector<uint32_t> gmap_ent;      // packed e<<10 | I*NN+J
  std::vector<uint32_t> blk_row, blk_k;
  nd::array<uint16_t, 2, cpu_mem> pair_k;   // (nelem, NN*NN) block-column index per pair
  uint32_t max_row_len = 0;
  double build_seconds = 0.0;
};

template < Geometry geom, uint32_t NC, uint32_t P >
HostMaps build_host_maps(const Domain<> & h_domain, const Field<Family::H1> & h_u,
                         const nd::array<int, 1, cpu_mem> & h_rowptr,
                         const nd::array<int, 1, cpu_mem> & h_colind) {
  constexpr uint32_t NN = nodes_per_element<geom, P>();
  constexpr uint32_t NNX = nodes_per_element<geom, 1>();

  femto::timer stopwatch;
  stopwatch.start();

  HostMaps m;
  auto conn = h_domain.mesh[geom];
  uint32_t nelem = conn.shape[0];
  FEMTO_ASSERT(nelem < (1u << 22), "element id packing assumes < 2^22 elements");
  FEMTO_ASSERT(NN <= 32 && NN * NN < 1024, "local index packing assumes small elements");

  FiniteElement<geom, Family::H1> el{P};
  FiniteElement<geom, Family::H1> X_el{1};
  m.ids = nd::array<uint32_t, 2, cpu_mem>({nelem, NN});
  m.X_ids = nd::array<uint32_t, 2, cpu_mem>({nelem, NNX});
  for (uint32_t e = 0; e < nelem; e++) {
    el.indices(h_u.offsets, &conn(e, 0), &m.ids(e, 0));
    X_el.indices(h_domain.mesh.X.offsets, &conn(e, 0), &m.X_ids(e, 0));
  }

  // node graph: for NC=1 it is the matrix graph; for NC=3, extract it from the
  // dof graph and verify the block (BSR) ordering assumption while doing so
  uint32_t nrows = h_rowptr.size() - 1;
  uint32_t num_nodes = nrows / NC;
  m.node_rowptr.assign(num_nodes + 1, 0);
  if constexpr (NC == 1) {
    m.node_rowptr.resize(num_nodes + 1);
    for (uint32_t r = 0; r <= num_nodes; r++) { m.node_rowptr[r] = h_rowptr(r); }
    m.node_cols.resize(h_colind.size());
    for (uint32_t i = 0; i < h_colind.size(); i++) { m.node_cols[i] = h_colind(i); }
  } else {
    for (uint32_t r = 0; r < num_nodes; r++) {
      int start = h_rowptr(r * NC);
      int end = h_rowptr(r * NC + 1);
      FEMTO_ASSERT((end - start) % NC == 0, "dof row length is not a multiple of NC");
      uint32_t len = (end - start) / NC;
      m.node_rowptr[r + 1] = m.node_rowptr[r] + len;
      for (uint32_t k = 0; k < len; k++) {
        int c = h_colind(start + int(k * NC));
        FEMTO_ASSERT(c % NC == 0, "block structure violated: first column of group not a block start");
        m.node_cols.push_back(c / NC);
        // verify position(i,j) = row_ptr[NC*r+i] + NC*k + j for the whole block
        for (uint32_t i = 0; i < NC; i++) {
          for (uint32_t j = 0; j < NC; j++) {
            FEMTO_ASSERT(h_colind(h_rowptr(r * NC + i) + int(k * NC + j)) == c + int(j),
                         "block structure violated: BSR position formula does not hold");
          }
        }
      }
    }
  }

  for (uint32_t r = 0; r < num_nodes; r++) {
    m.max_row_len = std::max(m.max_row_len, uint32_t(m.node_rowptr[r + 1] - m.node_rowptr[r]));
  }

  // node -> (element, local node) adjacency
  m.adj_off.assign(num_nodes + 1, 0);
  for (uint32_t e = 0; e < nelem; e++) {
    for (uint32_t I = 0; I < NN; I++) { m.adj_off[m.ids(e, I) + 1]++; }
  }
  for (uint32_t r = 0; r < num_nodes; r++) { m.adj_off[r + 1] += m.adj_off[r]; }
  m.adj_ent.resize(m.adj_off[num_nodes]);
  {
    std::vector<int> cursor(m.adj_off.begin(), m.adj_off.end() - 1);
    for (uint32_t e = 0; e < nelem; e++) {
      for (uint32_t I = 0; I < NN; I++) {
        m.adj_ent[cursor[m.ids(e, I)]++] = (e << 5) | I;
      }
    }
  }

  // node-pair block -> (element, pair) gather map
  uint32_t nblocks = m.node_rowptr[num_nodes];
  auto block_of = [&](uint32_t r, int c) {
    const int * begin = m.node_cols.data() + m.node_rowptr[r];
    const int * end = m.node_cols.data() + m.node_rowptr[r + 1];
    return uint32_t(m.node_rowptr[r] + (std::lower_bound(begin, end, c) - begin));
  };
  m.gmap_off.assign(nblocks + 1, 0);
  for (uint32_t e = 0; e < nelem; e++) {
    for (uint32_t I = 0; I < NN; I++) {
      uint32_t r = m.ids(e, I);
      for (uint32_t J = 0; J < NN; J++) {
        m.gmap_off[block_of(r, int(m.ids(e, J))) + 1]++;
      }
    }
  }
  for (uint32_t b = 0; b < nblocks; b++) { m.gmap_off[b + 1] += m.gmap_off[b]; }
  m.gmap_ent.resize(m.gmap_off[nblocks]);
  {
    std::vector<int> cursor(m.gmap_off.begin(), m.gmap_off.end() - 1);
    for (uint32_t e = 0; e < nelem; e++) {
      for (uint32_t I = 0; I < NN; I++) {
        uint32_t r = m.ids(e, I);
        for (uint32_t J = 0; J < NN; J++) {
          m.gmap_ent[cursor[block_of(r, int(m.ids(e, J)))]++] = (e << 10) | (I * NN + J);
        }
      }
    }
  }

  FEMTO_ASSERT(m.max_row_len < 65536, "pair_k assumes block-column indices fit uint16");
  m.pair_k = nd::array<uint16_t, 2, cpu_mem>({nelem, NN * NN});
  for (uint32_t e = 0; e < nelem; e++) {
    for (uint32_t I = 0; I < NN; I++) {
      uint32_t r = m.ids(e, I);
      for (uint32_t J = 0; J < NN; J++) {
        m.pair_k(e, I * NN + J) = uint16_t(block_of(r, int(m.ids(e, J))) - uint32_t(m.node_rowptr[r]));
      }
    }
  }

  m.blk_row.resize(nblocks);
  m.blk_k.resize(nblocks);
  for (uint32_t r = 0; r < num_nodes; r++) {
    for (int b = m.node_rowptr[r]; b < m.node_rowptr[r + 1]; b++) {
      m.blk_row[b] = r;
      m.blk_k[b] = uint32_t(b - m.node_rowptr[r]);
    }
  }

  stopwatch.stop();
  m.build_seconds = stopwatch.elapsed();
  return m;
}

////////////////////////////////////////////////////////////////////////////////
// per-case driver
////////////////////////////////////////////////////////////////////////////////

template < Geometry geom, uint32_t NC, uint32_t P, typename JacModel >
void run_case(CaseInfo info, const Mesh<> & mesh, JacModel jac_model, int warmup, int reps) {
  constexpr uint32_t NN = nodes_per_element<geom, P>();
  constexpr uint32_t NNX = nodes_per_element<geom, 1>();
  constexpr uint32_t NQ = qpts_per_element<geom, P>();
  constexpr uint32_t DSZ = NC * 3 * NC * 3;

  Domain<> h_domain(mesh, MeshQuadratureRule(P + 1));
  Field h_u = create_field<Family::H1>(mesh, P, NC);
  fill_solution_field(h_u, mesh);
  BasisFunction<Family::H1> phi(P, NC);

  Mesh<gpu_mem> d_mesh = mesh;
  Domain<gpu_mem> d_domain(d_mesh, MeshQuadratureRule(P + 1));
  Field<Family::H1, gpu_mem> d_u = h_u;

  info.degree = P;
  info.num_elements = h_domain.mesh[geom].shape[0];
  info.num_qpts = total(h_domain.num_qpts);
  info.num_unknowns = h_u.size();

  const uint32_t nelem = uint32_t(info.num_elements);
  const uint32_t nq = uint32_t(info.num_qpts);

  printf("# case: %s %s p%u: %llu elements, %llu qpts, %llu unknowns\n",
         info.physics.c_str(), info.geom.c_str(), P,
         (unsigned long long)info.num_elements, (unsigned long long)info.num_qpts,
         (unsigned long long)info.num_unknowns);

  //////////////////////////////////////////////////////////////////////////////
  // library baseline (+ its TestShapeMode knob, experiment 3)
  //////////////////////////////////////////////////////////////////////////////

  nd::array<double, 3, gpu_mem> du_q = evaluate(grad(d_u), d_domain);
  auto dflux_lib = forall(jac_model, du_q);

  sparse_matrix<gpu_mem> d_K = blank_sparse_matrix(grad(phi), grad(phi), d_domain);
  report_mem(info, "spmat", "all", "csr_matrix", d_K.nnz * 12 + (d_K.nrows + 1) * 4);

  auto fill = integrate(dot(grad(phi), dflux_lib, grad(phi)), d_domain);

  const std::pair<TestShapeMode, const char *> lib_modes[4] = {
    {TestShapeMode::MinimalGlobal, "lib:MinimalGlobal"},
    {TestShapeMode::MinimalShared, "lib:MinimalShared"},
    {TestShapeMode::OnTheFlyGlobal, "lib:OnTheFlyGlobal"},
    {TestShapeMode::OnTheFlyShared, "lib:OnTheFlyShared"},
  };

  // full warmup sweep before timing anything (GPU clock ramp!)
  for (auto [mode, name] : lib_modes) {
    set_test_shape_mode(mode);
    fill(d_K);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  const char * only_early = getenv("SPMAT_ONLY");
  if (!only_early || strncmp(only_early, "lib:", 4) == 0) {
    for (auto [mode, name] : lib_modes) {
      if (only_early && strcmp(only_early, name) != 0) { continue; }
      set_test_shape_mode(mode);
      double t = bench_gpu([&]() { fill(d_K); }, warmup, reps);
      report_time(info, "spmat", name, "total", t);
    }
  }
  set_test_shape_mode(TestShapeMode::MinimalGlobal);

  fill(d_K);
  nd::array<double, 1, cpu_mem> K_ref = d_K.values;

  auto verify = [&](const char * label) {
    nd::array<double, 1, cpu_mem> v = d_K.values;
    report_verify(info, "spmat", label, rel_l2_diff(&K_ref(0), &v(0), d_K.nnz), 1.0e-10);
  };

  //////////////////////////////////////////////////////////////////////////////
  // shared setup for the custom kernels
  //////////////////////////////////////////////////////////////////////////////

  // flat qdata, q-major (the library's layout)
  nd::array<double, 2, gpu_mem> qm({nq, DSZ});
  {
    uint32_t bs = 128;
    forall_kernel<<<(nq + bs - 1) / bs, bs>>>(nq, jac_model,
      reinterpret_cast<jac_t<NC> *>(qm.data()),
      reinterpret_cast<const grad_t<NC> *>(du_q.data()));
    CUDA_CHECK(cudaGetLastError());
  }

  nd::array<double, 3, cpu_mem> G, GX;
  nd::array<double, 1, cpu_mem> W, W_unused;
  build_qtables<geom>(h_domain, P, G, W);
  build_qtables<geom>(h_domain, 1, GX, W_unused);
  nd::array<double, 3, gpu_mem> d_G = G;
  nd::array<double, 3, gpu_mem> d_GX = GX;
  nd::array<double, 1, gpu_mem> d_W = W;

  nd::array<double, 2, cpu_mem> xi_pts({NQ, 3u});
  for (uint32_t q = 0; q < NQ; q++) {
    vec3 p = expanded_qpt<geom>(q, h_domain.rule[geom].points);
    xi_pts(q, 0) = p[0]; xi_pts(q, 1) = p[1]; xi_pts(q, 2) = p[2];
  }
  nd::array<double, 2, gpu_mem> d_xi_pts = xi_pts;

  nd::array<int, 1, cpu_mem> h_rowptr = d_K.row_ptr;
  nd::array<int, 1, cpu_mem> h_colind = d_K.col_ind;
  HostMaps maps = build_host_maps<geom, NC, P>(h_domain, h_u, h_rowptr, h_colind);
  printf("# host map build: %.3f s\n", maps.build_seconds);

  nd::array<uint32_t, 2, gpu_mem> d_ids = maps.ids;
  nd::array<uint32_t, 2, gpu_mem> d_X_ids = maps.X_ids;

  KArgs args{
    d_K.values, d_K.row_ptr, d_K.col_ind,
    d_ids, d_X_ids, d_mesh.X.data,
    qm, d_G, d_GX, d_W, d_xi_pts,
    {},   // pair_k, filled by the position-table variants
    nelem, nq
  };

  FiniteElement<geom, Family::H1> el{P};

  constexpr uint32_t pair_bs = std::min(256u, round_up_warp(std::max(NN * NN, std::max(NQ, DSZ))));

  // alternate qdata layouts, built once (used by experiments 1 and 2)
  nd::array<double, 2, gpu_mem> soa({DSZ, nq});
  nd::array<double, 2, gpu_mem> eb({nelem, DSZ * NQ});
  {
    uint32_t n = nq * DSZ;
    k_to_soa<<<(n + 255) / 256, 256>>>(nq, DSZ, qm, soa);
    k_to_eblock<<<(n + 255) / 256, 256>>>(nelem, NQ, DSZ, qm, eb);
    CUDA_CHECK(cudaGetLastError());
  }

  // SPMAT_ONLY=<variant> runs a single variant (for profiling under ncu)
  const char * only = getenv("SPMAT_ONLY");
  auto run = [&](const char * name, auto && launch) {
    if (only && strcmp(only, name) != 0) { return; }
    double t = bench_gpu([&]() { zero(d_K.values); launch(); CUDA_CHECK(cudaGetLastError()); }, warmup, reps);
    report_time(info, "spmat", name, "total", t);
    verify(name);
  };

  //////////////////////////////////////////////////////////////////////////////
  // experiment 1: work mappings
  //////////////////////////////////////////////////////////////////////////////

  run("map:pair", [&]() {
    k_pair<geom, NC, P, QL::QMajor, false, true, SM::Shared, SM::Shared, 1>
      <<<nelem, dim3(pair_bs, 1, 1)>>>(args, el);
  });

  run("map:scalar", [&]() {
    constexpr uint32_t bs = std::min(256u, round_up_warp(std::max(NN * NC * NN * NC, NQ)));
    k_scalar<geom, NC, P><<<nelem, bs>>>(args, el);
  });

  // multiple elements per block (small elements only)
  if constexpr (P == 1) {
    auto run_epb = [&](auto epb_c) {
      constexpr uint32_t E = decltype(epb_c)::value;
      constexpr uint32_t smem = E * sizeof(BlockSmem<geom, NC, P, true>);
      if constexpr (smem < 46 * 1024) {
        constexpr uint32_t tpe = std::min(std::max(NN * NN, NQ), std::max(1u, 256u / E));
        char name[64];
        snprintf(name, sizeof(name), "map:pair_epb%u", E);
        run(name, [&]() {
          k_pair<geom, NC, P, QL::QMajor, false, true, SM::Shared, SM::Shared, E>
            <<<(nelem + E - 1) / E, dim3(tpe, 1, E)>>>(args, el);
        });
      }
    };
    run_epb(std::integral_constant<uint32_t, 2>{});
    run_epb(std::integral_constant<uint32_t, 4>{});
    run_epb(std::integral_constant<uint32_t, 8>{});
    run_epb(std::integral_constant<uint32_t, 16>{});
    run_epb(std::integral_constant<uint32_t, 32>{});   // -> 8 threads per element

    run("map:elem_thread", [&]() {
      k_elem_thread<geom, NC, P, QL::QMajor><<<(nelem + 127) / 128, 128>>>(args);
    });
    {
      // element-blocked layout for the per-thread element sweep
      KArgs a2 = args;
      a2.qdata = eb;
      run("map:elem_thread_eblock", [&, a2]() {
        k_elem_thread<geom, NC, P, QL::EBlock><<<(nelem + 127) / 128, 128>>>(a2);
      });
    }
    {
      // element-minor layout: consecutive threads (elements) read consecutive
      // addresses for each (q, c) -- the coalesced layout for this mapping
      nd::array<double, 2, gpu_mem> esoa({DSZ * NQ, nelem});
      uint32_t n = nq * DSZ;
      k_to_esoa<<<(n + 255) / 256, 256>>>(nelem, NQ, DSZ, qm, esoa);
      CUDA_CHECK(cudaGetLastError());
      KArgs a2 = args;
      a2.qdata = esoa;
      run("map:elem_thread_esoa", [&, a2]() {
        k_elem_thread<geom, NC, P, QL::ESoA><<<(nelem + 127) / 128, 128>>>(a2);
      });
    }
  }

  // warp-per-row (atomic-free); geometry precompute included in the timing
  {
    nd::array<int, 1, gpu_mem> d_nrp = to_array(maps.node_rowptr);
    nd::array<int, 1, gpu_mem> d_ncols = to_array(maps.node_cols);
    nd::array<int, 1, gpu_mem> d_adj_off = to_array(maps.adj_off);
    nd::array<uint32_t, 1, gpu_mem> d_adj_ent = to_array(maps.adj_ent);
    nd::array<double, 1, gpu_mem> gWJ({nelem * NQ});
    nd::array<double, 2, gpu_mem> gJi({nelem * NQ, 9u});
    uint32_t num_node_rows = uint32_t(maps.node_rowptr.size() - 1);

    uint32_t acc_bytes = maps.max_row_len * NC * NC * 8;
    uint32_t warps = std::max(1u, std::min(8u, (46u * 1024) / std::max(acc_bytes, 1u)));
    uint32_t shmem = warps * acc_bytes;
    if (shmem <= 46 * 1024) {
      run("map:row_warp", [&]() {
        uint32_t n = nelem * NQ;
        k_geometry<geom, P><<<(n + 255) / 256, 256>>>(args, gWJ, gJi);
        dim3 block(32, warps, 1);
        uint32_t grid = (num_node_rows + warps - 1) / warps;
        k_row_warp<geom, NC, P><<<grid, block, shmem>>>(
          args, d_nrp, d_ncols, d_adj_off, d_adj_ent, gWJ, gJi, num_node_rows, maps.max_row_len);
      });
    }

    report_mem(info, "spmat", "map:row_warp", "adjacency+geometry",
               maps.adj_ent.size() * 4 + uint64_t(nelem) * NQ * 10 * 8);
  }

  //////////////////////////////////////////////////////////////////////////////
  // experiment 2: qdata layouts
  //////////////////////////////////////////////////////////////////////////////

  {
    auto with_layout = [&](QL l) {
      KArgs a2 = args;
      if (l == QL::SoA) { a2.qdata = soa; }
      if (l == QL::EBlock) { a2.qdata = eb; }
      return a2;
    };

    auto run_layout = [&](const char * name, auto layout_c, auto qfast_c, auto staged_c) {
      constexpr QL l = decltype(layout_c)::value;
      constexpr bool qfast = decltype(qfast_c)::value;
      constexpr bool staged = decltype(staged_c)::value;
      KArgs a2 = with_layout(l);
      run(name, [&, a2]() {
        k_pair<geom, NC, P, l, qfast, staged, SM::Shared, SM::Shared, 1>
          <<<nelem, dim3(pair_bs, 1, 1)>>>(a2, el);
      });
    };

    using yes = std::true_type;
    using no = std::false_type;
    using qmajor = std::integral_constant<QL, QL::QMajor>;
    using lsoa = std::integral_constant<QL, QL::SoA>;
    using leb = std::integral_constant<QL, QL::EBlock>;

    run_layout("layout:stage_qm_cfast", qmajor{}, no{}, yes{});
    run_layout("layout:stage_qm_qfast", qmajor{}, yes{}, yes{});
    run_layout("layout:stage_soa_cfast", lsoa{}, no{}, yes{});
    run_layout("layout:stage_soa_qfast", lsoa{}, yes{}, yes{});
    run_layout("layout:stage_eb_cfast", leb{}, no{}, yes{});
    run_layout("layout:stage_eb_qfast", leb{}, yes{}, yes{});
    run_layout("layout:direct_qm", qmajor{}, no{}, no{});
    run_layout("layout:direct_soa", lsoa{}, no{}, no{});
    run_layout("layout:direct_eb", leb{}, no{}, no{});
  }

  // experiment 2b: functor callbacks instead of stored qdata
  {
    // state functor: reads the compact per-qpt state (du_dX) and evaluates the
    // tangent inside the assembly kernel
    StateFunctor<NC, JacModel> sf{du_q.data(), jac_model};
    run("func:state", [&]() {
      k_pair_functor<geom, NC, P><<<nelem, pair_bs>>>(args, sf);
    });

    report_mem(info, "spmat", "func:state", "qpt_state", uint64_t(nq) * NC * 3 * 8);
    report_mem(info, "spmat", "func:stored", "qpt_tangent", uint64_t(nq) * DSZ * 8);
  }

  //////////////////////////////////////////////////////////////////////////////
  // experiment 3: shape function storage (test x trial)
  //////////////////////////////////////////////////////////////////////////////

  {
    auto run_shape = [&](const char * name, auto test_c, auto trial_c) {
      constexpr SM tm = decltype(test_c)::value;
      constexpr SM sm_ = decltype(trial_c)::value;
      run(name, [&]() {
        k_pair<geom, NC, P, QL::QMajor, false, true, tm, sm_, 1>
          <<<nelem, dim3(pair_bs, 1, 1)>>>(args, el);
      });
    };
    using sh = std::integral_constant<SM, SM::Shared>;
    using gl = std::integral_constant<SM, SM::Global>;
    using fl = std::integral_constant<SM, SM::Fly>;
    run_shape("shape:sh_sh", sh{}, sh{});
    run_shape("shape:sh_gl", sh{}, gl{});
    run_shape("shape:sh_fly", sh{}, fl{});
    run_shape("shape:gl_sh", gl{}, sh{});
    run_shape("shape:gl_gl", gl{}, gl{});
    run_shape("shape:gl_fly", gl{}, fl{});
    run_shape("shape:fly_sh", fl{}, sh{});
    run_shape("shape:fly_gl", fl{}, gl{});
    run_shape("shape:fly_fly", fl{}, fl{});
  }

  //////////////////////////////////////////////////////////////////////////////
  // experiment 4: two-pass assembly
  //////////////////////////////////////////////////////////////////////////////

  if (!only || strncmp(only, "twopass", 7) == 0) {
    uint64_t ae_bytes = uint64_t(nelem) * NN * NN * NC * NC * 8;
    uint64_t map_bytes = maps.gmap_ent.size() * 4 + maps.gmap_off.size() * 4
                       + maps.blk_row.size() * 8;
    report_mem(info, "spmat", "twopass", "element_matrices", ae_bytes);
    report_mem(info, "spmat", "twopass", "gather_map", map_bytes);

    size_t free_b = 0, total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    if (ae_bytes + map_bytes < (free_b * 4) / 5) {
      nd::array<double, 1, gpu_mem> A_e({uint32_t(uint64_t(nelem) * NN * NN * NC * NC)});
      nd::array<int, 1, gpu_mem> d_gmap_off = to_array(maps.gmap_off);
      nd::array<uint32_t, 1, gpu_mem> d_gmap_ent = to_array(maps.gmap_ent);
      nd::array<uint32_t, 1, gpu_mem> d_blk_row = to_array(maps.blk_row);
      nd::array<uint32_t, 1, gpu_mem> d_blk_k = to_array(maps.blk_k);
      uint32_t nblocks = uint32_t(maps.blk_row.size());
      constexpr uint32_t NN2 = NN * NN;

      auto pass1 = [&](auto ijmajor_c) {
        constexpr bool ijm = decltype(ijmajor_c)::value;
        k_pass1<geom, NC, P, ijm><<<nelem, pair_bs>>>(args, A_e);
      };

      using yes = std::true_type;
      using no = std::false_type;

      // pass 1 alone, both element-matrix layouts
      double t1_ij = bench_gpu([&]() { pass1(yes{}); CUDA_CHECK(cudaGetLastError()); }, warmup, reps);
      report_time(info, "spmat", "twopass:pass1_ij", "total", t1_ij);
      double t1_pair = bench_gpu([&]() { pass1(no{}); CUDA_CHECK(cudaGetLastError()); }, warmup, reps);
      report_time(info, "spmat", "twopass:pass1_pair", "total", t1_pair);

      // experiment 8: where does pass 1 fall off element_matrix.cu's
      // write-bound ceiling? (writes alone, then + staging, then the full
      // pass1_ij above)
      double t1w = bench_gpu([&]() {
        k_pass1_writes<geom, NC, P><<<nelem, pair_bs>>>(A_e);
        CUDA_CHECK(cudaGetLastError());
      }, warmup, reps);
      report_time(info, "spmat", "twopass:pass1_writes", "total", t1w);
      double t1s = bench_gpu([&]() {
        k_pass1_staged<geom, NC, P><<<nelem, pair_bs>>>(args, A_e);
        CUDA_CHECK(cudaGetLastError());
      }, warmup, reps);
      report_time(info, "spmat", "twopass:pass1_staged", "total", t1s);

      constexpr uint32_t E_AF = 16;
      constexpr uint32_t tpe_af = std::min(std::max(NN * NN, NC * NC * 9u), std::max(4u, 256u / E_AF));
      auto pass1_affine = [&]() {
        if constexpr (geom == Geometry::Tetrahedron && P == 1) {
          k_pass1_affine<NC, E_AF><<<(nelem + E_AF - 1) / E_AF, dim3(tpe_af, 1, E_AF)>>>(args, A_e);
        }
      };
      if constexpr (geom == Geometry::Tetrahedron && P == 1) {
        double t1a = bench_gpu([&]() { pass1_affine(); CUDA_CHECK(cudaGetLastError()); }, warmup, reps);
        report_time(info, "spmat", "twopass:pass1_affine", "total", t1a);
      }

      if constexpr (NC == 1) {
        uint32_t nnz = uint32_t(d_K.nnz);
        run("twopass:scalar", [&]() {
          pass1(yes{});
          k_gather_scalar<<<(nnz + 255) / 256, 256>>>(d_K.values, d_gmap_off, d_gmap_ent, A_e, nnz, NN2);
        });
        double t2 = bench_gpu([&]() {
          k_gather_scalar<<<(nnz + 255) / 256, 256>>>(d_K.values, d_gmap_off, d_gmap_ent, A_e, nnz, NN2);
          CUDA_CHECK(cudaGetLastError());
        }, warmup, reps);
        report_time(info, "spmat", "twopass:gather", "total", t2);

        if constexpr (geom == Geometry::Tetrahedron && P == 1) {
          run("twopass:affine", [&]() {
            pass1_affine();
            k_gather_scalar<<<(nnz + 255) / 256, 256>>>(d_K.values, d_gmap_off, d_gmap_ent, A_e, nnz, NN2);
          });
        }
      } else {
        auto gather9 = [&](auto ijmajor_c) {
          constexpr bool ijm = decltype(ijmajor_c)::value;
          uint32_t n = nblocks * 9;
          k_gather_block9<ijm><<<(n + 255) / 256, 256>>>(
            d_K.values, d_K.row_ptr, d_gmap_off, d_gmap_ent, d_blk_row, d_blk_k, A_e, nblocks, NN2);
        };
        auto gather1 = [&](auto ijmajor_c) {
          constexpr bool ijm = decltype(ijmajor_c)::value;
          k_gather_block1<ijm><<<(nblocks + 255) / 256, 256>>>(
            d_K.values, d_K.row_ptr, d_gmap_off, d_gmap_ent, d_blk_row, d_blk_k, A_e, nblocks, NN2);
        };

        run("twopass:ij_g9", [&]() { pass1(yes{}); gather9(yes{}); });
        run("twopass:pair_g9", [&]() { pass1(no{}); gather9(no{}); });
        run("twopass:ij_g1", [&]() { pass1(yes{}); gather1(yes{}); });

        double t2 = bench_gpu([&]() { gather9(no{}); CUDA_CHECK(cudaGetLastError()); }, warmup, reps);
        report_time(info, "spmat", "twopass:gather9_pair", "total", t2);
        t2 = bench_gpu([&]() { gather9(yes{}); CUDA_CHECK(cudaGetLastError()); }, warmup, reps);
        report_time(info, "spmat", "twopass:gather9_ij", "total", t2);
        t2 = bench_gpu([&]() { gather1(yes{}); CUDA_CHECK(cudaGetLastError()); }, warmup, reps);
        report_time(info, "spmat", "twopass:gather1_ij", "total", t2);

        if constexpr (geom == Geometry::Tetrahedron && P == 1) {
          run("twopass:affine", [&]() {
            pass1_affine();
            gather9(yes{});
          });
        }
      }
    } else {
      report_skip(info, "spmat_twopass", ae_bytes + map_bytes);
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // experiment 5: prototypes inspired by the fem_shootout libraries
  //////////////////////////////////////////////////////////////////////////////

  {
    // (a) precomputed scatter positions (DOLFINx/MFEM keep dof->position maps;
    //     here: one uint16 block-column index per (element, node pair))
    nd::array<uint16_t, 2, gpu_mem> d_pair_k = maps.pair_k;
    KArgs ap = args;
    ap.pair_k = d_pair_k;
    report_mem(info, "spmat", "map:pair_pos", "position_table", uint64_t(nelem) * NN * NN * 2);

    run("map:pair_pos", [&, ap]() {
      k_pair<geom, NC, P, QL::QMajor, false, true, SM::Shared, SM::Shared, 1, true>
        <<<nelem, dim3(pair_bs, 1, 1)>>>(ap, el);
    });

    // (b) affine collapse (FFCx-style): fold the quadrature loop into the
    //     qdata, one contraction per pair (p1 tets only)
    if constexpr (geom == Geometry::Tetrahedron && P == 1) {
      auto run_affine = [&](auto epb_c, auto pos_c, const char * name) {
        constexpr uint32_t E = decltype(epb_c)::value;
        constexpr bool POS = decltype(pos_c)::value;
        constexpr uint32_t tpe = std::min(std::max(NN * NN, DSZ), std::max(4u, 256u / E));
        run(name, [&, ap]() {
          k_pair_affine<geom, NC, P, E, POS>
            <<<(nelem + E - 1) / E, dim3(tpe, 1, E)>>>(ap);
        });
      };
      using yes = std::true_type;
      using no = std::false_type;
      run_affine(std::integral_constant<uint32_t, 8>{}, no{}, "affine:epb8");
      run_affine(std::integral_constant<uint32_t, 16>{}, no{}, "affine:epb16");
      run_affine(std::integral_constant<uint32_t, 32>{}, no{}, "affine:epb32");
      run_affine(std::integral_constant<uint32_t, 16>{}, yes{}, "affine:pos_epb16");
      run_affine(std::integral_constant<uint32_t, 32>{}, yes{}, "affine:pos_epb32");
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // experiment 6: residual integration (p1 tets), library vs affine collapse
  //////////////////////////////////////////////////////////////////////////////

  if constexpr (geom == Geometry::Tetrahedron && P == 1) {
    if (!only || strncmp(only, "res:", 4) == 0) {
      Residual<Family::H1, gpu_mem> d_r(FunctionSpace{Family::H1, P, NC}, d_mesh);

      // library path, using du_q as a stand-in flux array
      double t = bench_gpu([&]() {
        d_r = integrate(dot(du_q, grad(phi)), d_domain);
      }, warmup, reps);
      report_time(info, "spmat", "res:lib", "total", t);
      nd::array<double, 2, cpu_mem> r_ref = d_r.data;

      t = bench_gpu([&]() {
        zero(d_r.data);
        k_res_affine<NC, NQ><<<(nelem + 127) / 128, 128>>>(args, du_q, d_r.data);
        CUDA_CHECK(cudaGetLastError());
      }, warmup, reps);
      report_time(info, "spmat", "res:affine_ept", "total", t);
      {
        nd::array<double, 2, cpu_mem> r_h = d_r.data;
        report_verify(info, "spmat", "res:affine_ept",
                      rel_l2_diff(&r_ref(0, 0), &r_h(0, 0), r_ref.size()), 1.0e-10);
      }
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // experiment 7: access-pattern skeletons (data motion only, no pair flops)
  //////////////////////////////////////////////////////////////////////////////

  if (!only || strncmp(only, "skel:", 5) == 0) {
    // expected fill: every entry holds the number of elements touching it
    uint32_t num_nodes = uint32_t(h_rowptr.size() - 1) / NC;
    nd::array<double, 1, cpu_mem> expected({uint32_t(d_K.nnz)});
    for (uint32_t r = 0; r < num_nodes; r++) {
      for (int b = maps.node_rowptr[r]; b < maps.node_rowptr[r + 1]; b++) {
        double count = double(maps.gmap_off[b + 1] - maps.gmap_off[b]);
        uint32_t k = uint32_t(b - maps.node_rowptr[r]);
        for (uint32_t i = 0; i < NC; i++) {
          for (uint32_t j = 0; j < NC; j++) {
            expected(uint32_t(h_rowptr(r * NC + i)) + k * NC + j) = count;
          }
        }
      }
    }

    nd::array<uint16_t, 2, gpu_mem> d_pair_k = maps.pair_k;
    KArgs ap = args;
    ap.pair_k = d_pair_k;

    auto run_skel = [&](const char * name, auto && launch) {
      if (only && strcmp(only, name) != 0) { return; }
      double t = bench_gpu([&]() { zero(d_K.values); launch(); CUDA_CHECK(cudaGetLastError()); }, warmup, reps);
      report_time(info, "spmat", name, "total", t);
      nd::array<double, 1, cpu_mem> v = d_K.values;
      report_verify(info, "spmat", name, rel_l2_diff(&expected(0), &v(0), d_K.nnz), 1.0e-14);
    };

    run_skel("skel:traffic", [&]() {
      k_pair_skel_traffic<geom, NC, P><<<nelem, dim3(pair_bs, 1, 1)>>>(args);
    });
    run_skel("skel:incr", [&]() {
      k_pair_skel_incr<geom, NC, P, false><<<nelem, dim3(pair_bs, 1, 1)>>>(args);
    });
    run_skel("skel:incr_pos", [&, ap]() {
      k_pair_skel_incr<geom, NC, P, true><<<nelem, dim3(pair_bs, 1, 1)>>>(ap);
    });

    // the floor: flat grid, ids only, no shared / sync / block structure
    uint32_t flat_n = nelem * NN * NN;
    run_skel("skel:flat", [&]() {
      k_skel_flat<geom, NC, P, false><<<(flat_n + 255) / 256, 256>>>(args);
    });
    run_skel("skel:flat_pos", [&, ap]() {
      k_skel_flat<geom, NC, P, true><<<(flat_n + 255) / 256, 256>>>(ap);
    });

    // ablation: price each staged input at fixed (skel:traffic) occupancy
    run_skel("skel:smem", [&]() {
      k_pair_skel_stage<geom, NC, P, false, false><<<nelem, dim3(pair_bs, 1, 1)>>>(args);
    });
    run_skel("skel:smem_D", [&]() {
      k_pair_skel_stage<geom, NC, P, true, false><<<nelem, dim3(pair_bs, 1, 1)>>>(args);
    });
    run_skel("skel:smem_geom", [&]() {
      k_pair_skel_stage<geom, NC, P, false, true><<<nelem, dim3(pair_bs, 1, 1)>>>(args);
    });
    run_skel("skel:smem_full", [&]() {
      k_pair_skel_stage<geom, NC, P, true, true><<<nelem, dim3(pair_bs, 1, 1)>>>(args);
    });
  }

  // baseline repeated at the end to detect clock-ramp / thermal drift
  run("map:pair_repeat", [&]() {
    k_pair<geom, NC, P, QL::QMajor, false, true, SM::Shared, SM::Shared, 1>
      <<<nelem, dim3(pair_bs, 1, 1)>>>(args, el);
  });

  // constant-tangent functor: verified against the pair kernel run on qdata
  // generated by the same constant model (elasticity only; the poisson tangent
  // is already constant, so func:state covers it)
  if constexpr (NC == 3) if (!only || strncmp(only, "func:lin", 8) == 0) {
    LinearIsotropicJacobianModel lin{lambda0, mu0};
    nd::array<double, 2, gpu_mem> qm_lin({nq, DSZ});
    uint32_t bs = 128;
    forall_kernel<<<(nq + bs - 1) / bs, bs>>>(nq, lin,
      reinterpret_cast<jac_t<NC> *>(qm_lin.data()),
      reinterpret_cast<const grad_t<NC> *>(du_q.data()));
    CUDA_CHECK(cudaGetLastError());

    KArgs a2 = args;
    a2.qdata = qm_lin;

    double t = bench_gpu([&]() {
      zero(d_K.values);
      k_pair<geom, NC, P, QL::QMajor, false, true, SM::Shared, SM::Shared, 1>
        <<<nelem, dim3(pair_bs, 1, 1)>>>(a2, el);
      CUDA_CHECK(cudaGetLastError());
    }, warmup, reps);
    report_time(info, "spmat", "func:lin_stored", "total", t);
    nd::array<double, 1, cpu_mem> K_lin_ref = d_K.values;

    ConstFunctor<NC, LinearIsotropicJacobianModel> cf{lin};
    t = bench_gpu([&]() {
      zero(d_K.values);
      k_pair_functor<geom, NC, P><<<nelem, pair_bs>>>(args, cf);
      CUDA_CHECK(cudaGetLastError());
    }, warmup, reps);
    report_time(info, "spmat", "func:lin_const", "total", t);
    {
      nd::array<double, 1, cpu_mem> v = d_K.values;
      report_verify(info, "spmat", "func:lin_const", rel_l2_diff(&K_lin_ref(0), &v(0), d_K.nnz), 1.0e-10);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char * argv[]) {
  int size = (argc > 1) ? atoi(argv[1]) : 1;
  int warmup = (argc > 2) ? atoi(argv[2]) : 2;
  int reps = (argc > 3) ? atoi(argv[3]) : 5;

  uint32_t tet_n[2] = {16, 64};
  uint32_t tet2_n[2] = {12, 40};
  uint32_t hex_n[2] = {16, 64};
  uint32_t hex2_n[2] = {8, 32};
  int s = (size == 0) ? 0 : 1;

  // SPMAT_CASE=<substring of e.g. elasticity_tet_p1> runs matching cases only
  const char * case_filter = getenv("SPMAT_CASE");
  auto want = [&](const char * name) { return !case_filter || strstr(name, case_filter) != nullptr; };

  if (want("poisson_tet_p1") || want("elasticity_tet_p1")) {
    Mesh<> mesh = tet_cuboid(tet_n[s]);
    if (want("poisson_tet_p1"))
      run_case<Geometry::Tetrahedron, 1, 1>(CaseInfo{"gpu", "poisson", "tet", "p1"}, mesh,
                                            PoissonJacobianModel{kappa0}, warmup, reps);
    if (want("elasticity_tet_p1"))
      run_case<Geometry::Tetrahedron, 3, 1>(CaseInfo{"gpu", "elasticity", "tet", "p1"}, mesh,
                                            NeoHookeanJacobianModel{lambda0, mu0}, warmup, reps);
  }
  if (want("poisson_tet_p2") || want("elasticity_tet_p2")) {
    Mesh<> mesh = tet_cuboid(tet2_n[s]);
    if (want("poisson_tet_p2"))
      run_case<Geometry::Tetrahedron, 1, 2>(CaseInfo{"gpu", "poisson", "tet", "p2"}, mesh,
                                            PoissonJacobianModel{kappa0}, warmup, reps);
    if (want("elasticity_tet_p2"))
      run_case<Geometry::Tetrahedron, 3, 2>(CaseInfo{"gpu", "elasticity", "tet", "p2"}, mesh,
                                            NeoHookeanJacobianModel{lambda0, mu0}, warmup, reps);
  }
  if (want("poisson_hex_p1") || want("elasticity_hex_p1")) {
    Mesh<> mesh = Mesh<>::cuboid({hex_n[s], hex_n[s], hex_n[s]}, vec3{1.0, 1.0, 1.0});
    if (want("poisson_hex_p1"))
      run_case<Geometry::Hexahedron, 1, 1>(CaseInfo{"gpu", "poisson", "hex", "p1"}, mesh,
                                           PoissonJacobianModel{kappa0}, warmup, reps);
    if (want("elasticity_hex_p1"))
      run_case<Geometry::Hexahedron, 3, 1>(CaseInfo{"gpu", "elasticity", "hex", "p1"}, mesh,
                                           NeoHookeanJacobianModel{lambda0, mu0}, warmup, reps);
  }
  if (want("poisson_hex_p2") || want("elasticity_hex_p2")) {
    Mesh<> mesh = Mesh<>::cuboid({hex2_n[s], hex2_n[s], hex2_n[s]}, vec3{1.0, 1.0, 1.0});
    if (want("poisson_hex_p2"))
      run_case<Geometry::Hexahedron, 1, 2>(CaseInfo{"gpu", "poisson", "hex", "p2"}, mesh,
                                           PoissonJacobianModel{kappa0}, warmup, reps);
    if (want("elasticity_hex_p2"))
      run_case<Geometry::Hexahedron, 3, 2>(CaseInfo{"gpu", "elasticity", "hex", "p2"}, mesh,
                                           NeoHookeanJacobianModel{lambda0, mu0}, warmup, reps);
  }

  return 0;
}
