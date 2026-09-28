// Kernel fusion experiment, GPU driver.
//
// Compares three implementations of the residual / stiffness calculation:
//   separated : the existing library pipeline (evaluate / forall /
//               integrate) — 3+ kernel launches, intermediates allocated in
//               global memory on every call
//   staged    : the same three phases as custom kernels writing into
//               preallocated global buffers, timed with cudaEvents, so the
//               numbers contain only kernel execution
//   fused     : gather + interpolate + constitutive + integrate + scatter in a
//               single kernel launch
//
// The custom kernels follow the structure of the library kernels in
// src/field/evaluate.cu / integrate_residual.cu: several elements per block
// (threads_per_element x elements_per_block), with the shape function tables
// and all element-local data staged in shared memory.
//
// Compile-time isolation:
//   -DFUSION_ONLY_SEPARATED : compile only the 3-phase (separated + staged) code
//   -DFUSION_ONLY_FUSED     : compile only the fused kernels

#include "fusion_common.hpp"

#include "forall.hpp"
#include "misc/macros.hpp"

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

////////////////////////////////////////////////////////////////////////////////
// block-shape helpers (residual-style kernels: a few elements per block)
////////////////////////////////////////////////////////////////////////////////

template < Geometry geom, uint32_t P >
constexpr uint32_t threads_per_element() {
  constexpr uint32_t NN = nodes_per_element<geom, P>();
  constexpr uint32_t NQ = qpts_per_element<geom, P>();
  return (NN > NQ) ? NN : NQ;
}

// choose elements-per-block so ~256 threads/block, limited by a 44 KB static
// shared memory budget
template < Geometry geom, uint32_t NC, uint32_t P >
constexpr uint32_t elements_per_block_residual() {
  constexpr uint32_t NN = nodes_per_element<geom, P>();
  constexpr uint32_t NNX = nodes_per_element<geom, 1>();
  constexpr uint32_t NQ = qpts_per_element<geom, P>();
  constexpr uint32_t CM = conn_row_size<geom>();
  constexpr uint32_t per_element = CM * 4 + NN * 4 + NNX * 4
                                 + NNX * 3 * 8 + NN * NC * 8 + NQ * NC * 3 * 8;
  constexpr uint32_t base = NQ * NN * 3 * 8 + NQ * NNX * 3 * 8 + NQ * 8;
  uint32_t E = 256 / threads_per_element<geom, P>();
  if (E < 1) { E = 1; }
  while (E > 1 && base + E * per_element > 44 * 1024) { E--; }
  return E;
}

////////////////////////////////////////////////////////////////////////////////
// residual-style kernels
////////////////////////////////////////////////////////////////////////////////

// shared-memory staging common to the fused / staged residual kernels
template < Geometry geom, uint32_t NC, uint32_t P, uint32_t E >
struct ResidualBlockSmem {
  static constexpr uint32_t NN = nodes_per_element<geom, P>();
  static constexpr uint32_t NNX = nodes_per_element<geom, 1>();
  static constexpr uint32_t NQ = qpts_per_element<geom, P>();
  static constexpr uint32_t CM = conn_row_size<geom>();

  double G[NQ * NN * 3];      // solution-field shape gradients
  double GX[NQ * NNX * 3];    // (linear) coordinate-field shape gradients
  double W[NQ];
  Connection conn[E][CM];
  uint32_t ids[E][NN];
  uint32_t X_ids[E][NNX];
  double X_e[E][NNX][3];
  double u_e[E][NN][NC];
  double t[E][NQ][NC][3];     // w * adj(dX/dxi) * flux, per qpt/component
};

// cooperative load of tables / connectivity / element data. returns `active`
template < bool with_u, Geometry geom, uint32_t NC, uint32_t P, uint32_t E >
__device__ bool residual_block_setup(
  ResidualBlockSmem<geom, NC, P, E> & sm,
  FiniteElement< geom, Family::H1 > el,
  FiniteElement< geom, Family::H1 > X_el,
  GeometryInfo u_offsets, GeometryInfo X_offsets,
  nd::view<const double, 2, gpu_mem> u_data,
  nd::view<const double, 2, gpu_mem> X_data,
  nd::view<const Connection, 2, gpu_mem> conn,
  nd::view<const int, 1, gpu_mem> elements,
  nd::view<const double, 3, gpu_mem> G,
  nd::view<const double, 3, gpu_mem> GX,
  nd::view<const double, 1, gpu_mem> W) {

  constexpr uint32_t NN = sm.NN;
  constexpr uint32_t NNX = sm.NNX;
  constexpr uint32_t NQ = sm.NQ;

  const uint32_t tpe = blockDim.x;
  const uint32_t z = threadIdx.z;
  const uint32_t btid = threadIdx.x + blockDim.x * threadIdx.z;
  const uint32_t bstride = blockDim.x * blockDim.z;

  for (uint32_t i = btid; i < NQ * NN * 3; i += bstride) { sm.G[i] = G[i]; }
  for (uint32_t i = btid; i < NQ * NNX * 3; i += bstride) { sm.GX[i] = GX[i]; }
  for (uint32_t i = btid; i < NQ; i += bstride) { sm.W[i] = W[i]; }

  uint32_t e = blockIdx.x * E + z;
  bool active = e < elements.shape[0];
  uint32_t elem = elements(active ? e : 0);
  uint32_t cn = conn.shape[1];
  for (uint32_t i = threadIdx.x; i < cn; i += tpe) { sm.conn[z][i] = conn(elem, i); }
  __syncthreads();

  if (threadIdx.x == 0) {
    el.indices(u_offsets, sm.conn[z], sm.ids[z]);
    X_el.indices(X_offsets, sm.conn[z], sm.X_ids[z]);
  }
  __syncthreads();

  for (uint32_t i = threadIdx.x; i < NNX; i += tpe) {
    for (uint32_t d = 0; d < 3; d++) { sm.X_e[z][i][d] = X_data(sm.X_ids[z][i], d); }
  }
  if constexpr (with_u) {
    for (uint32_t i = threadIdx.x; i < NN; i += tpe) {
      for (uint32_t c = 0; c < NC; c++) { sm.u_e[z][i][c] = u_data(sm.ids[z][i], c); }
    }
  }
  __syncthreads();

  return active;
}

template < Geometry geom, uint32_t NC, uint32_t P, uint32_t E >
__device__ mat3 block_jacobian(const ResidualBlockSmem<geom, NC, P, E> & sm, uint32_t z, uint32_t q) {
  constexpr uint32_t NNX = sm.NNX;
  mat3 dX_dxi{};
  for (uint32_t i = 0; i < NNX; i++) {
    vec3 g = {sm.GX[(q * NNX + i) * 3 + 0], sm.GX[(q * NNX + i) * 3 + 1], sm.GX[(q * NNX + i) * 3 + 2]};
    for (uint32_t d = 0; d < 3; d++) {
      dX_dxi[d] += sm.X_e[z][i][d] * g;
    }
  }
  return dX_dxi;
}

template < Geometry geom, uint32_t NC, uint32_t P, uint32_t E >
__device__ grad_t<NC> block_gradient_xi(const ResidualBlockSmem<geom, NC, P, E> & sm, uint32_t z, uint32_t q) {
  constexpr uint32_t NN = sm.NN;
  grad_t<NC> du_dxi{};
  for (uint32_t i = 0; i < NN; i++) {
    vec3 g = {sm.G[(q * NN + i) * 3 + 0], sm.G[(q * NN + i) * 3 + 1], sm.G[(q * NN + i) * 3 + 2]};
    if constexpr (NC == 1) {
      du_dxi += sm.u_e[z][i][0] * g;
    } else {
      for (uint32_t c = 0; c < NC; c++) {
        du_dxi[c] += sm.u_e[z][i][c] * g;
      }
    }
  }
  return du_dxi;
}

// integrate sm.t against the test function gradients and scatter
template < Geometry geom, uint32_t NC, uint32_t P, uint32_t E >
__device__ void block_integrate_scatter(const ResidualBlockSmem<geom, NC, P, E> & sm, uint32_t z,
                                        nd::view<double, 2, gpu_mem> r_data) {
  constexpr uint32_t NN = sm.NN;
  constexpr uint32_t NQ = sm.NQ;
  for (uint32_t i = threadIdx.x; i < NN; i += blockDim.x) {
    for (uint32_t c = 0; c < NC; c++) {
      double sum = 0.0;
      for (uint32_t q = 0; q < NQ; q++) {
        for (uint32_t d = 0; d < 3; d++) {
          sum += sm.G[(q * NN + i) * 3 + d] * sm.t[z][q][c][d];
        }
      }
      atomicAdd(&r_data(sm.ids[z][i], c), sum);
    }
  }
}

#ifndef FUSION_ONLY_SEPARATED

template < Geometry geom, uint32_t NC, uint32_t P, uint32_t E, typename Material >
__global__ void fused_residual_kernel(
  nd::view<double, 2, gpu_mem> r_data,
  GeometryInfo u_offsets, GeometryInfo X_offsets,
  FiniteElement< geom, Family::H1 > el,
  FiniteElement< geom, Family::H1 > X_el,
  nd::view<const double, 2, gpu_mem> u_data,
  nd::view<const double, 2, gpu_mem> X_data,
  nd::view<const Connection, 2, gpu_mem> conn,
  nd::view<const int, 1, gpu_mem> elements,
  nd::view<const double, 3, gpu_mem> G,
  nd::view<const double, 3, gpu_mem> GX,
  nd::view<const double, 1, gpu_mem> W,
  Material material) {

  __shared__ ResidualBlockSmem<geom, NC, P, E> sm;
  constexpr uint32_t NQ = sm.NQ;
  const uint32_t z = threadIdx.z;

  bool active = residual_block_setup<true>(sm, el, X_el, u_offsets, X_offsets,
                                           u_data, X_data, conn, elements, G, GX, W);

  for (uint32_t q = threadIdx.x; q < NQ; q += blockDim.x) {
    mat3 dX_dxi = block_jacobian(sm, z, q);
    mat3 A = adj(dX_dxi);
    grad_t<NC> du_dX = dot(block_gradient_xi(sm, z, q), inv(dX_dxi));
    grad_t<NC> flux = material(du_dX);
    double w = sm.W[q];
    for (uint32_t c = 0; c < NC; c++) {
      vec3 fc;
      if constexpr (NC == 1) { fc = flux; } else { fc = flux[c]; }
      vec3 t = w * dot(A, fc);
      for (uint32_t d = 0; d < 3; d++) { sm.t[z][q][c][d] = t[d]; }
    }
  }
  __syncthreads();

  if (!active) return;
  block_integrate_scatter(sm, z, r_data);
}

template < Geometry geom, uint32_t NC, uint32_t P, typename Material >
void fused_residual_gpu(Residual<Family::H1, gpu_mem> & r,
                        const Field<Family::H1, gpu_mem> & u,
                        const Domain<gpu_mem> & domain,
                        Material material,
                        const nd::array<double, 3, gpu_mem> & G,
                        const nd::array<double, 3, gpu_mem> & GX,
                        const nd::array<double, 1, gpu_mem> & W) {
  constexpr uint32_t E = elements_per_block_residual<geom, NC, P>();
  FiniteElement< geom, Family::H1 > el{P};
  FiniteElement< geom, Family::H1 > X_el{1};
  nd::view<const int, 1, gpu_mem> elements = domain.active_elements[geom];

  zero(r.data);

  dim3 block(threads_per_element<geom, P>(), 1, E);
  uint32_t grid = (elements.shape[0] + E - 1) / E;
  fused_residual_kernel<geom, NC, P, E><<<grid, block>>>(
    r.data, u.offsets, domain.mesh.X.offsets, el, X_el, u.data, domain.mesh.X.data,
    domain.mesh[geom], elements, G, GX, W, material);
  CUDA_CHECK(cudaGetLastError());
}

#endif // !FUSION_ONLY_SEPARATED

#ifndef FUSION_ONLY_FUSED

template < Geometry geom, uint32_t NC, uint32_t P, uint32_t E >
__global__ void staged_interpolate_kernel(
  nd::view<double, 3, gpu_mem> du_q,           // (num_qpts, NC, 3)
  GeometryInfo u_offsets, GeometryInfo X_offsets,
  FiniteElement< geom, Family::H1 > el,
  FiniteElement< geom, Family::H1 > X_el,
  nd::view<const double, 2, gpu_mem> u_data,
  nd::view<const double, 2, gpu_mem> X_data,
  nd::view<const Connection, 2, gpu_mem> conn,
  nd::view<const int, 1, gpu_mem> elements,
  nd::view<const double, 3, gpu_mem> G,
  nd::view<const double, 3, gpu_mem> GX,
  nd::view<const double, 1, gpu_mem> W) {

  __shared__ ResidualBlockSmem<geom, NC, P, E> sm;
  constexpr uint32_t NQ = sm.NQ;
  const uint32_t z = threadIdx.z;

  bool active = residual_block_setup<true>(sm, el, X_el, u_offsets, X_offsets,
                                           u_data, X_data, conn, elements, G, GX, W);
  if (!active) return;

  uint32_t e = blockIdx.x * E + z;
  for (uint32_t q = threadIdx.x; q < NQ; q += blockDim.x) {
    mat3 dX_dxi = block_jacobian(sm, z, q);
    grad_t<NC> du_dX = dot(block_gradient_xi(sm, z, q), inv(dX_dxi));
    for (uint32_t c = 0; c < NC; c++) {
      for (uint32_t d = 0; d < 3; d++) {
        if constexpr (NC == 1) {
          du_q(e * NQ + q, c, d) = du_dX[d];
        } else {
          du_q(e * NQ + q, c, d) = du_dX[c][d];
        }
      }
    }
  }
}

template < Geometry geom, uint32_t NC, uint32_t P, uint32_t E >
__global__ void staged_integrate_residual_kernel(
  nd::view<double, 2, gpu_mem> r_data,
  GeometryInfo u_offsets, GeometryInfo X_offsets,
  FiniteElement< geom, Family::H1 > el,
  FiniteElement< geom, Family::H1 > X_el,
  nd::view<const double, 3, gpu_mem> flux_q,   // (num_qpts, NC, 3)
  nd::view<const double, 2, gpu_mem> u_data,
  nd::view<const double, 2, gpu_mem> X_data,
  nd::view<const Connection, 2, gpu_mem> conn,
  nd::view<const int, 1, gpu_mem> elements,
  nd::view<const double, 3, gpu_mem> G,
  nd::view<const double, 3, gpu_mem> GX,
  nd::view<const double, 1, gpu_mem> W) {

  __shared__ ResidualBlockSmem<geom, NC, P, E> sm;
  constexpr uint32_t NQ = sm.NQ;
  const uint32_t z = threadIdx.z;

  bool active = residual_block_setup<false>(sm, el, X_el, u_offsets, X_offsets,
                                            u_data, X_data, conn, elements, G, GX, W);

  uint32_t e = blockIdx.x * E + z;
  uint32_t source_e = active ? e : 0;
  for (uint32_t q = threadIdx.x; q < NQ; q += blockDim.x) {
    mat3 A = adj(block_jacobian(sm, z, q));
    double w = sm.W[q];
    for (uint32_t c = 0; c < NC; c++) {
      vec3 f = {flux_q(source_e * NQ + q, c, 0),
                flux_q(source_e * NQ + q, c, 1),
                flux_q(source_e * NQ + q, c, 2)};
      vec3 t = w * dot(A, f);
      for (uint32_t d = 0; d < 3; d++) { sm.t[z][q][c][d] = t[d]; }
    }
  }
  __syncthreads();

  if (!active) return;
  block_integrate_scatter(sm, z, r_data);
}

#endif // !FUSION_ONLY_FUSED

////////////////////////////////////////////////////////////////////////////////
// stiffness kernels: one element per block, one thread per (I,J) node pair
////////////////////////////////////////////////////////////////////////////////

// shared-memory staging common to the fused / staged stiffness kernels
template < Geometry geom, uint32_t NC, uint32_t P >
struct StiffnessBlockSmem {
  static constexpr uint32_t NN = nodes_per_element<geom, P>();
  static constexpr uint32_t NNX = nodes_per_element<geom, 1>();
  static constexpr uint32_t NQ = qpts_per_element<geom, P>();
  static constexpr uint32_t CM = conn_row_size<geom>();

  Connection conn[CM];
  uint32_t ids[NN];
  uint32_t X_ids[NNX];
  double X_e[NNX][3];
  double u_e[NN][NC];
  double wJ[NQ];
  vec3 dNdX[NQ][NN];
  grad_t<NC> du_dX[NQ];   // only used by the fused kernel
  jac_t<NC> D[NQ];
};

template < Geometry geom, uint32_t P >
constexpr uint32_t stiffness_block_size() {
  constexpr uint32_t NN = nodes_per_element<geom, P>();
  constexpr uint32_t NQ = qpts_per_element<geom, P>();
  uint32_t bs = NN * NN;
  if (bs < NQ) { bs = NQ; }
  if (bs < 32) { bs = 32; }
  return bs;
}

// cooperative gather + per-qpt geometry (wJ, dN/dX); the constitutive tangent
// D is filled in afterwards by the caller (computed in the fused kernel,
// loaded from global memory in the staged kernel)
template < bool with_u, Geometry geom, uint32_t NC, uint32_t P >
__device__ void stiffness_block_setup(
  StiffnessBlockSmem<geom, NC, P> & sm,
  FiniteElement< geom, Family::H1 > el,
  FiniteElement< geom, Family::H1 > X_el,
  GeometryInfo u_offsets, GeometryInfo X_offsets,
  nd::view<const double, 2, gpu_mem> u_data,
  nd::view<const double, 2, gpu_mem> X_data,
  nd::view<const Connection, 2, gpu_mem> conn,
  uint32_t elem,
  nd::view<const double, 3, gpu_mem> G,
  nd::view<const double, 3, gpu_mem> GX,
  nd::view<const double, 1, gpu_mem> W) {

  constexpr uint32_t NN = sm.NN;
  constexpr uint32_t NNX = sm.NNX;
  constexpr uint32_t NQ = sm.NQ;
  const uint32_t tid = threadIdx.x;

  uint32_t cn = conn.shape[1];
  for (uint32_t i = tid; i < cn; i += blockDim.x) { sm.conn[i] = conn(elem, i); }
  __syncthreads();

  if (tid == 0) {
    el.indices(u_offsets, sm.conn, sm.ids);
    X_el.indices(X_offsets, sm.conn, sm.X_ids);
  }
  __syncthreads();

  for (uint32_t i = tid; i < NNX; i += blockDim.x) {
    for (uint32_t d = 0; d < 3; d++) { sm.X_e[i][d] = X_data(sm.X_ids[i], d); }
  }
  if constexpr (with_u) {
    for (uint32_t i = tid; i < NN; i += blockDim.x) {
      for (uint32_t c = 0; c < NC; c++) { sm.u_e[i][c] = u_data(sm.ids[i], c); }
    }
  }
  __syncthreads();

  if (tid < NQ) {
    const uint32_t q = tid;
    mat3 dX_dxi{};
    for (uint32_t i = 0; i < NNX; i++) {
      vec3 g = {GX(q, i, 0), GX(q, i, 1), GX(q, i, 2)};
      for (uint32_t d = 0; d < 3; d++) {
        dX_dxi[d] += sm.X_e[i][d] * g;
      }
    }
    mat3 dxi_dX = inv(dX_dxi);
    sm.wJ[q] = W(q) * det(dX_dxi);
    for (uint32_t i = 0; i < NN; i++) {
      vec3 g = {G(q, i, 0), G(q, i, 1), G(q, i, 2)};
      sm.dNdX[q][i] = dot(g, dxi_dX);
    }
    if constexpr (with_u) {
      grad_t<NC> du_dxi{};
      for (uint32_t i = 0; i < NN; i++) {
        vec3 g = {G(q, i, 0), G(q, i, 1), G(q, i, 2)};
        if constexpr (NC == 1) {
          du_dxi += sm.u_e[i][0] * g;
        } else {
          for (uint32_t c = 0; c < NC; c++) {
            du_dxi[c] += sm.u_e[i][c] * g;
          }
        }
      }
      sm.du_dX[q] = dot(du_dxi, dxi_dX);
    }
  }
  __syncthreads();
}

// accumulate the (I,J) blocks assigned to this thread and scatter into the CSR
template < Geometry geom, uint32_t NC, uint32_t P >
__device__ void stiffness_block_accumulate_scatter(
  const StiffnessBlockSmem<geom, NC, P> & sm,
  nd::view<double, 1, gpu_mem> values,
  nd::view<const int, 1, gpu_mem> row_ptr,
  nd::view<const int, 1, gpu_mem> col_ind) {

  constexpr uint32_t NN = sm.NN;
  constexpr uint32_t NQ = sm.NQ;

  for (uint32_t pair = threadIdx.x; pair < NN * NN; pair += blockDim.x) {
    uint32_t I = pair / NN;
    uint32_t J = pair % NN;

    double B[NC][NC] = {};
    for (uint32_t q = 0; q < NQ; q++) {
      for (uint32_t i = 0; i < NC; i++) {
        for (uint32_t j = 0; j < NC; j++) {
          B[i][j] += sm.wJ[q] * jac_contract(sm.D[q], i, j, sm.dNdX[q][I], sm.dNdX[q][J]);
        }
      }
    }

    for (uint32_t i = 0; i < NC; i++) {
      int row = int(sm.ids[I] * NC + i);
      int row_start = row_ptr(row);
      int row_end = row_ptr(row + 1);
      for (uint32_t j = 0; j < NC; j++) {
        int col = int(sm.ids[J] * NC + j);
        int position = find_column_in_sparse_row(col_ind, row_start, row_end, col);
        atomicAdd(&values(position), B[i][j]);
      }
    }
  }
}

#ifndef FUSION_ONLY_SEPARATED

template < Geometry geom, uint32_t NC, uint32_t P, typename MaterialJac >
__global__ void fused_stiffness_kernel(
  nd::view<double, 1, gpu_mem> values,
  nd::view<const int, 1, gpu_mem> row_ptr,
  nd::view<const int, 1, gpu_mem> col_ind,
  GeometryInfo u_offsets, GeometryInfo X_offsets,
  FiniteElement< geom, Family::H1 > el,
  FiniteElement< geom, Family::H1 > X_el,
  nd::view<const double, 2, gpu_mem> u_data,
  nd::view<const double, 2, gpu_mem> X_data,
  nd::view<const Connection, 2, gpu_mem> conn,
  nd::view<const int, 1, gpu_mem> elements,
  nd::view<const double, 3, gpu_mem> G,
  nd::view<const double, 3, gpu_mem> GX,
  nd::view<const double, 1, gpu_mem> W,
  MaterialJac material_jac) {

  __shared__ StiffnessBlockSmem<geom, NC, P> sm;
  constexpr uint32_t NQ = sm.NQ;

  uint32_t e = blockIdx.x;
  if (e >= elements.shape[0]) return;

  stiffness_block_setup<true>(sm, el, X_el, u_offsets, X_offsets,
                              u_data, X_data, conn, elements(e), G, GX, W);

  if (threadIdx.x < NQ) {
    sm.D[threadIdx.x] = material_jac(sm.du_dX[threadIdx.x]);
  }
  __syncthreads();

  stiffness_block_accumulate_scatter(sm, values, row_ptr, col_ind);
}

template < Geometry geom, uint32_t NC, uint32_t P, typename MaterialJac >
void fused_stiffness_gpu(sparse_matrix<gpu_mem> & K,
                         const Field<Family::H1, gpu_mem> & u,
                         const Domain<gpu_mem> & domain,
                         MaterialJac material_jac,
                         const nd::array<double, 3, gpu_mem> & G,
                         const nd::array<double, 3, gpu_mem> & GX,
                         const nd::array<double, 1, gpu_mem> & W) {
  FiniteElement< geom, Family::H1 > el{P};
  FiniteElement< geom, Family::H1 > X_el{1};
  nd::view<const int, 1, gpu_mem> elements = domain.active_elements[geom];

  zero(K.values);

  fused_stiffness_kernel<geom, NC, P><<<elements.shape[0], stiffness_block_size<geom, P>()>>>(
    K.values, K.row_ptr, K.col_ind, u.offsets, domain.mesh.X.offsets, el, X_el,
    u.data, domain.mesh.X.data, domain.mesh[geom], elements, G, GX, W, material_jac);
  CUDA_CHECK(cudaGetLastError());
}

#endif // !FUSION_ONLY_SEPARATED

#ifndef FUSION_ONLY_FUSED

template < Geometry geom, uint32_t NC, uint32_t P >
__global__ void staged_integrate_stiffness_kernel(
  nd::view<double, 1, gpu_mem> values,
  nd::view<const int, 1, gpu_mem> row_ptr,
  nd::view<const int, 1, gpu_mem> col_ind,
  GeometryInfo u_offsets, GeometryInfo X_offsets,
  FiniteElement< geom, Family::H1 > el,
  FiniteElement< geom, Family::H1 > X_el,
  nd::view<const double, 2, gpu_mem> dflux_q,  // (num_qpts, NC*3*NC*3)
  nd::view<const double, 2, gpu_mem> u_data,
  nd::view<const double, 2, gpu_mem> X_data,
  nd::view<const Connection, 2, gpu_mem> conn,
  nd::view<const int, 1, gpu_mem> elements,
  nd::view<const double, 3, gpu_mem> G,
  nd::view<const double, 3, gpu_mem> GX,
  nd::view<const double, 1, gpu_mem> W) {

  __shared__ StiffnessBlockSmem<geom, NC, P> sm;
  constexpr uint32_t NQ = sm.NQ;
  constexpr uint32_t dsz = NC * 3 * NC * 3;

  uint32_t e = blockIdx.x;
  if (e >= elements.shape[0]) return;

  stiffness_block_setup<false>(sm, el, X_el, u_offsets, X_offsets,
                               u_data, X_data, conn, elements(e), G, GX, W);

  // stage the constitutive tangent from global memory into shared memory
  double * shD_flat = reinterpret_cast<double *>(sm.D);
  for (uint32_t idx = threadIdx.x; idx < NQ * dsz; idx += blockDim.x) {
    shD_flat[idx] = dflux_q(e * NQ + idx / dsz, idx % dsz);
  }
  __syncthreads();

  stiffness_block_accumulate_scatter(sm, values, row_ptr, col_ind);
}

////////////////////////////////////////////////////////////////////////////////
// staged launchers
////////////////////////////////////////////////////////////////////////////////

template < Geometry geom, uint32_t NC, uint32_t P >
void staged_interpolate_gpu(nd::array<double, 3, gpu_mem> & du_q,
                            const Field<Family::H1, gpu_mem> & u,
                            const Domain<gpu_mem> & domain,
                            const nd::array<double, 3, gpu_mem> & G,
                            const nd::array<double, 3, gpu_mem> & GX,
                            const nd::array<double, 1, gpu_mem> & W) {
  constexpr uint32_t E = elements_per_block_residual<geom, NC, P>();
  FiniteElement< geom, Family::H1 > el{P};
  FiniteElement< geom, Family::H1 > X_el{1};
  nd::view<const int, 1, gpu_mem> elements = domain.active_elements[geom];
  dim3 block(threads_per_element<geom, P>(), 1, E);
  uint32_t grid = (elements.shape[0] + E - 1) / E;
  staged_interpolate_kernel<geom, NC, P, E><<<grid, block>>>(
    du_q, u.offsets, domain.mesh.X.offsets, el, X_el, u.data, domain.mesh.X.data,
    domain.mesh[geom], elements, G, GX, W);
  CUDA_CHECK(cudaGetLastError());
}

template < typename in_t, typename out_t, typename Model >
void staged_material_gpu(uint32_t n, Model model, out_t * output, const in_t * input) {
  uint32_t block = 128;
  uint32_t grid = (n + block - 1) / block;
  forall_kernel<<<grid, block>>>(n, model, output, input);
  CUDA_CHECK(cudaGetLastError());
}

template < Geometry geom, uint32_t NC, uint32_t P >
void staged_integrate_residual_gpu(Residual<Family::H1, gpu_mem> & r,
                                   const nd::array<double, 3, gpu_mem> & flux_q,
                                   const Field<Family::H1, gpu_mem> & u,
                                   const Domain<gpu_mem> & domain,
                                   const nd::array<double, 3, gpu_mem> & G,
                                   const nd::array<double, 3, gpu_mem> & GX,
                                   const nd::array<double, 1, gpu_mem> & W) {
  constexpr uint32_t E = elements_per_block_residual<geom, NC, P>();
  FiniteElement< geom, Family::H1 > el{P};
  FiniteElement< geom, Family::H1 > X_el{1};
  nd::view<const int, 1, gpu_mem> elements = domain.active_elements[geom];
  zero(r.data);
  dim3 block(threads_per_element<geom, P>(), 1, E);
  uint32_t grid = (elements.shape[0] + E - 1) / E;
  staged_integrate_residual_kernel<geom, NC, P, E><<<grid, block>>>(
    r.data, u.offsets, domain.mesh.X.offsets, el, X_el, flux_q, u.data, domain.mesh.X.data,
    domain.mesh[geom], elements, G, GX, W);
  CUDA_CHECK(cudaGetLastError());
}

template < Geometry geom, uint32_t NC, uint32_t P >
void staged_integrate_stiffness_gpu(sparse_matrix<gpu_mem> & K,
                                    const nd::array<double, 2, gpu_mem> & dflux_q,
                                    const Field<Family::H1, gpu_mem> & u,
                                    const Domain<gpu_mem> & domain,
                                    const nd::array<double, 3, gpu_mem> & G,
                                    const nd::array<double, 3, gpu_mem> & GX,
                                    const nd::array<double, 1, gpu_mem> & W) {
  FiniteElement< geom, Family::H1 > el{P};
  FiniteElement< geom, Family::H1 > X_el{1};
  nd::view<const int, 1, gpu_mem> elements = domain.active_elements[geom];
  zero(K.values);
  staged_integrate_stiffness_kernel<geom, NC, P><<<elements.shape[0], stiffness_block_size<geom, P>()>>>(
    K.values, K.row_ptr, K.col_ind, u.offsets, domain.mesh.X.offsets, el, X_el,
    dflux_q, u.data, domain.mesh.X.data, domain.mesh[geom], elements, G, GX, W);
  CUDA_CHECK(cudaGetLastError());
}

#endif // !FUSION_ONLY_FUSED

////////////////////////////////////////////////////////////////////////////////
// case driver
////////////////////////////////////////////////////////////////////////////////

struct PhaseTimes {
  double interpolate = 0.0, material = 0.0, integrate = 0.0;
  double total() const { return interpolate + material + integrate; }
};

// cudaEvent-based timing: events are recorded on the default stream around each
// phase, and the host only synchronizes once, on the final event, after all
// phases of a repetition have been issued.
struct PhaseEvents {
  cudaEvent_t ev[4];
  PhaseEvents() { for (auto & e : ev) CUDA_CHECK(cudaEventCreate(&e)); }
  ~PhaseEvents() { for (auto & e : ev) cudaEventDestroy(e); }

  void mark(int i) { CUDA_CHECK(cudaEventRecord(ev[i])); }

  // elapsed seconds between consecutive marks; syncs on the last recorded event
  PhaseTimes elapsed(int n) {
    CUDA_CHECK(cudaEventSynchronize(ev[n - 1]));
    float ms[3] = {};
    for (int i = 0; i + 1 < n; i++) {
      CUDA_CHECK(cudaEventElapsedTime(&ms[i], ev[i], ev[i + 1]));
    }
    return PhaseTimes{ms[0] * 1.0e-3, ms[1] * 1.0e-3, ms[2] * 1.0e-3};
  }
};

// run f() `reps` times (after `warmup` calls), timing with events; returns the
// fastest device-side time in seconds
template < typename callable >
static double bench_gpu(callable && f, int warmup, int reps) {
  PhaseEvents events;
  for (int i = 0; i < warmup; i++) { f(); }
  CUDA_CHECK(cudaDeviceSynchronize());
  double best = 1.0e30;
  for (int i = 0; i < reps; i++) {
    events.mark(0);
    f();
    events.mark(1);
    best = std::min(best, events.elapsed(2).interpolate);
  }
  return best;
}

static uint64_t gpu_free_bytes() {
  size_t free_b = 0, total_b = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
  return free_b;
}

template < Geometry geom, uint32_t NC, uint32_t P, typename Flux, typename Jac >
void run_case_gpu(CaseInfo info, const Mesh<> & mesh, Flux flux_model, Jac jac_model,
                  int warmup, int reps) {

  // host-side setup (also used as reference solution)
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

  const uint64_t nq = info.num_qpts;
  const uint64_t nelem = info.num_elements;
  constexpr uint32_t NN = nodes_per_element<geom, P>();

  FEMTO_ASSERT(mesh.X.degree == 1, "experiment assumes linear mesh geometry");
  nd::array<double, 3, cpu_mem> G, GX;
  nd::array<double, 1, cpu_mem> W, W_unused;
  build_qtables<geom>(h_domain, P, G, W);
  build_qtables<geom>(h_domain, 1, GX, W_unused);
  nd::array<double, 3, gpu_mem> d_G = G;
  nd::array<double, 3, gpu_mem> d_GX = GX;
  nd::array<double, 1, gpu_mem> d_W = W;

  printf("# case: %s %s %s %s p%u: %llu elements, %llu qpts, %llu unknowns\n",
         info.device.c_str(), info.physics.c_str(), info.geom.c_str(), info.size.c_str(), P,
         (unsigned long long)nelem, (unsigned long long)nq, (unsigned long long)info.num_unknowns);

  // cpu reference values
  nd::array<double, 3, cpu_mem> h_du_q = evaluate(grad(h_u), h_domain);
  auto h_flux_q = forall(flux_model, h_du_q);
  Residual<Family::H1> h_r = integrate(dot(h_flux_q, grad(phi)), h_domain);

  //////////////////////////////////////////////////////////////////////////////
  // residual
  //////////////////////////////////////////////////////////////////////////////

  Residual<Family::H1, gpu_mem> d_r_sep;
  Residual<Family::H1, gpu_mem> d_r_staged(FunctionSpace{Family::H1, P, NC}, d_mesh);
  Residual<Family::H1, gpu_mem> d_r_fused(FunctionSpace{Family::H1, P, NC}, d_mesh);

#ifndef FUSION_ONLY_FUSED
  {
    PhaseTimes best;
    double best_total = 1.0e30;
    uint64_t measured_peak = 0;
    PhaseEvents events;
    for (int rep = 0; rep < warmup + reps; rep++) {
      uint64_t free0 = gpu_free_bytes();

      events.mark(0);
      nd::array<double, 3, gpu_mem> du_q = evaluate(grad(d_u), d_domain);
      events.mark(1);
      auto flux_q = forall(flux_model, du_q);
      events.mark(2);
      uint64_t free1 = gpu_free_bytes();
      d_r_sep = integrate(dot(flux_q, grad(phi)), d_domain);
      events.mark(3);
      PhaseTimes pt = events.elapsed(4);

      measured_peak = std::max(measured_peak, uint64_t(free0 - free1));
      if (rep >= warmup && pt.total() < best_total) { best_total = pt.total(); best = pt; }
    }
    report_time(info, "residual", "separated", "interpolate", best.interpolate);
    report_time(info, "residual", "separated", "material", best.material);
    report_time(info, "residual", "separated", "integrate", best.integrate);
    report_time(info, "residual", "separated", "total", best.total());

    report_mem(info, "residual", "separated", "du_dX_q", nq * NC * 3 * 8);
    report_mem(info, "residual", "separated", "flux_q", nq * NC * 3 * 8);
    report_mem(info, "residual", "separated", "element_residuals", nelem * NN * NC * 8);
    report_mem(info, "residual", "separated", "measured_intermediates", measured_peak);
  }

  // staged: preallocated intermediates, kernel-only timings
  {
    nd::array<double, 3, gpu_mem> du_q({uint32_t(nq), NC, 3u});
    nd::array<double, 3, gpu_mem> flux_q({uint32_t(nq), NC, 3u});

    PhaseTimes best;
    double best_total = 1.0e30;
    PhaseEvents events;
    for (int rep = 0; rep < warmup + reps; rep++) {
      events.mark(0);
      staged_interpolate_gpu<geom, NC, P>(du_q, d_u, d_domain, d_G, d_GX, d_W);
      events.mark(1);
      staged_material_gpu(uint32_t(nq), flux_model,
                          reinterpret_cast<grad_t<NC> *>(flux_q.data()),
                          reinterpret_cast<const grad_t<NC> *>(du_q.data()));
      events.mark(2);
      staged_integrate_residual_gpu<geom, NC, P>(d_r_staged, flux_q, d_u, d_domain, d_G, d_GX, d_W);
      events.mark(3);
      PhaseTimes pt = events.elapsed(4);
      if (rep >= warmup && pt.total() < best_total) { best_total = pt.total(); best = pt; }
    }
    report_time(info, "residual", "staged", "interpolate", best.interpolate);
    report_time(info, "residual", "staged", "material", best.material);
    report_time(info, "residual", "staged", "integrate", best.integrate);
    report_time(info, "residual", "staged", "total", best.total());
  }
#endif

#ifndef FUSION_ONLY_SEPARATED
  {
    double t = bench_gpu([&]() {
      fused_residual_gpu<geom, NC, P>(d_r_fused, d_u, d_domain, flux_model, d_G, d_GX, d_W);
    }, warmup, reps);
    report_time(info, "residual", "fused", "total", t);
    report_mem(info, "residual", "fused", "qtables", (G.size() + GX.size() + W.size()) * 8);
  }
#endif

#if !defined(FUSION_ONLY_FUSED) && !defined(FUSION_ONLY_SEPARATED)
  {
    Residual<Family::H1> r_sep_h = d_r_sep;
    Residual<Family::H1> r_staged_h = d_r_staged;
    Residual<Family::H1> r_fused_h = d_r_fused;
    report_verify(info, "residual", "gpu_separated_vs_cpu",
                  rel_l2_diff(&h_r.data[0], &r_sep_h.data[0], h_r.size()), 1.0e-8);
    report_verify(info, "residual", "gpu_staged_vs_cpu",
                  rel_l2_diff(&h_r.data[0], &r_staged_h.data[0], h_r.size()), 1.0e-8);
    report_verify(info, "residual", "gpu_fused_vs_cpu",
                  rel_l2_diff(&h_r.data[0], &r_fused_h.data[0], h_r.size()), 1.0e-8);
  }
#endif

  //////////////////////////////////////////////////////////////////////////////
  // stiffness (one shared CSR matrix; approaches verified via host copies)
  //////////////////////////////////////////////////////////////////////////////

  uint64_t est_nnz = info.num_unknowns * couplings_per_node(geom, P) * NC;
  uint64_t est_gpu_bytes = est_nnz * 12
                         + 2 * nq * NC * 3 * NC * 3 * 8
                         + nq * NC * 3 * 8;
  uint64_t est_host_bytes = est_nnz * 16;
  if (est_gpu_bytes > (gpu_free_bytes() * 4) / 5 || est_host_bytes > 40ull * 1000 * 1000 * 1000) {
    report_skip(info, "stiffness", est_gpu_bytes);
    return;
  }

  sparse_matrix<gpu_mem> d_K = blank_sparse_matrix(grad(phi), grad(phi), d_domain);

  report_mem(info, "stiffness", "both", "csr_matrix", d_K.nnz * 12 + (d_K.nrows + 1) * 4);

#ifndef FUSION_ONLY_FUSED
  {
    PhaseTimes best;
    double best_total = 1.0e30;
    uint64_t measured_peak = 0;
    PhaseEvents events;
    for (int rep = 0; rep < warmup + reps; rep++) {
      uint64_t free0 = gpu_free_bytes();

      events.mark(0);
      nd::array<double, 3, gpu_mem> du_q = evaluate(grad(d_u), d_domain);
      events.mark(1);
      auto dflux_q = forall(jac_model, du_q);
      events.mark(2);
      uint64_t free1 = gpu_free_bytes();
      auto fill = integrate(dot(grad(phi), dflux_q, grad(phi)), d_domain);
      fill(d_K);
      events.mark(3);
      PhaseTimes pt = events.elapsed(4);

      measured_peak = std::max(measured_peak, uint64_t(free0 - free1));
      if (rep >= warmup && pt.total() < best_total) { best_total = pt.total(); best = pt; }
    }
    report_time(info, "stiffness", "separated", "interpolate", best.interpolate);
    report_time(info, "stiffness", "separated", "material", best.material);
    report_time(info, "stiffness", "separated", "integrate", best.integrate);
    report_time(info, "stiffness", "separated", "total", best.total());

    report_mem(info, "stiffness", "separated", "du_dX_q", nq * NC * 3 * 8);
    report_mem(info, "stiffness", "separated", "dflux_q", nq * NC * 3 * NC * 3 * 8);
    report_mem(info, "stiffness", "separated", "measured_intermediates", measured_peak);
  }
#endif

#if !defined(FUSION_ONLY_FUSED) && !defined(FUSION_ONLY_SEPARATED)
  nd::array<double, 1, cpu_mem> K_ref = d_K.values;
#endif

#ifndef FUSION_ONLY_FUSED
  // staged: preallocated intermediates, kernel-only timings
  {
    nd::array<double, 3, gpu_mem> du_q({uint32_t(nq), NC, 3u});
    nd::array<double, 2, gpu_mem> dflux_q({uint32_t(nq), NC * 3 * NC * 3});

    PhaseTimes best;
    double best_total = 1.0e30;
    PhaseEvents events;
    for (int rep = 0; rep < warmup + reps; rep++) {
      events.mark(0);
      staged_interpolate_gpu<geom, NC, P>(du_q, d_u, d_domain, d_G, d_GX, d_W);
      events.mark(1);
      staged_material_gpu(uint32_t(nq), jac_model,
                          reinterpret_cast<jac_t<NC> *>(dflux_q.data()),
                          reinterpret_cast<const grad_t<NC> *>(du_q.data()));
      events.mark(2);
      staged_integrate_stiffness_gpu<geom, NC, P>(d_K, dflux_q, d_u, d_domain, d_G, d_GX, d_W);
      events.mark(3);
      PhaseTimes pt = events.elapsed(4);
      if (rep >= warmup && pt.total() < best_total) { best_total = pt.total(); best = pt; }
    }
    report_time(info, "stiffness", "staged", "interpolate", best.interpolate);
    report_time(info, "stiffness", "staged", "material", best.material);
    report_time(info, "stiffness", "staged", "integrate", best.integrate);
    report_time(info, "stiffness", "staged", "total", best.total());
  }
#endif

#if !defined(FUSION_ONLY_FUSED) && !defined(FUSION_ONLY_SEPARATED)
  {
    nd::array<double, 1, cpu_mem> K_staged = d_K.values;
    report_verify(info, "stiffness", "gpu_separated_vs_gpu_staged",
                  rel_l2_diff(&K_ref[0], &K_staged[0], d_K.nnz), 1.0e-10);
  }
#endif

#ifndef FUSION_ONLY_SEPARATED
  {
    double t = bench_gpu([&]() {
      fused_stiffness_gpu<geom, NC, P>(d_K, d_u, d_domain, jac_model, d_G, d_GX, d_W);
    }, warmup, reps);
    report_time(info, "stiffness", "fused", "total", t);
    report_mem(info, "stiffness", "fused", "qtables", (G.size() + GX.size() + W.size()) * 8);
  }
#endif

#if !defined(FUSION_ONLY_FUSED) && !defined(FUSION_ONLY_SEPARATED)
  {
    nd::array<double, 1, cpu_mem> K_fused = d_K.values;
    report_verify(info, "stiffness", "gpu_separated_vs_gpu_fused",
                  rel_l2_diff(&K_ref[0], &K_fused[0], d_K.nnz), 1.0e-10);
  }
#endif
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char * argv[]) {

  int max_size = (argc > 1) ? atoi(argv[1]) : 2;   // 0 = medium only, 2 = all
  int warmup = (argc > 2) ? atoi(argv[2]) : 1;
  int reps = (argc > 3) ? atoi(argv[3]) : 3;

  const char * size_labels[3] = {"medium", "large", "huge"};
  uint32_t hex_n[3] = {32, 64, 96};
  uint32_t tet_n[3] = {20, 40, 64};

  for (int s = 0; s <= max_size; s++) {
    // hexahedra
    {
      Mesh<> mesh = Mesh<>::cuboid({hex_n[s], hex_n[s], hex_n[s]}, vec3{1.0, 1.0, 1.0});
      CaseInfo poisson{"gpu", "poisson", "hex", size_labels[s]};
      CaseInfo elasticity{"gpu", "elasticity", "hex", size_labels[s]};
      run_case_gpu<Geometry::Hexahedron, 1, 1>(poisson, mesh, PoissonModel{kappa0}, PoissonJacobianModel{kappa0}, warmup, reps);
      run_case_gpu<Geometry::Hexahedron, 1, 2>(poisson, mesh, PoissonModel{kappa0}, PoissonJacobianModel{kappa0}, warmup, reps);
      run_case_gpu<Geometry::Hexahedron, 3, 1>(elasticity, mesh, NeoHookeanModel{lambda0, mu0}, NeoHookeanJacobianModel{lambda0, mu0}, warmup, reps);
      run_case_gpu<Geometry::Hexahedron, 3, 2>(elasticity, mesh, NeoHookeanModel{lambda0, mu0}, NeoHookeanJacobianModel{lambda0, mu0}, warmup, reps);
    }

    // tetrahedra
    {
      Mesh<> mesh = tet_cuboid(tet_n[s]);
      CaseInfo poisson{"gpu", "poisson", "tet", size_labels[s]};
      CaseInfo elasticity{"gpu", "elasticity", "tet", size_labels[s]};
      run_case_gpu<Geometry::Tetrahedron, 1, 1>(poisson, mesh, PoissonModel{kappa0}, PoissonJacobianModel{kappa0}, warmup, reps);
      run_case_gpu<Geometry::Tetrahedron, 1, 2>(poisson, mesh, PoissonModel{kappa0}, PoissonJacobianModel{kappa0}, warmup, reps);
      run_case_gpu<Geometry::Tetrahedron, 3, 1>(elasticity, mesh, NeoHookeanModel{lambda0, mu0}, NeoHookeanJacobianModel{lambda0, mu0}, warmup, reps);
      run_case_gpu<Geometry::Tetrahedron, 3, 2>(elasticity, mesh, NeoHookeanModel{lambda0, mu0}, NeoHookeanJacobianModel{lambda0, mu0}, warmup, reps);
    }
  }

  return 0;
}
