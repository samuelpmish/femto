// Tensor-core (FP64 wmma 8x8x4) experiment for residual and stiffness kernels.
//
// The element-level contractions are written as small GEMMs on shared-memory
// operands and evaluated two ways on identical data:
//   loops : nested-loop contraction, one output entry (or node pair) per thread
//   wmma  : 8x8x4 double fragments, one 8x8 output tile per warp-unit
// plus the library path as reference (sum-factorized on hexes, dense on tets).
//
// stiffness (pass 1, element matrices to global memory, no scatter):
//   A_ij (NN x NN) = B^T (NN x 3NQ) . T_ij (3NQ x NN),
//   B[(q,k)][J] = dN_J(q)[k],  T_ij[(q,k)][J] = wJ(q) sum_m C_q[i,k,j,m] B[(q,m)][J]
// residual, E elements batched per block so the "N" dimension is NC*E:
//   interpolate  D (3NQ x NC*E) = G (3NQ x NN) . U (NN x NC*E)
//   integrate    R (NN x NC*E)  = G^T (NN x 3NQ) . T (3NQ x NC*E)
//
// usage: cuda_wmma_perf [size(0=smoke,1=full)] [warmup] [reps]

#include "fusion_common.hpp"

#include "forall.hpp"
#include "misc/macros.hpp"

#include <mma.h>
#include <cstring>

using namespace femto;
using namespace fusion;
using namespace nvcuda;

static constexpr memory::space gpu_mem = memory::space::gpu;
static constexpr memory::space cpu_mem = memory::space::cpu;

constexpr uint32_t r8(uint32_t x) { return (x + 7) / 8 * 8; }

using frag_a_col = wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::col_major>;
using frag_a_row = wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major>;
using frag_b_row = wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major>;
using frag_c     = wmma::fragment<wmma::accumulator, 8, 8, 4, double>;

struct GeomArgs {
  nd::view<const uint32_t, 2, gpu_mem> ids;     // (nelem, NN)
  nd::view<const uint32_t, 2, gpu_mem> X_ids;   // (nelem, NNX)
  nd::view<const double, 2, gpu_mem> X;
  nd::view<const double, 3, gpu_mem> G;         // (NQ, NN, 3)
  nd::view<const double, 3, gpu_mem> GX;        // (NQ, NNX, 3)
  nd::view<const double, 1, gpu_mem> W;
  uint32_t nelem;
};

template < uint32_t NNX >
__device__ __forceinline__ void jacobian(const GeomArgs & g, uint32_t e, uint32_t q, mat3 & Ji, double & wJ) {
  mat3 dX{};
  for (uint32_t i = 0; i < NNX; i++) {
    vec3 gr = {g.GX(q, i, 0), g.GX(q, i, 1), g.GX(q, i, 2)};
    uint32_t xid = g.X_ids(e, i);
    for (uint32_t d = 0; d < 3; d++) { dX[d] += g.X(xid, d) * gr; }
  }
  Ji = inv(dX);
  wJ = g.W(q) * det(dX);
}

// C(q)[i,k,j,m] in the library's flat (i,k,j,m) order
template < uint32_t NC >
__device__ __forceinline__ uint32_t cidx(uint32_t i, uint32_t k, uint32_t j, uint32_t m) {
  return ((i * 3 + k) * NC + j) * 3 + m;
}

////////////////////////////////////////////////////////////////////////////////
// stiffness pass 1: one element per block, 256 threads
////////////////////////////////////////////////////////////////////////////////

template < Geometry geom, uint32_t NC, uint32_t P >
struct StiffDims {
  static constexpr uint32_t NN = nodes_per_element<geom, P>();
  static constexpr uint32_t NNX = nodes_per_element<geom, 1>();
  static constexpr uint32_t NQ = qpts_per_element<geom, P>();
  static constexpr uint32_t DSZ = NC * 3 * NC * 3;
  static constexpr uint32_t K = 3 * NQ;
  static constexpr uint32_t Kp = r8(K);        // multiple of 4 (and 8)
  static constexpr uint32_t Mp = r8(NN);       // padded NN
  static constexpr uint32_t NP = NC * NC;      // (i,j) pairs
  // (i,j) pairs whose T matrices are built together (shared-memory budget)
  static constexpr uint32_t GRP = (Kp * Mp * 8 * NP <= 24 * 1024) ? NP : ((Kp * Mp * 8 * 3 <= 24 * 1024) ? 3 : 1);
  static constexpr uint32_t smem_common = (Kp * Mp + r8(NQ * DSZ) + r8(NQ)) * 8;   // B, D, wJ (each 32B aligned)
  static constexpr uint32_t smem_wmma = smem_common + GRP * Kp * Mp * 8 + 8 * 64 * 8;
  static constexpr uint32_t useful = NN * NN * K;
  static constexpr uint32_t padded = Mp * Mp * Kp;
};

// shared setup: dN table B[(q,k)][J] (Kp x Mp, zero padded), staged D, wJ
template < Geometry geom, uint32_t NC, uint32_t P >
__device__ void stiff_setup(double * B, double * D, double * wJ, const GeomArgs & g,
                            nd::view<const double, 2, gpu_mem> qdata, uint32_t e) {
  using S = StiffDims<geom, NC, P>;
  const uint32_t tid = threadIdx.x, bs = blockDim.x;
  for (uint32_t idx = tid; idx < S::Kp * S::Mp; idx += bs) { B[idx] = 0.0; }
  for (uint32_t idx = tid; idx < S::NQ * S::DSZ; idx += bs) {
    D[idx] = qdata(e * S::NQ + idx / S::DSZ, idx % S::DSZ);
  }
  __syncthreads();
  for (uint32_t q = tid; q < S::NQ; q += bs) {
    mat3 Ji; double w;
    jacobian<S::NNX>(g, e, q, Ji, w);
    wJ[q] = w;
    for (uint32_t J = 0; J < S::NN; J++) {
      vec3 gr = {g.G(q, J, 0), g.G(q, J, 1), g.G(q, J, 2)};
      vec3 dN = dot(gr, Ji);
      for (uint32_t k = 0; k < 3; k++) { B[(q * 3 + k) * S::Mp + J] = dN[k]; }
    }
  }
  __syncthreads();
}

template < Geometry geom, uint32_t NC, uint32_t P >
__global__ void k_stiff_loops(GeomArgs g, nd::view<const double, 2, gpu_mem> qdata,
                              nd::view<double, 1, gpu_mem> A_e) {
  using S = StiffDims<geom, NC, P>;
  extern __shared__ __align__(32) double smem[];
  double * B = smem;
  double * D = B + S::Kp * S::Mp;
  double * wJ = D + r8(S::NQ * S::DSZ);

  uint32_t e = blockIdx.x;
  if (e >= g.nelem) { return; }
  stiff_setup<geom, NC, P>(B, D, wJ, g, qdata, e);

  for (uint32_t pair = threadIdx.x; pair < S::NN * S::NN; pair += blockDim.x) {
    uint32_t I = pair / S::NN, J = pair % S::NN;
    double acc[NC][NC] = {};
    for (uint32_t q = 0; q < S::NQ; q++) {
      const double * C = D + q * S::DSZ;
      double gI[3], gJ[3];
      for (uint32_t k = 0; k < 3; k++) { gI[k] = B[(q * 3 + k) * S::Mp + I]; gJ[k] = B[(q * 3 + k) * S::Mp + J]; }
      for (uint32_t i = 0; i < NC; i++) {
        for (uint32_t j = 0; j < NC; j++) {
          double s = 0.0;
          for (uint32_t k = 0; k < 3; k++) {
            for (uint32_t m = 0; m < 3; m++) { s += gI[k] * C[cidx<NC>(i, k, j, m)] * gJ[m]; }
          }
          acc[i][j] += wJ[q] * s;
        }
      }
    }
    for (uint32_t ij = 0; ij < S::NP; ij++) {
      A_e(((uint64_t(e) * S::NP + ij) * S::NN + I) * S::NN + J) = acc[ij / NC][ij % NC];
    }
  }
}

template < Geometry geom, uint32_t NC, uint32_t P >
__global__ void k_stiff_wmma(GeomArgs g, nd::view<const double, 2, gpu_mem> qdata,
                             nd::view<double, 1, gpu_mem> A_e) {
  using S = StiffDims<geom, NC, P>;
  extern __shared__ __align__(32) double smem[];
  double * B = smem;
  double * D = B + S::Kp * S::Mp;
  double * wJ = D + r8(S::NQ * S::DSZ);
  double * T = wJ + r8(S::NQ);
  double * scratch = T + S::GRP * S::Kp * S::Mp;   // 64 doubles per warp

  uint32_t e = blockIdx.x;
  if (e >= g.nelem) { return; }
  stiff_setup<geom, NC, P>(B, D, wJ, g, qdata, e);

  const uint32_t warp = threadIdx.x / 32, lane = threadIdx.x % 32, nwarps = blockDim.x / 32;
  constexpr uint32_t TM = S::Mp / 8;
  constexpr uint32_t TILES = TM * TM;

  for (uint32_t ij0 = 0; ij0 < S::NP; ij0 += S::GRP) {
    uint32_t ngrp = min(S::GRP, S::NP - ij0);

    // T_ij[(q,k)][J] = wJ(q) sum_m C_q[i,k,j,m] B[(q,m)][J]   (zero padded)
    for (uint32_t idx = threadIdx.x; idx < ngrp * S::Kp * S::Mp; idx += blockDim.x) {
      uint32_t gidx = idx / (S::Kp * S::Mp);
      uint32_t rem = idx % (S::Kp * S::Mp);
      uint32_t kk = rem / S::Mp, J = rem % S::Mp;
      double v = 0.0;
      if (kk < S::K && J < S::NN) {
        uint32_t q = kk / 3, k = kk % 3;
        uint32_t ij = ij0 + gidx, i = ij / NC, j = ij % NC;
        const double * C = D + q * S::DSZ;
        for (uint32_t m = 0; m < 3; m++) { v += C[cidx<NC>(i, k, j, m)] * B[(q * 3 + m) * S::Mp + J]; }
        v *= wJ[q];
      }
      T[idx] = v;
    }
    __syncthreads();

    // A_ij = B^T . T_ij, one 8x8 tile per unit
    for (uint32_t u = warp; u < ngrp * TILES; u += nwarps) {
      uint32_t gidx = u / TILES, tile = u % TILES;
      uint32_t m0 = (tile / TM) * 8, n0 = (tile % TM) * 8;
      const double * Tg = T + gidx * S::Kp * S::Mp;
      frag_c c;
      wmma::fill_fragment(c, 0.0);
      for (uint32_t k0 = 0; k0 < S::Kp; k0 += 4) {
        frag_a_col a;   // B^T(m, k) = B[k*Mp + m]: column-major with ldm = Mp
        frag_b_row b;
        wmma::load_matrix_sync(a, B + k0 * S::Mp + m0, S::Mp);
        wmma::load_matrix_sync(b, Tg + k0 * S::Mp + n0, S::Mp);
        wmma::mma_sync(c, a, b, c);
      }
      double * sc = scratch + warp * 64;
      wmma::store_matrix_sync(sc, c, 8, wmma::mem_row_major);
      __syncwarp();
      uint32_t ij = ij0 + gidx;
      for (uint32_t t = lane; t < 64; t += 32) {
        uint32_t I = m0 + t / 8, J = n0 + t % 8;
        if (I < S::NN && J < S::NN) {
          A_e(((uint64_t(e) * S::NP + ij) * S::NN + I) * S::NN + J) = sc[t];
        }
      }
      __syncwarp();
    }
    __syncthreads();
  }
}

////////////////////////////////////////////////////////////////////////////////
// residual: E elements per block, 256 threads
////////////////////////////////////////////////////////////////////////////////

template < Geometry geom, uint32_t NC, uint32_t P >
struct ResDims {
  static constexpr uint32_t NN = nodes_per_element<geom, P>();
  static constexpr uint32_t NNX = nodes_per_element<geom, 1>();
  static constexpr uint32_t NQ = qpts_per_element<geom, P>();
  static constexpr uint32_t Mr = r8(3 * NQ);    // rows of G (padded)
  static constexpr uint32_t Kc = r8(NN);        // cols of G (padded)
  static constexpr uint32_t Np = 24;            // batched columns = NC * E
  static constexpr uint32_t E = Np / NC;
  static constexpr uint32_t smem = (Mr * Kc + Kc * Np + Mr * Np) * 8 + E * NQ * (9 + 1) * 8;
};

// stage G (zero padded) and per-(element, qpt) geometry
template < Geometry geom, uint32_t NC, uint32_t P >
__device__ bool res_setup(double * Gs, double * Ji_s, double * wJ_s, const GeomArgs & g, uint32_t e0) {
  using R = ResDims<geom, NC, P>;
  const uint32_t tid = threadIdx.x, bs = blockDim.x;
  for (uint32_t idx = tid; idx < R::Mr * R::Kc; idx += bs) {
    uint32_t m = idx / R::Kc, I = idx % R::Kc;
    Gs[idx] = (m < 3 * R::NQ && I < R::NN) ? g.G(m / 3, I, m % 3) : 0.0;
  }
  for (uint32_t idx = tid; idx < R::E * R::NQ; idx += bs) {
    uint32_t le = idx / R::NQ, q = idx % R::NQ;
    uint32_t e = e0 + le;
    mat3 Ji{}; double w = 0.0;
    if (e < g.nelem) { jacobian<R::NNX>(g, e, q, Ji, w); }
    for (uint32_t d = 0; d < 9; d++) { Ji_s[idx * 9 + d] = Ji[d / 3][d % 3]; }
    wJ_s[idx] = w;
  }
  return true;
}

// ---- interpolate: du_dX at qpts from nodal u -------------------------------
template < Geometry geom, uint32_t NC, uint32_t P, bool use_wmma >
__global__ void k_interp(GeomArgs g, nd::view<const double, 2, gpu_mem> u,
                         nd::view<double, 3, gpu_mem> du_q) {
  using R = ResDims<geom, NC, P>;
  extern __shared__ __align__(32) double smem[];
  double * Gs = smem;
  double * Us = Gs + R::Mr * R::Kc;
  double * Ds = Us + R::Kc * R::Np;
  double * Ji_s = Ds + R::Mr * R::Np;
  double * wJ_s = Ji_s + R::E * R::NQ * 9;

  uint32_t e0 = blockIdx.x * R::E;
  res_setup<geom, NC, P>(Gs, Ji_s, wJ_s, g, e0);

  // gather U[I][(le,c)]
  for (uint32_t idx = threadIdx.x; idx < R::Kc * R::Np; idx += blockDim.x) {
    uint32_t I = idx / R::Np, n = idx % R::Np;
    uint32_t le = n / NC, c = n % NC, e = e0 + le;
    Us[idx] = (I < R::NN && e < g.nelem) ? u(g.ids(e, I), c) : 0.0;
  }
  __syncthreads();

  if constexpr (use_wmma) {
    const uint32_t warp = threadIdx.x / 32, nwarps = blockDim.x / 32;
    constexpr uint32_t TM = R::Mr / 8, TN = R::Np / 8;
    for (uint32_t t = warp; t < TM * TN; t += nwarps) {
      uint32_t m0 = (t / TN) * 8, n0 = (t % TN) * 8;
      frag_c c;
      wmma::fill_fragment(c, 0.0);
      for (uint32_t k0 = 0; k0 < R::Kc; k0 += 4) {
        frag_a_row a;
        frag_b_row b;
        wmma::load_matrix_sync(a, Gs + m0 * R::Kc + k0, R::Kc);
        wmma::load_matrix_sync(b, Us + k0 * R::Np + n0, R::Np);
        wmma::mma_sync(c, a, b, c);
      }
      wmma::store_matrix_sync(Ds + m0 * R::Np + n0, c, R::Np, wmma::mem_row_major);
    }
  } else {
    for (uint32_t idx = threadIdx.x; idx < R::Mr * R::Np; idx += blockDim.x) {
      uint32_t m = idx / R::Np, n = idx % R::Np;
      double s = 0.0;
      for (uint32_t I = 0; I < R::NN; I++) { s += Gs[m * R::Kc + I] * Us[I * R::Np + n]; }
      Ds[idx] = s;
    }
  }
  __syncthreads();

  // push through J^{-1} and write du_q(e*NQ+q, c, d)
  for (uint32_t idx = threadIdx.x; idx < R::E * R::NQ * NC; idx += blockDim.x) {
    uint32_t c = idx % NC, eq = idx / NC, le = eq / R::NQ, q = eq % R::NQ, e = e0 + le;
    if (e >= g.nelem) { continue; }
    const double * Ji = Ji_s + eq * 9;
    double dxi[3];
    for (uint32_t d = 0; d < 3; d++) { dxi[d] = Ds[(q * 3 + d) * R::Np + le * NC + c]; }
    for (uint32_t d = 0; d < 3; d++) {
      du_q(e * R::NQ + q, c, d) = dxi[0] * Ji[0 * 3 + d] + dxi[1] * Ji[1 * 3 + d] + dxi[2] * Ji[2 * 3 + d];
    }
  }
}

// ---- integrate: r_I(c) += sum_q wJ dN_I . f(q,c) ---------------------------
template < Geometry geom, uint32_t NC, uint32_t P, bool use_wmma >
__global__ void k_integ(GeomArgs g, nd::view<const double, 3, gpu_mem> f,
                        nd::view<double, 2, gpu_mem> r) {
  using R = ResDims<geom, NC, P>;
  extern __shared__ __align__(32) double smem[];
  double * Gs = smem;
  double * Rs = Gs + R::Mr * R::Kc;      // (Kc x Np)
  double * Ts = Rs + R::Kc * R::Np;      // (Mr x Np)
  double * Ji_s = Ts + R::Mr * R::Np;
  double * wJ_s = Ji_s + R::E * R::NQ * 9;

  uint32_t e0 = blockIdx.x * R::E;
  res_setup<geom, NC, P>(Gs, Ji_s, wJ_s, g, e0);
  __syncthreads();

  // T[(q,dd)][(le,c)] = wJ sum_d Ji[dd][d] f(q,c,d)
  for (uint32_t idx = threadIdx.x; idx < R::Mr * R::Np; idx += blockDim.x) {
    uint32_t m = idx / R::Np, n = idx % R::Np;
    uint32_t q = m / 3, dd = m % 3, le = n / NC, c = n % NC, e = e0 + le;
    double v = 0.0;
    if (m < 3 * R::NQ && e < g.nelem) {
      const double * Ji = Ji_s + (le * R::NQ + q) * 9;
      for (uint32_t d = 0; d < 3; d++) { v += Ji[dd * 3 + d] * f(e * R::NQ + q, c, d); }
      v *= wJ_s[le * R::NQ + q];
    }
    Ts[idx] = v;
  }
  __syncthreads();

  if constexpr (use_wmma) {
    const uint32_t warp = threadIdx.x / 32, nwarps = blockDim.x / 32;
    constexpr uint32_t TM = R::Kc / 8, TN = R::Np / 8;
    for (uint32_t t = warp; t < TM * TN; t += nwarps) {
      uint32_t m0 = (t / TN) * 8, n0 = (t % TN) * 8;
      frag_c c;
      wmma::fill_fragment(c, 0.0);
      for (uint32_t k0 = 0; k0 < R::Mr; k0 += 4) {
        frag_a_col a;   // G^T(I, m) = Gs[m*Kc + I]: column-major, ldm = Kc
        frag_b_row b;
        wmma::load_matrix_sync(a, Gs + k0 * R::Kc + m0, R::Kc);
        wmma::load_matrix_sync(b, Ts + k0 * R::Np + n0, R::Np);
        wmma::mma_sync(c, a, b, c);
      }
      wmma::store_matrix_sync(Rs + m0 * R::Np + n0, c, R::Np, wmma::mem_row_major);
    }
  } else {
    for (uint32_t idx = threadIdx.x; idx < R::Kc * R::Np; idx += blockDim.x) {
      uint32_t I = idx / R::Np, n = idx % R::Np;
      double s = 0.0;
      for (uint32_t m = 0; m < 3 * R::NQ; m++) { s += Gs[m * R::Kc + I] * Ts[m * R::Np + n]; }
      Rs[idx] = s;
    }
  }
  __syncthreads();

  for (uint32_t idx = threadIdx.x; idx < R::NN * R::Np; idx += blockDim.x) {
    uint32_t I = idx / R::Np, n = idx % R::Np, le = n / NC, c = n % NC, e = e0 + le;
    if (e < g.nelem) { atomicAdd(&r(g.ids(e, I), c), Rs[I * R::Np + n]); }
  }
}

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
  return best;
}

template < typename K >
static void allow_smem(K kernel, uint32_t bytes) {
  if (bytes > 48 * 1024) {
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes)));
  }
}

template < Geometry geom, uint32_t NC, uint32_t P, typename JacModel >
void run_case(CaseInfo info, const Mesh<> & mesh, JacModel jac_model, int warmup, int reps) {
  using S = StiffDims<geom, NC, P>;
  using R = ResDims<geom, NC, P>;
  constexpr uint32_t NN = S::NN, NNX = S::NNX, NQ = S::NQ, DSZ = S::DSZ;

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

  printf("# case: %s %s p%u: %u elements, %u qpts | stiffness GEMM %ux%ux%u padded %ux%ux%u (useful %.0f%%), %u pair(s) per T group"
         " | residual GEMM M=%u K=%u N=%u, E=%u\n",
         info.physics.c_str(), info.geom.c_str(), P, nelem, nq,
         NN, NN, S::K, S::Mp, S::Mp, S::Kp, 100.0 * S::useful / S::padded, S::GRP,
         R::Mr, R::Kc, R::Np, R::E);

  // library reference: du_q, tangent, residual
  nd::array<double, 3, gpu_mem> du_q = evaluate(grad(d_u), d_domain);
  double t_lib_eval = bench_gpu([&]() { du_q = evaluate(grad(d_u), d_domain); }, warmup, reps);
  report_time(info, "wmma", "interp:lib", "total", t_lib_eval);
  nd::array<double, 3, cpu_mem> du_ref = du_q;

  Residual<Family::H1, gpu_mem> d_r(FunctionSpace{Family::H1, P, NC}, d_mesh);
  double t_lib_int = bench_gpu([&]() { d_r = integrate(dot(du_q, grad(phi)), d_domain); }, warmup, reps);
  report_time(info, "wmma", "integ:lib", "total", t_lib_int);
  nd::array<double, 2, cpu_mem> r_ref = d_r.data;

  nd::array<double, 2, gpu_mem> qm({nq, DSZ});
  forall_kernel<<<(nq + 127) / 128, 128>>>(nq, jac_model,
    reinterpret_cast<jac_t<NC> *>(qm.data()),
    reinterpret_cast<const grad_t<NC> *>(du_q.data()));
  CUDA_CHECK(cudaGetLastError());

  nd::array<double, 3, cpu_mem> G, GX;
  nd::array<double, 1, cpu_mem> W, W_unused;
  build_qtables<geom>(h_domain, P, G, W);
  build_qtables<geom>(h_domain, 1, GX, W_unused);
  nd::array<double, 3, gpu_mem> d_G = G, d_GX = GX;
  nd::array<double, 1, gpu_mem> d_W = W;

  auto conn = h_domain.mesh[geom];
  FiniteElement<geom, Family::H1> el{P}, X_el{1};
  nd::array<uint32_t, 2, cpu_mem> ids({nelem, NN}), X_ids({nelem, NNX});
  for (uint32_t e = 0; e < nelem; e++) {
    el.indices(h_u.offsets, &conn(e, 0), &ids(e, 0));
    X_el.indices(h_domain.mesh.X.offsets, &conn(e, 0), &X_ids(e, 0));
  }
  nd::array<uint32_t, 2, gpu_mem> d_ids = ids, d_X_ids = X_ids;
  GeomArgs g{d_ids, d_X_ids, d_mesh.X.data, d_G, d_GX, d_W, nelem};

  //////////////////////////////////////////////////////////////////////////////
  // residual: interpolate + integrate, loops vs wmma
  //////////////////////////////////////////////////////////////////////////////
  {
    nd::array<double, 3, gpu_mem> du2({nq, NC, 3u});
    uint32_t grid = (nelem + R::E - 1) / R::E;
    allow_smem(k_interp<geom, NC, P, false>, R::smem);
    allow_smem(k_interp<geom, NC, P, true>, R::smem);
    allow_smem(k_integ<geom, NC, P, false>, R::smem);
    allow_smem(k_integ<geom, NC, P, true>, R::smem);

    auto run_interp = [&](const char * name, auto wmma_c) {
      constexpr bool W_ = decltype(wmma_c)::value;
      double t = bench_gpu([&]() {
        k_interp<geom, NC, P, W_><<<grid, 256, R::smem>>>(g, d_u.data, du2);
        CUDA_CHECK(cudaGetLastError());
      }, warmup, reps);
      report_time(info, "wmma", name, "total", t);
      nd::array<double, 3, cpu_mem> h = du2;
      report_verify(info, "wmma", name, rel_l2_diff(&du_ref(0, 0, 0), &h(0, 0, 0), du_ref.size()), 1.0e-10);
    };
    run_interp("interp:loops", std::false_type{});
    run_interp("interp:wmma", std::true_type{});

    auto run_integ = [&](const char * name, auto wmma_c) {
      constexpr bool W_ = decltype(wmma_c)::value;
      double t = bench_gpu([&]() {
        zero(d_r.data);
        k_integ<geom, NC, P, W_><<<grid, 256, R::smem>>>(g, du_q, d_r.data);
        CUDA_CHECK(cudaGetLastError());
      }, warmup, reps);
      report_time(info, "wmma", name, "total", t);
      nd::array<double, 2, cpu_mem> h = d_r.data;
      report_verify(info, "wmma", name, rel_l2_diff(&r_ref(0, 0), &h(0, 0), r_ref.size()), 1.0e-10);
    };
    run_integ("integ:loops", std::false_type{});
    run_integ("integ:wmma", std::true_type{});
  }

  //////////////////////////////////////////////////////////////////////////////
  // stiffness pass 1: loops vs wmma
  //////////////////////////////////////////////////////////////////////////////
  {
    uint64_t ae_n = uint64_t(nelem) * S::NP * NN * NN;
    size_t free_b = 0, total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    if (ae_n * 8 * 1.2 > free_b) { report_skip(info, "stiffness", ae_n * 8); return; }
    nd::array<double, 1, gpu_mem> A_e({uint32_t(ae_n)});

    allow_smem(k_stiff_loops<geom, NC, P>, S::smem_common);
    allow_smem(k_stiff_wmma<geom, NC, P>, S::smem_wmma);

    double t = bench_gpu([&]() {
      k_stiff_loops<geom, NC, P><<<nelem, 256, S::smem_common>>>(g, qm, A_e);
      CUDA_CHECK(cudaGetLastError());
    }, warmup, reps);
    report_time(info, "wmma", "stiff:loops", "total", t);
    nd::array<double, 1, cpu_mem> A_ref = A_e;

    t = bench_gpu([&]() {
      k_stiff_wmma<geom, NC, P><<<nelem, 256, S::smem_wmma>>>(g, qm, A_e);
      CUDA_CHECK(cudaGetLastError());
    }, warmup, reps);
    report_time(info, "wmma", "stiff:wmma", "total", t);
    nd::array<double, 1, cpu_mem> A_w = A_e;
    report_verify(info, "wmma", "stiff:wmma", rel_l2_diff(&A_ref(0), &A_w(0), ae_n), 1.0e-10);

    // flop rates of the contraction proper (useful flops, FMA = 2)
    double flops = 2.0 * double(nelem) * S::NP * NN * NN * S::K;
    printf("# stiffness contraction: %.1f GFLOP useful; loops %.0f GF/s, wmma %.0f GF/s (padded %.0f GF/s)\n",
           flops * 1e-9, flops / bench_gpu([&]() { k_stiff_loops<geom, NC, P><<<nelem, 256, S::smem_common>>>(g, qm, A_e); }, 0, 1) * 1e-9,
           flops / t * 1e-9, flops * S::padded / S::useful / t * 1e-9);
  }
}

int main(int argc, char * argv[]) {
  int size = (argc > 1) ? atoi(argv[1]) : 1;
  int warmup = (argc > 2) ? atoi(argv[2]) : 2;
  int reps = (argc > 3) ? atoi(argv[3]) : 5;
  int s = (size == 0) ? 0 : 1;
  uint32_t tet_n[2] = {16, 64}, tet2_n[2] = {12, 40}, hex_n[2] = {16, 64}, hex2_n[2] = {8, 40};

  {
    Mesh<> mesh = tet_cuboid(tet_n[s]);
    run_case<Geometry::Tetrahedron, 1, 1>(CaseInfo{"gpu", "poisson", "tet", "p1"}, mesh, PoissonJacobianModel{kappa0}, warmup, reps);
    run_case<Geometry::Tetrahedron, 3, 1>(CaseInfo{"gpu", "elasticity", "tet", "p1"}, mesh, NeoHookeanJacobianModel{lambda0, mu0}, warmup, reps);
  }
  {
    Mesh<> mesh = tet_cuboid(tet2_n[s]);
    run_case<Geometry::Tetrahedron, 1, 2>(CaseInfo{"gpu", "poisson", "tet", "p2"}, mesh, PoissonJacobianModel{kappa0}, warmup, reps);
    run_case<Geometry::Tetrahedron, 3, 2>(CaseInfo{"gpu", "elasticity", "tet", "p2"}, mesh, NeoHookeanJacobianModel{lambda0, mu0}, warmup, reps);
  }
  {
    Mesh<> mesh = Mesh<>::cuboid({hex_n[s], hex_n[s], hex_n[s]}, vec3{1.0, 1.0, 1.0});
    run_case<Geometry::Hexahedron, 1, 1>(CaseInfo{"gpu", "poisson", "hex", "p1"}, mesh, PoissonJacobianModel{kappa0}, warmup, reps);
    run_case<Geometry::Hexahedron, 3, 1>(CaseInfo{"gpu", "elasticity", "hex", "p1"}, mesh, NeoHookeanJacobianModel{lambda0, mu0}, warmup, reps);
  }
  {
    Mesh<> mesh = Mesh<>::cuboid({hex2_n[s], hex2_n[s], hex2_n[s]}, vec3{1.0, 1.0, 1.0});
    run_case<Geometry::Hexahedron, 1, 2>(CaseInfo{"gpu", "poisson", "hex", "p2"}, mesh, PoissonJacobianModel{kappa0}, warmup, reps);
    run_case<Geometry::Hexahedron, 3, 2>(CaseInfo{"gpu", "elasticity", "hex", "p2"}, mesh, NeoHookeanJacobianModel{lambda0, mu0}, warmup, reps);
  }
  return 0;
}
