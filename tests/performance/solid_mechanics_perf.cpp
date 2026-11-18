// SPDX-FileCopyrightText: Copyright (c) 2023-2026 Sam Mish
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT

// Residual and element stiffness evaluation for 3D solid mechanics, five
// constitutive models on two meshes at two polynomial orders
// (hidden/material_perftest.md):
//
//   meshes  : ball1000000.msh (tets), and a procedural cube of hexes,
//             100^3 for p = 1 and 50^3 for p = 2 (100^3 would need 52 GB of
//             element matrices)
//   fields  : H1 displacement, p = 1 and 2; quadrature exact for the linear
//             stiffness: 1 and 4 points on tets, 2^3 and 3^3 Gauss on hexes
//   models  : linear elastic, neo-Hookean, viscoelastic (standard linear
//             solid), J2 plasticity (small strain, isotropic hardening), and
//             corotational (the Hughes-Winget update, second order, with its
//             exact tangent); the last three carry state at every point
//
// The residual is timed in three phases, interpolate / q-function / integrate,
// all from the library.  The element stiffness matrices (not assembled) the
// same way, interpolate / tangent q-function / element matrix integration,
// and on the gpu also fused into one kernel that never writes a point's
// tangent to global memory, three ways (see the fused kernels below).  Every
// gpu q-function also runs through forall_staged().
//
// Every phase writes into arrays allocated ahead of time, and each time is
// the best of several repetitions after a warm up.  The stiffness phases see
// the same state as the residual (the last step's).  J2's yield stress is
// the median trial stress, so about half the points yield.
//
// Output: markdown tables.  solid_mechanics_perf.cu builds this same file for
// the gpu, where the element matrices of every fused kernel are checked
// against the library's element_integrate(); on the cpu, a sample of them
// against the direct sum over quadrature points.
//
// usage: solid_mechanics_perf [repetitions = 3 cpu, 10 gpu]

#include "femto/mesh.hpp"
#include "femto/field.hpp"
#include "femto/domain.hpp"

#include "forall.hpp"

#include "materials/linear_elasticity.hpp"
#include "materials/neohookean.hpp"
#include "materials/viscoelasticity.hpp"
#include "materials/J2_plasticity.hpp"
#include "materials/corotational.hpp"

#include "fusion_common.hpp"   // build_qtables, nodes_per_element, conn_row_size, jac_contract

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace femto;

#ifdef __CUDACC__
static constexpr memory::space mem = memory::space::gpu;
static const char * device = "gpu";
#else
static constexpr memory::space mem = memory::space::cpu;
static const char * device = "cpu";
#endif

using tensor4 = mat< 3, 3, mat3 >;
using CorotationalHWModel = CorotationalModel< HughesWinget >;

static const char * model_names[] = {"linear elastic", "neo-hookean", "viscoelastic", "J2 plasticity", "corotational"};
static constexpr int num_models = 5;

////////////////////////////////////////////////////////////////////////////////
// the q-functions of the separated pipeline
////////////////////////////////////////////////////////////////////////////////

template < typename model_t >
struct Stress {
  model_t model;
  __host__ __device__ void operator()(const mat3 & du_dX, mat3 & P) const { P = model(du_dX); }
};

template < typename model_t >
struct Tangent {
  model_t model;
  __host__ __device__ void operator()(const mat3 & du_dX, tensor4 & C) const { C = model.jac(du_dX); }
};

struct J2Tangent {
  J2PlasticityModel model;
  __host__ __device__ void operator()(const mat3 & du_dX, const mat3 & ep_old, const double & alpha_old, tensor4 & C) const {
    C = model.jac(du_dX, ep_old, alpha_old);
  }
};

struct CorotationalTangent {
  CorotationalHWModel model;
  __host__ __device__ void operator()(const mat3 & du_dX, const mat3 & du_dX_old, const mat3 & sigma_old, tensor4 & C) const {
    C = model.jac(du_dX, du_dX_old, sigma_old);
  }
};

// element_integrate() takes isoparametric data: the tangent against the
// reference gradients, J (dxi/dX) C (dxi/dX)^T in each (i, k) block
__host__ __device__ inline tensor4 to_reference(const tensor4 & C, const mat3 & dX_dxi) {
  mat3 dxi_dX = inv(dX_dxi);
  double J = det(dX_dxi);
  tensor4 K;
  for (int i = 0; i < 3; i++) {
    for (int k = 0; k < 3; k++) {
      mat3 C_ik;
      for (int j = 0; j < 3; j++) { for (int l = 0; l < 3; l++) { C_ik(j, l) = C[i][j][k][l]; } }
      mat3 K_ik = J * dot(dxi_dX, dot(C_ik, transpose(dxi_dX)));
      for (int j = 0; j < 3; j++) { for (int l = 0; l < 3; l++) { K[i][j][k][l] = K_ik(j, l); } }
    }
  }
  return K;
}

template < typename model_t >
struct ReferenceTangent {
  model_t model;
  __host__ __device__ void operator()(const mat3 & du_dxi, const mat3 & dX_dxi, tensor4 & K) const {
    K = to_reference(model.jac(dot(du_dxi, inv(dX_dxi))), dX_dxi);
  }
};

struct J2ReferenceTangent {
  J2PlasticityModel model;
  __host__ __device__ void operator()(const mat3 & du_dxi, const mat3 & dX_dxi, const mat3 & ep_old, const double & alpha_old, tensor4 & K) const {
    K = to_reference(model.jac(dot(du_dxi, inv(dX_dxi)), ep_old, alpha_old), dX_dxi);
  }
};

struct CorotationalReferenceTangent {
  CorotationalHWModel model;
  __host__ __device__ void operator()(const mat3 & du_dxi, const mat3 & dX_dxi, const mat3 & du_dX_old, const mat3 & sigma_old, tensor4 & K) const {
    K = to_reference(model.jac(dot(du_dxi, inv(dX_dxi)), du_dX_old, sigma_old), dX_dxi);
  }
};

////////////////////////////////////////////////////////////////////////////////
// the models at a quadrature point, for the fused kernels, reading any state
// from arrays indexed by the point: the whole tangent, or a linearization
// (whatever the directions have in common) applied to one direction dF
////////////////////////////////////////////////////////////////////////////////

struct ElasticPoint {
  LinearElasticModel model;
  struct linear_t {};
  __host__ __device__ tensor4 tangent(uint32_t, const mat3 & du_dX) const { return model.jac(du_dX); }
  __host__ __device__ linear_t linearize(uint32_t, const mat3 &) const { return {}; }
  __host__ __device__ mat3 jvp(const linear_t &, const mat3 & dF) const { return model(dF); }
};

struct NeoHookeanPoint {
  NeoHookeanModel model;
  struct linear_t { mat3 du_dX; };
  __host__ __device__ tensor4 tangent(uint32_t, const mat3 & du_dX) const { return model.jac(du_dX); }
  __host__ __device__ linear_t linearize(uint32_t, const mat3 & du_dX) const { return {du_dX}; }
  __host__ __device__ mat3 jvp(const linear_t & L, const mat3 & dF) const { return model.jvp(L.du_dX, dF); }
};

struct ViscoelasticPoint {
  LinearViscoelasticModel model;
  struct linear_t {};
  __host__ __device__ tensor4 tangent(uint32_t, const mat3 & du_dX) const { return model.jac(du_dX); }
  __host__ __device__ linear_t linearize(uint32_t, const mat3 &) const { return {}; }
  __host__ __device__ mat3 jvp(const linear_t &, const mat3 & dF) const {
    double g = model.mu_v / (1.0 + model.dt / model.tau);
    return LinearElasticModel{model.lambda - 2.0 * g / 3.0, model.mu + g}(dF);
  }
};

struct J2Point {
  J2PlasticityModel model;
  const mat3 * ep_old;
  const double * alpha_old;
  struct linear_t { mat3 du_dX, ep_old; double alpha_old; };
  __host__ __device__ tensor4 tangent(uint32_t q, const mat3 & du_dX) const { return model.jac(du_dX, ep_old[q], alpha_old[q]); }
  __host__ __device__ linear_t linearize(uint32_t q, const mat3 & du_dX) const { return {du_dX, ep_old[q], alpha_old[q]}; }
  __host__ __device__ mat3 jvp(const linear_t & L, const mat3 & dF) const { return model.jvp(L.du_dX, L.ep_old, L.alpha_old, dF); }
};

struct CorotationalPoint {
  CorotationalHWModel model;
  const mat3 * du_dX_old;
  const mat3 * sigma_old;
  // the step's inverses, the rotated stress, ...
  using linear_t = CorotationalHWModel::Linearization;
  __host__ __device__ tensor4 tangent(uint32_t q, const mat3 & du_dX) const { return model.jac(du_dX, du_dX_old[q], sigma_old[q]); }
  __host__ __device__ linear_t linearize(uint32_t q, const mat3 & du_dX) const { return model.linearize(du_dX, du_dX_old[q], sigma_old[q]); }
  __host__ __device__ mat3 jvp(const linear_t & L, const mat3 & dF) const { return L(dF); }
};

////////////////////////////////////////////////////////////////////////////////
// elements and quadrature
////////////////////////////////////////////////////////////////////////////////

// exact for the linear stiffness: degree 2(p - 1) on (affine) tets, and Gauss
// with p + 1 points a direction on hexes
template < Geometry geom >
constexpr uint32_t quadrature_order(uint32_t p) { return (geom == Geometry::Tetrahedron) ? p : p + 1; }

template < Geometry geom, uint32_t P >
constexpr uint32_t qpts_per_element() {
  constexpr uint32_t q = quadrature_order<geom>(P);
  return (geom == Geometry::Hexahedron) ? q * q * q : (q * (q + 1) * (q + 2)) / 6;
}

////////////////////////////////////////////////////////////////////////////////
// timing and the displacement
////////////////////////////////////////////////////////////////////////////////

template < typename callable >
double best_of(int reps, const callable & f) {
  double best = 1.0e30;
  for (int k = 0; k <= reps; k++) {
    double t = femto::time([&]() {
      f();
#ifdef __CUDACC__
      CUDA_CHECK(cudaDeviceSynchronize());
#endif
    });
    if (k > 0) { best = std::min(best, t); }
  }
  return best;
}

// u = Q (X + w(X)) - X: a rotation by angle about (1, 1, 1) of a smooth
// displacement w, whose gradient has entries of order amplitude
void fill_displacement(Field<Family::H1> & u, const Mesh<> & mesh, double amplitude, double angle) {
  mat3 Q = RotationMatrix(normalize(vec3{1.0, 1.0, 1.0}) * angle);
  nd::cpu_array<double, 2> X = nodes_for(u, mesh);
  for (uint32_t i = 0; i < X.shape[0]; i++) {
    vec3 Xi = load<vec3>(X, i);
    vec3 w;
    for (int c = 0; c < 3; c++) {
      w[c] = amplitude * std::sin(1.7 * Xi[0] + 0.4 * c) * std::cos(1.3 * Xi[1] - 0.2 * c) * (0.5 + 0.5 * Xi[2]);
    }
    save(u.data, i, vec3(dot(Q, Xi + w) - Xi));
  }
}

////////////////////////////////////////////////////////////////////////////////
// markdown tables, with every column padded to its widest cell
////////////////////////////////////////////////////////////////////////////////

struct Table {
  std::vector< std::string > header;
  std::vector< std::vector< std::string > > rows;

  void print() const {
    std::vector< size_t > width(header.size());
    for (size_t c = 0; c < header.size(); c++) {
      width[c] = std::max< size_t >(header[c].size(), 3);
      for (auto & row : rows) { width[c] = std::max(width[c], row[c].size()); }
    }
    // the first column is text, left aligned; the rest are numbers, right aligned
    auto line = [&](const std::vector< std::string > & cells) {
      printf("|");
      for (size_t c = 0; c < cells.size(); c++) {
        printf(c == 0 ? " %-*s |" : " %*s |", int(width[c]), cells[c].c_str());
      }
      printf("\n");
    };
    line(header);
    printf("|");
    for (size_t c = 0; c < header.size(); c++) {
      printf(" %s%s |", std::string(width[c] - 1, '-').c_str(), c == 0 ? "-" : ":");
    }
    printf("\n");
    for (auto & row : rows) { line(row); }
    printf("\n");
    fflush(stdout);
  }
};

std::string fixed(double value, int digits) {
  if (value < 0.0) { return "-"; }
  char buffer[32];
  snprintf(buffer, 32, "%.*f", digits, value);
  return buffer;
}

std::string with_commas(uint64_t n) {
  std::string s = std::to_string(n);
  for (int i = int(s.size()) - 3; i > 0; i -= 3) { s.insert(size_t(i), ","); }
  return s;
}

std::string gigabytes(double bytes) { return fixed(bytes * 1e-9, 2) + " GB"; }

////////////////////////////////////////////////////////////////////////////////
// gpu: the fused element stiffness kernels, and a check of element matrices
////////////////////////////////////////////////////////////////////////////////

#ifdef __CUDACC__

// The fused kernels: E elements a block, each gathered, interpolated at its
// quadrature points, its tangent evaluated there into shared memory, and its
// element matrix accumulated by one thread per (I, J) node pair and written
// K(e, i * 3 + j, I * NN + J), the layout of element_integrate().  They differ
// in how a block evaluates the tangents, and how many elements it takes:
//   F1 : one element, a thread a point for its whole tangent (the fusion
//        experiment's design)
//   F2 : E = elements_per_block() elements (about 256 threads of node pairs)
//   F3 : F2, a point's tangent split over 9 threads, one direction dF_kl each,
//        each thread linearizing the model itself
//   F4 : F3 with registers capped at 128 (__launch_bounds__(256, 2)): one
//        kernel takes the register count of its heaviest phase, so an
//        expensive tangent otherwise sets the occupancy of the whole kernel
//   F5 : F4, but a point is linearized once, into shared memory, and its 9
//        directions applied to that
enum class Fused { F1, F2, F3, F4, F5 };

template < Fused v > inline constexpr bool split_tangent = (v == Fused::F3 || v == Fused::F4 || v == Fused::F5);
template < Fused v > inline constexpr bool capped_registers = (v == Fused::F4 || v == Fused::F5);

struct Nothing {};

// what one element keeps in shared memory: its connectivity and dof ids, its
// coordinates and displacements, and at each quadrature point the weight
// times J, the physical shape function gradients, du/dX, the tangent, and
// for F5 the model's linearization
template < Geometry geom, uint32_t P, typename linear_t >
struct ElementShared {
  static constexpr uint32_t NN = fusion::nodes_per_element<geom, P>();
  static constexpr uint32_t NNX = fusion::nodes_per_element<geom, 1>();
  static constexpr uint32_t NQ = qpts_per_element<geom, P>();
  static constexpr uint32_t CM = fusion::conn_row_size<geom>();
  Connection conn[CM];
  uint32_t ids[NN], X_ids[NNX];
  double X_e[NNX][3], u_e[NN][3];
  double wJ[NQ];
  vec3 dNdX[NQ][NN];
  mat3 du_dX[NQ];
  tensor4 D[NQ];
  linear_t linear[NQ];
};

template < Fused v, typename point_t >
using linear_storage_t = std::conditional_t< v == Fused::F5, typename point_t::linear_t, Nothing >;

// enough elements a block to give it about 256 threads of node pairs, within
// the 48 KB of shared memory a kernel gets without opting in to more; 0 when
// one element doesn't fit (F5's linearizations, for the corotational model on p = 2 hexes)
template < Geometry geom, uint32_t P, Fused v, typename point_t >
constexpr uint32_t elements_per_block() {
  using shared_t = ElementShared<geom, P, linear_storage_t<v, point_t>>;
  constexpr uint32_t NN = shared_t::NN;
  constexpr uint32_t by_threads = (256 / (NN * NN) > 1) ? 256 / (NN * NN) : 1;
  constexpr uint32_t by_memory = (48 * 1024) / sizeof(shared_t);
  return (v == Fused::F1) ? std::min< uint32_t >(1, by_memory) : std::min(by_threads, by_memory);
}

template < Geometry geom, uint32_t P, Fused v, typename point_t >
__device__ void fused_stiffness_block(
  nd::view<double, 3, memory::space::gpu> K_e,
  GeometryInfo u_offsets, GeometryInfo X_offsets,
  FiniteElement< geom, Family::H1 > el, FiniteElement< geom, Family::H1 > X_el,
  nd::view<const double, 2, memory::space::gpu> u_data,
  nd::view<const double, 2, memory::space::gpu> X_data,
  nd::view<const Connection, 2, memory::space::gpu> conn,
  nd::view<const int, 1, memory::space::gpu> elements,
  nd::view<const double, 3, memory::space::gpu> G,
  nd::view<const double, 3, memory::space::gpu> GX,
  nd::view<const double, 1, memory::space::gpu> W,
  point_t point) {

  using shared_t = ElementShared<geom, P, linear_storage_t<v, point_t>>;
  constexpr uint32_t NN = shared_t::NN, NNX = shared_t::NNX, NQ = shared_t::NQ, CM = shared_t::CM;
  constexpr uint32_t E = elements_per_block<geom, P, v, point_t>();
  if constexpr (E == 0) { return; } else {
  // raw storage: a __shared__ variable can't run constructors, and nothing here needs one
  __shared__ alignas(shared_t) unsigned char storage[E * sizeof(shared_t)];
  shared_t * sm = reinterpret_cast< shared_t * >(storage);

  const uint32_t first = blockIdx.x * E;
  const uint32_t ne = elements.shape[0];
  auto active = [&](uint32_t le) { return first + le < ne; };

  for (uint32_t t = threadIdx.x; t < E * CM; t += blockDim.x) {
    uint32_t le = t / CM;
    if (active(le)) { sm[le].conn[t % CM] = conn(elements(first + le), t % CM); }
  }
  __syncthreads();

  if (threadIdx.x < E && active(threadIdx.x)) {
    el.indices(u_offsets, sm[threadIdx.x].conn, sm[threadIdx.x].ids);
    X_el.indices(X_offsets, sm[threadIdx.x].conn, sm[threadIdx.x].X_ids);
  }
  __syncthreads();

  for (uint32_t t = threadIdx.x; t < E * NNX * 3; t += blockDim.x) {
    uint32_t le = t / (NNX * 3), i = (t / 3) % NNX, d = t % 3;
    if (active(le)) { sm[le].X_e[i][d] = X_data(sm[le].X_ids[i], d); }
  }
  for (uint32_t t = threadIdx.x; t < E * NN * 3; t += blockDim.x) {
    uint32_t le = t / (NN * 3), i = (t / 3) % NN, c = t % 3;
    if (active(le)) { sm[le].u_e[i][c] = u_data(sm[le].ids[i], c); }
  }
  __syncthreads();

  // geometry and du/dX at each point
  for (uint32_t t = threadIdx.x; t < E * NQ; t += blockDim.x) {
    uint32_t le = t / NQ, q = t % NQ;
    if (!active(le)) { continue; }
    shared_t & s = sm[le];
    mat3 dX_dxi{}, du_dxi{};
    for (uint32_t i = 0; i < NNX; i++) {
      vec3 g = {GX(q, i, 0), GX(q, i, 1), GX(q, i, 2)};
      for (uint32_t d = 0; d < 3; d++) { dX_dxi[d] += s.X_e[i][d] * g; }
    }
    for (uint32_t i = 0; i < NN; i++) {
      vec3 g = {G(q, i, 0), G(q, i, 1), G(q, i, 2)};
      for (uint32_t c = 0; c < 3; c++) { du_dxi[c] += s.u_e[i][c] * g; }
    }
    mat3 dxi_dX = inv(dX_dxi);
    s.wJ[q] = W(q) * det(dX_dxi);
    for (uint32_t i = 0; i < NN; i++) { s.dNdX[q][i] = dot(vec3{G(q, i, 0), G(q, i, 1), G(q, i, 2)}, dxi_dX); }
    s.du_dX[q] = dot(du_dxi, dxi_dX);
    if constexpr (v == Fused::F5) { s.linear[q] = point.linearize((first + le) * NQ + q, s.du_dX[q]); }
  }
  __syncthreads();

  // the tangent at each point, never leaving shared memory
  if constexpr (split_tangent<v>) {
    for (uint32_t t = threadIdx.x; t < E * NQ * 9; t += blockDim.x) {
      uint32_t le = t / (NQ * 9), q = (t / 9) % NQ, k = (t % 9) / 3, l = t % 3;
      if (!active(le)) { continue; }
      mat3 dF{};
      dF(k, l) = 1.0;
      mat3 column;
      if constexpr (v == Fused::F5) {
        column = point.jvp(sm[le].linear[q], dF);
      } else {
        column = point.jvp(point.linearize((first + le) * NQ + q, sm[le].du_dX[q]), dF);
      }
      for (uint32_t i = 0; i < 3; i++) {
        for (uint32_t j = 0; j < 3; j++) { sm[le].D[q][i][j][k][l] = column(i, j); }
      }
    }
  } else {
    for (uint32_t t = threadIdx.x; t < E * NQ; t += blockDim.x) {
      uint32_t le = t / NQ, q = t % NQ;
      if (active(le)) { sm[le].D[q] = point.tangent((first + le) * NQ + q, sm[le].du_dX[q]); }
    }
  }
  __syncthreads();

  for (uint32_t t = threadIdx.x; t < E * NN * NN; t += blockDim.x) {
    uint32_t le = t / (NN * NN), pair = t % (NN * NN), I = pair / NN, J = pair % NN;
    if (!active(le)) { continue; }
    const shared_t & s = sm[le];
    double B[3][3] = {};
    for (uint32_t q = 0; q < NQ; q++) {
      for (uint32_t i = 0; i < 3; i++) {
        for (uint32_t j = 0; j < 3; j++) { B[i][j] += s.wJ[q] * fusion::jac_contract(s.D[q], i, j, s.dNdX[q][I], s.dNdX[q][J]); }
      }
    }
    for (uint32_t i = 0; i < 3; i++) {
      for (uint32_t j = 0; j < 3; j++) { K_e(first + le, i * 3 + j, pair) = B[i][j]; }
    }
  }
  }
}

#define FUSED_STIFFNESS_PARAMETERS \
  nd::view<double, 3, memory::space::gpu> K_e, GeometryInfo u_offsets, GeometryInfo X_offsets, \
  FiniteElement< geom, Family::H1 > el, FiniteElement< geom, Family::H1 > X_el, \
  nd::view<const double, 2, memory::space::gpu> u_data, nd::view<const double, 2, memory::space::gpu> X_data, \
  nd::view<const Connection, 2, memory::space::gpu> conn, nd::view<const int, 1, memory::space::gpu> elements, \
  nd::view<const double, 3, memory::space::gpu> G, nd::view<const double, 3, memory::space::gpu> GX, \
  nd::view<const double, 1, memory::space::gpu> W, point_t point
#define FUSED_STIFFNESS_ARGUMENTS K_e, u_offsets, X_offsets, el, X_el, u_data, X_data, conn, elements, G, GX, W, point

template < Geometry geom, uint32_t P, Fused v, typename point_t >
__global__ void fused_stiffness_kernel(FUSED_STIFFNESS_PARAMETERS) {
  fused_stiffness_block<geom, P, v>(FUSED_STIFFNESS_ARGUMENTS);
}

template < Geometry geom, uint32_t P, Fused v, typename point_t >
__global__ void __launch_bounds__(256, 2) fused_stiffness_kernel_capped(FUSED_STIFFNESS_PARAMETERS) {
  fused_stiffness_block<geom, P, v>(FUSED_STIFFNESS_ARGUMENTS);
}

// what the fused kernels read besides the field: the shape function
// gradient tables of the field and of the (linear) coordinates, and weights
struct ShapeTables {
  nd::array<double, 3, memory::space::gpu> G, GX;
  nd::array<double, 1, memory::space::gpu> W;
};

template < Geometry geom, uint32_t P, Fused v, typename point_t >
void fused_stiffness(nd::array<double, 3, memory::space::gpu> & K_e, const Field<Family::H1, mem> & u,
                     const Domain<mem> & domain, const ShapeTables & tables, point_t point) {
  constexpr uint32_t NN = fusion::nodes_per_element<geom, P>();
  constexpr uint32_t E = elements_per_block<geom, P, v, point_t>();
  if constexpr (E == 0) { return; } else {
  nd::view<const int, 1, memory::space::gpu> elements = domain.active_elements[geom];
  // a thread a node pair, as many as this instantiation's registers allow
  // (the threads loop over the pairs, so fewer is fine)
  auto kernel = capped_registers<v> ? fused_stiffness_kernel_capped<geom, P, v, point_t> : fused_stiffness_kernel<geom, P, v, point_t>;
  cudaFuncAttributes attributes;
  CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
  uint32_t threads = std::min< uint32_t >((attributes.maxThreadsPerBlock / 32) * 32, ((E * NN * NN + 31) / 32) * 32);
  uint32_t blocks = (elements.shape[0] + E - 1) / E;
  kernel<<< blocks, threads >>>(
    K_e, u.offsets, domain.mesh.X.offsets, FiniteElement< geom, Family::H1 >{P}, FiniteElement< geom, Family::H1 >{1},
    u.data, domain.mesh.X.data, domain.mesh[geom], elements, tables.G, tables.GX, tables.W, point);
  CUDA_CHECK(cudaGetLastError());
  }
}

__global__ void squared_difference_kernel(uint64_t n, const double * a, const double * b, double * sums) {
  double d2 = 0.0, b2 = 0.0;
  for (uint64_t k = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x; k < n; k += uint64_t(gridDim.x) * blockDim.x) {
    double d = a[k] - b[k];
    d2 += d * d;
    b2 += b[k] * b[k];
  }
  atomicAdd(&sums[0], d2);
  atomicAdd(&sums[1], b2);
}

// |a - b| / |b| over two arrays of element matrices
double relative_difference(const nd::array<double, 3, memory::space::gpu> & a, const nd::array<double, 3, memory::space::gpu> & b) {
  nd::array<double, 1, memory::space::gpu> sums({2});
  zero(sums);
  squared_difference_kernel<<< 1024, 256 >>>(a.size(), a.data(), b.data(), sums.data());
  CUDA_CHECK(cudaDeviceSynchronize());
  nd::array<double, 1, memory::space::cpu> h = sums;
  return std::sqrt(h(0) / h(1));
}

#else

////////////////////////////////////////////////////////////////////////////////
// cpu: element stiffness matrices from the tangent at the quadrature points,
// in element_integrate()'s layout.  Per element and point, DB_J = C dN_J/dX is
// staged first (the strategy of the library's gpu element matrix kernel), so a
// node pair costs 27 multiply-adds a point rather than 81
////////////////////////////////////////////////////////////////////////////////

template < Geometry geom, uint32_t P >
void element_stiffness_cpu(nd::array<double, 3, memory::space::cpu> & K_e, const nd::array<double, 5, memory::space::cpu> & C_q,
                           const Domain<> & domain, const nd::array<double, 3, memory::space::cpu> & G,
                           const nd::array<double, 3, memory::space::cpu> & GX, const nd::array<double, 1, memory::space::cpu> & W) {
  constexpr uint32_t NN = fusion::nodes_per_element<geom, P>(), NNX = fusion::nodes_per_element<geom, 1>();
  constexpr uint32_t NQ = qpts_per_element<geom, P>();
  FiniteElement< geom, Family::H1 > X_el{1};
  nd::view<const Connection, 2> conn = domain.mesh[geom];
  nd::view<const int> elements = domain.active_elements[geom];
  const tensor4 * C = reinterpret_cast< const tensor4 * >(C_q.data());

  threadpool::block_parallel_for(elements.size(), [&](uint32_t e_start, uint32_t e_end) {
    std::vector< vec3 > dNdX(NQ * NN);
    std::vector< mat3 > DB(NQ * NN * 3);   // DB(q, J, i)(k, j) = sum_m C_q[i][k][j][m] dN_J/dX_m
    std::vector< double > wJ(NQ);
    uint32_t X_ids[NNX];
    for (uint32_t e = e_start; e < e_end; e++) {
      X_el.indices(domain.mesh.X.offsets, &conn(elements(e), 0), X_ids);
      for (uint32_t q = 0; q < NQ; q++) {
        mat3 dX_dxi{};
        for (uint32_t i = 0; i < NNX; i++) {
          vec3 g = {GX(q, i, 0), GX(q, i, 1), GX(q, i, 2)};
          for (uint32_t d = 0; d < 3; d++) { dX_dxi[d] += domain.mesh.X.data(X_ids[i], d) * g; }
        }
        mat3 dxi_dX = inv(dX_dxi);
        wJ[q] = W(q) * det(dX_dxi);
        const tensor4 & Cq = C[e * NQ + q];
        for (uint32_t J = 0; J < NN; J++) {
          vec3 dN = dot(vec3{G(q, J, 0), G(q, J, 1), G(q, J, 2)}, dxi_dX);
          dNdX[q * NN + J] = dN;
          for (uint32_t i = 0; i < 3; i++) {
            mat3 & db = DB[(q * NN + J) * 3 + i];
            for (uint32_t k = 0; k < 3; k++) {
              for (uint32_t j = 0; j < 3; j++) { db(k, j) = dot(Cq[i][k][j], dN); }
            }
          }
        }
      }
      for (uint32_t I = 0; I < NN; I++) {
        for (uint32_t J = 0; J < NN; J++) {
          double B[3][3] = {};
          for (uint32_t q = 0; q < NQ; q++) {
            vec3 w_dN = wJ[q] * dNdX[q * NN + I];
            for (uint32_t i = 0; i < 3; i++) {
              vec3 row = dot(w_dN, DB[(q * NN + J) * 3 + i]);
              for (uint32_t j = 0; j < 3; j++) { B[i][j] += row[j]; }
            }
          }
          for (uint32_t i = 0; i < 3; i++) {
            for (uint32_t j = 0; j < 3; j++) { K_e(e, i * 3 + j, I * NN + J) = B[i][j]; }
          }
        }
      }
    }
  });
}

// the same element matrices directly, sum_q wJ dN_I . C . dN_J (no DB_J), for
// the first n elements: |K_e - direct| / |direct| over them
template < Geometry geom, uint32_t P >
double check_element_stiffness_cpu(const nd::array<double, 3, memory::space::cpu> & K_e, const nd::array<double, 5, memory::space::cpu> & C_q,
                                   const Domain<> & domain, const nd::array<double, 3, memory::space::cpu> & G,
                                   const nd::array<double, 3, memory::space::cpu> & GX, const nd::array<double, 1, memory::space::cpu> & W, uint32_t n) {
  constexpr uint32_t NN = fusion::nodes_per_element<geom, P>(), NNX = fusion::nodes_per_element<geom, 1>();
  constexpr uint32_t NQ = qpts_per_element<geom, P>();
  FiniteElement< geom, Family::H1 > X_el{1};
  nd::view<const Connection, 2> conn = domain.mesh[geom];
  nd::view<const int> elements = domain.active_elements[geom];
  const tensor4 * C = reinterpret_cast< const tensor4 * >(C_q.data());
  double d2 = 0.0, r2 = 0.0;
  uint32_t X_ids[NNX];
  for (uint32_t e = 0; e < n; e++) {
    X_el.indices(domain.mesh.X.offsets, &conn(elements(e), 0), X_ids);
    for (uint32_t I = 0; I < NN; I++) {
      for (uint32_t J = 0; J < NN; J++) {
        for (uint32_t i = 0; i < 3; i++) {
          for (uint32_t j = 0; j < 3; j++) {
            double direct = 0.0;
            for (uint32_t q = 0; q < NQ; q++) {
              mat3 dX_dxi{};
              for (uint32_t a = 0; a < NNX; a++) {
                for (uint32_t d = 0; d < 3; d++) { dX_dxi[d] += domain.mesh.X.data(X_ids[a], d) * vec3{GX(q, a, 0), GX(q, a, 1), GX(q, a, 2)}; }
              }
              mat3 dxi_dX = inv(dX_dxi);
              vec3 dN_I = dot(vec3{G(q, I, 0), G(q, I, 1), G(q, I, 2)}, dxi_dX), dN_J = dot(vec3{G(q, J, 0), G(q, J, 1), G(q, J, 2)}, dxi_dX);
              direct += W(q) * det(dX_dxi) * fusion::jac_contract(C[e * NQ + q], i, j, dN_I, dN_J);
            }
            double diff = K_e(e, i * 3 + j, I * NN + J) - direct;
            d2 += diff * diff;
            r2 += direct * direct;
          }
        }
      }
    }
  }
  return std::sqrt(d2 / r2);
}

#endif

////////////////////////////////////////////////////////////////////////////////
// one case: a mesh and an order, all five models
////////////////////////////////////////////////////////////////////////////////

struct ModelTimes {
  // residual: interpolate, q-function (forall, forall_staged), integrate
  double r_interpolate, r_qf, r_qf_staged = -1.0, r_integrate;
  // stiffness: interpolate, tangent q-function (forall, forall_staged), integrate
  double s_interpolate, s_qf, s_qf_staged = -1.0, s_integrate;
  static constexpr int variants = 5;
  double fused[variants] = {-1.0, -1.0, -1.0, -1.0, -1.0};
  double fused_error[variants] = {-1.0, -1.0, -1.0, -1.0, -1.0};
  double cpu_error = -1.0;   // element_stiffness_cpu against the direct sum, on a sample
  uint32_t fused_elements[variants] = {};
};

struct CaseResult {
  std::string name;
  uint32_t p, elements, qpts_per_element, qpts, unknowns;
  double bytes_du, bytes_tangent, bytes_K_e;
  ModelTimes models[num_models];
};

std::vector< CaseResult > results;

template < Geometry geom, uint32_t P >
void run_case(const char * name, const Mesh<> & host_mesh, int reps) {

  constexpr uint32_t NQ = qpts_per_element<geom, P>();
  constexpr double lambda = 10.0, mu = 10.0;

  // the last of 32 steps to a 30 degree rotation on a 5% strain field
  Field host_u = create_field<Family::H1>(host_mesh, P, 3), host_u_old = host_u;
  fill_displacement(host_u, host_mesh, 0.05, M_PI / 6.0);
  fill_displacement(host_u_old, host_mesh, 0.05 * 31.0 / 32.0, (M_PI / 6.0) * 31.0 / 32.0);

  MeshQuadratureRule rule(quadrature_order<geom>(P));
  Domain<> host_domain(host_mesh, rule);
  nd::cpu_array<double, 3> host_du = evaluate(grad(host_u), host_domain);
  const uint32_t nq = host_du.shape[0];
  FEMTO_ASSERT(nq == host_domain.mesh[geom].shape[0] * NQ, "unexpected quadrature point count");

  // J2's yield stress: the median trial stress, so about half the points yield
  std::vector< double > s_trial(nq);
  for (uint32_t q = 0; q < nq; q++) {
    mat3 H = load_mat(host_du, q, mat3{});
    mat3 e = 0.5 * (H + transpose(H));
    s_trial[q] = 2.0 * mu * norm(e - (tr(e) / 3.0) * Identity<3>());
  }
  std::nth_element(s_trial.begin(), s_trial.begin() + nq / 2, s_trial.end());
  double sigma_y = std::sqrt(1.5) * s_trial[nq / 2];

  LinearElasticModel elastic{lambda, mu};
  NeoHookeanModel neohookean{lambda, mu};
  LinearViscoelasticModel viscoelastic{lambda, mu, 2.0 * mu, 1.0, 0.1};
  J2PlasticityModel j2{lambda, mu, sigma_y, 0.1 * mu};
  CorotationalHWModel corotational{lambda, mu};

  Mesh<mem> mesh = host_mesh;
  Field<Family::H1, mem> u = host_u, u_old = host_u_old;
  Domain<mem> domain(mesh, rule);
  BasisFunction<Family::H1> phi(P, 3);

  // quadrature point data, allocated once: the gradient and the last step's,
  // the stress, a mat3 of state and a scalar one (in and out), the tangent
  nd::array<double, 3, mem> du_q = evaluate(grad(u), domain);
  nd::array<double, 3, mem> du_old_q = evaluate(grad(u_old), domain);
  nd::array<double, 3, mem> P_q({nq, 3u, 3u}), state_old({nq, 3u, 3u}), state({nq, 3u, 3u});
  nd::array<double, 2, mem> scalar_old({nq, 1u}), scalar({nq, 1u});
  nd::array<double, 5, mem> C_q({nq, 3u, 3u, 3u, 3u});
  zero(state_old);
  zero(scalar_old);

  CaseResult result;
  result.name = name;
  result.p = P;
  result.elements = host_mesh.hex.shape[0] + host_mesh.tet.shape[0];
  result.qpts_per_element = NQ;
  result.qpts = nq;
  result.unknowns = host_u.size();
  result.bytes_du = 72.0 * nq;
  result.bytes_tangent = 648.0 * nq;
  constexpr uint32_t NN = fusion::nodes_per_element<geom, P>();
  result.bytes_K_e = 8.0 * 9 * NN * NN * result.elements;

  printf("# %s, %s, p = %u: %u elements, %u quadrature points, %u unknowns, J2 yield stress %.3f\n",
         device, name, P, result.elements, nq, result.unknowns, sigma_y);
  fflush(stdout);

  // the old state: none for the viscous and plastic strains, and for the
  // corotational model's Cauchy stress the last step's linear elastic one
  auto prepare_state = [&](int m) {
    zero(state_old);
    zero(scalar_old);
    if (m == 4) { forall(Stress<LinearElasticModel>{elastic}, du_old_q, state_old); }
  };

  //////////////////////////////////////////////////////////////////////////////
  // residual: the library's evaluate / forall / integrate
  //////////////////////////////////////////////////////////////////////////////

  Residual<Family::H1, mem> r = integrate(dot(P_q, grad(phi)), domain);
  double r_interpolate = best_of(reps, [&]() { du_q = evaluate(grad(u), domain); });
  double r_integrate = best_of(reps, [&]() { r = integrate(dot(P_q, grad(phi)), domain); });

  auto residual_qf = [&](int m, auto launch) {
    switch (m) {
      case 0: return best_of(reps, [&]() { launch(Stress<LinearElasticModel>{elastic}, du_q, P_q); });
      case 1: return best_of(reps, [&]() { launch(Stress<NeoHookeanModel>{neohookean}, du_q, P_q); });
      case 2: return best_of(reps, [&]() { launch(viscoelastic, du_q, state_old, P_q, state); });
      case 3: return best_of(reps, [&]() { launch(j2, du_q, state_old, scalar_old, P_q, state, scalar); });
      default: return best_of(reps, [&]() { launch(corotational, du_q, du_old_q, state_old, P_q, state); });
    }
  };
  auto plain = [](auto f, auto & ... arrays) { forall(f, arrays ...); };
  [[maybe_unused]] auto staged = [](auto f, auto & ... arrays) { forall_staged(f, arrays ...); };

  for (int m = 0; m < num_models; m++) {
    prepare_state(m);
    ModelTimes & t = result.models[m];
    t.r_interpolate = r_interpolate;
    t.r_integrate = r_integrate;
    t.r_qf = residual_qf(m, plain);
    if constexpr (mem != memory::space::cpu) { t.r_qf_staged = residual_qf(m, staged); }
  }

  //////////////////////////////////////////////////////////////////////////////
  // element stiffness matrices
  //////////////////////////////////////////////////////////////////////////////

#ifdef __CUDACC__

  // separated: the library's isoparametric evaluate / forall / element_integrate
  // (the coordinate gradients are the mesh's, so computed once, untimed)
  nd::array<double, 3, mem> dX_dxi_q = evaluate(grad(mesh.X), isoparametric(domain));
  nd::array<double, 3, mem> du_dxi_q = evaluate(grad(u), isoparametric(domain));
  forall(ReferenceTangent<LinearElasticModel>{elastic}, du_dxi_q, dX_dxi_q, C_q);
  ElementMatrices<mem> library = element_integrate(dot(grad(phi), C_q, grad(phi)), isoparametric(domain));
  nd::view<const double, 5, mem> C5{C_q.data(), C_q.shape};
  FunctionSpace space = phi.space;
  auto element_matrices = [&]() {
    femto::impl::element_matrices_into<Family::H1, DerivedQuantity::GRAD, Family::H1, DerivedQuantity::GRAD>(
      space, C5, space, domain, DomainType::ISOPARAMETRIC, library.K, library.test_ids, library.trial_ids);
  };
  double s_interpolate = best_of(reps, [&]() { du_dxi_q = evaluate(grad(u), isoparametric(domain)); });
  double s_integrate = best_of(reps, [&]() { element_matrices(); });

  auto tangent_qf = [&](int m, auto launch) {
    switch (m) {
      case 0: return best_of(reps, [&]() { launch(ReferenceTangent<LinearElasticModel>{elastic}, du_dxi_q, dX_dxi_q, C_q); });
      case 1: return best_of(reps, [&]() { launch(ReferenceTangent<NeoHookeanModel>{neohookean}, du_dxi_q, dX_dxi_q, C_q); });
      case 2: return best_of(reps, [&]() { launch(ReferenceTangent<LinearViscoelasticModel>{viscoelastic}, du_dxi_q, dX_dxi_q, C_q); });
      case 3: return best_of(reps, [&]() { launch(J2ReferenceTangent{j2}, du_dxi_q, dX_dxi_q, state_old, scalar_old, C_q); });
      default: return best_of(reps, [&]() { launch(CorotationalReferenceTangent{corotational}, du_dxi_q, dX_dxi_q, du_old_q, state_old, C_q); });
    }
  };

  // fused
  ShapeTables tables;
  {
    nd::array<double, 3, memory::space::cpu> G, GX;
    nd::array<double, 1, memory::space::cpu> W, W_unused;
    fusion::build_qtables<geom>(host_domain, P, G, W);
    fusion::build_qtables<geom>(host_domain, 1, GX, W_unused);
    tables.G = G;
    tables.GX = GX;
    tables.W = W;
  }
  nd::array<double, 3, mem> K_fused(library.K.shape);
  auto point_of = [&](int m, auto use) {
    const mat3 * s_old = reinterpret_cast< const mat3 * >(state_old.data());
    switch (m) {
      case 0: return use(ElasticPoint{elastic});
      case 1: return use(NeoHookeanPoint{neohookean});
      case 2: return use(ViscoelasticPoint{viscoelastic});
      case 3: return use(J2Point{j2, s_old, scalar_old.data()});
      default: return use(CorotationalPoint{corotational, reinterpret_cast< const mat3 * >(du_old_q.data()), s_old});
    }
  };

  for (int m = 0; m < num_models; m++) {
    prepare_state(m);
    ModelTimes & t = result.models[m];
    t.s_interpolate = s_interpolate;
    t.s_integrate = s_integrate;
    t.s_qf = tangent_qf(m, plain);
    t.s_qf_staged = tangent_qf(m, staged);
    element_matrices();
    point_of(m, [&](auto point) {
      using point_t = decltype(point);
      auto variant = [&](auto v, int f) {
        constexpr Fused which = decltype(v)::value;
        t.fused_elements[f] = elements_per_block<geom, P, which, point_t>();
        if (t.fused_elements[f] == 0) { return; }
        t.fused[f] = best_of(reps, [&]() { fused_stiffness<geom, P, which>(K_fused, u, domain, tables, point); });
        t.fused_error[f] = relative_difference(K_fused, library.K);
      };
      variant(std::integral_constant< Fused, Fused::F1 >{}, 0);
      variant(std::integral_constant< Fused, Fused::F2 >{}, 1);
      variant(std::integral_constant< Fused, Fused::F3 >{}, 2);
      variant(std::integral_constant< Fused, Fused::F4 >{}, 3);
      variant(std::integral_constant< Fused, Fused::F5 >{}, 4);
      return 0;
    });
  }

#else

  // separated: the library's evaluate / forall, then element_stiffness_cpu
  nd::cpu_array<double, 3> G, GX;
  nd::cpu_array<double, 1> W, W_unused;
  fusion::build_qtables<geom>(host_domain, P, G, W);
  fusion::build_qtables<geom>(host_domain, 1, GX, W_unused);
  nd::cpu_array<double, 3> K_e({result.elements, 9u, NN * NN});
  forall(Tangent<LinearElasticModel>{elastic}, du_q, C_q);
  double s_interpolate = r_interpolate;
  double s_integrate = best_of(reps, [&]() { element_stiffness_cpu<geom, P>(K_e, C_q, domain, G, GX, W); });

  for (int m = 0; m < num_models; m++) {
    prepare_state(m);
    ModelTimes & t = result.models[m];
    t.s_interpolate = s_interpolate;
    t.s_integrate = s_integrate;
    switch (m) {
      case 0: t.s_qf = best_of(reps, [&]() { forall(Tangent<LinearElasticModel>{elastic}, du_q, C_q); }); break;
      case 1: t.s_qf = best_of(reps, [&]() { forall(Tangent<NeoHookeanModel>{neohookean}, du_q, C_q); }); break;
      case 2: t.s_qf = best_of(reps, [&]() { forall(Tangent<LinearViscoelasticModel>{viscoelastic}, du_q, C_q); }); break;
      case 3: t.s_qf = best_of(reps, [&]() { forall(J2Tangent{j2}, du_q, state_old, scalar_old, C_q); }); break;
      default: t.s_qf = best_of(reps, [&]() { forall(CorotationalTangent{corotational}, du_q, du_old_q, state_old, C_q); }); break;
    }
    element_stiffness_cpu<geom, P>(K_e, C_q, domain, G, GX, W);
    t.cpu_error = check_element_stiffness_cpu<geom, P>(K_e, C_q, domain, G, GX, W, 2000);
  }

#endif

  results.push_back(result);
}

////////////////////////////////////////////////////////////////////////////////

void print_results() {
  bool gpu = (mem != memory::space::cpu);

  printf("## %s: cases\n\n", device);
  Table cases{{"case", "elements", "qpts/elem", "qpts", "unknowns", "du/dX at qpts", "tangent at qpts", "element matrices"}};
  for (auto & c : results) {
    cases.rows.push_back({c.name + " p=" + std::to_string(c.p), with_commas(c.elements), std::to_string(c.qpts_per_element),
                          with_commas(c.qpts), with_commas(c.unknowns), gigabytes(c.bytes_du), gigabytes(c.bytes_tangent), gigabytes(c.bytes_K_e)});
  }
  cases.print();

  printf("## %s: residual (ms)\n\n", device);
  Table residual{{"case / model", "interpolate", "q-function", "integrate", "total"}};
  if (gpu) { residual.header.insert(residual.header.end(), {"q-function, staged", "total, staged"}); }
  for (auto & c : results) {
    for (int m = 0; m < num_models; m++) {
      const ModelTimes & t = c.models[m];
      std::vector< std::string > row = {c.name + " p=" + std::to_string(c.p) + " / " + model_names[m],
                                        fixed(1e3 * t.r_interpolate, 2), fixed(1e3 * t.r_qf, 2), fixed(1e3 * t.r_integrate, 2),
                                        fixed(1e3 * (t.r_interpolate + t.r_qf + t.r_integrate), 2)};
      if (gpu) {
        row.push_back(fixed(1e3 * t.r_qf_staged, 2));
        row.push_back(fixed(1e3 * (t.r_interpolate + t.r_qf_staged + t.r_integrate), 2));
      }
      residual.rows.push_back(row);
    }
  }
  residual.print();

  printf("## %s: element stiffness matrices (ms)\n\n", device);
  Table stiffness{{"case / model", "interpolate", "tangent q-function", "integrate", "separated total"}};
  if (gpu) {
    stiffness.header.insert(stiffness.header.end(), {"q-function, staged", "separated, staged", "fused F1", "fused F2", "fused F3", "fused F4", "fused F5",
                                                     "best fused vs separated", "vs separated, staged"});
  }
  for (auto & c : results) {
    for (int m = 0; m < num_models; m++) {
      const ModelTimes & t = c.models[m];
      double separated = t.s_interpolate + t.s_qf + t.s_integrate;
      std::vector< std::string > row = {c.name + " p=" + std::to_string(c.p) + " / " + model_names[m],
                                        fixed(1e3 * t.s_interpolate, 2), fixed(1e3 * t.s_qf, 2), fixed(1e3 * t.s_integrate, 2),
                                        fixed(1e3 * separated, 2)};
      if (gpu) {
        double best = 1.0e30;
        for (double f : t.fused) { if (f >= 0.0) { best = std::min(best, f); } }
        double separated_staged = t.s_interpolate + t.s_qf_staged + t.s_integrate;
        row.push_back(fixed(1e3 * t.s_qf_staged, 2));
        row.push_back(fixed(1e3 * separated_staged, 2));
        for (int f = 0; f < ModelTimes::variants; f++) { row.push_back(fixed(1e3 * t.fused[f], 2)); }
        row.push_back(fixed(separated / best, 2) + "x");
        row.push_back(fixed(separated_staged / best, 2) + "x");
      }
      stiffness.rows.push_back(row);
    }
  }
  stiffness.print();

  if (!gpu) {
    printf("## cpu: element matrices against the direct sum over quadrature points (first 2,000 elements)\n\n");
    Table check{{"case / model", "relative difference"}};
    for (auto & c : results) {
      for (int m = 0; m < num_models; m++) {
        char b[32];
        snprintf(b, 32, "%.1e", c.models[m].cpu_error);
        check.rows.push_back({c.name + " p=" + std::to_string(c.p) + " / " + model_names[m], b});
      }
    }
    check.print();
  }

  if (gpu) {
    printf("## gpu: fused element matrices against element_integrate(), |K_fused - K_library| / |K_library|\n\n");
    Table check{{"case / model", "F1", "F2", "F3", "F4", "F5", "elements a block (F1 / F2-F4 / F5)"}};
    auto sci = [](double x) { if (x < 0.0) { return std::string("-"); } char b[32]; snprintf(b, 32, "%.1e", x); return std::string(b); };
    for (auto & c : results) {
      for (int m = 0; m < num_models; m++) {
        const ModelTimes & t = c.models[m];
        std::vector< std::string > row = {c.name + " p=" + std::to_string(c.p) + " / " + model_names[m]};
        for (int f = 0; f < ModelTimes::variants; f++) { row.push_back(sci(t.fused_error[f])); }
        row.push_back(std::to_string(t.fused_elements[0]) + " / " + std::to_string(t.fused_elements[1]) + " / " + std::to_string(t.fused_elements[4]));
        check.rows.push_back(row);
      }
    }
    check.print();
  }
}

int main(int argc, char ** argv) {

  int reps = (argc > 1) ? atoi(argv[1]) : (mem == memory::space::cpu ? 3 : 10);

  {
    Mesh<> ball = Mesh<>::load(FEMTO_MESH_DIR "ball1000000.msh");
    run_case<Geometry::Tetrahedron, 1>("ball (tet)", ball, reps);
    run_case<Geometry::Tetrahedron, 2>("ball (tet)", ball, reps);
  }
  {
    Mesh<> cube = Mesh<>::cuboid({100, 100, 100}, vec3{1.0, 1.0, 1.0});
    run_case<Geometry::Hexahedron, 1>("cube 100^3 (hex)", cube, reps);
  }
  {
    Mesh<> cube = Mesh<>::cuboid({50, 50, 50}, vec3{1.0, 1.0, 1.0});
    run_case<Geometry::Hexahedron, 2>("cube 50^3 (hex)", cube, reps);
  }

  printf("\n");
  print_results();
}
