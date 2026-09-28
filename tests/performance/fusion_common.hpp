#pragma once

// Shared pieces of the kernel-fusion experiment:
// separated (3-phase) vs fused (1-phase) residual and stiffness evaluation.
//
// Cases: {poisson (scalar), neohookean solid (vector)} x {tet, hex} x {cpu, gpu}

#include "femto/mesh.hpp"
#include "femto/field.hpp"
#include "femto/domain.hpp"

#include "fm/macros.hpp"
#include "fm/types/vec.hpp"
#include "fm/types/matrix.hpp"
#include "fm/operations/adjugate.hpp"

#include "misc/timer.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>

namespace fusion {

using namespace femto;
using namespace fm;

// each case uses H1 elements of degree P with a MeshQuadratureRule(P + 1)

template < Geometry geom, uint32_t P >
__host__ __device__ constexpr uint32_t nodes_per_element() {
  return (geom == Geometry::Hexahedron) ? (P + 1) * (P + 1) * (P + 1)
                                        : ((P + 1) * (P + 2) * (P + 3)) / 6;
}

template < Geometry geom, uint32_t P >
__host__ __device__ constexpr uint32_t qpts_per_element() {
  constexpr uint32_t q = P + 1;
  return (geom == Geometry::Hexahedron) ? q * q * q : (q * (q + 1) * (q + 2)) / 6;
}

// number of Connection entries in one element's connectivity row
template < Geometry geom >
__host__ __device__ constexpr uint32_t conn_row_size() {
  return (geom == Geometry::Hexahedron) ? 27 : 15;
}

////////////////////////////////////////////////////////////////////////////////
// material models
////////////////////////////////////////////////////////////////////////////////

struct PoissonModel {
  __host__ __device__ vec3 operator()(vec3 du_dX) const {
    return kappa * du_dX;
  }
  double kappa;
};

// "jacobian" of the poisson flux w.r.t. the temperature gradient.
// (constant for a linear conductivity, but evaluated per-qpt to mimic
//  the cost/structure of a general nonlinear constitutive model)
struct PoissonJacobianModel {
  __host__ __device__ mat3 operator()(vec3 /*du_dX*/) const {
    return kappa * Identity<3>();
  }
  double kappa;
};

struct NeoHookeanModel {
  __host__ __device__ mat3 operator()(mat3 du_dX) const {
    mat3 F = Identity<3>() + du_dX;
    double J = det(F);
    mat3 invFT = transpose(inv(F));
    return (lambda * log(J) - mu) * invFT + mu * F;
  }
  double lambda, mu;
};

struct NeoHookeanJacobianModel {
  // returns dP_dF, where dP_dF[i][j][k][l] = dP_ij / dF_kl
  __host__ __device__ mat<3,3,mat<3,3>> operator()(mat3 du_dX) const {
    mat3 F = Identity<3>() + du_dX;
    double logJ = log(det(F));
    mat3 invFT = transpose(inv(F));
    mat<3,3,mat<3,3>> dP_dF;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
          for (int l = 0; l < 3; l++) {
            dP_dF[i][j][k][l] = lambda * invFT[i][j] * invFT[k][l]
                              + (mu - lambda * logJ) * invFT[k][j] * invFT[i][l]
                              + mu * (i == k) * (j == l);
          }
        }
      }
    }
    return dP_dF;
  }
  double lambda, mu;
};

inline constexpr double kappa0 = 1.7;
inline constexpr double lambda0 = 10.0;
inline constexpr double mu0 = 10.0;

////////////////////////////////////////////////////////////////////////////////
// type helpers for scalar (NC=1) vs vector (NC=3) fields
////////////////////////////////////////////////////////////////////////////////

template < uint32_t NC > struct grad_type_helper { using type = mat<NC,3>; };
template <> struct grad_type_helper<1> { using type = vec3; };
template < uint32_t NC > using grad_t = typename grad_type_helper<NC>::type;

template < uint32_t NC > struct jac_type_helper { using type = mat<3,3,mat<3,3>>; };
template <> struct jac_type_helper<1> { using type = mat3; };
template < uint32_t NC > using jac_t = typename jac_type_helper<NC>::type;

__host__ __device__ inline double flux_component_dot(const vec3 & flux, uint32_t /*c*/, const vec3 & g) {
  return dot(flux, g);
}

__host__ __device__ inline double flux_component_dot(const mat3 & flux, uint32_t c, const vec3 & g) {
  return dot(flux[c], g);
}

// contribution of the constitutive jacobian to the (i,j) component pair of a
// stiffness block:  dN_I[k] * (dflux_ik / dgrad_jm) * dN_J[m]
__host__ __device__ inline double jac_contract(const mat3 & C, uint32_t /*i*/, uint32_t /*j*/,
                                               const vec3 & dN_I, const vec3 & dN_J) {
  return dot(dN_I, dot(C, dN_J));
}

__host__ __device__ inline double jac_contract(const mat<3,3,mat<3,3>> & D, uint32_t i, uint32_t j,
                                               const vec3 & dN_I, const vec3 & dN_J) {
  // D[a][b][c][d] = dP_ab / dF_cd ; we need dN_I[k] * dP_ik/dF_jm * dN_J[m]
  double sum = 0.0;
  for (uint32_t k = 0; k < 3; k++) {
    for (uint32_t m = 0; m < 3; m++) {
      sum += dN_I[k] * D[i][k][j][m] * dN_J[m];
    }
  }
  return sum;
}

////////////////////////////////////////////////////////////////////////////////
// expanded quadrature tables (uniform layout for tet and hex)
////////////////////////////////////////////////////////////////////////////////

template < Geometry geom >
inline vec3 expanded_qpt(uint32_t q, nd::view<const double, 2> pts) {
  if constexpr (geom == Geometry::Hexahedron) {
    uint32_t q1D = pts.shape[0];
    return vec3{pts(q % q1D, 0), pts((q % (q1D * q1D)) / q1D, 0), pts(q / (q1D * q1D), 0)};
  } else {
    return vec3{pts(q, 0), pts(q, 1), pts(q, 2)};
  }
}

template < Geometry geom >
inline double expanded_qwt(uint32_t q, nd::view<const double, 1> w) {
  if constexpr (geom == Geometry::Hexahedron) {
    uint32_t q1D = w.shape[0];
    return w(q % q1D) * w((q % (q1D * q1D)) / q1D) * w(q / (q1D * q1D));
  } else {
    return w(q);
  }
}

// shape function gradient table G(q, i, d) at every (expanded) quadrature
// point, and expanded weights W(q)
template < Geometry geom >
inline void build_qtables(const Domain<> & domain, uint32_t p,
                          nd::array<double, 3, memory::space::cpu> & G,
                          nd::array<double, 1, memory::space::cpu> & W) {
  FiniteElement< geom, Family::H1 > el{p};
  nd::view<const double, 2> pts = domain.rule[geom].points;
  nd::view<const double, 1> wts = domain.rule[geom].weights;
  uint32_t nq = femto::impl::qpe<geom>(pts.shape[0]);
  uint32_t nn = el.num_nodes();

  G = nd::array<double, 3, memory::space::cpu>({nq, nn, 3u});
  W = nd::array<double, 1, memory::space::cpu>({nq});
  for (uint32_t q = 0; q < nq; q++) {
    vec3 xi_q = expanded_qpt<geom>(q, pts);
    W(q) = expanded_qwt<geom>(q, wts);
    for (uint32_t i = 0; i < nn; i++) {
      vec3 g = el.shape_function_gradient(xi_q, i);
      G(q, i, 0) = g[0];
      G(q, i, 1) = g[1];
      G(q, i, 2) = g[2];
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// tet-only cuboid mesh: Kuhn subdivision (6 positively-oriented tets per cube)
////////////////////////////////////////////////////////////////////////////////

inline Mesh<> tet_cuboid(uint32_t n) {
  uint32_t nv = n + 1;
  uint32_t dx = 1, dy = nv, dz = nv * nv;

  nd::array<double, 2, memory::space::cpu> nodes({nv * nv * nv, 3u});
  for (uint32_t z = 0; z < nv; z++) {
    for (uint32_t y = 0; y < nv; y++) {
      for (uint32_t x = 0; x < nv; x++) {
        uint32_t id = x * dx + y * dy + z * dz;
        nodes(id, 0) = double(x) / n;
        nodes(id, 1) = double(y) / n;
        nodes(id, 2) = double(z) / n;
      }
    }
  }

  // the 6 permutations of (x,y,z), with parity
  const uint32_t perms[6][3] = {{0,1,2}, {1,2,0}, {2,0,1}, {0,2,1}, {2,1,0}, {1,0,2}};
  const bool even[6] = {true, true, true, false, false, false};
  const uint32_t step[3] = {dx, dy, dz};

  nd::array<uint32_t, 2, memory::space::cpu> tets({6 * n * n * n, 4u});
  nd::array<uint32_t, 2, memory::space::cpu> hexes({0, 0});

  uint32_t t = 0;
  for (uint32_t z = 0; z < n; z++) {
    for (uint32_t y = 0; y < n; y++) {
      for (uint32_t x = 0; x < n; x++) {
        uint32_t v0 = x * dx + y * dy + z * dz;
        for (uint32_t p = 0; p < 6; p++) {
          uint32_t v1 = v0 + step[perms[p][0]];
          uint32_t v2 = v1 + step[perms[p][1]];
          uint32_t v3 = v2 + step[perms[p][2]];
          tets(t, 0) = v0;
          tets(t, 1) = v1;
          tets(t, 2) = even[p] ? v2 : v3;
          tets(t, 3) = even[p] ? v3 : v2;
          t++;
        }
      }
    }
  }

  return Mesh<>::create_3D(nodes, 1, tets, hexes);
}

////////////////////////////////////////////////////////////////////////////////
// initial guess: smooth in space, identical on cpu/gpu
////////////////////////////////////////////////////////////////////////////////

inline void fill_solution_field(Field<Family::H1> & u, const Mesh<> & mesh) {
  uint32_t n = u.data.shape[0];
  uint32_t nc = u.data.shape[1];
  nd::array<double, 2, memory::space::cpu> X_nodes = nodes_for(u, mesh);
  FEMTO_ASSERT(n == X_nodes.shape[0], "node coordinate count mismatch");
  for (uint32_t i = 0; i < n; i++) {
    double x = X_nodes(i, 0);
    double y = X_nodes(i, 1);
    double z = X_nodes(i, 2);
    for (uint32_t c = 0; c < nc; c++) {
      u.data(i, c) = 0.05 * sin(1.7 * x + 0.4 * c) * cos(1.3 * y - 0.2 * c) * (0.5 + 0.5 * z);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// comparison helpers
////////////////////////////////////////////////////////////////////////////////

inline double rel_l2_diff(const double * a, const double * b, size_t n) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < n; i++) {
    double d = a[i] - b[i];
    num += d * d;
    den += std::max(a[i] * a[i], b[i] * b[i]);
  }
  return (den > 0.0) ? std::sqrt(num / den) : std::sqrt(num);
}

////////////////////////////////////////////////////////////////////////////////
// result reporting (CSV on stdout, collected later into markdown tables)
////////////////////////////////////////////////////////////////////////////////

struct CaseInfo {
  std::string device;    // cpu | gpu
  std::string physics;   // poisson | elasticity
  std::string geom;      // tet | hex
  std::string size;      // medium | large | huge
  uint32_t degree;       // polynomial degree p
  uint64_t num_elements;
  uint64_t num_qpts;
  uint64_t num_unknowns;
};

inline void report_time(const CaseInfo & c, const std::string & quantity,
                        const std::string & approach, const std::string & phase, double seconds) {
  printf("RESULT,%s,%s,%s,%s,%u,%llu,%llu,%llu,%s,%s,%s,%.6f\n",
         c.device.c_str(), c.physics.c_str(), c.geom.c_str(), c.size.c_str(), c.degree,
         (unsigned long long)c.num_elements, (unsigned long long)c.num_qpts,
         (unsigned long long)c.num_unknowns,
         quantity.c_str(), approach.c_str(), phase.c_str(), seconds * 1.0e3);
  fflush(stdout);
}

inline void report_mem(const CaseInfo & c, const std::string & quantity,
                       const std::string & approach, const std::string & what, uint64_t bytes) {
  printf("MEMORY,%s,%s,%s,%s,%u,%llu,%llu,%llu,%s,%s,%s,%llu\n",
         c.device.c_str(), c.physics.c_str(), c.geom.c_str(), c.size.c_str(), c.degree,
         (unsigned long long)c.num_elements, (unsigned long long)c.num_qpts,
         (unsigned long long)c.num_unknowns,
         quantity.c_str(), approach.c_str(), what.c_str(), (unsigned long long)bytes);
  fflush(stdout);
}

inline void report_verify(const CaseInfo & c, const std::string & quantity,
                          const std::string & label, double err, double tol) {
  printf("VERIFY,%s,%s,%s,%s,%u,%s,%s,%.3e,%s\n",
         c.device.c_str(), c.physics.c_str(), c.geom.c_str(), c.size.c_str(), c.degree,
         quantity.c_str(), label.c_str(), err, (err <= tol) ? "PASS" : "FAIL");
  fflush(stdout);
  FEMTO_ASSERT(err <= tol, "verification failed: " + label);
}

inline void report_skip(const CaseInfo & c, const std::string & quantity, uint64_t est_bytes) {
  printf("SKIP,%s,%s,%s,%s,%u,%s,estimated_bytes=%llu\n",
         c.device.c_str(), c.physics.c_str(), c.geom.c_str(), c.size.c_str(), c.degree,
         quantity.c_str(), (unsigned long long)est_bytes);
  fflush(stdout);
}

// rough upper bound on couplings per node, used only to decide whether a
// stiffness case fits in memory
inline uint64_t couplings_per_node(Geometry g, uint32_t p) {
  if (g == Geometry::Hexahedron) { return (2 * p + 1) * (2 * p + 1) * (2 * p + 1); }
  return (p == 1) ? 18 : 55;
}

// run f() `reps` times (after `warmup` calls), return the fastest time
template < typename callable >
double bench(callable && f, int warmup = 1, int reps = 3) {
  for (int i = 0; i < warmup; i++) { f(); }
  double best = 1.0e30;
  for (int i = 0; i < reps; i++) {
    femto::timer t;
    t.start();
    f();
    t.stop();
    best = std::min(best, t.elapsed());
  }
  return best;
}

} // namespace fusion
