#include "gtest/gtest.h"

#include "../../examples/cpp/mug_analysis.hpp"

using namespace femto;

// CGS units: cm, g/cm^3, dyn/cm^2
static constexpr double rho = 2.4;      // g/cm^3
static constexpr double E = 1.0e12;     // 100 GPa in dyn/cm^2
static constexpr double nu = 0.3;

// volume of a trilinear hex by 2x2x2 Gauss quadrature
static double hex_volume(const Mesh<> & mesh, uint32_t e) {
  vec3 v[8];
  for (int c = 0; c < 8; c++) {
    uint32_t vid = mesh.hex(e, c).index;
    v[c] = vec3{mesh.X.data(vid, 0), mesh.X.data(vid, 1), mesh.X.data(vid, 2)};
  }
  double g = 1.0 / std::sqrt(3.0);
  double volume = 0.0;
  for (int q = 0; q < 8; q++) {
    double xi = 0.5 + 0.5 * g * ((q & 1) ? 1 : -1);
    double eta = 0.5 + 0.5 * g * ((q & 2) ? 1 : -1);
    double zeta = 0.5 + 0.5 * g * ((q & 4) ? 1 : -1);
    // corner ordering: (0,0,0),(1,0,0),(1,1,0),(0,1,0), then +zeta
    double sx[8] = {(1-xi)*(1-eta), xi*(1-eta), xi*eta, (1-xi)*eta,
                    (1-xi)*(1-eta), xi*(1-eta), xi*eta, (1-xi)*eta};
    vec3 d_dxi{}, d_deta{}, d_dzeta{};
    double dxi[8] = {-(1-eta), (1-eta), eta, -eta, -(1-eta), (1-eta), eta, -eta};
    double deta[8] = {-(1-xi), -xi, xi, (1-xi), -(1-xi), -xi, xi, (1-xi)};
    for (int c = 0; c < 8; c++) {
      double sz = (c < 4) ? (1 - zeta) : zeta;
      double dz = (c < 4) ? -1.0 : 1.0;
      d_dxi = d_dxi + v[c] * (dxi[c] * sz);
      d_deta = d_deta + v[c] * (deta[c] * sz);
      d_dzeta = d_dzeta + v[c] * (sx[c] * dz);
    }
    volume += dot(cross(d_dxi, d_deta), d_dzeta) * 0.125;
  }
  return volume;
}

TEST(mug_analysis, mass_and_stiffness) {

  Mesh<> mesh = Mesh<>::coffee_mug(3.0, 4.0, 9.0, 0.5, 1, 1);
  mug::Analysis a = mug::analyze(mesh, rho, E, nu);

  EXPECT_EQ(a.ndof, 3 * mesh.vert.shape[0]);
  EXPECT_EQ(a.M.nnz, a.K.nnz);

  // total mass: 1^T M 1 counts each of the three components once,
  // so it should equal 3 * rho * volume
  double volume = 0.0;
  for (uint32_t e = 0; e < mesh.hex.shape[0]; e++) { volume += hex_volume(mesh, e); }

  vector ones(a.ndof);
  for (uint32_t i = 0; i < a.ndof; i++) { ones[i] = 1.0; }
  vector M1 = a.M(ones);
  double total = 0.0;
  for (uint32_t i = 0; i < a.ndof; i++) { total += M1[i]; }
  EXPECT_NEAR(total, 3 * rho * volume, 1e-6 * 3 * rho * volume);

  // rigid body motions (translations and linearized rotations) produce no
  // elastic force: K u_rigid = 0
  auto check_nullspace = [&](auto make_u) {
    vector u(a.ndof);
    for (uint32_t n = 0; n < a.ndof / 3; n++) {
      vec3 X{a.nodes(n, 0), a.nodes(n, 1), a.nodes(n, 2)};
      vec3 val = make_u(X);
      for (int d = 0; d < 3; d++) { u[3 * n + d] = val[d]; }
    }
    vector Ku = a.K(u);
    double num = 0, den = 0, unorm = 0;
    for (uint32_t i = 0; i < a.ndof; i++) { num += Ku[i] * Ku[i]; unorm += u[i] * u[i]; }
    for (size_t i = 0; i < a.K.nnz; i++) { den += a.K.values(i) * a.K.values(i); }
    return std::sqrt(num) / (std::sqrt(den) * std::sqrt(unorm));
  };

  EXPECT_LT(check_nullspace([](vec3) { return vec3{1, 0, 0}; }), 1e-12);
  EXPECT_LT(check_nullspace([](vec3) { return vec3{0, 0, 1}; }), 1e-12);
  EXPECT_LT(check_nullspace([](vec3 X) { return vec3{-X[1], X[0], 0}; }), 1e-12);
  EXPECT_LT(check_nullspace([](vec3 X) { return vec3{0, -X[2], X[1]}; }), 1e-12);

  // the row-sum lumped mass (used by the explicit integrator) is strictly
  // positive
  double dmin = 1e30;
  for (uint32_t i = 0; i < a.ndof; i++) { dmin = std::min(dmin, M1[i]); }
  EXPECT_GT(dmin, 0.0);

}

TEST(mug_analysis, transient) {

  Mesh<> mesh = Mesh<>::coffee_mug(3.0, 4.0, 9.0, 0.5, 1, 1);
  auto a = std::make_shared<mug::Analysis>(mug::analyze(mesh, rho, E, nu));
  mug::TransientSim sim(a, mesh, 1.0e6);

  EXPECT_GT(sim.dt, 0.0);
  EXPECT_LT(sim.dt, 1e-4); // ~h/c is well under a microsecond for stiff ceramic

  // the face pressure is already on at t = 0
  double acc_max = 0;
  for (uint32_t i = 0; i < a->ndof; i++) { acc_max = std::max(acc_max, std::fabs(sim.acc[i])); }
  EXPECT_GT(acc_max, 0.0);

  // step well past the end of the 5 us load: the response is nonzero and
  // remains bounded (stability)
  sim.advance(uint32_t(2.0 * sim.force_duration / sim.dt));
  EXPECT_GT(sim.max_displacement, 0.0);
  EXPECT_LT(sim.max_displacement, 1.0); // cm; 1 bar for 5 us shouldn't blow up

  // the held nodes at the center of the base never move
  uint32_t nv = mesh.vert.shape[0];
  double zmax = 0, r_base = 0;
  for (uint32_t n = 0; n < nv; n++) { zmax = std::max(zmax, a->nodes(n, 2)); }
  for (uint32_t n = 0; n < nv; n++) {
    if (a->nodes(n, 2) < 1e-9 * zmax) { r_base = std::max(r_base, std::hypot(a->nodes(n, 0), a->nodes(n, 1))); }
  }
  uint32_t held = 0;
  for (uint32_t n = 0; n < nv; n++) {
    if (a->nodes(n, 2) < 1e-9 * zmax && std::hypot(a->nodes(n, 0), a->nodes(n, 1)) < 0.35 * r_base) {
      held++;
      for (int d = 0; d < 3; d++) { EXPECT_EQ(sim.u[3 * n + d], 0.0); }
    }
  }
  EXPECT_GT(held, 2u);

}

TEST(mug_analysis, modal) {

  Mesh<> mesh = Mesh<>::coffee_mug(3.0, 4.0, 9.0, 0.5, 1, 1);
  mug::Analysis a = mug::analyze(mesh, rho, E, nu);

  uint32_t nmodes = 10;
  mug::ModalResult r = mug::modal(a, nmodes);

  EXPECT_EQ(r.eigenvalues.size(), nmodes);
  EXPECT_EQ(r.modes.size(), size_t(nmodes) * mesh.vert.shape[0] * 3);

  // elastic eigenvalues are positive, ascending, and correspond to
  // audible-range frequencies for a ceramic mug (hundreds of Hz to ~100 kHz)
  for (uint32_t i = 0; i < nmodes; i++) {
    EXPECT_GT(r.eigenvalues[i], 0.0);
    if (i > 0) { EXPECT_GE(r.eigenvalues[i], r.eigenvalues[i - 1]); }
    double hz = std::sqrt(r.eigenvalues[i]) / (2 * M_PI);
    EXPECT_GT(hz, 100.0);
    EXPECT_LT(hz, 1e6);
  }

  // the Lanczos iteration converged before hitting its basis cap
  EXPECT_LT(r.iterations, 120u);

}
