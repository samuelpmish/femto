// ScalarPML: the update solves its implicit equation and jac() is the total
// derivative of the flux through it, at a point in 2D and 3D; and on a mesh
// the assembled tangent agrees with perturbations of the residual across a
// Newton loop that starts every iterate from the previous step's state.  The
// wave dynamics carry a nonlinearity, c^2 = c0^2 (1 + beta |grad u|^2), the
// PML part unchanged.
#include <gtest/gtest.h>

#include "femto/mesh.hpp"
#include "femto/field.hpp"
#include "femto/domain.hpp"
#include "linear_algebra/sparse_matrix.hpp"
#include "linear_algebra/sparse_direct.hpp"
#include "forall.hpp"
#include "materials/pml.hpp"

using namespace femto;

template < int dim >
static vec<dim> point() { vec<dim> x{}; x[0] = 0.9; x[1] = -0.8; if constexpr (dim == 3) { x[2] = 0.5; } return x; }

template < int dim >
static void check_point_level() {
  pml::Radial<dim> layer{vec<dim>{}, 1.0, 0.5, 4.0};
  vec<dim> x = point<dim>();
  mat<dim, dim> R = layer.frame(x);
  vec<dim> alpha = layer.alpha(x);
  ASSERT_GT(alpha[0], 0.0);
  ScalarPML<dim> model{2.25, 0.3};
  constexpr int ns = ScalarPML<dim>::nstate;

  vec<dim> du{};
  vec<ns> aux_old{};
  for (int i = 0; i < dim; i++) { du[i] = 0.3 - 0.7 * i; }
  for (int k = 0; k < ns; k++) { aux_old[k] = 0.1 * (k + 1); }

  // the update leaves the old state alone, and the rate form agrees with it:
  // (aux - aux_old) / dt is the rate at (du, aux) stage by stage
  vec<dim> flux, flux_rate;
  vec<ns> aux, auxdot;
  model(R, alpha, du, aux_old, flux, aux);
  typename ScalarPML<dim>::Rate{model.c2}(R, alpha, du, aux, flux_rate, auxdot);
  EXPECT_NEAR(norm((aux - aux_old) / model.dt - auxdot), 0.0, 1e-12);
  EXPECT_NEAR(norm(flux - flux_rate), 0.0, 1e-13);
  EXPECT_EQ(aux_old[0], 0.1);

  // jac against central differences of the update
  mat<dim, dim> K = model.jac(R, alpha);
  double h = 1e-6;
  for (int j = 0; j < dim; j++) {
    vec<dim> e{}; e[j] = h;
    vec<dim> fp, fm; vec<ns> unused;
    model(R, alpha, du + e, aux_old, fp, unused);
    model(R, alpha, du - e, aux_old, fm, unused);
    vec<dim> dK = (fp - fm) / (2 * h);
    for (int i = 0; i < dim; i++) { EXPECT_NEAR(K(i, j), dK[i], 1e-8) << "dim " << dim << " entry " << i << ", " << j; }
  }
  // frozen state: c^2 I
  EXPECT_NEAR(norm(ScalarPML<dim>{2.25, 0.0}.jac(R, alpha) - mat<dim, dim>(2.25 * Identity<dim>())), 0.0, 1e-14);
  // the frame is orthonormal
  EXPECT_NEAR(norm(dot(R, transpose(R)) - mat<dim, dim>(Identity<dim>())), 0.0, 1e-14);
}

TEST(ScalarPML, point_level_2D) { check_point_level<2>(); }
TEST(ScalarPML, point_level_3D) { check_point_level<3>(); }

TEST(ScalarPML, alpha_t_is_the_running_mean_of_alpha_r) {
  pml::Radial<3> layer{vec3{0.0, 0.0, 0.0}, 1.5, 0.5, 4.0};
  EXPECT_EQ(layer.alpha(vec3{1.2, 0.0, 0.0})[0], 0.0);
  double r = 1.85, sum = 0;
  int n = 2000;
  for (int i = 0; i < n; i++) {
    double s = 1.5 + (r - 1.5) * (i + 0.5) / n;
    sum += layer.alpha(vec3{s, 0.0, 0.0})[0] * (r - 1.5) / n;
  }
  vec3 s = layer.alpha(vec3{0.0, r, 0.0});
  EXPECT_NEAR(s[1], sum / r, 1e-7);
  EXPECT_EQ(s[2], s[1]);
}

////////////////////////////////////////////////////////////////////////////////

template < int dim >
struct Kerr {
  ScalarPML<dim> pml;
  double beta;
  double c2(const vec<dim> & du) const { return 1.0 + beta * norm_squared(du); }
  void operator()(const mat<dim, dim> & R, const vec<dim> & alpha, const vec<dim> & du, const vec<ScalarPML<dim>::nstate> & aux_old,
                  vec<dim> & flux, vec<ScalarPML<dim>::nstate> & aux) const {
    pml(R, alpha, du, aux_old, flux, aux);
    flux = c2(du) * flux;
  }
  mat<dim, dim> jac(const mat<dim, dim> & R, const vec<dim> & alpha, const vec<dim> & du, const vec<ScalarPML<dim>::nstate> & aux) const {
    vec<dim> f; vec<ScalarPML<dim>::nstate> unused;
    typename ScalarPML<dim>::Rate{pml.c2}(R, alpha, du, aux, f, unused);
    return c2(du) * pml.jac(R, alpha) + outer(2.0 * beta * f, du);
  }
};

template < int dim >
struct ImplicitWave {
  static constexpr int ns = ScalarPML<dim>::nstate;
  Mesh<> mesh;
  Field<Family::H1> u_field;
  BasisFunction<Family::H1> phi;
  Domain<> domain;
  uint32_t nn, nq;
  double dt = 0.05;
  Kerr<dim> model{ScalarPML<dim>{1.0, dt}, 0.3};
  nd::cpu_array<double, 3> frame_q;
  nd::cpu_array<double, 2> alpha_q, alpha_n, Ml;
  vector u_n, v_n, psi_n;               // the previous step, psi = int u dt for the 3D mass side
  nd::cpu_array<double, 2> aux_n;     // its auxiliary state, never written by an iterate
  nd::cpu_array<double, 3> du_q;
  nd::cpu_array<double, 2> flux_q, aux_q;

  static Mesh<> make_mesh() {
    if constexpr (dim == 2) { return Mesh<>::disk(vec2{0.0, 0.0}, 1.5, 0.25, 1, Geometry::Triangle); }
    else { return Mesh<>::ball(vec3{0.0, 0.0, 0.0}, 1.5, 0.45, 1, Geometry::Tetrahedron); }
  }

  ImplicitWave() : mesh(make_mesh()), u_field(create_field<Family::H1>(mesh, 1, 1)), phi(u_field), domain(mesh, MeshQuadratureRule(2)) {
    nn = mesh.vert.shape[0];
    nq = total(domain.num_qpts);
    double h = dim == 2 ? 0.25 : 0.45;
    pml::Radial<dim> layer{vec<dim>{}, 1.0, 0.5, 1.4 / h};
    nd::cpu_array<double, 3> x_q = evaluate(mesh.X, domain);
    frame_q = forall(std::function<mat<dim, dim>(const vec<dim> &)>([&](const vec<dim> & x) { return layer.frame(x); }), x_q);
    alpha_q = forall(std::function<vec<dim>(const vec<dim> &)>([&](const vec<dim> & x) { return layer.alpha(x); }), x_q);
    alpha_n = forall(std::function<vec<dim>(const vec<dim> &)>([&](const vec<dim> & x) { return layer.alpha(x); }), mesh.X.data);
    nd::cpu_array<double, 1> ones_q({nq});
    for (uint32_t q = 0; q < nq; q++) { ones_q(q) = 1.0; }
    Residual<Family::H1> m = integrate(dot(ones_q, phi), domain);
    Ml = m.data;

    u_n = vector(nn); v_n = vector(nn); psi_n = vector(nn);
    vec<dim> x0{}; x0[0] = 0.4; x0[1] = 0.1;
    for (uint32_t i = 0; i < nn; i++) {
      u_n[i] = std::exp(-norm_squared(load<vec<dim>>(mesh.X.data, i) - x0) / 0.09);
      v_n[i] = 0.0; psi_n[i] = 0.0;
    }
    aux_n.resize({nq, uint32_t(ns)});
  }

  // det S applied to u_tt at a node: u_tt + e1 u_t + e2 u + e3 int u, with
  // e1 = sum alpha, e2 = sum_{i<j} alpha_i alpha_j, e3 = prod alpha (3D only)
  void mass_side(uint32_t i, double & e1, double & e2, double & e3) const {
    vec<dim> s = load<vec<dim>>(alpha_n, i);
    e1 = 0; e2 = 0; e3 = 1;
    for (int a = 0; a < dim; a++) { e1 += s[a]; e3 *= s[a]; for (int b = a + 1; b < dim; b++) { e2 += s[a] * s[b]; } }
    if constexpr (dim == 2) { e3 = 0; }
  }

  vector residual(const vector & u) {
    u_field.v() = u;
    du_q = evaluate(grad(u_field), domain);
    forall(model, frame_q, alpha_q, du_q, aux_n, flux_q, aux_q);
    Residual<Family::H1> r = integrate(dot(flux_q, grad(phi)), domain);
    vector R(nn);
    for (uint32_t i = 0; i < nn; i++) {
      double e1, e2, e3; mass_side(i, e1, e2, e3);
      R[i] = Ml(i, 0) * ((u[i] - u_n[i] - dt * v_n[i]) / (dt * dt) + e1 * (u[i] - u_n[i]) / dt + e2 * u[i] + e3 * (psi_n[i] + dt * u[i])) + r.data(i, 0);
    }
    return R;
  }

  sparse_matrix<> jacobian() {
    nd::cpu_array<double, 3> K_q = forall(std::function<mat<dim, dim>(const mat<dim, dim> &, const vec<dim> &, const vec<dim> &, const vec<ns> &)>(
      [&](const mat<dim, dim> & R, const vec<dim> & alpha, const vec<dim> & du, const vec<ns> & aux) { return model.jac(R, alpha, du, aux); }),
      frame_q, alpha_q, du_q, aux_q);
    sparse_matrix<> J = integrate(dot(grad(phi), K_q, grad(phi)), domain);
    for (uint32_t i = 0; i < nn; i++) {
      double e1, e2, e3; mass_side(i, e1, e2, e3);
      J.at(i, i) += Ml(i, 0) * (1.0 / (dt * dt) + e1 / dt + e2 + e3 * dt);
    }
    J.symmetry = Symmetry::Unsymmetric;
    J.definiteness = Definiteness::Indefinite;
    return J;
  }
};

template < int dim >
static void run_newton() {
  ImplicitWave<dim> w;
  vector u = w.u_n;
  vector d(w.nn);
  for (uint32_t i = 0; i < w.nn; i++) { d[i] = std::sin(0.37 * (i + 1)); }
  double alpha_max = 0;
  for (uint32_t k = 0; k < w.alpha_q.sz; k++) { alpha_max = std::max(alpha_max, w.alpha_q.data()[k]); }
  ASSERT_GT(alpha_max, 1.0) << "the layer is not active on this mesh";
  double h = 1e-5, r0 = 0;
  int iterations = 0;
  for (; iterations < 10; iterations++) {
    vector R = w.residual(u);
    if (iterations == 0) { r0 = norm(R); }
    if (norm(R) < 1e-10 * r0) { break; }
    sparse_matrix<> J = w.jacobian();

    vector Jd = dot(J, d);
    vector Rp = w.residual(vector(u + h * d)), Rm = w.residual(vector(u - h * d));
    vector fd = vector((Rp - Rm) / (2 * h));
    EXPECT_LT(norm(vector(fd - Jd)), 1e-7 * norm(Jd)) << "dim " << dim << " iterate " << iterations;

    w.residual(u);
    nd::cpu_array<double, 2> aux_first = w.aux_q;
    w.residual(u);
    for (uint32_t k = 0; k < aux_first.sz; k++) {
      ASSERT_EQ(w.aux_q.data()[k], aux_first.data()[k]) << "aux accumulated across evaluations";
      ASSERT_EQ(w.aux_n.data()[k], 0.0) << "the previous step's state was modified";
    }

    sparse_factorization invJ = inv(J);
    u = vector(u - dot(invJ, R));
  }
  EXPECT_LE(iterations, 6) << "Newton did not converge quadratically";
  EXPECT_LT(norm(w.residual(u)), 1e-10 * r0);
  EXPECT_GT(norm(vector(u - w.u_n)), 1e-3);
}

TEST(ScalarPML, newton_tangent_matches_residual_perturbations_2D) { run_newton<2>(); }
TEST(ScalarPML, newton_tangent_matches_residual_perturbations_3D) { run_newton<3>(); }
