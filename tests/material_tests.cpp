// SPDX-FileCopyrightText: Copyright (c) 2023-2026 Sam Mish
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "common.hpp"

#include "misc/timer.hpp"
#include "materials/plane_strain.hpp"
#include "materials/neohookean.hpp"
#include "materials/corotational.hpp"
#include "materials/viscoelasticity.hpp"
#include "materials/J2_plasticity.hpp"

#include <cmath>
#include <iostream>
#include <vector>

using namespace fm;
using namespace femto;

template < int dim >
mat<dim,dim> e(int i, int j) {
  mat<dim,dim> out{};
  out(i,j) = 1.0;
  return out;
}

double mu(mat3 J) {
  return (ddot(J, J) / (3 * std::pow(det(J), 2.0 / 3.0))) - 1.0;
}

mat3 dmu_dJ(mat3 J) {
  double JJ = ddot(J, J);
  mat3 invJT = inv(transpose(J));
  double scale = (2.0 / (3.0 * std::pow(det(J), 2.0 / 3.0)));
  return scale * (J - (JJ/3.0) * invJT);
}

TEST(TMOP, check_derivatives) {
  constexpr int dim = 3;
  double eps = 1.0e-6;

  mat3 J = Identity<3>() + 0.1 * femto::random_mat<3,3>();

  mat3 dmudJ = dmu_dJ(J);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      double dmudJij = (mu(J + eps * e<dim>(i,j)) - mu(J - eps * e<dim>(i,j))) / (2.0 * eps);
      EXPECT_NEAR(dmudJij, dmudJ[i][j], 1.0e-9);
    }
  }
}

TEST(NeoHookean, check_derivatives) {
  constexpr int dim = 3;
  double eps = 1.0e-7;

  double lambda = 2.0;
  double mu = 1.0;
  NeoHookeanModel material{lambda, mu};

  mat3 du_dX = 0.1 * femto::random_mat<3,3>();
  mat3 ddu_dX = 0.1 * femto::random_mat<3,3>();
  mat<3,3, mat3> dP_dF = material.jac(du_dX);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        for (int l = 0; l < 3; l++) {
          mat3 dP = material.jvp(du_dX, e<dim>(k,l));
          EXPECT_NEAR(dP_dF[i][j][k][l], dP[i][j], 1.0e-14);

          mat3 d_ddudX = material.vjp(du_dX, e<dim>(i,j));
          EXPECT_NEAR(dP_dF[i][j][k][l], d_ddudX[k][l], 1.0e-14);
        }
      }
    }
  }

  mat3 dP_fd = (material(du_dX + eps * ddu_dX) - material(du_dX - eps * ddu_dX)) / (2 * eps); 
  mat3 dP_expected{};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      // note: looser tolerance, since finite difference stencil isn't as accurate
      EXPECT_NEAR(dP_fd[i][j], ddot(dP_dF[i][j], ddu_dX), 1.0e-8);
    }
  }


}

TEST(PlaneStrainNeoHookean, check_derivatives) {
  constexpr int dim = 2;
  double eps = 1.0e-7;

  double lambda = 2.0;
  double mu = 1.0;
  PlaneStrain<NeoHookeanModel> material{{lambda, mu}};

  mat<dim,dim> du_dX = 0.1 * femto::random_mat<dim,dim>();
  mat<dim,dim> ddu_dX = 0.1 * femto::random_mat<dim,dim>();
  mat<dim,dim,mat<dim,dim>> dP_dF = material.jac(du_dX);

  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      for (int k = 0; k < dim; k++) {
        for (int l = 0; l < dim; l++) {
          mat<dim,dim> dP = material.jvp(du_dX, e<dim>(k,l));
          EXPECT_NEAR(dP_dF[i][j][k][l], dP[i][j], 1.0e-14);

          mat<dim,dim> d_ddudX = material.vjp(du_dX, e<dim>(i,j));
          EXPECT_NEAR(dP_dF[i][j][k][l], d_ddudX[k][l], 1.0e-14);
        }
      }
    }
  }

  mat<dim,dim> dP_fd = (material(du_dX + eps * ddu_dX) - material(du_dX - eps * ddu_dX)) / (2 * eps); 
  mat<dim,dim> dP_expected{};
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      // note: looser tolerance, since finite difference stencil isn't as accurate
      EXPECT_NEAR(dP_fd[i][j], ddot(dP_dF[i][j], ddu_dX), 1.0e-8);
    }
  }
}

// tol is how orthogonal R comes out: polar_eig's degrades as cond(F)^2
template < typename polar_t >
void check_polar_decomposition(polar_t polar, double tol) {
  mat3 Q = RotationMatrix(vec3{0.3, -0.5, 0.4});

  // a modest strain, a step's worth of one, and stretches of 0.2 and 6 that
  // start Newton's iteration far from its quadratic regime
  mat3 small = Identity<3>() + 0.1 * femto::random_mat<3,3>();
  mat3 tiny = Identity<3>() + 1.0e-4 * femto::random_mat<3,3>();
  mat3 large = dot(Q, dot(diag3{{6.0, 0.2, 1.3}}, transpose(mat3(RotationMatrix(vec3{-0.2, 0.7, 0.1})))));

  for (mat3 F : {small, tiny, large}) {
    auto [R, U] = polar(F);
    EXPECT_NEAR(norm(dot(transpose(R), R) - Identity<3>()), 0.0, tol);
    EXPECT_NEAR(det(R), 1.0, tol);
    EXPECT_NEAR(norm(dot(R, U) - F), 0.0, 1.0e-14 * norm(F));
    EXPECT_GT(min(eig(U).lambda), 0.0);
    EXPECT_NEAR(norm(polar(dot(Q, F)).R - dot(Q, R)), 0.0, 1.0e-14);
  }
}

TEST(PolarDecomposition, newton) { check_polar_decomposition(polar_newton, 1.0e-14); }
TEST(PolarDecomposition, eig) { check_polar_decomposition(polar_eig, 1.0e-13); }

TEST(Logm, is_the_log_of_the_eigenvalues) {
  mat3 Q = RotationMatrix(vec3{0.3, -0.5, 0.4});
  vec3 a = {0.5, 2.0, 3.0};
  mat3 expected = dot(Q, dot(diag3{{std::log(a[0]), std::log(a[1]), std::log(a[2])}}, transpose(Q)));
  EXPECT_NEAR(norm(logm(sym3(dot(Q, dot(diag3{a}, transpose(Q))))) - expected), 0.0, 1.0e-14);

  // near the identity, where a step's stretch is: log(I + E) = E - E^2 / 2 + E^3 / 3 - ...
  mat3 E = 1.0e-4 * mat3(sym3(femto::random_mat<3,3>()));
  mat3 series = E - 0.5 * dot(E, E) + dot(E, dot(E, E)) / 3.0;
  EXPECT_NEAR(norm(logm(sym3(Identity<3>() + E)) - series), 0.0, 1.0e-15);
}

TEST(Logm, series_matches_the_eigendecomposition) {
  for (double scale : {0.0, 1.0e-6, 1.0e-3, 0.05, 0.3}) {
    sym3 A = sym3(Identity<3>() + scale * mat3(sym3(femto::random_mat<3,3>())));
    EXPECT_NEAR(norm(logm_series(A) - logm(A)), 0.0, 1.0e-15) << "scale " << scale;
  }
  // too far from I for the series, so logm's own answer
  sym3 A = sym3(diag3{{6.0, 0.2, 1.3}});
  EXPECT_EQ(norm(logm_series(A) - logm(A)), 0.0);
}

// a model driven through F(t), t from 0 to 1 in n steps from sigma0: the stress
// after each step, and the PK1 stress of the last
template < typename model_t, typename callable >
std::vector< mat3 > drive(model_t model, callable F, mat3 sigma0, int n, mat3 & P) {
  std::vector< mat3 > sigma = {sigma0};
  mat3 du_dX_old = F(0.0) - Identity<3>();
  for (int k = 1; k <= n; k++) {
    mat3 du_dX = F(k / (1.0 * n)) - Identity<3>(), sigma_new;
    model(du_dX, du_dX_old, sigma.back(), P, sigma_new);
    sigma.push_back(sigma_new);
    du_dX_old = du_dX;
  }
  return sigma;
}

using HughesWingetModel = CorotationalModel< HughesWinget >;
using RashidEigModel = CorotationalModel< RashidEig >;
using RashidNewtonModel = CorotationalModel< RashidNewton >;

// the notebook's first problem: a rigid rotation by pi about z, in 32 steps,
// carries the stress along exactly
template < typename model_t >
void check_rigid_rotation() {
  model_t model{1.0, 1.0};
  auto Q = [](double t) { double c = std::cos(M_PI * t), s = std::sin(M_PI * t); return mat3{{{c, -s, 0.0}, {s, c, 0.0}, {0.0, 0.0, 1.0}}}; };
  mat3 sigma0 = diag3{{3.0, 2.0, 1.0}}, P;
  std::vector< mat3 > sigma = drive(model, Q, sigma0, 32, P);
  for (int k = 0; k <= 32; k++) {
    mat3 expected = dot(Q(k / 32.0), dot(sigma0, transpose(Q(k / 32.0))));
    EXPECT_NEAR(norm(sigma[k] - expected), 0.0, 1.0e-13) << "step " << k;
  }
  // and P = J sigma F^-T, at F = Q(1)
  EXPECT_NEAR(norm(P - dot(sigma.back(), transpose(inv(Q(1.0))))), 0.0, 1.0e-13);
}

TEST(Corotational, rigid_rotation_hughes_winget) { check_rigid_rotation< HughesWingetModel >(); }
TEST(Corotational, rigid_rotation_rashid_eig) { check_rigid_rotation< RashidEigModel >(); }
TEST(Corotational, rigid_rotation_rashid_newton) { check_rigid_rotation< RashidNewtonModel >(); }

// the notebook's convergence study: simple shear, F = I + (pi/2) t e1 e2,
// against the exact solution of the rate equation at t = 1, which for a
// shear gamma is sigma_12 = mu sin(gamma), sigma_11 = -sigma_22 = mu (1 - cos(gamma)).
// All converge at second order (the notebook's own Hughes-Winget, at first)
template < typename model_t >
void check_simple_shear_convergence(const char * name, double order) {
  double mu = 1.0, gamma = M_PI / 2;
  model_t model{1.0, mu};
  auto F = [=](double t) { mat3 F = Identity<3>(); F(0, 1) = gamma * t; return F; };
  mat3 exact{}, P;
  exact(0, 0) = mu * (1 - std::cos(gamma));
  exact(1, 1) = -exact(0, 0);
  exact(0, 1) = exact(1, 0) = mu * std::sin(gamma);

  double error_last = 0.0;
  for (int n : {8, 16, 32, 64, 128, 256}) {
    double error = norm(drive(model, F, mat3{}, n, P).back() - exact) / norm(exact);
    if (n > 8) {
      EXPECT_NEAR(std::log2(error_last / error), order, 0.1) << name << ", n = " << n << ", relative error " << error;
    }
    error_last = error;
  }
}

TEST(Corotational, simple_shear_converges_hughes_winget) { check_simple_shear_convergence< HughesWingetModel >("hughes-winget", 2.0); }
TEST(Corotational, simple_shear_converges_rashid_eig) { check_simple_shear_convergence< RashidEigModel >("rashid-eig", 2.0); }
TEST(Corotational, simple_shear_converges_rashid_newton) { check_simple_shear_convergence< RashidNewtonModel >("rashid-newton", 2.0); }

// both Rashids give what polar_newton and logm do separately
template < typename kinematics >
void check_rashid_kinematics() {
  for (double scale : {1.0e-6, 1.0e-3, 0.1}) {
    mat3 f = dot(mat3(RotationMatrix(vec3{0.3, -0.5, 0.4})), Identity<3>() + scale * femto::random_mat<3,3>());
    auto [R, U] = polar_newton(f);
    Increment step = kinematics{}(f);
    EXPECT_NEAR(norm(step.R - R), 0.0, 1.0e-14) << "scale " << scale;
    EXPECT_NEAR(norm(step.d - logm(U)), 0.0, 1.0e-14) << "scale " << scale;
  }
}

TEST(Corotational, rashid_eig_is_the_polar_decomposition_and_log) { check_rashid_kinematics< RashidEig >(); }
TEST(Corotational, rashid_newton_is_the_polar_decomposition_and_log) { check_rashid_kinematics< RashidNewton >(); }

// one small step from no stress is linear elasticity, with P = sigma to first order
template < typename model_t >
void check_small_step() {
  double lambda = 2.0, mu = 1.0;
  model_t model{lambda, mu};
  mat3 du_dX = 1.0e-6 * femto::random_mat<3,3>(), P, sigma;
  model(du_dX, mat3{}, mat3{}, P, sigma);
  mat3 linear = LinearElasticModel{lambda, mu}(du_dX);
  EXPECT_LT(norm(sigma - linear), 1.0e-5 * norm(linear));
  EXPECT_LT(norm(P - linear), 1.0e-5 * norm(linear));
}

TEST(Corotational, small_step_is_linear_elastic_hughes_winget) { check_small_step< HughesWingetModel >(); }
TEST(Corotational, small_step_is_linear_elastic_rashid_eig) { check_small_step< RashidEigModel >(); }
TEST(Corotational, small_step_is_linear_elastic_rashid_newton) { check_small_step< RashidNewtonModel >(); }

TEST(LinearViscoelastic, check_derivatives) {
  double eps = 1.0e-7;

  LinearViscoelasticModel material{2.0, 1.0, 3.0, 0.5, 0.1};

  mat3 du_dX = 0.1 * femto::random_mat<3,3>();
  mat3 ddu_dX = 0.1 * femto::random_mat<3,3>();
  mat3 ev_old = 0.05 * femto::random_mat<3,3>();
  mat<3,3, mat3> dP_dF = material.jac(du_dX);

  mat3 Pp, Pm, ev;
  material(du_dX + eps * ddu_dX, ev_old, Pp, ev);
  material(du_dX - eps * ddu_dX, ev_old, Pm, ev);
  mat3 dP_fd = (Pp - Pm) / (2 * eps);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      EXPECT_NEAR(dP_fd[i][j], ddot(dP_dF[i][j], ddu_dX), 1.0e-8);
    }
  }
}

TEST(LinearViscoelastic, relaxes_to_the_equilibrium_modulus) {
  double lambda = 2.0, mu = 1.0, mu_v = 3.0, tau = 0.5, dt = 0.05;
  LinearViscoelasticModel material{lambda, mu, mu_v, tau, dt};

  // hold a shear: backward Euler takes 1 / (1 + dt / tau) off the branch's
  // stress each step, from mu + mu_v toward mu
  mat3 du_dX{};
  du_dX(0, 1) = 0.01;
  mat3 ev{}, stress;
  for (int n = 1; n <= 400; n++) {
    mat3 ev_old = ev;
    material(du_dX, ev_old, stress, ev);
    EXPECT_NEAR(stress(0, 1), 0.01 * (mu + mu_v * std::pow(1.0 + dt / tau, -n)), 1.0e-15);
  }
  EXPECT_NEAR(stress(0, 1), 0.01 * mu, 1.0e-10);

  // and a dilatation never meets the branch
  mat3 dilatation = 0.01 * Identity<3>();
  material(dilatation, mat3{}, stress, ev);
  EXPECT_NEAR(norm(stress - LinearElasticModel{lambda, mu}(dilatation)), 0.0, 1.0e-15);
}

// central differences of a stress against a tangent, componentwise, relative to the tangent's size
template < typename stress_t >
void check_tangent(stress_t stress, mat3 du_dX, const mat< 3, 3, mat3 > & dP_dF, double tol) {
  double eps = 1.0e-6, scale = 0.0;
  for (int i = 0; i < 3; i++) { for (int j = 0; j < 3; j++) { scale = std::max(scale, norm(dP_dF[i][j])); } }
  for (int k = 0; k < 3; k++) {
    for (int l = 0; l < 3; l++) {
      mat3 dP_fd = (stress(du_dX + eps * e<3>(k, l)) - stress(du_dX - eps * e<3>(k, l))) / (2 * eps);
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          EXPECT_NEAR(dP_fd(i, j), dP_dF[i][j][k][l], tol * scale) << "dP_" << i << j << " / dF_" << k << l;
        }
      }
    }
  }
}

TEST(J2Plasticity, tangent_elastic_and_plastic) {
  J2PlasticityModel model{2.0, 1.0, 0.05, 0.3};
  mat3 ep_old = 0.01 * mat3(sym3(femto::random_mat<3,3>()));
  ep_old = ep_old - (tr(ep_old) / 3.0) * Identity<3>();
  double alpha_old = 0.02;
  for (double scale : {0.001, 0.1}) {
    mat3 du_dX = scale * femto::random_mat<3,3>();
    bool plastic = model.radial_return(du_dX, ep_old, alpha_old).dgamma > 0.0;
    EXPECT_EQ(plastic, scale > 0.01) << "scale " << scale;
    auto stress = [&](mat3 H) { mat3 sigma, ep; double alpha; model(H, ep_old, alpha_old, sigma, ep, alpha); return sigma; };
    check_tangent(stress, du_dX, model.jac(du_dX, ep_old, alpha_old), 1.0e-8);
  }
}

TEST(J2Plasticity, returns_to_the_hardened_yield_surface) {
  double sigma_y = 0.05, H = 0.3;
  J2PlasticityModel model{2.0, 1.0, sigma_y, H};
  mat3 du_dX = 0.1 * femto::random_mat<3,3>(), sigma, ep;
  double alpha;
  model(du_dX, mat3{}, 0.0, sigma, ep, alpha);
  mat3 s = sigma - (tr(sigma) / 3.0) * Identity<3>();
  EXPECT_GT(alpha, 0.0);
  EXPECT_NEAR(norm(s), std::sqrt(2.0 / 3.0) * (sigma_y + H * alpha), 1.0e-14);
  // the plastic flow is deviatoric, along s, and alpha is its equivalent measure
  EXPECT_NEAR(tr(ep), 0.0, 1.0e-15);
  EXPECT_NEAR(norm(ep / norm(ep) - s / norm(s)), 0.0, 1.0e-12);
  EXPECT_NEAR(alpha, std::sqrt(2.0 / 3.0) * norm(ep), 1.0e-15);
}

template < typename kinematics >
void check_corotational_tangent() {
  CorotationalModel< kinematics > model{2.0, 1.0};
  mat3 Q = RotationMatrix(vec3{0.3, -0.5, 0.4});
  mat3 du_dX_old = dot(Q, Identity<3>() + 0.05 * femto::random_mat<3,3>()) - Identity<3>();
  mat3 step = dot(mat3(RotationMatrix(vec3{0.02, 0.01, -0.01})), Identity<3>() + 0.003 * femto::random_mat<3,3>());
  mat3 du_dX = dot(step, Identity<3>() + du_dX_old) - Identity<3>();
  mat3 sigma_old = 0.1 * mat3(sym3(femto::random_mat<3,3>()));
  auto stress = [&](mat3 H) { mat3 P, sigma; model(H, du_dX_old, sigma_old, P, sigma); return P; };
  check_tangent(stress, du_dX, model.jac(du_dX, du_dX_old, sigma_old), 1.0e-7);
}

TEST(Corotational, hughes_winget_tangent) { check_corotational_tangent< HughesWinget >(); }
TEST(Corotational, rashid_tangent) { check_corotational_tangent< RashidNewton >(); }

// jvp() is jac() applied to a direction
template < typename jvp_t >
void check_jvp(jvp_t jvp, const mat< 3, 3, mat3 > & dP_dF) {
  mat3 ddu_dX = femto::random_mat<3,3>(), expected{};
  for (int i = 0; i < 3; i++) { for (int j = 0; j < 3; j++) { expected(i, j) = ddot(dP_dF[i][j], ddu_dX); } }
  EXPECT_NEAR(norm(jvp(ddu_dX) - expected), 0.0, 1.0e-13 * norm(expected));
}

TEST(J2Plasticity, jvp_is_jac) {
  J2PlasticityModel model{2.0, 1.0, 0.05, 0.3};
  for (double scale : {0.001, 0.1}) {
    mat3 du_dX = scale * femto::random_mat<3,3>();
    check_jvp([&](mat3 d) { return model.jvp(du_dX, mat3{}, 0.0, d); }, model.jac(du_dX, mat3{}, 0.0));
  }
}

template < typename kinematics >
void check_corotational_jvp() {
  CorotationalModel< kinematics > model{2.0, 1.0};
  mat3 du_dX_old = 0.05 * femto::random_mat<3,3>(), sigma_old = 0.1 * mat3(sym3(femto::random_mat<3,3>()));
  mat3 du_dX = du_dX_old + 0.003 * femto::random_mat<3,3>();
  check_jvp([&](mat3 d) { return model.jvp(du_dX, du_dX_old, sigma_old, d); }, model.jac(du_dX, du_dX_old, sigma_old));
}

TEST(Corotational, hughes_winget_jvp_is_jac) { check_corotational_jvp< HughesWinget >(); }
TEST(Corotational, rashid_jvp_is_jac) { check_corotational_jvp< RashidNewton >(); }

TEST(Dev, is_traceless_and_removes_only_the_mean) {
  mat3 A = femto::random_mat<3,3>();
  mat3 D = dev(A);
  EXPECT_NEAR(tr(D), 0.0, 1.0e-15);
  EXPECT_NEAR(norm(A - D - (tr(A) / 3.0) * Identity<3>()), 0.0, 1.0e-15);
  mat2 B = femto::random_mat<2,2>();
  EXPECT_NEAR(tr(dev(B)), 0.0, 1.0e-15);
}

TEST(J2PlasticityFiniteStrain, elastic_then_returns_to_the_yield_surface) {
  double K = 2.0, mu = 1.0, sigma_y = 0.05;
  J2PlasticityFiniteStrainModel model{K, mu, sigma_y};
  J2PlasticityFiniteStrainModel::State state{Identity<3>(), Identity<3>(), 0.0};

  // a small shear stays elastic
  mat3 small{};
  small(0, 1) = 0.01;
  model(state, small);
  EXPECT_EQ(state.alpha, 0.0);
  EXPECT_NEAR(norm(state.F_old - (Identity<3>() + small)), 0.0, 1.0e-15);

  // a large one yields, and the deviatoric Kirchhoff stress lands on the hardened yield surface
  mat3 large{};
  large(0, 1) = 0.2;
  mat3 tau = model(state, large);
  EXPECT_GT(state.alpha, 0.0);
  EXPECT_NEAR(norm(dev(tau)), std::sqrt(2.0 / 3.0) * (sigma_y + K * state.alpha), 1.0e-14);
  EXPECT_NEAR(norm(state.F_old - (Identity<3>() + large)), 0.0, 1.0e-15);
}
