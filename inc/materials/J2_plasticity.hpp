// SPDX-FileCopyrightText: Copyright (c) 2023-2026 Sam Mish
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>

#include "fm/types/matrix.hpp"

#include "linear_algebra/dev.hpp"
#include "materials/linear_elasticity.hpp"

namespace femto {

using namespace fm;

// small strain J2 plasticity with linear isotropic hardening, by the radial
// return of Simo & Hughes, Computational Inelasticity (box 3.1): the trial
// deviatoric stress s* = 2 mu (dev(e) - ep_old), and when it lies outside the
// yield surface |s| = sqrt(2/3) (sigma_y + H alpha), a return along its own
// direction n = s* / |s*| by dgamma = f / (2 mu + 2 H / 3), f the excess.  The
// state is the (deviatoric) plastic strain ep and the equivalent plastic
// strain alpha.
//
// Same convention as ScalarPML: operator() gives the stress and the new state
// from the old one, which it never modifies, so a Newton iterate can start
// again from it.  jac() is the consistent tangent (box 3.2).
struct J2PlasticityModel {

  double lambda, mu;    // elastic moduli
  double sigma_y, H;    // initial yield stress and hardening modulus

  // the trial state and how far the return goes
  struct Return {
    mat3 s_trial;
    double norm_s_trial;
    double dgamma;        // 0 for an elastic step
  };

  __host__ __device__ Return radial_return(const mat3 & du_dX, const mat3 & ep_old, double alpha_old) const {
    mat3 e = 0.5 * (du_dX + transpose(du_dX));
    mat3 s_trial = 2.0 * mu * (dev(e) - ep_old);
    double norm_s = norm(s_trial);
    double f = norm_s - sqrt(2.0 / 3.0) * (sigma_y + H * alpha_old);
    return {s_trial, norm_s, (f > 0.0) ? f / (2.0 * mu + 2.0 * H / 3.0) : 0.0};
  }

  __host__ __device__ void operator()(const mat3 & du_dX, const mat3 & ep_old, const double & alpha_old,
                                      mat3 & sigma, mat3 & ep, double & alpha) const {
    Return r = radial_return(du_dX, ep_old, alpha_old);
    mat3 n = (r.dgamma > 0.0) ? r.s_trial / r.norm_s_trial : mat3{};
    double kappa = lambda + 2.0 * mu / 3.0;
    sigma = kappa * tr(du_dX) * Identity<3>() + r.s_trial - 2.0 * mu * r.dgamma * n;
    ep = ep_old + r.dgamma * n;
    alpha = alpha_old + sqrt(2.0 / 3.0) * r.dgamma;
  }

  // kappa 1 x 1 + 2 mu theta (I - 1 x 1 / 3) - 2 mu theta_bar n x n, with
  // theta = 1 - 2 mu dgamma / |s*| and theta_bar = 1 / (1 + H / (3 mu)) - (1 - theta):
  // linear elasticity with shear modulus mu theta, less the n x n part
  __host__ __device__ mat< 3, 3, mat3 > jac(const mat3 & du_dX, const mat3 & ep_old, const double & alpha_old) const {
    Return r = radial_return(du_dX, ep_old, alpha_old);
    if (r.dgamma == 0.0) { return LinearElasticModel{lambda, mu}.jac(du_dX); }
    double theta = 1.0 - 2.0 * mu * r.dgamma / r.norm_s_trial;
    double theta_bar = 1.0 / (1.0 + H / (3.0 * mu)) - (1.0 - theta);
    double kappa = lambda + 2.0 * mu / 3.0;
    mat< 3, 3, mat3 > C = LinearElasticModel{kappa - 2.0 * mu * theta / 3.0, mu * theta}.jac(du_dX);
    mat3 n = r.s_trial / r.norm_s_trial;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) { C[i][j] = C[i][j] - (2.0 * mu * theta_bar * n(i, j)) * n; }
    }
    return C;
  }

  // the same tangent applied to a direction ddu_dX
  __host__ __device__ mat3 jvp(const mat3 & du_dX, const mat3 & ep_old, const double & alpha_old, const mat3 & ddu_dX) const {
    Return r = radial_return(du_dX, ep_old, alpha_old);
    if (r.dgamma == 0.0) { return LinearElasticModel{lambda, mu}(ddu_dX); }
    double theta = 1.0 - 2.0 * mu * r.dgamma / r.norm_s_trial;
    double theta_bar = 1.0 / (1.0 + H / (3.0 * mu)) - (1.0 - theta);
    double kappa = lambda + 2.0 * mu / 3.0;
    mat3 n = r.s_trial / r.norm_s_trial;
    return LinearElasticModel{kappa - 2.0 * mu * theta / 3.0, mu * theta}(ddu_dX) - (2.0 * mu * theta_bar * ddot(n, ddu_dX)) * n;
  }

};

// finite strain J2 plasticity (Simo's multiplicative split, with the elastic
// left Cauchy-Green tensor be_bar's isochoric part as the state), returning
// the Kirchhoff stress tau.  It updates its state in place.  Note K is both
// the bulk modulus (the pressure) and the hardening modulus.
struct J2PlasticityFiniteStrainModel {

  struct State {
    mat3 F_old;
    mat3 be_bar;
    double alpha;
  };

  __host__ __device__ mat3 operator()(State & state, mat3 du_dx) const {

    mat3 I = Identity<3>();

    mat3 F_new = I + du_dx;

    // compute elastic predictor
    mat3 fbar_new = dot(F_new, inv(state.F_old));
    fbar_new /= cbrt(det(fbar_new));
    mat3 be_bar_new = dot(fbar_new, dot(state.be_bar, transpose(fbar_new)));
    double Ie_bar_new = tr(be_bar_new) / 3.0;
    mat3 s_new = mu * dev(be_bar_new);

    // check for plastic loading
    double y_new = norm(s_new) - sqrt(2.0 / 3.0) * (K * state.alpha + sigma_y);

    // return mapping algorithm
    if (y_new > 0) {
      double mu_bar = mu / Ie_bar_new;
      double delta_gamma = (3 * y_new) / (2 * K + 6 * mu_bar);
      mat3 n = s_new / norm(s_new);
      s_new = s_new - 2 * mu_bar * delta_gamma * n;
      state.alpha += sqrt(2.0 / 3.0) * delta_gamma;
    }

    state.F_old = F_new;
    state.be_bar = (s_new / mu) + Ie_bar_new * I;

    double J_new = det(F_new);
    double p_new = (0.5 * K) * (J_new - (1.0 / J_new));
    return (J_new * p_new) * I + s_new; // tau

  }

  double K;
  double mu;
  double sigma_y;
};

} // namespace femto
