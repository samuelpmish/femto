#pragma once

#include "fm/types/matrix.hpp"

struct J2PlasticityModel {

  struct State {
    mat3 F_old;
    mat3 be_bar;
    double alpha;
  };

  auto operator()(State & state, mat3 du_dx) {

    mat3 I = Identity<3>();

    mat3 F_new = I + du_dx;

    // compute elastic predictor
    mat3 fbar_new = dot(F_new, inv(state.F_old));
    fbar_new /= cbrt(det(fbar_new));
    mat3 be_bar_new = dot(fbar_new, dot(state.be_bar, transpose(fbar_new)));
    double Ie_bar_new = tr(be_bar_new) / 3.0;
    mat3 s_new = mu * dev(be_bar_new);

    // check for plastic loading
    double y_new = norm(s_new) - sqrt(2.0 / 3.0) * (K * alpha + sigma_y);

    // return mapping algorithm
    if (y_new > 0) {
      double mu_bar = mu / Ie_bar_new;
      double delta_gamma = (3 * y_new) / (2 * K + 6 * mu_bar);
      mat3 n = s_new / norm(s_new);
      s_new -= 2 * mu_bar * delta_gamma * n;
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
}