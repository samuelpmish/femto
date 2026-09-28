#pragma once

#include "fm/types/matrix.hpp"

namespace femto {

using namespace fm;

// isotropic linear elasticity, with the same interface as NeoHookeanModel
// (the stress is linear, so its derivatives are the model itself)
struct LinearElasticModel {
  auto operator()(mat3 du_dx) const {
    mat3 I = Identity<3>();
    return lambda * tr(du_dx) * I + mu * (du_dx + transpose(du_dx));
  }

  auto jvp(mat3 /*du_dx*/, mat3 ddu_dx) const { return (*this)(ddu_dx); }

  // the stress operator is self-adjoint
  auto vjp(mat3 /*du_dx*/, mat3 dO_dP) const { return (*this)(dO_dP); }

  auto jac(mat3 /*du_dx*/) const {
    mat< 3, 3, mat<3, 3> > C;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
          for (int l = 0; l < 3; l++) {
            C[i][j][k][l] = lambda * (i == j) * (k == l) + mu * ((i == k) * (j == l) + (i == l) * (j == k));
          }
        }
      }
    }
    return C;
  }

  double lambda;
  double mu;
};

} // namespace femto
