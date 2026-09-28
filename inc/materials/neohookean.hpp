#pragma once

#include "fm/types/matrix.hpp"

namespace femto {

struct NeoHookeanModel {

  static constexpr auto dim = 3;
  static constexpr auto I = Identity<dim>();

  // returns P (PK1 stress)
  auto operator()(mat3 du_dX) const {
    mat3 F = I + du_dX;
    double J = det(F);
    mat3 invFT = transpose(inv(F));
    return (lambda * log(J) - mu) * invFT + mu * F;
  }

  // returns dP
  auto jvp(mat3 du_dX, mat3 ddu_dX) const {
    mat3 F = I + du_dX;
    double J = det(F);
    mat3 invFT = transpose(inv(F));
    return lambda * ddot(invFT, ddu_dX) * invFT + (mu - lambda * log(J)) * dot(invFT, dot(transpose(ddu_dX), invFT)) + mu * ddu_dX;
  }

  // returns dO/dF
  auto vjp(mat3 du_dX, mat3 dO_dP) const {
    mat3 F = I + du_dX;
    double J = det(F);
    mat3 invFT = transpose(inv(F));
    return lambda * ddot(invFT, dO_dP) * invFT + (mu - lambda * log(J)) * dot(invFT, dot(transpose(dO_dP), invFT)) + mu * dO_dP;
  }

  // returns dP/dF
  auto jac(mat3 du_dX) const {
    mat3 F = I + du_dX;
    double logJ = log(det(F));
    mat3 invFT = transpose(inv(F));

    mat< 3,3, mat<3,3> > dP_dF;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
          for (int l = 0; l < 3; l++) {
            dP_dF[i][j][k][l] = lambda * invFT[i][j] * invFT[k][l] + (mu - lambda * logJ) * invFT[k][j] * invFT[i][l] + mu * (i == k) * (j == l);
          }
        }
      }
    }
    return dP_dF;
  }

  double lambda, mu;

};

} // namespace femto