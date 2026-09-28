#pragma once

#include <cmath>

#include "fm/types/matrix.hpp"

namespace femto {

using namespace fm;

// plane stress from a 3D material model: the out-of-plane stretch is the one
// that makes the out-of-plane stress vanish, found by Newton per evaluation,
// and the in-plane tangent is the 3D tangent condensed with that constraint
template <typename material_model>
struct PlaneStress {

  // the 3D displacement gradient: in-plane from du_dX, du3/dX3 solved for
  mat3 embed(const mat2 & du_dX) const {
    mat3 F = to_3x3(du_dX);
    mat3 e{};
    e[2][2] = 1.0;
    for (int it = 0; it < 20; it++) {
      double P33 = model(F)[2][2];
      double dP33 = model.jvp(F, e)[2][2];
      double step = P33 / dP33;
      F[2][2] -= step;
      if (std::fabs(step) < 1e-14) break;
    }
    return F;
  }

  mat2 operator()(const mat2 & du_dX) const {
    return to_2x2(model(embed(du_dX)));
  }

  // condensed tangent: dP_ij/dF_kl - (dP_ij/dF_33)(dP_33/dF_kl)/(dP_33/dF_33)
  mat<2,2,mat2> jac(const mat2 & du_dX) const {
    auto C = model.jac(embed(du_dX));
    mat<2,2, mat2> output;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        for (int k = 0; k < 2; k++) {
          for (int l = 0; l < 2; l++) {
            output[i][j][k][l] = C[i][j][k][l] - C[i][j][2][2] * C[2][2][k][l] / C[2][2][2][2];
          }
        }
      }
    }
    return output;
  }

  material_model model;

};

template <typename T>
PlaneStress(T) -> PlaneStress<T>;

} // namespace femto
