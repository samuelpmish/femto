#pragma once

#include "fm/types/matrix.hpp"

namespace femto {

using namespace fm;

template <typename material_model>
struct PlaneStrain {

  mat2 operator()(const mat2 & du_dX) const {
    return to_2x2(model(to_3x3(du_dX)));
  }

  mat2 jvp(const mat2 & du_dX, const mat2 & ddu_dX) const {
    return to_2x2(model.jvp(to_3x3(du_dX), to_3x3(ddu_dX)));
  }

  mat2 vjp(const mat2 & du_dX, const mat2 & dO_dP) const {
    return to_2x2(model.vjp(to_3x3(du_dX), to_3x3(dO_dP)));
  }

  mat<2,2,mat2> jac(const mat2 & du_dX) const {
    mat<2,2, mat2> output;
    mat3 e{};
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        e[i][j] = 1.0;
        output[i][j] = to_2x2(model.vjp(to_3x3(du_dX), e));
        e[i][j] = 0.0;
      }
    }
    return output;
  }

  material_model model;

};

template <typename T>
PlaneStrain(T) -> PlaneStrain<T>;

} // namespace femto