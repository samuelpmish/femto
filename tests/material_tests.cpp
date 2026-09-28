#include <gtest/gtest.h>

#include "common.hpp"

#include "misc/timer.hpp"
#include "materials/plane_strain.hpp"
#include "materials/neohookean.hpp"

#include <iostream>

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