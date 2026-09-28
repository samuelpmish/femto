#include "common.hpp"

#include <gtest/gtest.h>

#include <iomanip>
#include <iostream>
#include <functional>

#include "femto/mesh.hpp"
#include "femto/domain.hpp"

#include <gtest/gtest.h>

#include "forall.hpp"
#include "fm/types/vec.hpp"
#include "fm/types/matrix.hpp"
#include "containers/ndarray_conversions.hpp"

#include "materials/neohookean.hpp"

using namespace femto;

////////////////////////////////////////////////////////////////////////////////

double lambda = 3.0;
double mu = 3.0;

template < uint32_t dim >
mat<dim,dim> qfunction(const mat<dim, dim> & du_dX) {
  mat<dim,dim> I = Identity<dim>();
  mat<dim,dim> F = I + du_dX;
  mat<dim,dim> eps = 0.5 * (du_dX + transpose(du_dX) + dot(transpose(du_dX), du_dX));
  return lambda * tr(eps) * I + mu * eps;
}

template < uint32_t dim >
mat< dim,dim, mat<dim,dim> > qfunction_jac(const mat<dim,dim> & du_dX) {

  /*

    eps[i][j] = 0.5 * (F[m][i] * F[m][j] - delta[i][j])

    deps[i][j]_dF[k][l] = 0.5 * (dF[m][i]_dF[k][l] * F[m][j] + F[m][i] * dF[m][j]_dF[k][l])
      = 0.5 * ((m==k) * (i == l) * F[m][j] + F[m][i] * (m == k) * (j == l))
      = 0.5 * ((i == l) * F[k][j] + F[k][i] * (j == l))

    deps[m][m]_dF[k][l] = 0.5 * ((m == l) * F[k][m] + F[k][m] * (m == l)) = F[k][l]

    --------------------------------------------------------------------------------

    sigma[i][j] = lambda * eps[m][m] * delta[i][j] + mu * eps[i][j];

    dsigma[i][j]_dF[k][l] = d_dF[k][l](lambda * eps[m][m] * delta[i][j] + mu * eps[i][j])

      = lambda * deps[m][m]_dF[k][l] * delta[i][j] + mu * deps[i][j]_dF[k][l]

      = lambda * F[k][l] * delta[i][j] + 0.5 * mu * ((i == l) * F[k][j] + F[k][i] * (j == l))

  */

  mat< dim,dim, mat<dim,dim> > output{};

  mat<dim,dim> F = Identity<dim>() + du_dX;

  for (uint32_t i = 0; i < dim; i++) {
    for (uint32_t j = 0; j < dim; j++) {
      for (uint32_t k = 0; k < dim; k++) {
        for (uint32_t l = 0; l < dim; l++) {
          output[i][j][k][l] = lambda * F[k][l] * (i == j) + 0.5 * mu * ((i == l) * F[k][j] + F[k][i] * (j == l));
        }
      }
    }
  }

  return output;
}

template < uint32_t dim >
mat<dim,dim> qfunction_xi(const mat<dim,dim> & du_dxi, const mat<dim,dim> & dX_dxi) {
  mat<dim,dim> dxi_dX = inv(dX_dxi);
  mat<dim,dim> du_dX = dot(du_dxi, dxi_dX);
  mat<dim,dim> sigma = qfunction(du_dX);
  return dot(sigma, transpose(dxi_dX)) * det(dX_dxi);
}

template < uint32_t dim >
mat< dim,dim, mat<dim,dim> > qfunction_xi_jac(const mat<dim,dim> & du_dxi, const mat<dim,dim> & dX_dxi) {
  mat<dim,dim> dxi_dX = inv(dX_dxi);
  mat<dim,dim> du_dX = dot(du_dxi, dxi_dX);
  mat<dim,dim, mat<dim,dim>> dsigma_dF = qfunction_jac(du_dX);

  double wt = det(dX_dxi);
  mat<dim,dim, mat<dim,dim>> output{};
  for (uint32_t i = 0; i < dim; i++) {
    for (uint32_t j = 0; j < dim; j++) {
      for (uint32_t k = 0; k < dim; k++) {
        for (uint32_t l = 0; l < dim; l++) {
          double sum = 0.0;
          for (uint32_t m = 0; m < dim; m++) {
            for (uint32_t n = 0; n < dim; n++) {
              sum += dsigma_dF[i][m][k][n] * dxi_dX[j][m] * dxi_dX[l][n];
            }
          }
          output[i][j][k][l] = sum * wt;
        }
      }
    }
  }
  return output;
}

template < uint32_t dim >
mat<dim, dim> qfunction_xi_jvp(const mat<dim,dim> & du_dxi, const mat<dim,dim> & ddu_dxi, const mat<dim,dim> & dX_dxi) {
  return chain_rule(qfunction_xi_jac(du_dxi, dX_dxi), ddu_dxi);
}

template < uint32_t dim >
mat<dim, dim> qfunction_jvp(const mat<dim,dim> & du_dX, const mat<dim,dim> & ddu_dX) {
  return chain_rule(qfunction_jac(du_dX), ddu_dX);
}

////////////////////////////////////////////////////////////////////////////////

template < uint32_t dim >
void verify_dH1v_dH1v_derivatives() {
  double eps = 1.0e-5;
  mat<dim,dim> du_dxi = femto::random_mat<dim,dim>();
  mat<dim,dim> ddu_dxi = femto::random_mat<dim,dim>();
  mat<dim,dim> dX_dxi = femto::random_mat<dim,dim>() + Identity<dim>();

  mat<dim,dim> dsigma0 = (qfunction(du_dxi + eps * ddu_dxi) -
                          qfunction(du_dxi - eps * ddu_dxi)) / (2 * eps);
  mat<dim,dim> dsigma1 = chain_rule(qfunction_jac(du_dxi), ddu_dxi);
  mat<dim,dim> dsigma2 = qfunction_jvp(du_dxi, ddu_dxi);

  for (uint32_t i = 0; i < dim; i++) {
    for (uint32_t j = 0; j < dim; j++) {
      EXPECT_NEAR(dsigma0[i][j], dsigma1[i][j], 2.0e-9);
      EXPECT_NEAR(dsigma0[i][j], dsigma2[i][j], 2.0e-9);
    }
  }

  mat<dim,dim> dflux0 = (qfunction_xi(du_dxi + eps * ddu_dxi, dX_dxi) -
                         qfunction_xi(du_dxi - eps * ddu_dxi, dX_dxi)) / (2 * eps);
  mat<dim,dim> dflux1 = chain_rule(qfunction_xi_jac(du_dxi, dX_dxi), ddu_dxi);

  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      EXPECT_NEAR(dflux0[i][j], dflux1[i][j], 5.0e-10);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

template < uint32_t dim >
void dH1v_dH1v_test(std::string filename, double tolerance) {

  double epsilon = 1.0e-6;

  using vecd = vec<dim>;
  using matd = mat<dim,dim>;

  auto mesh = Mesh<>::load(FEMTO_MESH_DIR + filename);

  for (int p1 = 1; p1 < 4; p1++) {

    Field u = create_field<Family::H1>(mesh, p1, dim);
    Field du = create_field<Family::H1>(mesh, p1, dim);

    auto nodes = nodes_for(u, mesh);

    auto u0 = forall(+[](const vec<dim> & X) {
      double sum = 0.0;
      for (int i = 0; i < dim; i++) { sum += (i + 1) * X[i] * X[i]; }

      vec<dim> output{};
      for (int i = 0; i < dim; i++) { output[i] = sum / (i + 1); }
      return output;
    }, nodes);

    du.data = femto::random(u.data.shape);

    BasisFunction<Family::H1> psi(p1, dim);

    for (int p2 = 1; p2 < 4; p2++) {

      BasisFunction<Family::H1> phi(p2, dim);

      for (int q = 1; q < 5; q++) {

        Domain domain(mesh, MeshQuadratureRule(q));

        nd::cpu_array<double, 3> dX_dxi_q = evaluate(grad(mesh.X), isoparametric(domain));

        auto r_xi = [&](const Field<Family::H1> & u_) -> Residual<Family::H1> {
          nd::cpu_array<double, 3> du_dxi_q = evaluate(grad(u_), isoparametric(domain));
          auto P_xi_q = forall(qfunction_xi<dim>, du_dxi_q, dX_dxi_q);
          return integrate(dot(P_xi_q, grad(phi)), isoparametric(domain));
        };

        auto r = [&](const Field<Family::H1> & u_) -> Residual<Family::H1> {
          nd::cpu_array<double, 3> du_dX_q = evaluate(grad(u_), domain);
          auto P_q = forall(qfunction<dim>, du_dX_q);
          return integrate(dot(P_q, grad(phi)), domain);
        };

        // finite difference approximation of jvp
        u = u0 + epsilon * du.data;
        Residual<Family::H1> rp1 = r_xi(u);
        Residual<Family::H1> rp2 = r(u);

        u = u0 - epsilon * du.data;
        Residual<Family::H1> rm1 = r_xi(u);
        Residual<Family::H1> rm2 = r(u);

        auto dr1 = (rp1.data - rm1.data) / (2 * epsilon);
        auto dr2 = (rp2.data - rm2.data) / (2 * epsilon);

        {
          SCOPED_TRACE("||dr1 - dr2|| p1 = " + std::to_string(p1) + ", p2 = " + std::to_string(p2) + ", q = " + std::to_string(q));
          EXPECT_NEAR(relative_error(dr1, dr2), 0.0, tolerance);
        }

        // "matrix-free" jvp
        u = u0;
        nd::cpu_array<double, 3> du_dxi_q = evaluate(grad(u), isoparametric(domain));
        nd::cpu_array<double, 3> ddu_dxi_q = evaluate(grad(du), isoparametric(domain));
        auto df1_q = forall(qfunction_xi_jvp<dim>, du_dxi_q, ddu_dxi_q, dX_dxi_q);
        Residual<Family::H1> dr3_r = integrate(dot(df1_q, grad(phi)), isoparametric(domain));
        auto dr3 = dr3_r.data;

        nd::cpu_array<double, 3> du_dX_q = evaluate(grad(u), domain);
        nd::cpu_array<double, 3> ddu_dX_q = evaluate(grad(du), domain);
        auto df2_q = forall(qfunction_jvp<dim>, du_dX_q, ddu_dX_q);
        Residual<Family::H1> dr4_r = integrate(dot(df2_q, grad(phi)), domain);
        auto dr4 = dr4_r.data;

        {
          SCOPED_TRACE("||dr3 - dr4|| p1 = " + std::to_string(p1) + ", p2 = " + std::to_string(p2) + ", q = " + std::to_string(q));
          EXPECT_NEAR(relative_error(dr3, dr4), 0.0, 1.0e-14);
        }

        {
          SCOPED_TRACE("||dr1 - dr3|| p1 = " + std::to_string(p1) + ", p2 = " + std::to_string(p2) + ", q = " + std::to_string(q));
          EXPECT_NEAR(relative_error(dr1, dr3), 0.0, tolerance);
        }

        // "sparse matrix" jvp
        auto df_ddudxi_q = forall(qfunction_xi_jac<dim>, du_dxi_q, dX_dxi_q);
        femto::sparse_matrix K_xi = integrate(dot(grad(psi), df_ddudxi_q, grad(phi)), isoparametric(domain));

        Residual<Family::H1> dr5(FunctionSpace(u.family, p2, u.data.shape[1]), mesh);
        dr5.v() = K_xi(du.v());

        {
          SCOPED_TRACE("||dr1 - dr5|| p1 = " + std::to_string(p1) + ", p2 = " + std::to_string(p2) + ", q = " + std::to_string(q));
          EXPECT_NEAR(relative_error(dr1, dr5.data), 0.0, tolerance);
        }

        auto dsigma_ddudX_q = forall(qfunction_jac<dim>, du_dX_q);
        femto::sparse_matrix K = integrate(dot(grad(psi), dsigma_ddudX_q, grad(phi)), domain);

        Residual<Family::H1> dr6(FunctionSpace(u.family, p2, u.data.shape[1]), mesh);
        dr6.v() = K(du.v());

        {
          SCOPED_TRACE("||dr5 - dr6|| p1 = " + std::to_string(p1) + ", p2 = " + std::to_string(p2) + ", q = " + std::to_string(q));
          EXPECT_NEAR(relative_error(dr5.data, dr6.data), 0.0, 1.0e-14);
        }

        if (p1 == p2) {
          auto D1 = diagonal(K_xi);
          auto D2 = integrate(diagonal(dot(grad(psi), df_ddudxi_q, grad(phi))), isoparametric(domain));
          auto D3 = integrate(diagonal(dot(grad(psi), dsigma_ddudX_q, grad(phi))), domain);
          EXPECT_NEAR(relative_error(D1, nd::flatten(D2)), 0.0, 1.0e-14);
          EXPECT_NEAR(relative_error(D1, nd::flatten(D3)), 0.0, 1.0e-14);
        }

      }

    }

  }

}

// ----------------------------------------------------------------------------

TEST(IntegrateTest, verify_dH1v_dH1v_derivatives2D) { verify_dH1v_dH1v_derivatives<2>(); }
TEST(IntegrateTest, verify_dH1v_dH1v_derivatives3D) { verify_dH1v_dH1v_derivatives<3>(); }

// ----------------------------------------------------------------------------

TEST(IntegrateTest, dH1v_dH1v_edges) { dH1v_dH1v_test<1>("patch_test_edges.json", 2.0e-8); }

TEST(IntegrateTest, dH1v_dH1v_tris) { dH1v_dH1v_test<2>("patch_test_tris.json", 2.0e-8); }
TEST(IntegrateTest, dH1v_dH1v_quads) { dH1v_dH1v_test<2>("patch_test_quads.json", 2.0e-8); }
TEST(IntegrateTest, dH1v_dH1v_tris_and_quads) { dH1v_dH1v_test<2>("patch_test_tris_and_quads.json", 2.0e-8); }

TEST(IntegrateTest, dH1v_dH1v_tets) { dH1v_dH1v_test<3>("patch_test_tets.json", 2.0e-8); }
TEST(IntegrateTest, dH1v_dH1v_hexes) { dH1v_dH1v_test<3>("patch_test_hexes.json", 2.0e-8); }
TEST(IntegrateTest, dH1v_dH1v_tets_and_hexes) { dH1v_dH1v_test<3>("patch_test_tets_and_hexes.json", 2.0e-8); }

//TEST(IntegrateTest, big_dH1v_dH1v_tets) { dH1v_dH1v_test<3>("octane_fine.msh", 1.5e-6); }