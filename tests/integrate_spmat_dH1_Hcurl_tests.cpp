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

using namespace fm;
using namespace femto;

////////////////////////////////////////////////////////////////////////////////

double sigma = 3.0;

template < uint32_t dim >
vec<dim> material(const vec<dim> & E) {
  return (sigma + E[0]) * E;
}

template < uint32_t dim >
mat<dim,dim> material_jac(const vec<dim> & E) {
  return outer(E, Identity<dim>()[0]) + (sigma + E[0]) * Identity<dim>();
}

template < uint32_t dim >
vec<dim> qfunction(const vec<dim> & E_xi, const mat<dim,dim> & dX_dxi) {
  mat<dim,dim> dxi_dX = inv(dX_dxi);
  vec<dim> E = dot(E_xi, dxi_dX);
  vec<dim> J = material(E);
  return dot(J, transpose(dxi_dX)) * det(dX_dxi);
}

template < uint32_t dim >
mat<dim,dim> qfunction_jac(const vec<dim> & E_xi, const mat<dim,dim> & dX_dxi) {
  mat<dim,dim> dxi_dX = inv(dX_dxi);
  vec<dim> E = dot(E_xi, dxi_dX);
  return dot(dxi_dX, dot(material_jac(E), transpose(dxi_dX))) * det(dX_dxi);
}

template < uint32_t dim >
vec<dim> qfunction_jvp(const vec<dim> & E_xi, const vec<dim> & dE_xi, const mat<dim,dim> & dX_dxi) {
  return dot(qfunction_jac(E_xi, dX_dxi), dE_xi);
}

////////////////////////////////////////////////////////////////////////////////

template < uint32_t dim >
void verify_dH1_Hcurl_derivatives() {
  double eps = 1.0e-5;
  vec<dim>     E_xi = femto::random_vec<dim>();
  vec<dim>     dE_xi = femto::random_vec<dim>();
  mat<dim,dim> dX_dxi = femto::random_mat<dim,dim>() + Identity<dim>();

  vec<dim> dJ0 = (qfunction(E_xi + eps * dE_xi, dX_dxi) -
                  qfunction(E_xi - eps * dE_xi, dX_dxi)) / (2 * eps);
  vec<dim> dJ1 = dot(qfunction_jac(E_xi, dX_dxi), dE_xi);

  for (int i = 0; i < dim; i++) {
    EXPECT_NEAR(dJ0[i], dJ1[i], 1.0e-10);
  }
}

////////////////////////////////////////////////////////////////////////////////

template < uint32_t dim >
void dH1_Hcurl_test(std::string filename, double tolerance) {

  double epsilon = 1.0e-6;

  using vecd = vec<dim>;
  using matd = mat<dim,dim>;

  auto mesh = Mesh<>::load(FEMTO_MESH_DIR + filename);

  auto f = [](const vecd & X) {
    double sum = 0.0;
    for (uint32_t i = 0; i < dim; i++) {
      sum += (i + 1) * X[i];
    }
    return sum;
  };

  for (int p1 = 1; p1 < 4; p1++) {

    Field u = create_field<Family::H1>(mesh, p1, 1);
    Field du = create_field<Family::H1>(mesh, p1, 1);

    auto nodes = nodes_for(u, mesh);
    auto u0 = forall(+f, nodes);

    du.data = femto::random(u.data.shape);

    BasisFunction psi(u);

    for (int p2 = 1; p2 < 4; p2++) {

      BasisFunction<Family::Hcurl> phi(p2, 1);

      for (int q = 1; q < 5; q++) {

        Domain domain(mesh, MeshQuadratureRule(q));

        nd::cpu_array<double, 3> dX_dxi_q = evaluate(grad(mesh.X), isoparametric(domain));

        auto r = [&](const Field<Family::H1> & u_) -> Residual<Family::Hcurl> {
          nd::cpu_array<double, 3> du_dxi_q = evaluate(grad(u_), isoparametric(domain));
          auto J_q = forall(qfunction<dim>, du_dxi_q, dX_dxi_q);
          return integrate(dot(J_q, phi), isoparametric(domain));
        };

        // finite difference approximation of jvp
        u = u0 + epsilon * du.data;
        Residual<Family::Hcurl> rp = r(u);

        u = u0 - epsilon * du.data;
        Residual<Family::Hcurl> rm = r(u);

        auto dr1 = (rp.data - rm.data) / (2 * epsilon);

        // "matrix-free" jvp
        u = u0;
        nd::cpu_array<double, 3> du_dxi_q = evaluate(grad(u), isoparametric(domain));
        nd::cpu_array<double, 3> ddu_dxi_q = evaluate(grad(du), isoparametric(domain));
        auto dJ_q = forall(qfunction_jvp<dim>, du_dxi_q, ddu_dxi_q, dX_dxi_q);
        Residual<Family::Hcurl> dr2_r = integrate(dot(dJ_q, phi), isoparametric(domain));
        auto dr2 = dr2_r.data;

        {
          SCOPED_TRACE("||dr1 - dr2|| p1 = " + std::to_string(p1) + ", p2 = " + std::to_string(p2) + ", q = " + std::to_string(q));
          EXPECT_NEAR(relative_error(dr1, dr2), 0.0, tolerance);
        }

        // "sparse matrix" jvp
        auto dJ_ddu_dxi_q = forall(qfunction_jac<dim>, du_dxi_q, dX_dxi_q);
        femto::sparse_matrix K = integrate(dot(grad(psi), dJ_ddu_dxi_q, phi), isoparametric(domain));

        Residual<Family::Hcurl> dr3(phi.space, mesh);

        dr3.v() = K(du.v());

        //std::string filename = "K" + std::to_string(p1) + std::to_string(p2) + std::to_string(q) + ".mtx";
        //export_matrix_market(K, filename);

        {
          SCOPED_TRACE("||dr1 - dr3|| p1 = " + std::to_string(p1) + ", p2 = " + std::to_string(p2) + ", q = " + std::to_string(q));
          EXPECT_NEAR(relative_error(dr1, dr3.data), 0.0, tolerance);
        }

        {
          SCOPED_TRACE("||dr2 - dr3|| p1 = " + std::to_string(p1) + ", p2 = " + std::to_string(p2) + ", q = " + std::to_string(q));
          EXPECT_NEAR(relative_error(dr2, dr3.data), 0.0, 1.0e-14);
        }

      }

    }

  }

}

// ----------------------------------------------------------------------------

TEST(IntegrateTest, verify_dH1_Hcurl_derivatives2D) { verify_dH1_Hcurl_derivatives<2>(); }
TEST(IntegrateTest, verify_dH1_Hcurl_derivatives3D) { verify_dH1_Hcurl_derivatives<3>(); }

// ----------------------------------------------------------------------------

TEST(IntegrateTest, dH1_Hcurl_edges) { dH1_Hcurl_test<1>("patch_test_edges.json", 1.0e-8); }

TEST(IntegrateTest, dH1_Hcurl_tris) { dH1_Hcurl_test<2>("patch_test_tris.json", 1.0e-8); }
TEST(IntegrateTest, dH1_Hcurl_quads) { dH1_Hcurl_test<2>("patch_test_quads.json", 1.0e-8); }
TEST(IntegrateTest, dH1_Hcurl_tris_and_quads) { dH1_Hcurl_test<2>("patch_test_tris_and_quads.json", 1.0e-8); }

TEST(IntegrateTest, dH1_Hcurl_tets) { dH1_Hcurl_test<3>("patch_test_tets.json", 1.0e-8); }
TEST(IntegrateTest, dH1_Hcurl_hexes) { dH1_Hcurl_test<3>("patch_test_hexes.json", 1.0e-8); }
TEST(IntegrateTest, dH1_Hcurl_tets_and_hexes) { dH1_Hcurl_test<3>("patch_test_tets_and_hexes.json", 1.0e-8); }
