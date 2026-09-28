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
vec<dim> qfunction(const vec<dim> & E, const mat<dim,dim> & dX_dxi) {
  mat<dim,dim> dxi_dX = inv(dX_dxi);
  vec<dim> J = material(E);
  return dot(J, transpose(dxi_dX)) * det(dX_dxi);
}

template < uint32_t dim >
mat<dim,dim> qfunction_jac(const vec<dim> & E, const mat<dim,dim> & dX_dxi) {
  mat<dim,dim> dxi_dX = inv(dX_dxi);
  return dot(dxi_dX, material_jac(E)) * det(dX_dxi);
}

template < uint32_t dim >
vec<dim> qfunction_jvp(const vec<dim> & E, const vec<dim> & dE, const mat<dim,dim> & dX_dxi) {
  return dot(qfunction_jac(E, dX_dxi), dE);
}

////////////////////////////////////////////////////////////////////////////////

template < uint32_t dim >
void verify_H1v_Hcurl_derivatives() {
  double eps = 1.0e-5;
  vec<dim>     E = femto::random_vec<dim>();
  vec<dim>     dE = femto::random_vec<dim>();
  mat<dim,dim> dX_dxi = femto::random_mat<dim,dim>() + Identity<dim>();

  vec<dim> dJ0 = (qfunction(E + eps * dE, dX_dxi) -
                  qfunction(E - eps * dE, dX_dxi)) / (2 * eps);
  vec<dim> dJ1 = dot(qfunction_jac(E, dX_dxi), dE);

  for (uint32_t i = 0; i < dim; i++) {
    EXPECT_NEAR(dJ0[i], dJ1[i], 1.0e-10);
  }
}

////////////////////////////////////////////////////////////////////////////////

template < uint32_t dim >
void H1v_Hcurl_test(std::string filename, double tolerance) {

  double epsilon = 1.0e-6;

  using vecd = vec<dim>;
  using matd = mat<dim,dim>;

  auto mesh = Mesh<>::load(FEMTO_MESH_DIR + filename);

  auto f = [](const vecd & X) { return X; };

  for (int p1 = 1; p1 < 4; p1++) {

    Field E = create_field<Family::H1>(mesh, p1, dim);
    Field dE = create_field<Family::H1>(mesh, p1, dim);

    auto nodes = nodes_for(E, mesh);
    auto E0 = forall(+f, nodes);

    dE.data = femto::random(E.data.shape);

    BasisFunction psi(E);

    for (int p2 = 1; p2 < 4; p2++) {

      BasisFunction<Family::Hcurl> phi(p2, 1);

      for (int q = 1; q < 5; q++) {

        Domain domain(mesh, MeshQuadratureRule(q));

        nd::cpu_array<double, 3> dX_dxi_q = evaluate(grad(mesh.X), isoparametric(domain));

        auto r = [&](const Field<Family::H1> & E_) -> Residual<Family::Hcurl> {
          nd::cpu_array<double, 3> E_q = evaluate(E_, isoparametric(domain));
          auto J_q = forall(qfunction<dim>, E_q, dX_dxi_q);
          return integrate(dot(J_q, phi), isoparametric(domain));
        };

        // finite difference approximation of jvp
        E = E0 + epsilon * dE.data;
        Residual<Family::Hcurl> rp = r(E);

        E = E0 - epsilon * dE.data;
        Residual<Family::Hcurl> rm = r(E);

        auto dr1 = (rp.data - rm.data) / (2 * epsilon);

        // "matrix-free" jvp
        E = E0;
        nd::cpu_array<double, 3> E_q = evaluate(E, domain);
        nd::cpu_array<double, 3> dE_q = evaluate(dE, domain);
        auto dJ_q = forall(qfunction_jvp<dim>, E_q, dE_q, dX_dxi_q);
        Residual<Family::Hcurl> dr2_r = integrate(dot(dJ_q, phi), isoparametric(domain));
        auto dr2 = dr2_r.data;

        {
          SCOPED_TRACE("||dr1 - dr2|| p1 = " + std::to_string(p1) + ", p2 = " + std::to_string(p2) + ", q = " + std::to_string(q));
          EXPECT_NEAR(relative_error(dr1, dr2), 0.0, tolerance);
        }

        // "sparse matrix" jvp
        auto dJ_dE_q = forall(qfunction_jac<dim>, E_q, dX_dxi_q);
        femto::sparse_matrix K = integrate(dot(psi, dJ_dE_q, phi), isoparametric(domain));

        Residual<Family::Hcurl> dr3(phi.space, mesh);

        dr3.v() = K(dE.v());

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

TEST(IntegrateTest, verify_H1v_Hcurl_derivatives2D) { verify_H1v_Hcurl_derivatives<2>(); }
TEST(IntegrateTest, verify_H1v_Hcurl_derivatives3D) { verify_H1v_Hcurl_derivatives<3>(); }

// ----------------------------------------------------------------------------

TEST(IntegrateTest, H1v_Hcurl_edges) { H1v_Hcurl_test<1>("patch_test_edges.json", 1.0e-8); }

TEST(IntegrateTest, H1v_Hcurl_tris) { H1v_Hcurl_test<2>("patch_test_tris.json", 1.0e-8); }
TEST(IntegrateTest, H1v_Hcurl_quads) { H1v_Hcurl_test<2>("patch_test_quads.json", 1.0e-8); }
TEST(IntegrateTest, H1v_Hcurl_tris_and_quads) { H1v_Hcurl_test<2>("patch_test_tris_and_quads.json", 1.0e-8); }

TEST(IntegrateTest, H1v_Hcurl_tets) { H1v_Hcurl_test<3>("patch_test_tets.json", 1.0e-8); }
TEST(IntegrateTest, H1v_Hcurl_hexes) { H1v_Hcurl_test<3>("patch_test_hexes.json", 1.0e-8); }
TEST(IntegrateTest, H1v_Hcurl_tets_and_hexes) { H1v_Hcurl_test<3>("patch_test_tets_and_hexes.json", 1.0e-8); }
