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
void verify_Hcurl_Hcurl_derivatives() {
  double eps = 1.0e-5;

  vec<dim> E = femto::random_vec<dim>();
  vec<dim> dE = femto::random_vec<dim>();

  vec<dim> dJ0 = (material(E + eps * dE) - material(E - eps * dE)) / (2 * eps);
  vec<dim> dJ1 = dot(material_jac(E), dE);

  for (int i = 0; i < dim; i++) {
    EXPECT_NEAR(dJ0[i], dJ1[i], 1.0e-10);
  }

////////////////////////////////////////////////////////////////////////////////

  vec<dim>     E_xi = femto::random_vec<dim>();
  vec<dim>     dE_xi = femto::random_vec<dim>();
  mat<dim,dim> dX_dxi = femto::random_mat<dim,dim>() + Identity<dim>();

  vec<dim> dJ_xi_0 = (qfunction(E_xi + eps * dE_xi, dX_dxi) -
                      qfunction(E_xi - eps * dE_xi, dX_dxi)) / (2 * eps);
  vec<dim> dJ_xi_1 = dot(qfunction_jac(E_xi, dX_dxi), dE_xi);

  for (int i = 0; i < dim; i++) {
    EXPECT_NEAR(dJ_xi_0[i], dJ_xi_1[i], 3.0e-9);
  }
}

////////////////////////////////////////////////////////////////////////////////

template < uint32_t dim >
void Hcurl_Hcurl_test(std::string filename, double tolerance) {

  threadpool::set_num_threads(1);

  double epsilon = 1.0e-6;

  using vecd = vec<dim>;
  using matd = mat<dim,dim>;

  auto mesh = Mesh<>::load(FEMTO_MESH_DIR + filename);

  auto f = [](const vecd & X, const vecd & n) { return dot(X, n); };

  for (int p1 = 1; p1 < 4; p1++) {

    Field E = create_field<Family::Hcurl>(mesh, p1, 1);
    Field dE = create_field<Family::Hcurl>(mesh, p1, 1);

    auto nodes = nodes_for(E, mesh);
    auto directions = directions_for(E, mesh);
    auto E0 = forall(+f, nodes, directions);

    dE.data = femto::random(E.data.shape);

    BasisFunction<Family::Hcurl> psi(p1, 1);

    for (int p2 = 1; p2 < 4; p2++) {

      BasisFunction<Family::Hcurl> phi(p2, 1);

      for (int q = 1; q < 5; q++) {

        Domain domain(mesh, MeshQuadratureRule(q));

        nd::cpu_array<double, 3> dX_dxi_q = evaluate(grad(mesh.X), isoparametric(domain));

        auto r = [&](const Field<Family::Hcurl> & u_) -> Residual<Family::Hcurl> {
          nd::cpu_array<double, 3> E_xi_q = evaluate(u_, isoparametric(domain));
          auto J_xi_q = forall(qfunction<dim>, E_xi_q, dX_dxi_q);
          return integrate(dot(J_xi_q, phi), isoparametric(domain));
        };

        auto rSpatial = [&](const Field<Family::Hcurl> & u_) -> Residual<Family::Hcurl> {
          nd::cpu_array<double, 3> E_q = evaluate(u_, domain);
          auto J_q = forall(material<dim>, E_q);
          return integrate(dot(J_q, phi), domain);
        };

        E = E0;

        Residual<Family::Hcurl> r1 = r(E);
        Residual<Family::Hcurl> r2 = rSpatial(E);

        {
          SCOPED_TRACE("||r1 - r2|| p1 = " + std::to_string(p1) + ", p2 = " + std::to_string(p2) + ", q = " + std::to_string(q));
          EXPECT_NEAR(relative_error(r1.data, r2.data), 0.0, tolerance);
        }

        // finite difference approximation of jvp
        E = E0 + epsilon * dE.data;
        Residual<Family::Hcurl> rp = r(E);

        E = E0 - epsilon * dE.data;
        Residual<Family::Hcurl> rm = r(E);

        auto dr1 = (rp.data - rm.data) / (2 * epsilon);

        // "matrix-free" jvp
        E = E0;
        nd::cpu_array<double, 3> E_xi_q = evaluate(E, isoparametric(domain));
        nd::cpu_array<double, 3> dE_xi_q = evaluate(dE, isoparametric(domain));
        auto dJ_q = forall(qfunction_jvp<dim>, E_xi_q, dE_xi_q, dX_dxi_q);
        Residual<Family::Hcurl> dr2_r = integrate(dot(dJ_q, phi), isoparametric(domain));
        auto dr2 = dr2_r.data;

        {
          SCOPED_TRACE("||dr1 - dr2|| p1 = " + std::to_string(p1) + ", p2 = " + std::to_string(p2) + ", q = " + std::to_string(q));
          EXPECT_NEAR(relative_error(dr1, dr2), 0.0, tolerance);
        }

        // "sparse matrix" jvp
        auto dJ_dExi_q = forall(qfunction_jac<dim>, E_xi_q, dX_dxi_q);
        femto::sparse_matrix K = integrate(dot(psi, dJ_dExi_q, phi), isoparametric(domain));

        Residual<Family::Hcurl> dr3(FunctionSpace(E.family, p2, E.data.shape[1]), mesh);
        dr3.v() = K(dE.v());

        {
          SCOPED_TRACE("||dr1 - dr3|| p1 = " + std::to_string(p1) + ", p2 = " + std::to_string(p2) + ", q = " + std::to_string(q));
          EXPECT_NEAR(relative_error(dr1, dr3.data), 0.0, tolerance);
        }

        {
          SCOPED_TRACE("||dr2 - dr3|| p1 = " + std::to_string(p1) + ", p2 = " + std::to_string(p2) + ", q = " + std::to_string(q));
          EXPECT_NEAR(relative_error(dr2, dr3.data), 0.0, 1.0e-14);
        }

        nd::cpu_array<double, 3> E_q = evaluate(E, domain);
        auto dJ_dE_q = forall(material_jac<dim>, E_q);
        femto::sparse_matrix K2 = integrate(dot(psi, dJ_dE_q, phi), domain);

        Residual<Family::Hcurl> dr4(FunctionSpace(E.family, p2, E.data.shape[1]), mesh);
        dr4.v() = K2(dE.v());

        {
          SCOPED_TRACE("||dr2 - dr4|| p1 = " + std::to_string(p1) + ", p2 = " + std::to_string(p2) + ", q = " + std::to_string(q));
          EXPECT_NEAR(relative_error(dr2, dr4.data), 0.0, 1.0e-14);
        }

        {
          SCOPED_TRACE("||dr3 - dr4|| p1 = " + std::to_string(p1) + ", p2 = " + std::to_string(p2) + ", q = " + std::to_string(q));
          EXPECT_NEAR(relative_error(dr3.data, dr4.data), 0.0, 1.0e-14);
        }

        if (p1 == p2) {
          auto D1 = diagonal(K);
          auto D2 = integrate(diagonal(dot(psi, dJ_dExi_q, phi)), isoparametric(domain));
          auto D3 = integrate(diagonal(dot(psi, dJ_dE_q, phi)), domain);
          //print(D1);
          //print(nd::flatten(D2));
          //print(nd::flatten(D3));
          SCOPED_TRACE("||D1 - D2|| p1 = " + std::to_string(p1) + ", p2 = " + std::to_string(p2) + ", q = " + std::to_string(q));
          EXPECT_NEAR(relative_error(D1, nd::flatten(D2)), 0.0, 1.0e-14);

          SCOPED_TRACE("||D1 - D3|| p1 = " + std::to_string(p1) + ", p2 = " + std::to_string(p2) + ", q = " + std::to_string(q));
          EXPECT_NEAR(relative_error(D1, nd::flatten(D3)), 0.0, 1.0e-14);
        }

        //std::string filename = "K1_" + std::to_string(p1) + std::to_string(p2) + std::to_string(q) + ".mtx";
        //export_matrix_market(K, filename);

        //filename = "K2_" + std::to_string(p1) + std::to_string(p2) + std::to_string(q) + ".mtx";
        //export_matrix_market(K, filename);

        //dE.data = dE.data * 0;
        //for (int k = 0; k < E.data.shape[0]; k++) {
        //  dE.data[k] = 1.0;
        //  std::cout << "dE" << std::endl;
        //  print(dE.data);
        //  nd::cpu_array<double, 3> dE_xi_q = evaluate(dE, isoparametric(domain));
        //  std::cout << "dE_xi_q" << std::endl;
        //  print(dE_xi_q);
        //  auto dJ_q = forall(qfunction_jvp<dim>, E_xi_q, dE_xi_q, dX_dxi_q);
        //  std::cout << "dJ_q" << std::endl;
        //  print(dJ_q);
        //  std::cout << "dr" << std::endl;
        //  print(integrate(dot(dJ_q, phi), isoparametric(domain)).data);
        //  dE.data[k] = 0.0;
        //}

      }

    }

  }

}

// ----------------------------------------------------------------------------

TEST(IntegrateTest, verify_Hcurl_Hcurl_derivatives2D) { verify_Hcurl_Hcurl_derivatives<2>(); }
TEST(IntegrateTest, verify_Hcurl_Hcurl_derivatives3D) { verify_Hcurl_Hcurl_derivatives<3>(); }

// ----------------------------------------------------------------------------

TEST(IntegrateTest, Hcurl_Hcurl_edges) { Hcurl_Hcurl_test<1>("patch_test_edges.json", 1.0e-8); }

TEST(IntegrateTest, Hcurl_Hcurl_tris) { Hcurl_Hcurl_test<2>("patch_test_tris.json", 1.0e-8); }
TEST(IntegrateTest, Hcurl_Hcurl_quads) { Hcurl_Hcurl_test<2>("patch_test_quads.json", 1.0e-8); }
TEST(IntegrateTest, Hcurl_Hcurl_tris_and_quads) { Hcurl_Hcurl_test<2>("patch_test_tris_and_quads.json", 1.0e-8); }

TEST(IntegrateTest, Hcurl_Hcurl_one_tet) { Hcurl_Hcurl_test<3>("one_tet.json", 1.0e-8); }
TEST(IntegrateTest, Hcurl_Hcurl_tets) { Hcurl_Hcurl_test<3>("patch_test_tets.json", 1.0e-8); }
TEST(IntegrateTest, Hcurl_Hcurl_hexes) { Hcurl_Hcurl_test<3>("patch_test_hexes.json", 1.0e-8); }
TEST(IntegrateTest, Hcurl_Hcurl_tets_and_hexes) { Hcurl_Hcurl_test<3>("patch_test_tets_and_hexes.json", 1.0e-8); }

//TEST(IntegrateTest, Hcurl_Hcurl_big_tet_mesh) { Hcurl_Hcurl_test<3>("octane_fine.msh", 1.0e-8); }
