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

double rho = 3.0;

double material_jac(const double & u) {
  return 2 * rho * u;
}

template < uint32_t dim >
double qfunction(const double & u, const mat<dim,dim> & dX_dxi) {
  return rho * u * u * det(dX_dxi);
}

template < uint32_t dim >
double qfunction_jac(const double & u, const mat<dim,dim> & dX_dxi) {
  return 2 * rho * u * det(dX_dxi);
}

template < uint32_t dim >
double qfunction_jvp(const double & u, const double & du, const mat<dim,dim> & dX_dxi) {
  return qfunction_jac(u, dX_dxi) * du;
}

////////////////////////////////////////////////////////////////////////////////

template < uint32_t dim >
void verify_H1_H1_derivatives() {
  double eps = 1.0e-5;
  double u = femto::random();
  double du = femto::random();
  mat<dim,dim> dX_dxi = femto::random_mat<dim,dim>() + Identity<dim>();

  double dsource0 = (qfunction(u + eps * du, dX_dxi) - qfunction(u - eps * du, dX_dxi)) / (2 * eps);
  double dsource1 = dot(qfunction_jac(u, dX_dxi), du);
  EXPECT_NEAR(dsource0, dsource1, 1.0e-10);
}

////////////////////////////////////////////////////////////////////////////////

template < uint32_t dim >
void H1_H1_test(std::string filename, double tolerance) {

  threadpool::set_num_threads(1);

  double epsilon = 1.0e-6;

  using vecd = vec<dim>;
  using matd = mat<dim,dim>;

  auto mesh = Mesh<>::load(FEMTO_MESH_DIR + filename);

  for (int p1 = 1; p1 < 4; p1++) {

    Field u = create_field<Family::H1>(mesh, p1, 1);
    Field du = create_field<Family::H1>(mesh, p1, 1);

    auto nodes = nodes_for(u, mesh);

    auto u0 = forall(+[](const vec<dim> & X) {
      double sum = 0.0;
      for (int i = 0; i < dim; i++) { sum += (i + 1) * X[i]; }
      return sum;
    }, nodes);

    du.data = femto::random(u.data.shape);

    BasisFunction<Family::H1> psi(p1, 1);

    for (int p2 = 1; p2 < 4; p2++) {

      BasisFunction<Family::H1> phi(p2, 1);

      for (int q = 1; q < 5; q++) {

        Domain domain(mesh, MeshQuadratureRule(q));

        nd::cpu_array<double, 3> dX_dxi_q = evaluate(grad(mesh.X), isoparametric(domain));

        auto r = [&](const Field<Family::H1> & u_) -> Residual<Family::H1> {
          nd::cpu_array<double, 3> u_q = evaluate(u_, domain);
          auto s_q = forall(qfunction<dim>, u_q, dX_dxi_q);
          return integrate(dot(s_q, phi), isoparametric(domain));
        };

        // finite difference approximation of jvp
        u = u0 + epsilon * du.data;
        Residual<Family::H1> rp = r(u);

        u = u0 - epsilon * du.data;
        Residual<Family::H1> rm = r(u);

        auto dr1 = (rp.data - rm.data) / (2 * epsilon);

        // "matrix-free" jvp
        u = u0;
        nd::cpu_array<double, 3> u_q = evaluate(u, domain);
        nd::cpu_array<double, 3> du_q = evaluate(du, domain);
        auto ds_q = forall(qfunction_jvp<dim>, u_q, du_q, dX_dxi_q);
        Residual<Family::H1> dr2_r = integrate(dot(ds_q, phi), isoparametric(domain));
        auto dr2 = dr2_r.data;

        {
          SCOPED_TRACE("||dr1 - dr2|| p1 = " + std::to_string(p1) + ", p2 = " + std::to_string(p2) + ", q = " + std::to_string(q));
          EXPECT_NEAR(relative_error(dr1, dr2), 0.0, tolerance);
        }

        // "sparse matrix" jvp
        auto ds_du_q = forall(qfunction_jac<dim>, u_q, dX_dxi_q);
        femto::sparse_matrix M = integrate(dot(psi, ds_du_q, phi), isoparametric(domain));

        Residual<Family::H1> dr3(FunctionSpace(u.family, p2, u.data.shape[1]), mesh);

        dr3.v() = M(du.v());

        {
          SCOPED_TRACE("||dr1 - dr3|| p1 = " + std::to_string(p1) + ", p2 = " + std::to_string(p2) + ", q = " + std::to_string(q));
          EXPECT_NEAR(relative_error(dr1, dr3.data), 0.0, tolerance);
        }

        {
          SCOPED_TRACE("||dr2 - dr3|| p1 = " + std::to_string(p1) + ", p2 = " + std::to_string(p2) + ", q = " + std::to_string(q));
          EXPECT_NEAR(relative_error(dr2, dr3.data), 0.0, 1.0e-14);
        }

        if (p1 == p2) {
          auto D1 = diagonal(M);
          auto D2 = integrate(diagonal(dot(psi, ds_du_q, phi)), isoparametric(domain));
          //print(D1);
          //print(nd::flatten(D2));
          EXPECT_NEAR(relative_error(D1, nd::flatten(D2)), 0.0, 1.0e-14);
        }

      }

    }


  }

}

// ----------------------------------------------------------------------------

TEST(IntegrateTest, verify_H1_H1_derivatives2D) { verify_H1_H1_derivatives<2>(); }
TEST(IntegrateTest, verify_H1_H1_derivatives3D) { verify_H1_H1_derivatives<3>(); }

// ----------------------------------------------------------------------------

TEST(IntegrateTest, H1_H1_edges) { H1_H1_test<1>("patch_test_edges.json", 1.0e-8); }

TEST(IntegrateTest, H1_H1_tris) { H1_H1_test<2>("patch_test_tris.json", 1.0e-8); }
TEST(IntegrateTest, H1_H1_quads) { H1_H1_test<2>("patch_test_quads.json", 1.0e-8); }
TEST(IntegrateTest, H1_H1_tris_and_quads) { H1_H1_test<2>("patch_test_tris_and_quads.json", 1.0e-8); }

TEST(IntegrateTest, H1_H1_tets) { H1_H1_test<3>("patch_test_tets.json", 1.0e-8); }
TEST(IntegrateTest, H1_H1_hexes) { H1_H1_test<3>("patch_test_hexes.json", 1.0e-8); }
TEST(IntegrateTest, H1_H1_tets_and_hexes) { H1_H1_test<3>("patch_test_tets_and_hexes.json", 1.0e-8); }
