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

double k = 3.0;

template < uint32_t dim >
vec<dim> material(const vec<dim> & du_dX) {
  return (k + du_dX[0]) * du_dX;
}

template < uint32_t dim >
mat<dim,dim> material_jac(const vec<dim> & du_dX) {
  return outer(du_dX, Identity<dim>()[0]) + (k + du_dX[0]) * Identity<dim>();
}

template < uint32_t dim >
vec<dim> qfunction(const vec<dim> & du_dxi, const mat<dim,dim> & dX_dxi) {
  mat<dim,dim> dxi_dX = inv(dX_dxi);
  vec<dim> du_dX = dot(du_dxi, dxi_dX);
  vec<dim> flux = material(du_dX);
  return dot(flux, transpose(dxi_dX)) * det(dX_dxi);
}

template < uint32_t dim >
mat<dim,dim> qfunction_jac(const vec<dim> & du_dxi, const mat<dim,dim> & dX_dxi) {
  mat<dim,dim> dxi_dX = inv(dX_dxi);
  vec<dim> du_dX = dot(du_dxi, dxi_dX);
  return dot(dxi_dX, dot(material_jac(du_dX), transpose(dxi_dX))) * det(dX_dxi);
}

template < uint32_t dim >
vec<dim> qfunction_jvp(const vec<dim> & du_dxi, const vec<dim> & ddu_dxi, const mat<dim,dim> & dX_dxi) {
  return dot(qfunction_jac(du_dxi, dX_dxi), ddu_dxi);
}

////////////////////////////////////////////////////////////////////////////////

template < uint32_t dim >
void verify_dH1_dH1_derivatives() {
  double eps = 1.0e-5;
  vec<dim>     du_dxi = femto::random_vec<dim>();
  vec<dim>     ddu_dxi = femto::random_vec<dim>();
  mat<dim,dim> dX_dxi = femto::random_mat<dim,dim>() + Identity<dim>();

  vec<dim> dflux0 = (qfunction(du_dxi + eps * ddu_dxi, dX_dxi) -
                     qfunction(du_dxi - eps * ddu_dxi, dX_dxi)) / (2 * eps);
  vec<dim> dflux1 = dot(qfunction_jac(du_dxi, dX_dxi), ddu_dxi);

  for (uint32_t i = 0; i < dim; i++) {
    EXPECT_NEAR(dflux0[i], dflux1[i], 1.0e-10);
  }
}

////////////////////////////////////////////////////////////////////////////////

template < uint32_t dim >
void dH1_dH1_test(std::string filename, double tolerance) {

  double epsilon = 1.0e-6;

  using vecd = vec<dim>;
  using matd = mat<dim,dim>;

  auto mesh = Mesh< memory::space::cpu >::load(FEMTO_MESH_DIR + filename);

  print_timings = true;

  for (int p1 = 3; p1 < 4; p1++) {

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

    for (int p2 = 3; p2 < 4; p2++) {

      BasisFunction<Family::H1> phi(p2, 1);

      for (int q = 1; q < 5; q++) {

        Domain domain(mesh, MeshQuadratureRule(q));

        nd::cpu_array<double, 3> dX_dxi_q = evaluate(grad(mesh.X), isoparametric(domain));

        auto r = [&](const Field<Family::H1> & u_) -> Residual<Family::H1> {
          nd::cpu_array<double, 3> du_dxi_q = evaluate(grad(u_), isoparametric(domain));
          auto f_q = forall(qfunction<dim>, du_dxi_q, dX_dxi_q);
          return integrate(dot(f_q, grad(phi)), isoparametric(domain));
        };

        // finite difference approximation of jvp
        u = u0 + epsilon * du.data;
        Residual<Family::H1> rp = r(u);

        u = u0 - epsilon * du.data;
        Residual<Family::H1> rm = r(u);

        auto dr1 = (rp.data - rm.data) / (2 * epsilon);

        // "matrix-free" jvp
        u = u0;
        nd::cpu_array<double, 3> du_dxi_q = evaluate(grad(u), isoparametric(domain));
        nd::cpu_array<double, 3> ddu_dxi_q = evaluate(grad(du), isoparametric(domain));
        auto df_q = forall(qfunction_jvp<dim>, du_dxi_q, ddu_dxi_q, dX_dxi_q);
        Residual<Family::H1> dr2_r = integrate(dot(df_q, grad(phi)), isoparametric(domain));
        auto dr2 = dr2_r.data;

        {
          SCOPED_TRACE("||dr1 - dr2|| p1 = " + std::to_string(p1) + ", p2 = " + std::to_string(p2) + ", q = " + std::to_string(q));
          EXPECT_NEAR(relative_error(dr1, dr2), 0.0, tolerance);
        }

        // "sparse matrix" jvp
        auto df_ddudxi_q = forall(qfunction_jac<dim>, du_dxi_q, dX_dxi_q);
        femto::sparse_matrix K = integrate(dot(grad(psi), df_ddudxi_q, grad(phi)), isoparametric(domain));

        Residual<Family::H1> dr3(FunctionSpace(u.family, p2, u.data.shape[1]), mesh);

        dr3.v() = K(du.v());

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

TEST(IntegrateTest, verify_dH1_dH1_derivatives2D) { verify_dH1_dH1_derivatives<2>(); }
TEST(IntegrateTest, verify_dH1_dH1_derivatives3D) { verify_dH1_dH1_derivatives<3>(); }

// ----------------------------------------------------------------------------

TEST(IntegrateTest, dH1_dH1_edges) { dH1_dH1_test<1>("patch_test_edges.json", 1.0e-8); }

TEST(IntegrateTest, dH1_dH1_tris) { dH1_dH1_test<2>("patch_test_tris.json", 1.0e-8); }
TEST(IntegrateTest, dH1_dH1_quads) { dH1_dH1_test<2>("patch_test_quads.json", 1.0e-8); }
TEST(IntegrateTest, dH1_dH1_tris_and_quads) { dH1_dH1_test<2>("patch_test_tris_and_quads.json", 1.0e-8); }

TEST(IntegrateTest, dH1_dH1_tets) { dH1_dH1_test<3>("patch_test_tets.json", 1.0e-8); }
TEST(IntegrateTest, dH1_dH1_hexes) { dH1_dH1_test<3>("patch_test_hexes.json", 1.0e-8); }
TEST(IntegrateTest, dH1_dH1_tets_and_hexes) { dH1_dH1_test<3>("patch_test_tets_and_hexes.json", 1.0e-8); }

//TEST(IntegrateTest, big_dH1_dH1_tets) { dH1_dH1_test<3>("octane_fine.msh", 1.0e-8); }