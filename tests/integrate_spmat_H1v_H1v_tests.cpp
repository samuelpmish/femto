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
#include "fm/operations/random.hpp"
#include "containers/ndarray_conversions.hpp"

using namespace fm;
using namespace femto;

////////////////////////////////////////////////////////////////////////////////

double rho = 3.0;

template < uint32_t dim >
vec<dim> qfunction(const vec<dim> & u, const mat<dim,dim> & dX_dxi) {
  return rho * dot(u, u) * u * det(dX_dxi);
}

/*

--------------------------------------------------------------------------------

  s[i] = rho * u[j] * u[j] * u[i] * det(dX_dxi);

--------------------------------------------------------------------------------

  ds[i]_du[j] = d_du[j] (rho * u[k] * u[k] * u[i] * det(dX_dxi));

  = rho * det(dX_dxi) * (
    du[k]_du[j] * u[k] * u[i] +
    u[k] * du[k]_du[j] * u[i] +
    u[k] * u[k] * du[i]_du[j]
  )

  = rho * det(dX_dxi) * (
    (j == k) * u[k] * u[i] +
    u[k] * (k == j) * u[i] +
    u[k] * u[k] * (i == j)
  )

  = rho * det(dX_dxi) * (u[j] * u[i] + u[j] * u[i] + (u[k] * u[k]) * (i == j))

--------------------------------------------------------------------------------

*/
template < uint32_t dim >
mat<dim,dim> qfunction_jac(const vec<dim> & u, const mat<dim,dim> & dX_dxi) {
  double uTu = dot(u, u);
  double scale = rho * det(dX_dxi);
  mat<dim,dim> ds_du{};
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      ds_du[i][j] = scale * (u[i] * u[j] + u[j] * u[i] + uTu * (i == j));
    }
  }
  return ds_du;
}

template < int dim >
vec<dim> qfunction_jvp(const vec<dim> & u, const vec<dim> & du, const mat<dim,dim> & dX_dxi) {
  return dot(qfunction_jac(u, dX_dxi), du);
}

////////////////////////////////////////////////////////////////////////////////

template < uint32_t dim >
void verify_H1v_H1v_derivatives() {
  double eps = 1.0e-6;
  vec<dim> u = random_vec<dim>();
  vec<dim> du = random_vec<dim>();
  mat<dim,dim> dX_dxi = femto::random_mat<dim,dim>() + Identity<dim>();

  vec<dim> dsource0 = (qfunction(u + eps * du, dX_dxi) - qfunction(u - eps * du, dX_dxi)) / (2 * eps);
  vec<dim> dsource1 = dot(qfunction_jac(u, dX_dxi), du);
  for (int k = 0; k < dim; k++) {
    EXPECT_NEAR(dsource0[k], dsource1[k], 3.0e-9);
  }
}

////////////////////////////////////////////////////////////////////////////////

template < uint32_t dim >
void H1v_H1v_test(std::string filename, double tolerance) {

  threadpool::set_num_threads(1);

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

TEST(IntegrateTest, verify_H1v_H1v_derivatives2D) { verify_H1v_H1v_derivatives<2>(); }
TEST(IntegrateTest, verify_H1v_H1v_derivatives3D) { verify_H1v_H1v_derivatives<3>(); }

// ----------------------------------------------------------------------------

TEST(IntegrateTest, H1v_H1v_edges) { H1v_H1v_test<1>("patch_test_edges.json", 1.0e-8); }

TEST(IntegrateTest, H1v_H1v_tris) { H1v_H1v_test<2>("patch_test_tris.json", 1.0e-8); }
TEST(IntegrateTest, H1v_H1v_quads) { H1v_H1v_test<2>("patch_test_quads.json", 1.0e-8); }
TEST(IntegrateTest, H1v_H1v_tris_and_quads) { H1v_H1v_test<2>("patch_test_tris_and_quads.json", 1.0e-8); }

TEST(IntegrateTest, H1v_H1v_tets) { H1v_H1v_test<3>("patch_test_tets.json", 1.0e-8); }
TEST(IntegrateTest, H1v_H1v_hexes) { H1v_H1v_test<3>("patch_test_hexes.json", 1.0e-8); }
TEST(IntegrateTest, H1v_H1v_tets_and_hexes) { H1v_H1v_test<3>("patch_test_tets_and_hexes.json", 1.0e-8); }
