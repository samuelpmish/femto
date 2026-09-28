#include "common.hpp"

#include <gtest/gtest.h>

#include <functional>

#include "femto/mesh.hpp"
#include "femto/domain.hpp"

#include "forall.hpp"
#include "fm/types/vec.hpp"
#include "fm/types/matrix.hpp"
#include "fm/operations/random.hpp"
#include "containers/ndarray_conversions.hpp"

using namespace fm;
using namespace femto;

////////////////////////////////////////////////////////////////////////////////

// a nonlinear vector-valued boundary source s(u) = rho (u . u) u and its
// derivative ds_du = rho (2 u uᵀ + (u . u) I), see integrate_spmat_H1v_H1v_tests
double rho = 3.0;

template < uint32_t dim >
vec<dim> qfunction(const vec<dim> & u) { return rho * dot(u, u) * u; }

template < uint32_t dim >
mat<dim,dim> qfunction_jac(const vec<dim> & u) {
  return rho * (2.0 * outer(u, u) + dot(u, u) * Identity<dim>());
}

template < uint32_t dim >
vec<dim> qfunction_jvp(const vec<dim> & u, const vec<dim> & du) { return dot(qfunction_jac(u), du); }

// the same in parent coordinates, weighted by the facet measure of J
template < uint32_t dim >
double facet_measure(const mat<dim, dim - 1> & J) { return sqrt(det(dot(transpose(J), J))); }

template < uint32_t dim >
vec<dim> qfunction_xi(const vec<dim> & u, const mat<dim, dim - 1> & J) { return qfunction(u) * facet_measure<dim>(J); }

template < uint32_t dim >
mat<dim,dim> qfunction_jac_xi(const vec<dim> & u, const mat<dim, dim - 1> & J) { return qfunction_jac(u) * facet_measure<dim>(J); }

template < uint32_t dim >
vec<dim> qfunction_jvp_xi(const vec<dim> & u, const vec<dim> & du, const mat<dim, dim - 1> & J) { return qfunction_jvp(u, du) * facet_measure<dim>(J); }

////////////////////////////////////////////////////////////////////////////////

template < uint32_t dim >
void verify_H1v_H1v_derivatives() {
  double eps = 1.0e-6;
  vec<dim> u = random_vec<dim>();
  vec<dim> du = random_vec<dim>();

  vec<dim> dsource0 = (qfunction(u + eps * du) - qfunction(u - eps * du)) / (2 * eps);
  vec<dim> dsource1 = qfunction_jvp(u, du);
  for (int k = 0; k < dim; k++) {
    EXPECT_NEAR(dsource0[k], dsource1[k], 3.0e-9);
  }
}

////////////////////////////////////////////////////////////////////////////////

// the finite difference of the boundary residual, its matrix-free jvp and the
// assembled boundary matrix times du must all agree
template < uint32_t dim >
void H1v_H1v_bdr_test(std::string filename, double tolerance) {

  threadpool::set_num_threads(1);

  double epsilon = 1.0e-6;

  auto mesh = Mesh<>::load(FEMTO_MESH_DIR + filename);

  for (int p1 = 1; p1 < 4; p1++) {

    Field u = create_field<Family::H1>(mesh, p1, dim);
    Field du = create_field<Family::H1>(mesh, p1, dim);

    auto nodes = nodes_for(u, mesh);

    auto u0 = forall(+[](const vec<dim> & X) {
      vec<dim> output;
      for (int i = 0; i < dim; i++) { output[i] = (i + 1) * X[i] + 1.0; }
      return output;
    }, nodes);

    du.data = femto::random(u.data.shape);

    BasisFunction<Family::H1> psi(p1, dim);

    for (int p2 = 1; p2 < 4; p2++) {

      BasisFunction<Family::H1> phi(p2, dim);

      for (int q = 1; q < 5; q++) {

        Domain bdr(boundary_of(mesh), MeshQuadratureRule(q));

        nd::cpu_array<double, 3> J_q = evaluate(grad(mesh.X), isoparametric(bdr));

        // s, ds and ds_du take (u, J...) with J present only in parent coordinates
        auto check = [&](const DomainWithType<> & d, auto s, auto ds, auto ds_du, const auto & ... J) {

          auto r = [&](const Field<Family::H1> & u_) -> Residual<Family::H1> {
            nd::cpu_array<double, 3> u_q = evaluate(u_, d.domain);
            auto s_q = forall(s, u_q, J...);
            return integrate(dot(s_q, phi), d);
          };

          // finite difference approximation of jvp
          u = u0 + epsilon * du.data;
          Residual<Family::H1> rp = r(u);

          u = u0 - epsilon * du.data;
          Residual<Family::H1> rm = r(u);

          auto dr1 = (rp.data - rm.data) / (2 * epsilon);

          // "matrix-free" jvp
          u = u0;
          nd::cpu_array<double, 3> u_q = evaluate(u, d.domain);
          nd::cpu_array<double, 3> du_q = evaluate(du, d.domain);
          auto ds_q = forall(ds, u_q, du_q, J...);
          Residual<Family::H1> dr2_r = integrate(dot(ds_q, phi), d);
          auto dr2 = dr2_r.data;

          // "sparse matrix" jvp
          auto ds_du_q = forall(ds_du, u_q, J...);
          femto::sparse_matrix M = integrate(dot(psi, ds_du_q, phi), d);

          Residual<Family::H1> dr3(FunctionSpace(Family::H1, p2, dim), mesh);
          dr3.v() = M(du.v());

          EXPECT_NEAR(relative_error(dr1, dr2), 0.0, tolerance);
          EXPECT_NEAR(relative_error(dr1, dr3.data), 0.0, tolerance);
          EXPECT_NEAR(relative_error(dr2, dr3.data), 0.0, 1.0e-14);

          if (p1 == p2) {
            auto D1 = diagonal(M);
            auto D2 = integrate(diagonal(dot(psi, ds_du_q, phi)), d);
            EXPECT_NEAR(relative_error(D1, nd::flatten(D2)), 0.0, 1.0e-14);
          }
        };

        {
          SCOPED_TRACE("spatial, p1 = " + std::to_string(p1) + ", p2 = " + std::to_string(p2) + ", q = " + std::to_string(q));
          check(bdr, qfunction<dim>, qfunction_jvp<dim>, qfunction_jac<dim>);
        }

        {
          SCOPED_TRACE("isoparametric, p1 = " + std::to_string(p1) + ", p2 = " + std::to_string(p2) + ", q = " + std::to_string(q));
          check(isoparametric(bdr), qfunction_xi<dim>, qfunction_jvp_xi<dim>, qfunction_jac_xi<dim>, J_q);
        }

      }

    }

  }

}

// ----------------------------------------------------------------------------

TEST(IntegrateTest, verify_H1v_H1v_derivatives2D) { verify_H1v_H1v_derivatives<2>(); }
TEST(IntegrateTest, verify_H1v_H1v_derivatives3D) { verify_H1v_H1v_derivatives<3>(); }

// ----------------------------------------------------------------------------

TEST(IntegrateTest, H1v_H1v_bdr_tris) { H1v_H1v_bdr_test<2>("patch_test_tris.json", 1.0e-8); }
TEST(IntegrateTest, H1v_H1v_bdr_quads) { H1v_H1v_bdr_test<2>("patch_test_quads.json", 1.0e-8); }
TEST(IntegrateTest, H1v_H1v_bdr_tris_and_quads) { H1v_H1v_bdr_test<2>("patch_test_tris_and_quads.json", 1.0e-8); }

TEST(IntegrateTest, H1v_H1v_bdr_tets) { H1v_H1v_bdr_test<3>("patch_test_tets.json", 1.0e-8); }
TEST(IntegrateTest, H1v_H1v_bdr_hexes) { H1v_H1v_bdr_test<3>("patch_test_hexes.json", 1.0e-8); }
TEST(IntegrateTest, H1v_H1v_bdr_tets_and_hexes) { H1v_H1v_bdr_test<3>("patch_test_tets_and_hexes.json", 1.0e-8); }
