#include "common.hpp"

#include <gtest/gtest.h>

#include <functional>

#include "femto/mesh.hpp"
#include "femto/domain.hpp"

#include "forall.hpp"
#include "fm/types/vec.hpp"
#include "fm/types/matrix.hpp"
#include "containers/ndarray_conversions.hpp"

using namespace femto;

////////////////////////////////////////////////////////////////////////////////

// a nonlinear boundary source s(u) = rho u^2, and its derivatives.  Applied
// on a spatial domain the kernels supply the facet measure; in parent
// coordinates it is carried explicitly from the (dim x dim - 1) jacobian J
double rho = 3.0;

double qfunction(const double & u) { return rho * u * u; }
double qfunction_jac(const double & u) { return 2 * rho * u; }
double qfunction_jvp(const double & u, const double & du) { return qfunction_jac(u) * du; }

template < uint32_t dim >
double facet_measure(const mat<dim, dim - 1> & J) { return sqrt(det(dot(transpose(J), J))); }

template < uint32_t dim >
double qfunction_xi(const double & u, const mat<dim, dim - 1> & J) { return qfunction(u) * facet_measure<dim>(J); }

template < uint32_t dim >
double qfunction_jac_xi(const double & u, const mat<dim, dim - 1> & J) { return qfunction_jac(u) * facet_measure<dim>(J); }

template < uint32_t dim >
double qfunction_jvp_xi(const double & u, const double & du, const mat<dim, dim - 1> & J) { return qfunction_jvp(u, du) * facet_measure<dim>(J); }

////////////////////////////////////////////////////////////////////////////////

// the finite difference of the boundary residual, its matrix-free jvp and the
// assembled boundary matrix times du must all agree
template < uint32_t dim >
void H1_H1_bdr_test(std::string filename, double tolerance) {

  threadpool::set_num_threads(1);

  double epsilon = 1.0e-6;

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

          Residual<Family::H1> dr3(FunctionSpace(Family::H1, p2, 1), mesh);
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
          check(bdr, qfunction, qfunction_jvp, qfunction_jac);
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

TEST(IntegrateTest, H1_H1_bdr_tris) { H1_H1_bdr_test<2>("patch_test_tris.json", 1.0e-8); }
TEST(IntegrateTest, H1_H1_bdr_quads) { H1_H1_bdr_test<2>("patch_test_quads.json", 1.0e-8); }
TEST(IntegrateTest, H1_H1_bdr_tris_and_quads) { H1_H1_bdr_test<2>("patch_test_tris_and_quads.json", 1.0e-8); }

TEST(IntegrateTest, H1_H1_bdr_tets) { H1_H1_bdr_test<3>("patch_test_tets.json", 1.0e-8); }
TEST(IntegrateTest, H1_H1_bdr_hexes) { H1_H1_bdr_test<3>("patch_test_hexes.json", 1.0e-8); }
TEST(IntegrateTest, H1_H1_bdr_tets_and_hexes) { H1_H1_bdr_test<3>("patch_test_tets_and_hexes.json", 1.0e-8); }
