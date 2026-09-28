#include "common.hpp"

#include <gtest/gtest.h>

#include <iomanip>
#include <iostream>
#include <functional>

#include "femto/mesh.hpp"
#include "femto/domain.hpp"
#include "femto/piola_transformations.hpp"

#include <gtest/gtest.h>

#include "forall.hpp"
#include "fm/types/vec.hpp"
#include "fm/types/matrix.hpp"
#include "containers/ndarray_conversions.hpp"

using namespace femto;

////////////////////////////////////////////////////////////////////////////////

double inv_mu = 3.0;

template < uint32_t dim >
vec<dim> material(const vec<dim> & B) {
  return (inv_mu + B[0]) * B;
}

template < uint32_t dim >
mat<dim,dim> material_jac(const vec<dim> & B) {
  return outer(B, Identity<dim>()[0]) + (inv_mu + B[0]) * Identity<dim>();
}

template < uint32_t dim >
auto qfunction(const vec< (dim == 2) ? 1 : dim > & B_xi, const mat<dim,dim> & dX_dxi) {
  auto Q = covariant_piola(dX_dxi);
  auto B = dot(Q, B_xi);
  auto H = material(B);
  return dot(transpose(Q), H) * det(dX_dxi);
}

template < uint32_t dim >
auto qfunction_jac(const vec< (dim == 2) ? 1 : dim > & B_xi, const mat<dim,dim> & dX_dxi) {
  auto Q = covariant_piola(dX_dxi);
  auto B = dot(Q, B_xi);
  return dot(transpose(Q), dot(material_jac(B), Q)) * det(dX_dxi);
}

template < uint32_t dim >
auto qfunction_jvp(const vec< (dim == 2) ? 1 : dim > & B_xi, const vec< (dim == 2) ? 1 : dim > & dB_xi, const mat<dim,dim> & dX_dxi) {
  return dot(qfunction_jac(B_xi, dX_dxi), dB_xi);
}

////////////////////////////////////////////////////////////////////////////////

template < uint32_t dim >
void verify_dHcurl_dHcurl_derivatives() {
  double eps = 1.0e-5;
  constexpr uint32_t c = (dim == 2) ? 1 : dim;
  vec<c>     B_xi = femto::random_vec<c>();
  vec<c>     dB_xi = femto::random_vec<c>();
  mat<dim,dim> dX_dxi = femto::random_mat<dim,dim>() + Identity<dim>();

  vec<c> dH0 = (qfunction(B_xi + eps * dB_xi, dX_dxi) -
                qfunction(B_xi - eps * dB_xi, dX_dxi)) / (2 * eps);
  vec<c> dH1 = dot(qfunction_jac(B_xi, dX_dxi), dB_xi);

  for (uint32_t i = 0; i < c; i++) {
    EXPECT_NEAR(dH0[i], dH1[i], 1.0e-9);
  }
}

// ----------------------------------------------------------------------------

TEST(IntegrateTest, verify_dHcurl_dHcurl_derivatives2D) { verify_dHcurl_dHcurl_derivatives<2>(); }
TEST(IntegrateTest, verify_dHcurl_dHcurl_derivatives3D) { verify_dHcurl_dHcurl_derivatives<3>(); }

////////////////////////////////////////////////////////////////////////////////
#if 1
template < uint32_t dim >
void dHcurl_dHcurl_test(std::string filename, double tolerance) {

  double epsilon = 1.0e-6;

  using vecd = vec<dim>;
  using matd = mat<dim,dim>;

  auto mesh = Mesh<>::load(FEMTO_MESH_DIR + filename);

  auto f = [](const vecd & X, const vecd & n) { return dot(X, n); };

  for (int p1 = 1; p1 < 4; p1++) {

    Field A = create_field<Family::Hcurl>(mesh, p1, 1);
    Field dA = create_field<Family::Hcurl>(mesh, p1, 1);

    auto nodes = nodes_for(A, mesh);
    auto directions = directions_for(A, mesh);
    auto A0 = forall(+f, nodes, directions);

    dA.data = femto::random(A.data.shape);

    BasisFunction<Family::Hcurl> psi(p1, 1);

    for (int p2 = 1; p2 < 4; p2++) {

      BasisFunction<Family::Hcurl> phi(p2, 1);

      for (int q = 1; q < 5; q++) {

        Domain domain(mesh, MeshQuadratureRule(q));

        nd::cpu_array<double, 3> dX_dxi_q = evaluate(grad(mesh.X), isoparametric(domain));

        auto r = [&](const Field<Family::Hcurl> & A_) -> Residual<Family::Hcurl> {
          nd::cpu_array<double, 3> B_xi_q = evaluate(curl(A_), isoparametric(domain));
          auto H_xi_q = forall(qfunction<dim>, B_xi_q, dX_dxi_q);
          return integrate(dot(H_xi_q, curl(phi)), isoparametric(domain));
        };

        // finite difference approximation of jvp
        A = A0 + epsilon * dA.data;
        Residual<Family::Hcurl> rp = r(A);

        A = A0 - epsilon * dA.data;
        Residual<Family::Hcurl> rm = r(A);

        auto dr1 = (rp.data - rm.data) / (2 * epsilon);

        // "matrix-free" jvp
        A = A0;
        nd::cpu_array<double, 3> B_xi_q = evaluate(curl(A), isoparametric(domain));
        nd::cpu_array<double, 3> dB_xi_q = evaluate(curl(dA), isoparametric(domain));
        auto dH_xi_q = forall(qfunction_jvp<dim>, B_xi_q, dB_xi_q, dX_dxi_q);
        Residual<Family::Hcurl> dr2_r = integrate(dot(dH_xi_q, curl(phi)), isoparametric(domain));
        auto dr2 = dr2_r.data;

        {
          SCOPED_TRACE("||dr1 - dr2|| p1 = " + std::to_string(p1) + ", p2 = " + std::to_string(p2) + ", q = " + std::to_string(q));
          EXPECT_NEAR(relative_error(dr1, dr2), 0.0, tolerance);
        }

        // "sparse matrix" jvp
        auto dHxi_dBxi_q = forall(qfunction_jac<dim>, B_xi_q, dX_dxi_q);
        femto::sparse_matrix K = integrate(dot(curl(psi), dHxi_dBxi_q, curl(phi)), isoparametric(domain));

        Residual<Family::Hcurl> dr3(FunctionSpace(A.family, p2, A.data.shape[1]), mesh);

        dr3.v() = K(dA.v());

        //std::string filename = "K" + std::to_string(p1) + std::to_string(p2) + std::to_string(q) + ".mtx";
        //export_matrix_market(K, filename);

        {
          SCOPED_TRACE("||dr1 - dr3|| p1 = " + std::to_string(p1) + ", p2 = " + std::to_string(p2) + ", q = " + std::to_string(q));
          EXPECT_NEAR(relative_error(dr1, dr3.data), 0.0, tolerance);
        }

        {
          // the matrix-free path integrates with sum factorization on tensor-product
          // elements while the assembled-matrix path sums per shape function, so the
          // two accumulate roundoff in different orders (worst at cubic hexes)
          SCOPED_TRACE("||dr2 - dr3|| p1 = " + std::to_string(p1) + ", p2 = " + std::to_string(p2) + ", q = " + std::to_string(q));
          EXPECT_NEAR(relative_error(dr2, dr3.data), 0.0, 1.0e-13);
        }

        if (p1 == p2) {
          auto D1 = diagonal(K);
          auto D2 = integrate(diagonal(dot(curl(psi), dHxi_dBxi_q, curl(phi))), isoparametric(domain));
          SCOPED_TRACE("||D1 - D2|| p1 = " + std::to_string(p1) + ", p2 = " + std::to_string(p2) + ", q = " + std::to_string(q));
          EXPECT_NEAR(relative_error(D1, nd::flatten(D2)), 0.0, 1.0e-14);
        }

      }

    }

  }

}

// ----------------------------------------------------------------------------

TEST(IntegrateTest, dHcurl_dHcurl_tris) { dHcurl_dHcurl_test<2>("patch_test_tris.json", 1.0e-8); }
TEST(IntegrateTest, dHcurl_dHcurl_quads) { dHcurl_dHcurl_test<2>("patch_test_quads.json", 1.0e-8); }
TEST(IntegrateTest, dHcurl_dHcurl_tris_and_quads) { dHcurl_dHcurl_test<2>("patch_test_tris_and_quads.json", 1.0e-8); }

TEST(IntegrateTest, dHcurl_dHcurl_one_tet) { dHcurl_dHcurl_test<3>("one_tet.json", 1.0e-8); }
TEST(IntegrateTest, dHcurl_dHcurl_tets) { dHcurl_dHcurl_test<3>("patch_test_tets.json", 1.0e-8); }
TEST(IntegrateTest, dHcurl_dHcurl_hexes) { dHcurl_dHcurl_test<3>("patch_test_hexes.json", 1.0e-8); }
TEST(IntegrateTest, dHcurl_dHcurl_tets_and_hexes) { dHcurl_dHcurl_test<3>("patch_test_tets_and_hexes.json", 1.0e-8); }

//TEST(IntegrateTest, dHcurl_dHcurl_big_tet_mesh) { dHcurl_dHcurl_test<3>("octane_fine.msh", 1.0e-8); }
#endif