#include "common.hpp"

#include <gtest/gtest.h>

#include "femto/mesh.hpp"
#include "femto/domain.hpp"

using namespace femto;

// sum of the domain's precomputed facet measures against the quadrature
// weights: an independent route to the boundary measure from integrate()
static double measure_from_tables(const Domain<> & d) {
  double sum = 0.0;
  uint32_t q = 0;
  foreach_geometry([&](auto geom) {
    uint32_t num_elements = d.active_elements[geom].shape[0];
    if (num_elements == 0) return;
    nd::view<const double, 1> w = d.rule[geom].weights;
    uint32_t qpe = femto::impl::qpe<geom>(d.rule[geom].points.shape[0]);
    for (uint32_t e = 0; e < num_elements; e++) {
      for (uint32_t i = 0; i < qpe; i++) {
        sum += d.det_dX_dxi(q++) * femto::impl::integration_weight<geom>(i, w);
      }
    }
  });
  EXPECT_EQ(q, total(d.num_qpts));
  return sum;
}

static nd::cpu_array<double, 1> ones(const Domain<> & d) {
  nd::cpu_array<double, 1> f({total(d.num_qpts)});
  for (uint32_t q = 0; q < f.shape[0]; q++) { f(q) = 1.0; }
  return f;
}

// the pseudo-inverse stored in dxi_dX must be a left inverse of dX_dxi
static void check_pseudo_inverse(const Domain<> & d) {
  nd::cpu_array<double, 3> J = evaluate(grad(d.mesh.X), isoparametric(d));
  uint32_t sdim = J.shape[1];
  uint32_t gdim = J.shape[2];
  ASSERT_EQ(gdim, d.geometry_dimension);
  ASSERT_EQ(d.dxi_dX.shape[1], gdim);
  ASSERT_EQ(d.dxi_dX.shape[2], sdim);
  for (uint32_t q = 0; q < J.shape[0]; q++) {
    for (uint32_t i = 0; i < gdim; i++) {
      for (uint32_t k = 0; k < gdim; k++) {
        double PJ = 0.0;
        for (uint32_t j = 0; j < sdim; j++) { PJ += d.dxi_dX(q, i, j) * J(q, j, k); }
        ASSERT_NEAR(PJ, double(i == k), 1e-12);
      }
    }
  }
}

// perimeter / surface area of a box, and the boundary integrals of a
// constant (residual and mass matrix) that must reproduce it
template < uint32_t dim >
static void box_test(const Mesh<> & mesh, vec<dim> L, double expected) {

  for (int q = 1; q < 4; q++) {

    Domain bdr(boundary_of(mesh), MeshQuadratureRule(q));
    EXPECT_EQ(bdr.geometry_dimension, dim - 1);

    EXPECT_NEAR(integrate(ones(bdr), bdr), expected, 1e-12);
    EXPECT_NEAR(measure_from_tables(bdr), expected, 1e-12);
    if constexpr (dim == 2) {
      EXPECT_NEAR(integrate(std::function<double(vec2)>([](vec2){ return 1.0; }), bdr), expected, 1e-12);
    } else {
      EXPECT_NEAR(integrate(std::function<double(vec3)>([](vec3){ return 1.0; }), bdr), expected, 1e-12);
    }

    // every quadrature point sits on a face of the box
    nd::cpu_array<double, 3> x_q = evaluate(mesh.X, bdr);
    ASSERT_EQ(x_q.shape[0], total(bdr.num_qpts));
    for (uint32_t i = 0; i < x_q.shape[0]; i++) {
      bool on_face = false;
      for (uint32_t c = 0; c < dim; c++) {
        on_face |= (std::abs(x_q(i, c, 0)) < 1e-12) || (std::abs(x_q(i, c, 0) - L[c]) < 1e-12);
      }
      EXPECT_TRUE(on_face);
    }

    for (int p = 1; p < 4; p++) {
      BasisFunction<Family::H1> phi(p, 1);
      Field<Family::H1> u = create_field<Family::H1>(mesh, p, 1);
      nd::cpu_array<double, 2> nodes = nodes_for(u, mesh);

      // partition of unity: the residual of a unit load sums to the measure,
      // and vanishes at nodes that are not on the boundary
      Residual<Family::H1> r = integrate(dot(ones(bdr), phi), bdr);
      double sum = 0.0;
      for (uint32_t i = 0; i < r.data.shape[0]; i++) {
        sum += r.data(i, 0);
        bool interior = true;
        for (uint32_t c = 0; c < dim; c++) {
          interior &= (nodes(i, c) > 1e-12) && (nodes(i, c) < L[c] - 1e-12);
        }
        if (interior) { EXPECT_EQ(r.data(i, 0), 0.0); }
      }
      EXPECT_NEAR(sum, expected, 1e-12);

      // boundary mass matrix times the constant 1 is that same residual
      femto::sparse_matrix M = integrate(dot(phi, ones(bdr), phi), bdr);
      for (uint32_t i = 0; i < u.data.shape[0]; i++) { u.data(i, 0) = 1.0; }
      Residual<Family::H1> Mu(FunctionSpace(Family::H1, p, 1), mesh);
      Mu.v() = M(u.v());
      for (uint32_t i = 0; i < r.data.shape[0]; i++) {
        EXPECT_NEAR(Mu.data(i, 0), r.data(i, 0), 1e-12);
      }

      // and its diagonal matches the assembled matrix's
      nd::cpu_array<double, 2> D = integrate(diagonal(dot(phi, ones(bdr), phi)), bdr);
      for (int i = 0; i < M.nrows; i++) {
        double Mii = 0.0;
        for (int k = M.row_ptr[i]; k < M.row_ptr[i + 1]; k++) {
          if (M.col_ind[k] == i) { Mii = M.values[k]; }
        }
        EXPECT_NEAR(D(i, 0), Mii, 1e-12);
      }
    }
  }
}

TEST(boundary_domain, box_2d) {
  vec2 L{2.0, 1.0};
  Mesh<> quads = Mesh<>::cuboid({4, 3}, L);
  box_test<2>(quads, L, 6.0);
  box_test<2>(convert_to_simplices(quads), L, 6.0);
}

TEST(boundary_domain, box_3d) {
  vec3 L{1.0, 2.0, 3.0};
  Mesh<> hexes = Mesh<>::cuboid({2, 3, 4}, L);
  box_test<3>(hexes, L, 22.0);
  box_test<3>(convert_to_simplices(hexes), L, 22.0);
}

// curved facets: the boundary of a p = 2 disk / ball converges to the circle /
// sphere, and the facet jacobian tables are consistent with evaluate()
TEST(boundary_domain, disk) {
  double r = 1.5;
  for (Geometry g : {Geometry::Triangle, Geometry::Quadrilateral}) {
    double errors[2];
    for (int p = 1; p < 3; p++) {
      Mesh<> mesh = Mesh<>::disk(vec2{0.0, 0.0}, r, 0.3, p, g);
      Domain bdr(boundary_of(mesh), MeshQuadratureRule(3));
      errors[p - 1] = std::abs(integrate(ones(bdr), bdr) - 2 * M_PI * r) / (2 * M_PI * r);
      EXPECT_NEAR(integrate(ones(bdr), bdr), measure_from_tables(bdr), 1e-12);
      check_pseudo_inverse(bdr);
    }
    EXPECT_LT(errors[0], 5e-3);
    EXPECT_LT(errors[1], 1e-4);
  }
}

TEST(boundary_domain, ball) {
  double r = 1.0;
  for (Geometry g : {Geometry::Tetrahedron, Geometry::Hexahedron}) {
    double errors[2];
    for (int p = 1; p < 3; p++) {
      Mesh<> mesh = Mesh<>::ball(vec3{0.0, 0.0, 0.0}, r, 0.4, p, g);
      Domain bdr(boundary_of(mesh), MeshQuadratureRule(3));
      errors[p - 1] = std::abs(integrate(ones(bdr), bdr) - 4 * M_PI * r * r) / (4 * M_PI * r * r);
      EXPECT_NEAR(integrate(ones(bdr), bdr), measure_from_tables(bdr), 1e-12);
      check_pseudo_inverse(bdr);
    }
    // p = 1 is a polyhedron (6% off at this h); p = 2 gets the curvature
    EXPECT_LT(errors[1], errors[0] / 10);
    EXPECT_LT(errors[1], 5e-3);
  }
}

// DG fields have no dofs on facets, so a boundary domain refuses them
TEST(boundary_domain, dg_is_rejected) {
  Mesh<> mesh = Mesh<>::cuboid({2, 2}, vec2{1.0, 1.0});
  Domain bdr(boundary_of(mesh), MeshQuadratureRule(2));
  Field<Family::DG> u = create_field<Family::DG>(mesh, 1, 1);
  // re-exec the binary rather than fork: a forked child has no threadpool workers
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(({ nd::cpu_array<double, 3> u_q = evaluate(u, bdr); }), ::testing::ExitedWithCode(1), "DG fields are not supported");
}
