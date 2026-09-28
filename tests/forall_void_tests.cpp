// forall() over a q-function that returns void and writes through its non-const
// reference arguments.  It has to agree with the value-returning overload, size
// an output that has not been allocated yet, and leave an output that is already
// the right shape where it is -- that last one is the whole point: it lets a
// residual evaluated over and over reuse its quadrature-point buffers.
#include "common.hpp"

#include <gtest/gtest.h>

#include "femto/mesh.hpp"
#include "femto/domain.hpp"

#include "forall.hpp"

using namespace femto;

// the same kernel, written both ways
static vec<3> flux(const vec<3> & du_dxi, const mat<3,3> & dX_dxi) {
  mat<3,3> dxi_dX = inv(dX_dxi);
  return dot(dot(du_dxi, dxi_dX), transpose(dxi_dX)) * det(dX_dxi);
}
static void flux_void(const vec<3> & du_dxi, const mat<3,3> & dX_dxi, vec<3> & f) {
  f = flux(du_dxi, dX_dxi);
}

TEST(ForallVoid, matches_the_value_returning_overload) {

  auto mesh = Mesh< memory::space::cpu >::load(FEMTO_MESH_DIR + std::string("patch_test_tets.json"));

  Field u = create_field<Family::H1>(mesh, 2, 1);
  u.data = femto::random(u.data.shape);
  Domain domain(mesh, MeshQuadratureRule(2));

  nd::cpu_array<double, 3> dX_dxi_q = evaluate(grad(mesh.X), isoparametric(domain));
  nd::cpu_array<double, 3> du_dxi_q = evaluate(grad(u), isoparametric(domain));

  nd::cpu_array<double, 2> expected = forall(flux, du_dxi_q, dX_dxi_q);

  // an unallocated output is sized here
  nd::cpu_array<double, 2> f_q;
  EXPECT_EQ(f_q.sz, 0u);
  forall(flux_void, du_dxi_q, dX_dxi_q, f_q);
  ASSERT_EQ(f_q.sz, expected.sz);
  ASSERT_EQ(f_q.shape[0], expected.shape[0]);
  ASSERT_EQ(f_q.shape[1], expected.shape[1]);
  for (uint32_t i = 0; i < expected.sz; i++) {
    EXPECT_EQ(f_q.data()[i], expected.data()[i]) << "differs at " << i;
  }

  // a second pass over the same output must not move its buffer
  const double * values_before = f_q.data();
  forall(flux_void, du_dxi_q, dX_dxi_q, f_q);
  EXPECT_EQ(f_q.data(), values_before) << "forall() reallocated an output that was already the right size";
  for (uint32_t i = 0; i < expected.sz; i++) {
    EXPECT_EQ(f_q.data()[i], expected.data()[i]) << "differs at " << i;
  }
}

// a struct functor taking its arguments by const reference goes through the
// value-returning overload (a function pointer or std::function strips the
// reference itself; the functor overload has to)
struct ScaledFlux {
  double scale;
  vec<3> operator()(const vec<3> & du_dxi, const mat<3,3> & dX_dxi) const { return scale * flux(du_dxi, dX_dxi); }
};

TEST(Forall, functor_with_reference_parameters) {

  auto mesh = Mesh< memory::space::cpu >::load(FEMTO_MESH_DIR + std::string("patch_test_tets.json"));

  Field u = create_field<Family::H1>(mesh, 2, 1);
  u.data = femto::random(u.data.shape);
  Domain domain(mesh, MeshQuadratureRule(2));

  nd::cpu_array<double, 3> dX_dxi_q = evaluate(grad(mesh.X), isoparametric(domain));
  nd::cpu_array<double, 3> du_dxi_q = evaluate(grad(u), isoparametric(domain));

  nd::cpu_array<double, 2> expected = forall(flux, du_dxi_q, dX_dxi_q);
  nd::cpu_array<double, 2> actual = forall(ScaledFlux{1.0}, du_dxi_q, dX_dxi_q);
  ASSERT_EQ(actual.sz, expected.sz);
  for (uint32_t i = 0; i < expected.sz; i++) {
    EXPECT_EQ(actual.data()[i], expected.data()[i]) << "differs at " << i;
  }
}

// save() and load() of one entry of a (n, 3) and a (n, 3, 3) array: a vec or
// mat template parameter that does not deduce leaves save() doing nothing
TEST(Forall, save_and_load_round_trip) {
  vec<3> b{1.0, 2.0, 3.0};
  mat<3,3> A{{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}}};

  nd::cpu_array<double, 2> v({2, 3});
  nd::cpu_array<double, 3> m({2, 3, 3});
  save(v, 1, b);
  save(m, 1, A);

  vec<3> b1 = load<vec<3>>(v, 1);
  mat<3,3> A1 = load<mat<3,3>>(m, 1);
  for (uint32_t i = 0; i < 3; i++) {
    EXPECT_EQ(v(0, i), 0.0);
    EXPECT_EQ(b1[i], b[i]);
    for (uint32_t j = 0; j < 3; j++) {
      EXPECT_EQ(m(0, i, j), 0.0);
      EXPECT_EQ(A1(i, j), A(i, j));
    }
  }
}

// the host loop runs on the calling thread below impl::serial_below entries
// and on the pool above it; both have to give the same answer as a plain loop
static double twice_the_sum(const vec<3> & v) { return 2.0 * (v[0] + v[1] + v[2]); }

TEST(Forall, both_sides_of_the_serial_threshold) {
  for (uint32_t n : {100u, 2 * femto::impl::serial_below}) {
    nd::cpu_array<double, 2> v({n, 3});
    for (uint32_t i = 0; i < 3 * n; i++) { v.data()[i] = 0.25 * (i % 11); }
    nd::cpu_array<double, 2> out = forall(twice_the_sum, v);
    ASSERT_EQ(out.shape[0], n);
    for (uint32_t i = 0; i < n; i++) {
      EXPECT_EQ(out(i, 0), 2.0 * (v(i, 0) + v(i, 1) + v(i, 2))) << "n = " << n << ", i = " << i;
    }
  }
}
