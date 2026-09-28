// evaluate() and the residual form of integrate() return writers rather than
// freshly allocated results (nd::array::writer, Residual::writer), so that
// assigning into a destination that is already the right size refills it in
// place.  Two things have to hold, and this checks both: the destination's
// buffer is not reallocated, and the values are exactly what a newly
// constructed destination would have received.
#include "common.hpp"

#include <gtest/gtest.h>

#include "femto/mesh.hpp"
#include "femto/domain.hpp"

#include "forall.hpp"

using namespace femto;

static vec<3> flux(const vec<3> & du_dxi, const mat<3,3> & dX_dxi) {
  mat<3,3> dxi_dX = inv(dX_dxi);
  return dot(dot(du_dxi, dxi_dX), transpose(dxi_dX)) * det(dX_dxi);
}

static void reuses_destination(std::string filename) {

  auto mesh = Mesh< memory::space::cpu >::load(FEMTO_MESH_DIR + filename);

  Field u = create_field<Family::H1>(mesh, 2, 1);
  u.data = femto::random(u.data.shape);

  BasisFunction<Family::H1> phi(2, 1);
  Domain domain(mesh, MeshQuadratureRule(2));

  nd::cpu_array<double, 3> dX_dxi_q = evaluate(grad(mesh.X), isoparametric(domain));

  // first assignment allocates, second must not
  nd::cpu_array<double, 3> du_dxi_q = evaluate(grad(u), isoparametric(domain));
  const double * values_before = du_dxi_q.data();
  du_dxi_q = evaluate(grad(u), isoparametric(domain));
  EXPECT_EQ(du_dxi_q.data(), values_before) << "evaluate() reallocated its destination";

  auto f_q = forall(flux, du_dxi_q, dX_dxi_q);

  Residual<Family::H1> r = integrate(dot(f_q, grad(phi)), isoparametric(domain));
  const double * data_before = r.data.data();
  r = integrate(dot(f_q, grad(phi)), isoparametric(domain));
  EXPECT_EQ(r.data.data(), data_before) << "integrate() reallocated its destination";

  // and a refilled destination has to match a brand new one -- in particular
  // the residual must have been zeroed, since integrate_residual accumulates
  nd::cpu_array<double, 3> du_dxi_fresh = evaluate(grad(u), isoparametric(domain));
  Residual<Family::H1> r_fresh = integrate(dot(f_q, grad(phi)), isoparametric(domain));

  ASSERT_EQ(du_dxi_q.sz, du_dxi_fresh.sz);
  for (uint32_t i = 0; i < du_dxi_q.sz; i++) {
    EXPECT_EQ(du_dxi_q.data()[i], du_dxi_fresh.data()[i]) << "evaluate() differs at " << i;
  }

  ASSERT_EQ(r.size(), r_fresh.size());
  for (uint32_t i = 0; i < r.size(); i++) {
    EXPECT_EQ(r.v()[i], r_fresh.v()[i]) << "integrate() differs at " << i;
  }
}

TEST(DeferredWriter, tets)  { reuses_destination("patch_test_tets.json"); }
TEST(DeferredWriter, hexes) { reuses_destination("patch_test_hexes.json"); }
