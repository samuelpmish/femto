#include <gtest/gtest.h>

#include <fstream>
#include <cstdio>

#include "containers/dual.hpp"
#include "fm/types/vec.hpp"
#include "fm/types/matrix.hpp"

using namespace fm;

static constexpr double tolerance = 1.0e-14;

bool approximately_equal(double x, double y) {
  return fabs(x - y) < tolerance;
}

template < uint32_t n >
bool approximately_equal(vec<n> x, vec<n> y) {
  return norm(x - y) < tolerance;
}

template < uint32_t m, uint32_t n, typename T >
bool approximately_equal(mat<m,n> x, mat<m,n> y) {
  return norm(x - y) < tolerance;
}

template < typename S, typename T >
bool approximately_equal(dual< S, T > x, dual< S, T > y) {
  return approximately_equal(x.value, y.value) && 
         approximately_equal(x.gradient, y.gradient);
}

TEST(DualNumbers, ScalarAddition) {
  EXPECT_TRUE(approximately_equal(dual{1.0, 2.0}, dual{0.0, 2.0} + 1.0));
  EXPECT_TRUE(approximately_equal(dual{1.0, 2.0}, 1.0 + dual{0.0, 2.0}));
  EXPECT_TRUE(approximately_equal(dual{1.0, 2.0}, dual{0.5, 0.5} + dual{0.5, 1.5}));
}

TEST(DualNumbers, VectorAddition) {
  EXPECT_TRUE(approximately_equal(dual{vec3{2.0, 4.0, 6.0}, 2.0}, 
                                  dual{vec3{1.0, 2.0, 3.0}, 2.0} + vec3{1.0, 2.0, 3.0}));
  EXPECT_TRUE(approximately_equal(dual{vec3{2.0, 4.0, 6.0}, 2.0}, 
                                  vec3{1.0, 2.0, 3.0} + dual{vec3{1.0, 2.0, 3.0}, 2.0}));
  EXPECT_TRUE(approximately_equal(dual{vec3{2.0, 4.0, 6.0}, 2.0}, 
                                  dual{vec3{1.0, 2.0, 3.0}, 1.0} + dual{vec3{1.0, 2.0, 3.0}, 1.0}));
}

TEST(DualNumbers, ScalarSubtraction) {
  EXPECT_TRUE(approximately_equal(dual{-1.0, 2.0}, dual{0.0, 2.0} - 1.0));
  EXPECT_TRUE(approximately_equal(dual{1.0, -2.0}, 1.0 - dual{0.0, 2.0}));
  EXPECT_TRUE(approximately_equal(dual{0.0, -1.0}, dual{0.5, 0.5} - dual{0.5, 1.5}));
}

TEST(DualNumbers, VectorSubtraction) {
  EXPECT_TRUE(approximately_equal(dual{vec3{0.0, 0.0, 0.0}, 2.0}, 
                                  dual{vec3{1.0, 2.0, 3.0}, 2.0} - vec3{1.0, 2.0, 3.0}));
  EXPECT_TRUE(approximately_equal(dual{vec3{0.0, 0.0, 0.0}, -2.0}, 
                                  vec3{1.0, 2.0, 3.0} - dual{vec3{1.0, 2.0, 3.0}, 2.0}));
  EXPECT_TRUE(approximately_equal(dual{vec3{0.0, 0.0, 0.0}, 0.0}, 
                                  dual{vec3{1.0, 2.0, 3.0}, 1.0} - dual{vec3{1.0, 2.0, 3.0}, 1.0}));
}

TEST(DualNumbers, UnaryNegation) {
  EXPECT_TRUE(approximately_equal(dual{1.0, 2.0}, -dual{-1.0, -2.0}));
  EXPECT_TRUE(approximately_equal(dual{vec2{1.0, 2.0}, 2.0}, -dual{vec2{-1.0, -2.0}, -2.0}));
}

TEST(DualNumbers, ScalarMultiplication) {
  EXPECT_TRUE(approximately_equal(dual{2.0, 4.0}, dual{1.0, 2.0} * 2.0));
  EXPECT_TRUE(approximately_equal(dual{2.0, 7.0}, dual{1.0, 2.0} * dual{2.0, 3.0}));
  EXPECT_TRUE(approximately_equal(dual{2.0, 4.0}, 2.0 * dual{1.0, 2.0}));
}

TEST(DualNumbers, ScalarDivision) {
  EXPECT_TRUE(approximately_equal(dual{0.5, 1.0}, dual{1.0, 2.0} / 2.0));
  EXPECT_TRUE(approximately_equal(dual{0.5, 0.25}, dual{1.0, 2.0} / dual{2.0, 3.0}));
  EXPECT_TRUE(approximately_equal(dual{2.0, -4.0}, 2.0 / dual{1.0, 2.0}));
}

TEST(DualNumbers, ComparisonOperators) {
  EXPECT_TRUE((dual{-1.0, 2.0} <  dual{1.0, 3.0}));
  EXPECT_TRUE((dual{ 1.0, 2.0} <= dual{1.0, 4.0}));
  EXPECT_TRUE((dual{ 1.0, 2.0} == dual{1.0, 5.0}));
  EXPECT_TRUE((dual{ 1.0, 2.0} >= dual{1.0, 6.0}));
  EXPECT_TRUE((dual{ 2.0, 2.0} >  dual{1.0, 7.0}));
}

TEST(DualNumbers, CompoundAssignmentOperators) {
  dual x{1.0, 2.0};

  x += dual{1.0, 1.0};
  EXPECT_NEAR(x.value, 2.0, tolerance);
  EXPECT_NEAR(x.gradient, 3.0, tolerance);

  x -= dual{1.0, 1.0};
  EXPECT_NEAR(x.value, 1.0, tolerance);
  EXPECT_NEAR(x.gradient, 2.0, tolerance);

  x += 1.0;
  EXPECT_NEAR(x.value, 2.0, tolerance);
  EXPECT_NEAR(x.gradient, 2.0, tolerance);

  x -= 1.0;
  EXPECT_NEAR(x.value, 1.0, tolerance);
  EXPECT_NEAR(x.gradient, 2.0, tolerance);
}

TEST(DualNumbers, TrigFunctions) {
  dual x{1.0, 2.0};

  EXPECT_NEAR(sin(x).value, sin(1.0), tolerance);
  EXPECT_NEAR(sin(x).gradient, 2.0 * cos(1.0), tolerance);

  EXPECT_NEAR(cos(x).value, cos(1.0), tolerance);
  EXPECT_NEAR(cos(x).gradient, -2.0 * sin(1.0), tolerance);
}

TEST(DualNumbers, ExpLog) {
  dual x{2.0, 3.0};

  EXPECT_NEAR(exp(x).value, exp(2.0), tolerance);
  EXPECT_NEAR(exp(x).gradient, 3.0 * exp(2.0), tolerance);

  EXPECT_NEAR(log(x).value, log(2.0), tolerance);
  EXPECT_NEAR(log(x).gradient, 3.0 / 2.0, tolerance);
}

TEST(DualNumbers, PowSqrtAbs) {
  dual x{2.0, 3.0};
  dual y{3.0, 4.0};

  EXPECT_NEAR(pow(x, 2.0).value, 4.0, tolerance);
  EXPECT_NEAR(pow(x, 2.0).gradient, 12.0, tolerance);

  EXPECT_NEAR(pow(2.0, x).value, 4.0, tolerance);
  EXPECT_NEAR(pow(2.0, x).gradient, 12.0 * log(2.0), tolerance);

  EXPECT_NEAR(pow(y, x).value, 9.0, tolerance);
  EXPECT_NEAR(pow(y, x).gradient, 24 + 27 * log(3.0), 10.0 * tolerance); // TODO: why is this op less accurate

  EXPECT_NEAR(sqrt(x).value, sqrt(2.0), tolerance);
  EXPECT_NEAR(sqrt(x).gradient, 3.0 * (0.5 / sqrt(2.0)), tolerance);

  EXPECT_NEAR(abs(x).value, 2.0, tolerance);
  EXPECT_NEAR(abs(x).gradient, 3.0, tolerance);

  EXPECT_NEAR(abs(-x).value, 2.0, tolerance);
  EXPECT_NEAR(abs(-x).gradient, 3.0, tolerance);
}

TEST(DualNumbers, HessianScalar) {
  auto f = [](auto x) { return sin(x * 0.5); };

  {
    auto output = f(dual{2.0, 3.0});
    EXPECT_NEAR(output.value, sin(1.0), tolerance);
    EXPECT_NEAR(output.gradient, 1.5 * cos(1.0), tolerance);
  }

  {
    auto output = f(hessian_wrt(2.0));
    EXPECT_NEAR(output.value.value, sin(1.0), tolerance);
    EXPECT_NEAR(output.value.gradient, 0.5 * cos(1.0), tolerance);
    EXPECT_NEAR(output.gradient.gradient, -0.25 * sin(1.0), tolerance);
  }
}

#if 0
TEST(DualNumbers, GradientVector) {

  double k = 2.0;
  vec2 v{1.0, 2.0};

  auto f = [&](auto du_dxi, auto dx_dxi) {
    mat2 dxi_dx = inv(dx_dxi);
    auto du_dx = dot(du_dxi, dxi_dx);
    auto heat_flux = -k * du_dx;
    return dot(heat_flux, transpose(dxi_dx)) * det(dx_dxi);
  };

  vec2 du_dxi{1.0, 2.0};
  mat2 dx_dxi{{{1.0, 0.1},{-0.2, 1.0}}};

  auto output = f(gradient_wrt(du_dxi), dx_dxi);
  mat2 answer = -k * dot(inv(dx_dxi), inv(transpose(dx_dxi))) * det(dx_dxi);

  EXPECT_NEAR(output[0].gradient[0], answer(0,0), tolerance);
  EXPECT_NEAR(output[1].gradient[0], answer(1,0), tolerance);
  EXPECT_NEAR(output[0].gradient[1], answer(0,1), tolerance);
  EXPECT_NEAR(output[1].gradient[1], answer(1,1), tolerance);

}

TEST(DualNumbers, HessianVector) {

  vec2 v{1.0, 2.0};

  auto f = [](auto v) { return sin(dot(v,v) * 0.5); };

  EXPECT_NEAR(f(hessian_wrt(v)).value.value, f(v), tolerance);

  // f[v_] := Sin[1/2 v.v];
  // FullSimplify[D[f[{v1, v2}], {{v1, v2}}] /. {v1 -> 1, v2 -> 2}] 
  // 
  // {Cos[5/2], 2 Cos[5/2]}
  EXPECT_NEAR(f(hessian_wrt(v)).gradient[0].value, cos(2.5), tolerance);
  EXPECT_NEAR(f(hessian_wrt(v)).gradient[1].value, 2.0 * cos(2.5), tolerance);
  EXPECT_NEAR(f(hessian_wrt(v)).value.gradient[0], cos(2.5), tolerance);
  EXPECT_NEAR(f(hessian_wrt(v)).value.gradient[1], 2.0 * cos(2.5), tolerance);

  // FullSimplify[D[f[{v1, v2}], {{v1, v2}, 2}] /. {v1 -> 1, v2 -> 2}]
  // 
  // {{Cos[5/2] - Sin[5/2], -2 Sin[5/2]}, {-2 Sin[5/2],  Cos[5/2] - 4 Sin[5/2]}}
  EXPECT_NEAR(f(hessian_wrt(v)).gradient[0].gradient[0], cos(2.5) - sin(2.5), tolerance);
  EXPECT_NEAR(f(hessian_wrt(v)).gradient[0].gradient[1], -2.0 * sin(2.5), tolerance);
  EXPECT_NEAR(f(hessian_wrt(v)).gradient[1].gradient[0], -2.0 * sin(2.5), tolerance);
  EXPECT_NEAR(f(hessian_wrt(v)).gradient[1].gradient[1], cos(2.5) - 4.0 * sin(2.5), tolerance);

}
#endif