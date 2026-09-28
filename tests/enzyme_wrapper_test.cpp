#if FEMTO_ENABLE_ENZYME
#include "gtest/gtest.h"

#include "misc/enzyme_wrapper.hpp"

#include <stdio.h>

extern double __enzyme_autodiff(void*, double);

using namespace femto;

TEST(enzyme, basic_test) {

  constexpr auto square = [](double x) { return x * x; };
  auto dsquare = [=](double x){ return jvp<+square>(x)(1.0); };

  for(double i=1; i<5; i++) {
    EXPECT_EQ(square(i), i*i);
    EXPECT_EQ(dsquare(i), 2*i);
  }
}

double f1(double x) { return x * x; }
double f2(double x, double y) { return x * sin(y); }

TEST(enzyme, 1x1_jacobian_test) {
  for(double i=1; i<5; i++) {
    // jacobian<> takes a function, f, 
    // and returns a lambda that evaluates the jacobian of f
    auto J1 = jacobian<f1>(); 
    EXPECT_EQ(f1(i), i*i);
    EXPECT_EQ(J1(i) * 1, 2*i); // jvp
    EXPECT_EQ(1 * J1(i), 2*i); // vjp
  }
}

TEST(enzyme, 1x2_jacobian_test) {
  for(double i=1; i<5; i++) {
    double x = i;
    double y = 2.0;
    auto J2 = jacobian<f2>()(x, y); // this time, we evaluate J2 immediately
    EXPECT_EQ(f2(i, i), i * sin(i));
    EXPECT_EQ((J2 * tuple{1.0, 0.0}), sin(y));      // jvp
    EXPECT_EQ(1 * J2, (tuple{sin(y), x * cos(y)})); // vjp
  }
}
#endif