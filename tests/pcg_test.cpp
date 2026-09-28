#include <gtest/gtest.h>

#include "linear_algebra/vector.hpp"
#include "linear_algebra/krylov.hpp"
#include "misc/timer.hpp"

#include <iostream>

auto HilbertOperator(int n) {
  return [=](const femto::vector & x){
    femto::vector Ax(n);
    for (int i = 0; i < n; i++) {
      double total = 0.0;
      for (int j = 0; j < n; j++) {
        double A_ij = 1.0 / (i + j + 1);
        total += A_ij * x[j];
      }
      Ax[i] = total;
    }
    return Ax;
  };
}

auto LaplaceOperator1D(int n) {
  return [=](const femto::vector & x){
    femto::vector Ax(n);
    for (int i = 0; i < n; i++) {
      double total = 2 * x[i];
      if (i-1 >= 0) total -= x[i-1];
      if (i+1  < n) total -= x[i+1];
      Ax[i] = total;
    }
    return Ax;
  };
}

////////////////////////////////////////////////////////////////////////////////

TEST(cg, Hilbert8) {

  int n = 8;

  auto H = HilbertOperator(n);

  femto::vector ones(n);
  for (int i = 0; i < n; i++) {
    ones[i] = 1;
  }

  femto::vector b = H(ones);

  femto::vector x = cg(H, b, n, 1.0e-16);

  EXPECT_NEAR(norm(x - ones) / norm(ones), 0.0, 1.0e-3);

}

TEST(cg, Laplace64) {

  int n = 64;

  auto H = LaplaceOperator1D(n);

  femto::vector ones(n);
  for (int i = 0; i < n; i++) {
    ones[i] = 1;
  }

  femto::vector b = H(ones);

  femto::vector x = cg(H, b, n / 2, 1.0e-16);

  EXPECT_NEAR(norm(x - ones) / norm(ones), 0.0, 1.0e-8);

}

TEST(pcg, Laplace64) {

  int n = 64;

  auto H = LaplaceOperator1D(n);

  femto::vector ones(n);
  for (int i = 0; i < n; i++) {
    ones[i] = 1;
  }

  femto::vector b = H(ones);

  auto M = [n](const femto::vector & x){
    return x;
  };

  femto::vector x = pcg(M, H, b, n / 2, 1.0e-16);

  EXPECT_NEAR(norm(x - ones) / norm(ones), 0.0, 1.0e-8);

}
