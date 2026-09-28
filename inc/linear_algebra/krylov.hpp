#pragma once

#include <tuple>
#include <vector>

template< typename LinearOperatorTypeM, typename LinearOperatorTypeA, typename VectorType >
VectorType pcg(
  const LinearOperatorTypeM & M, 
  const LinearOperatorTypeA & A, 
  const VectorType & b,
  int imax, 
  double epsilon) {

  VectorType x = b * 0.0;
  VectorType q = b;
  VectorType r = b;
  VectorType d = M(r);
  double delta = dot(r, d);
  double delta0 = delta;

  int i = 0;
  while (i < imax && delta > ((epsilon * epsilon) * delta0)) {
    q = A(d);
    double alpha = delta / dot(d, q);
    x = x + alpha * d;

    if (i % 50 == 0) {
      r = b - A(x);
    } else {
      r = r - alpha * q;
    }

    q = M(r);

    double delta_old = delta;
    delta = dot(r, q);

    double beta = delta / delta_old;
    d = q + beta * d;
    i++;
  }

  return x;
}

template< typename LinearOperatorTypeA, typename VectorType >
VectorType cg(const LinearOperatorTypeA & A, const VectorType & b, int imax, double epsilon) {
  return pcg([](const VectorType & x){ return x; }, A, b, imax, epsilon);
}