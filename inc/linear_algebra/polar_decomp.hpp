#pragma once

#include <cmath>

#include "fm/types/matrix.hpp"

// The polar decomposition F = R U of a 3x3 F with det(F) > 0, R a rotation
// and U symmetric positive definite, two ways, and the logarithm of a
// symmetric positive definite matrix.  Nothing here but fm, so it can move
// there.
namespace femto {

using namespace fm;

struct PolarDecomposition {
  mat3 R;
  sym3 U;
};

// Newton's iteration R <- (R + R^-T) / 2 from R = F, which converges
// quadratically.  The iterates are R_k = R X_k, X_k the same iteration on U
// alone, so the iteration count depends on the stretch and not on the
// rotation: each stretch s goes to (s + 1/s) / 2, so 3 iterations for strains
// under 1%, 4 up to 10%, and 4-5 at 30%.
//
// ponytail: unscaled Newton, so a stretch far from 1 spends extra iterations
// getting into the quadratic regime (s = 10 takes 4 just to reach 1.1).  The
// upgrade is Higham's scaling, R <- (g R + (g R)^-T) / 2 with g = |det R|^(-1/3).
__host__ __device__ inline PolarDecomposition polar_newton(const mat3 & F) {
  mat3 R = F;
  for (int k = 0; k < 32; k++) {
    mat3 R_next = 0.5 * (R + transpose(inv(R)));
    double change = norm(R_next - R);
    R = R_next;
    // the error after a step is about half the square of the change
    if (change < 1.0e-8) break;
  }
  return {R, sym3(dot(transpose(R), F))};
}

// from the eigendecomposition F^T F = U^2 = V^T diag(s^2) V, fm::eig giving
// the eigenvectors as the rows of V: U = V^T diag(s) V and R = F U^-1, at a
// fixed cost whatever the stretch.  Working with F^T F squares the condition
// number, so R is orthogonal to about eps cond(F)^2 where Newton's is to eps:
// the same for F near a rotation, like a step's relative deformation
// gradient, but 2e-14 for stretches of 6 and 0.2.
__host__ __device__ inline PolarDecomposition polar_eig(const mat3 & F) {
  Eigensystem<3> C = eig(sym3(dot(transpose(F), F)));
  vec3 s = {sqrt(C.lambda[0]), sqrt(C.lambda[1]), sqrt(C.lambda[2])};
  mat3 Vt = transpose(C.X);
  mat3 inv_U = dot(Vt, dot(diag3{{1.0 / s[0], 1.0 / s[1], 1.0 / s[2]}}, C.X));
  return {dot(F, inv_U), sym3(dot(Vt, dot(diag3{s}, C.X)))};
}

// the logarithm of a symmetric positive definite A = V^T diag(a) V, V^T diag(log(a)) V
__host__ __device__ inline sym3 logm(const sym3 & A) {
  Eigensystem<3> e = eig(A);
  vec3 log_a = {log(e.lambda[0]), log(e.lambda[1]), log(e.lambda[2])};
  return sym3(dot(transpose(e.X), dot(diag3{log_a}, e.X)));
}

// the same for A near I, by the series log(A) = 2 artanh(Z) = 2 (Z + Z^3/3 +
// Z^5/5 + ...), Z = (A - I)(A + I)^-1: each eigenvalue a of A contributes
// terms shrinking by ((a - 1)/(a + 1))^2, so 4 terms for a step's strain of
// 0.2% and 6 for 2%, where logm's eigendecomposition costs more than 16 would.
// Past 16 (an eigenvalue outside about 0.53 to 1.87) it hands A to logm.
__host__ __device__ inline sym3 logm_series(const sym3 & A) {
  auto I = Identity<3>();
  mat3 a = A;   // fm's inv has no sym3 case
  mat3 Z = dot(a - I, inv(a + I));
  mat3 Z2 = dot(Z, Z), term = Z, sum = Z;
  for (int k = 1; k < 16; k++) {
    term = dot(term, Z2);
    mat3 next = term / (2.0 * k + 1.0);
    sum = sum + next;
    if (norm(next) <= 1.0e-17 * norm(sum)) { return sym3(2.0 * sum); }
  }
  return logm(A);
}

} // namespace femto
