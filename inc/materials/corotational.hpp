#pragma once

#include "fm/types/matrix.hpp"
#include "fm/operations/adjugate.hpp"

#include "linear_algebra/polar_decomp.hpp"

namespace femto {

using namespace fm;

// a step's rotation and strain increment
struct Increment {
  mat3 R;
  mat3 d;
};

// Linear hypoelasticity integrated one step at a time, with the stress updates
// of hidden/IncrementalKinematicsExampleProblems.nb.  The Cauchy stress follows
//
//   sigma' - W sigma + sigma W = lambda tr(D) I + 2 mu D,
//
// linear elasticity in a frame spinning with W, and a step takes the stress
// at the last step and the step's relative deformation gradient
// f = F_{n+1} F_n^-1 to
//
//   sigma_{n+1} = R (sigma_n + lambda tr(d) I + 2 mu d) R^T,
//
// where the kinematics take a rotation R and a strain increment d ~ D dt from f:
//   HughesWinget : the midpoint velocity gradient l = 2 (f - I)(f + I)^-1,
//                  d = sym(l), and R = (I + w/2)(I - w/2)^-1 for w = skew(l)
//   Rashid       : f = R U and d = log(U), with the polar decomposition and
//                  the log two ways: RashidEig and RashidNewton
// All are exact for a rigid rotation and second order in dt.
//
// Same convention as ScalarPML: operator() takes the last step's du_dX and
// sigma, which it never modifies, and gives the PK1 stress P = J sigma F^-T
// that integrate(dot(P, grad(phi))) wants, and the new sigma.  No jac() yet.
template < typename kinematics >
struct CorotationalModel {

  __host__ __device__ void operator()(const mat3 & du_dX, const mat3 & du_dX_old, const mat3 & sigma_old, mat3 & P, mat3 & sigma) const {
    auto I = Identity<3>();
    mat3 F = I + du_dX;
    auto [R, d] = kinematics{}(dot(F, inv(I + du_dX_old)));
    sigma = dot(R, dot(sigma_old + lambda * tr(d) * I + 2.0 * mu * d, transpose(R)));
    // J F^-T = adj(F)^T
    P = dot(sigma, transpose(adj(F)));
  }

  double lambda, mu;

};

// d = sym(l) belongs to the step's midpoint configuration, and the update
// carries it forward by the whole step's R, so it goes in rotated back half a
// step first, by cay(-w/2) with cay(v) = (I + v/2)(I - v/2)^-1: carried
// forward by R, it lands as rotated by half a step to second order.  (The
// notebook adds d as it is, which leaves it half a step's rotation off and
// the update first order.)  With (f - I)(f + I)^-1 = I - 2 (f + I)^-1 and
// cay(v) = 2 (I - v/2)^-1 - I, that is three 3x3 inverses.
struct HughesWinget {
  __host__ __device__ Increment operator()(const mat3 & f) const {
    auto I = Identity<3>();
    mat3 l = 2.0 * I - 4.0 * inv(f + I);
    mat3 w = 0.5 * (l - transpose(l));
    mat3 back = 2.0 * inv(I + 0.25 * w) - I;
    return {2.0 * inv(I - 0.5 * w) - I, dot(back, dot(0.5 * (l + transpose(l)), transpose(back)))};
  }
};

// R and log(U) from one eigendecomposition, f^T f = U^2 = V^T diag(c) V:
// log(U) = V^T diag(log(c) / 2) V and R = f U^-1, U^-1 = V^T diag(c^(-1/2)) V.
// Newton's R alone is 3.5x cheaper than polar_eig's, but with logm(U) after
// it, 10-15% dearer than this (tests/performance/material_perf.cpp).
struct RashidEig {
  __host__ __device__ Increment operator()(const mat3 & f) const {
    Eigensystem<3> C = eig(sym3(dot(transpose(f), f)));
    vec3 inv_s = {1.0 / sqrt(C.lambda[0]), 1.0 / sqrt(C.lambda[1]), 1.0 / sqrt(C.lambda[2])};
    vec3 log_s = {0.5 * log(C.lambda[0]), 0.5 * log(C.lambda[1]), 0.5 * log(C.lambda[2])};
    mat3 Vt = transpose(C.X);
    return {dot(f, dot(Vt, dot(diag3{inv_s}, C.X))), dot(Vt, dot(diag3{log_s}, C.X))};
  }
};

// R from polar_newton and log(U) from logm_series, neither an
// eigendecomposition: a step's U is close enough to I for the series to
// take a few terms
struct RashidNewton {
  __host__ __device__ Increment operator()(const mat3 & f) const {
    auto [R, U] = polar_newton(f);
    return {R, logm_series(U)};
  }
};

} // namespace femto
