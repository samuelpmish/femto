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
// that integrate(dot(P, grad(phi))) wants, and the new sigma.  jac() is dP/dF
// through the update, for kinematics with a Linearization (HughesWinget,
// RashidNewton).
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

  // dP = dsigma J F^-T + sigma d(J F^-T), with d(J F^-T) = tr(F^-1 dF) J F^-T
  // - J F^-T dF^T F^-T, dsigma = dR S R^T + R dS R^T + R S dR^T for
  // S = sigma_old + C : d, and (dR, dd) from the kinematics along df = dF F_n^-1.
  // Everything but dF is set up once, for the kinematics with a Linearization
  // (RashidNewton): jac() applies it in the 9 directions, jvp() in one
  struct Linearization {
    double lambda, mu;
    typename kinematics::Linearization step;
    mat3 inv_Fn, inv_F, RS, SRt, sigma, cof;

    __host__ __device__ Linearization(double lambda_, double mu_, const mat3 & du_dX, const mat3 & du_dX_old)
      : lambda(lambda_), mu(mu_), step(dot(Identity<3>() + du_dX, inv(Identity<3>() + du_dX_old))) {
      mat3 F = Identity<3>() + du_dX;
      inv_Fn = inv(Identity<3>() + du_dX_old);
      inv_F = inv(F);
      cof = transpose(adj(F));
    }

    __host__ __device__ void set_stress(const mat3 & sigma_old) {
      mat3 S = sigma_old + lambda * tr(step.d) * Identity<3>() + 2.0 * mu * step.d;
      RS = dot(step.R, S);
      SRt = dot(S, transpose(step.R));
      sigma = dot(RS, transpose(step.R));
    }

    __host__ __device__ mat3 operator()(const mat3 & dF) const {
      auto [dR, dd] = step.derivative(dot(dF, inv_Fn));
      mat3 dS = lambda * tr(dd) * Identity<3>() + 2.0 * mu * dd;
      mat3 dsigma = dot(dR, SRt) + dot(step.R, dot(dS, transpose(step.R))) + dot(RS, transpose(dR));
      mat3 dcof = tr(dot(inv_F, dF)) * cof - dot(cof, dot(transpose(dF), transpose(inv_F)));
      return dot(dsigma, cof) + dot(sigma, dcof);
    }
  };

  __host__ __device__ Linearization linearize(const mat3 & du_dX, const mat3 & du_dX_old, const mat3 & sigma_old) const {
    Linearization L(lambda, mu, du_dX, du_dX_old);
    L.set_stress(sigma_old);
    return L;
  }

  __host__ __device__ mat3 jvp(const mat3 & du_dX, const mat3 & du_dX_old, const mat3 & sigma_old, const mat3 & ddu_dX) const {
    return linearize(du_dX, du_dX_old, sigma_old)(ddu_dX);
  }

  // not self-adjoint, so the (k, l) direction fills dP_dF[.][.][k][l]
  __host__ __device__ mat< 3, 3, mat3 > jac(const mat3 & du_dX, const mat3 & du_dX_old, const mat3 & sigma_old) const {
    Linearization L = linearize(du_dX, du_dX_old, sigma_old);
    mat< 3, 3, mat3 > dP_dF;
    for (int k = 0; k < 3; k++) {
      for (int l = 0; l < 3; l++) {
        mat3 dF{};
        dF(k, l) = 1.0;
        mat3 dP = L(dF);
        for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 3; j++) { dP_dF[i][j][k][l] = dP(i, j); }
        }
      }
    }
    return dP_dF;
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

  // R and d at f, and their derivative along a direction df, all rational:
  // with A = (f + I)^-1, B = (I - w/2)^-1, C = (I + w/4)^-1 and s = sym(l),
  // dl = 4 A df A, dR = B dw B, d(back) = -C dw C / 2, and
  // dd = d(back) s back^T + back ds back^T + back s d(back)^T
  struct Linearization {
    mat3 R, d, A, B, C, back, s;

    __host__ __device__ Linearization(const mat3 & f) {
      auto I = Identity<3>();
      A = inv(f + I);
      mat3 l = 2.0 * I - 4.0 * A;
      mat3 w = 0.5 * (l - transpose(l));
      s = 0.5 * (l + transpose(l));
      B = inv(I - 0.5 * w);
      C = inv(I + 0.25 * w);
      R = 2.0 * B - I;
      back = 2.0 * C - I;
      d = dot(back, dot(s, transpose(back)));
    }

    __host__ __device__ Increment derivative(const mat3 & df) const {
      mat3 dl = 4.0 * dot(A, dot(df, A));
      mat3 dw = 0.5 * (dl - transpose(dl)), ds = 0.5 * (dl + transpose(dl));
      mat3 dback = -0.5 * dot(C, dot(dw, C));
      mat3 back_s = dot(back, s);
      mat3 dd = dot(dback, dot(s, transpose(back))) + dot(back, dot(ds, transpose(back))) + dot(back_s, transpose(dback));
      return {dot(B, dot(dw, B)), dd};
    }
  };
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

  // R and d at f, and their derivative along a direction df: differentiating
  // f = R U, dR = R W for the skew W whose axial vector solves
  // (tr(U) I - U) w = axial(A - A^T), A = R^T df, and dU = A - W U; then
  // d(log U) in U's eigenbasis U = V^T diag(u) V is dU's components there
  // scaled by the divided differences of log (Daleckii-Krein), one
  // eigendecomposition for the tangent where the stress needs none
  struct Linearization {
    mat3 R, d, U, inv_M, V;
    mat3 G;   // (log u_a - log u_b) / (u_a - u_b), 1 / u_a on the diagonal

    __host__ __device__ static double divided_log(double a, double b) {
      double x = (a - b) / b;
      // log1p(x) / x to rounding for |x| < 1e-4
      if (fabs(x) < 1.0e-4) { return (1.0 - x / 2.0 + x * x / 3.0 - x * x * x / 4.0) / b; }
      return log1p(x) / (a - b);
    }

    __host__ __device__ Linearization(const mat3 & f) {
      PolarDecomposition p = polar_newton(f);
      R = p.R;
      U = p.U;
      d = logm_series(p.U);
      inv_M = inv(tr(U) * Identity<3>() - U);
      Eigensystem<3> e = eig(p.U);
      V = e.X;
      for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) { G(a, b) = divided_log(e.lambda[a], e.lambda[b]); }
      }
    }

    __host__ __device__ Increment derivative(const mat3 & df) const {
      mat3 A = dot(transpose(R), df);
      vec3 w = dot(inv_M, vec3{A(2, 1) - A(1, 2), A(0, 2) - A(2, 0), A(1, 0) - A(0, 1)});
      mat3 W = skew3{w};
      mat3 dU = dot(V, dot(A - dot(W, U), transpose(V)));
      for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) { dU(a, b) *= G(a, b); }
      }
      return {dot(R, W), dot(transpose(V), dot(dU, V))};
    }
  };
};

} // namespace femto
