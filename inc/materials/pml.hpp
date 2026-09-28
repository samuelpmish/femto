#pragma once

#include <cmath>

#include "fm/types/vec.hpp"
#include "fm/types/matrix.hpp"

namespace femto {

using namespace fm;

// The perfectly matched layer for the scalar wave equation u_tt = c^2 div(grad u),
// in the auxiliary-differential-equation form of docs/absorbing_boundaries.
//
// In the frame where the coordinate stretch is diagonal (the rows of R are
// the stretch directions, for a radial layer rhat and its tangents), each
// coordinate is stretched by s_i = 1 + alpha_i / s, with s the Laplace
// variable and alpha_i >= 0 the damping profile.  The stretched equation has
// the mass factor det S = prod_i s_i and the flux
//
//   c^2 R^T diag(Lambda_i) R grad u,    Lambda_i = prod_{j != i} s_j / s_i,
//
// so the PML at a quadrature point is the coefficient Lambda_i applied to each
// frame component of grad u.  Lambda_i is a product of dim - 1 factors
// (s + a) / (s + b), and each factor is the transfer function of one
// first-order ODE with one auxiliary variable,
//
//   y = x + eta,    eta' = -b eta + (a - b) x,
//
// the factors of one component applied in cascade, each driven by the output
// of the one before.  The state per point is dim (dim - 1) auxiliaries: in 2D
// one per component, Lambda_r = s_t / s_r; in 3D two, Lambda_r = s_t^2 / s_r.
//
// Two ways to advance it:
//   ScalarPML::Rate   the flux and the rate of the auxiliaries, for an explicit
//                     integrator that advances them itself (wave_disk_2D)
//   ScalarPML         the flux and the new auxiliaries after a backward Euler
//                     step over dt from the old ones, which it never modifies,
//                     so a Newton iterate can start again from them.  jac() is
//                     the total derivative of the flux through that update,
//                     c^2 R^T diag(Lambda_i(1/dt)) R, the coefficient K_q of
//                     integrate(dot(grad(phi), K_q, grad(phi)))
//
// The mass side, det S applied to u_tt, is a nodal term the caller applies
// with alpha at the nodes: u_tt + (alpha_r + alpha_t) u_t + alpha_r alpha_t u
// in 2D, with an extra alpha_r alpha_t^2 int u dt in 3D.
//
// ponytail: backward Euler in the auxiliaries, first order in dt.  The upgrade
// is a theta-method in pml::Factor::update / tangent.
namespace pml {

// one factor (s + a) / (s + b): y = x + eta, eta' = -b eta + (a - b) x
struct Factor {
  double a, b;
  double rate(double x, double eta) const { return (a - b) * x - b * eta; }
  double update(double x, double eta_old, double dt) const { return (eta_old + dt * (a - b) * x) / (1.0 + dt * b); }
  double tangent(double dt) const { return (1.0 + dt * a) / (1.0 + dt * b); }
};

// the radial layer: a frame of the radial direction and its tangents, and
// the profile alpha_r, quadratic across a layer of the given thickness outside
// radius R, with its running mean alpha_t = (1/r) int_R^r alpha_r on every
// tangential direction
template < int dim >
struct Radial {
  vec<dim> center;
  double R, thickness, alpha_max;

  vec<dim> alpha(const vec<dim> & x) const {
    double r = norm(x - center), d = r - R;
    vec<dim> a{};
    if (d <= 0 || thickness <= 0) { return a; }
    a[0] = alpha_max * std::pow(d / thickness, 2);
    for (int i = 1; i < dim; i++) { a[i] = alpha_max * std::pow(d, 3) / (3.0 * thickness * thickness * r); }
    return a;
  }

  mat<dim, dim> frame(const vec<dim> & x) const {
    vec<dim> rel = x - center;
    vec<dim> rhat = rel / std::max(norm(rel), 1e-12);
    mat<dim, dim> F{};
    F[0] = rhat;
    if constexpr (dim == 2) {
      F[1] = cross(rhat);
    } else {
      // the axis least aligned with rhat seeds the first tangent
      int k = 0;
      for (int i = 1; i < 3; i++) { if (std::fabs(rhat[i]) < std::fabs(rhat[k])) { k = i; } }
      vec3 e{}; e[k] = 1.0;
      F[1] = normalize(cross(rhat, e));
      F[2] = cross(rhat, F[1]);
    }
    return F;
  }
};

} // namespace pml

template < int dim >
struct ScalarPML {

  static constexpr int nfactors = dim - 1;
  static constexpr int nstate = dim * nfactors;

  double c2;    // wave speed squared
  double dt;    // the step the auxiliaries are advanced over

  // the k-th factor of Lambda_i = prod_{j != i} (s + alpha_j) / (s^(dim - 2) (s + alpha_i)):
  // (s + alpha_j) / (s + alpha_i) for the first j != i, (s + alpha_j) / s for the rest
  static pml::Factor factor(const vec<dim> & alpha, int i, int k) {
    int j = k < i ? k : k + 1;
    return pml::Factor{alpha[j], k == 0 ? alpha[i] : 0.0};
  }

  // the flux c^2 (grad u + the auxiliaries, back in x), and the new auxiliaries
  void operator()(const mat<dim, dim> & R, const vec<dim> & alpha, const vec<dim> & du, const vec<nstate> & aux_old, vec<dim> & flux, vec<nstate> & aux) const {
    flux = du;
    for (int i = 0; i < dim; i++) {
      double y = dot(du, R[i]), sum = 0.0;
      for (int k = 0; k < nfactors; k++) {
        int m = i * nfactors + k;
        aux[m] = factor(alpha, i, k).update(y, aux_old[m], dt);
        y += aux[m];
        sum += aux[m];
      }
      flux = flux + sum * R[i];
    }
    flux = c2 * flux;
  }

  // the same system as a rate: the flux and the rate of the auxiliaries
  struct Rate {
    double c2;
    void operator()(const mat<dim, dim> & R, const vec<dim> & alpha, const vec<dim> & du, const vec<nstate> & aux, vec<dim> & flux, vec<nstate> & auxdot) const {
      flux = du;
      for (int i = 0; i < dim; i++) {
        double y = dot(du, R[i]), sum = 0.0;
        for (int k = 0; k < nfactors; k++) {
          int m = i * nfactors + k;
          auxdot[m] = factor(alpha, i, k).rate(y, aux[m]);
          y += aux[m];
          sum += aux[m];
        }
        flux = flux + sum * R[i];
      }
      flux = c2 * flux;
    }
  };

  // the total derivative of the flux with respect to grad u through the update:
  // Lambda_i at s = 1 / dt on each frame component
  mat<dim, dim> jac(const mat<dim, dim> & R, const vec<dim> & alpha) const {
    vec<dim> t{};
    for (int i = 0; i < dim; i++) {
      t[i] = 1.0;
      for (int k = 0; k < nfactors; k++) { t[i] *= factor(alpha, i, k).tangent(dt); }
    }
    return c2 * dot(transpose(R), dot(diag<dim>{t}, R));
  }

  // its action on a direction, and the adjoint (jac is symmetric)
  vec<dim> jvp(const mat<dim, dim> & R, const vec<dim> & alpha, const vec<dim> & ddu) const { return dot(jac(R, alpha), ddu); }
  vec<dim> vjp(const mat<dim, dim> & R, const vec<dim> & alpha, const vec<dim> & dO_dflux) const { return dot(jac(R, alpha), dO_dflux); }

};

} // namespace femto
