#pragma once

#include "fm/types/matrix.hpp"

#include "materials/linear_elasticity.hpp"

namespace femto {

using namespace fm;

// small strain viscoelasticity, the standard linear solid: an equilibrium
// spring (lambda, mu) in parallel with a Maxwell branch, a spring of shear
// modulus mu_v in series with a dashpot of relaxation time tau, acting on the
// deviatoric strain.  The branch's state is its viscous strain ev, with
//
//   d(ev)/dt = (dev(e) - ev) / tau,    sigma = C(lambda, mu) e + 2 mu_v (dev(e) - ev),
//
// so a sudden shear meets mu + mu_v, relaxing to mu over a few tau.
//
// Same convention as ScalarPML: operator() gives the stress and the viscous
// strain after a backward Euler step over dt from the old one, which it never
// modifies, so a Newton iterate can start again from it.  jac() is the total
// derivative of the stress through that update.
//
// ponytail: backward Euler, first order in dt, and a single Maxwell branch,
// where a Prony series fit to measured data has several, each with its own
// (mu_v, tau) and viscous strain.  Also: ev is stored as a full mat3, 9 values
// where a deviatoric sym3 has 5 independent ones.
struct LinearViscoelasticModel {

  double lambda, mu;   // the equilibrium (fully relaxed) moduli
  double mu_v, tau;    // the Maxwell branch: shear modulus and relaxation time
  double dt;           // the step the viscous strain is advanced over

  __host__ __device__ void operator()(const mat3 & du_dX, const mat3 & ev_old, mat3 & stress, mat3 & ev) const {
    mat3 e = 0.5 * (du_dX + transpose(du_dX));
    mat3 e_dev = e - (tr(e) / 3.0) * Identity<3>();
    double x = dt / tau;
    ev = (ev_old + x * e_dev) / (1.0 + x);
    stress = LinearElasticModel{lambda, mu}(du_dX) + 2.0 * mu_v * (e_dev - ev);
  }

  // through the update, d(dev(e) - ev) = d(dev(e)) / (1 + dt / tau): linear
  // elasticity with the branch adding shear stiffness g = mu_v / (1 + dt / tau)
  // and nothing to the bulk modulus lambda + 2 mu / 3
  __host__ __device__ mat< 3, 3, mat3 > jac(const mat3 & du_dX) const {
    double g = mu_v / (1.0 + dt / tau);
    return LinearElasticModel{lambda - 2.0 * g / 3.0, mu + g}.jac(du_dX);
  }

};

} // namespace femto
