#pragma once

// solid mechanics analysis routines for the coffee mug modal example.
//
// header-only so it can be compiled both into the example (native or wasm) and
// into native unit tests. Everything is expressed with femto's public API
// (fields, integrate, sparse matrices, direct solvers).
//
// units are CGS: lengths in cm, density in g/cm^3, moduli in dyn/cm^2
// (1 GPa = 1e10 dyn/cm^2), so time is in seconds and the eigenvalues of
// (K, M) are squared angular frequencies in (rad/s)^2.

#include <cmath>
#include <chrono>
#include <memory>
#include <vector>
#include <algorithm>

#include "femto/mesh.hpp"
#include "femto/field.hpp"
#include "femto/domain.hpp"

#include "linear_algebra/sparse_matrix.hpp"
#include "linear_algebra/sparse_direct.hpp"

#include "forall.hpp"
#include "materials/linear_elasticity.hpp"

namespace mug {

using femto::vector;
using femto::vec3;
using femto::mat3;

inline double milliseconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

struct Analysis {
  femto::sparse_matrix<> M;
  femto::sparse_matrix<> K;

  nd::cpu_array<double, 2> nodes;  // dof node coordinates (vertices first)
  uint32_t num_verts;
  uint32_t ndof;

  double assemble_ms;   // field setup + both assemblies
  double setup_ms;      // field / domain construction
  double K_ms;          // stiffness assembly
  double M_ms;          // mass assembly
  double lambda_max = 0;  // largest eigenvalue of (K, M_L), for timestep selection
};

// assemble the consistent mass matrix and the linear elastic stiffness matrix
// on the mesh's own H1 space
inline Analysis analyze(const femto::Mesh<> & mesh, double rho, double E, double nu) {

  Analysis out;

  double lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
  double mu = E / (2 * (1 + nu));

  auto t0 = std::chrono::steady_clock::now();

  uint32_t p = mesh.X.degree;
  femto::Field u = femto::create_field<femto::Family::H1>(mesh, p, 3);
  nd::zero(u.data);

  femto::BasisFunction phi(u);
  femto::Domain domain(mesh, femto::MeshQuadratureRule(p + 1));

  out.setup_ms = milliseconds_since(t0);
  auto t_K = std::chrono::steady_clock::now();

  // stiffness: C_ijkl = lambda d_ij d_kl + mu (d_ik d_jl + d_il d_jk)
  {
    nd::cpu_array<double, 3> du_dX_q = femto::evaluate(femto::grad(u), domain);
    femto::LinearElasticModel material{lambda, mu};
    auto C_q = femto::forall(std::function<fm::mat<3, 3, mat3>(const mat3 &)>(
      [=](const mat3 & du_dX) { return material.jac(du_dX); }), du_dX_q);
    out.K = femto::sparse_matrix<>(femto::integrate(femto::dot(femto::grad(phi), C_q, femto::grad(phi)), domain));
  }

  out.K_ms = milliseconds_since(t_K);
  auto t_M = std::chrono::steady_clock::now();

  // mass: rho * identity coupling the displacement components
  {
    nd::cpu_array<double, 3> u_q = femto::evaluate(u, domain);
    auto rho_q = femto::forall(std::function<mat3(const vec3 &)>(
      [=](const vec3 &) { return mat3(rho * fm::Identity<3>()); }), u_q);
    out.M = femto::sparse_matrix<>(femto::integrate(femto::dot(phi, rho_q, phi), domain));
  }

  out.M_ms = milliseconds_since(t_M);
  out.assemble_ms = milliseconds_since(t0);

  out.nodes = femto::nodes_for(u, mesh);
  out.num_verts = mesh.vert.shape[0];
  out.ndof = u.size();

  return out;

}

// largest eigenvalue of (K, M_L), where M_L is the row-sum lumped mass used
// by the explicit integrator, by power iteration on M_L^{-1} K
inline double estimate_lambda_max(Analysis & a) {
  if (a.lambda_max > 0) { return a.lambda_max; }

  vector d = a.M(femto::ones(int(a.ndof)));

  vector x(a.ndof);
  for (uint32_t i = 0; i < a.ndof; i++) { x[i] = std::sin(0.7 * i + 0.3); }

  double lam = 0;
  for (int it = 0; it < 30; it++) {
    vector Kx = a.K(x);
    double xKx = 0, xdx = 0;
    for (uint32_t i = 0; i < a.ndof; i++) { xKx += x[i] * Kx[i]; xdx += x[i] * d[i] * x[i]; }
    lam = xKx / xdx;
    double nrm = 0;
    for (uint32_t i = 0; i < a.ndof; i++) { Kx[i] /= d[i]; nrm += Kx[i] * Kx[i]; }
    nrm = std::sqrt(nrm);
    for (uint32_t i = 0; i < a.ndof; i++) { x[i] = Kx[i] / nrm; }
  }

  return a.lambda_max = lam;
}

// transient simulation, stepped on demand (advance()) so the state can be
// displayed while it evolves: a uniform pressure is applied to one quad face
// of the outer wall (directly between the top of the handle and the rim) for
// the first 5 microseconds, and the (undamped, lumped-mass) equations of
// motion are integrated with the central difference method, holding a few
// nodes at the center of the base fixed
struct TransientSim {

  std::shared_ptr<Analysis> a;
  femto::sparse_matrix<> Kmod;
  vector Ml;                       // row-sum lumped mass (bc rows are 1)
  vector u, v, acc, f;             // f: consistent nodal loads of the face pressure
  double dt;
  double force_duration = 5.0e-6;  // seconds the pressure stays on
  uint32_t step_count = 0;
  double max_displacement = 0;
  double setup_ms;     // matrix modification, mass lumping, dt estimate
  double step_ms = 0;  // time integration, accumulated over advance() calls

  TransientSim(std::shared_ptr<Analysis> a_, const femto::Mesh<> & mesh, double pressure)
    : a(std::move(a_)), Kmod(a->K), u(a->ndof), v(a->ndof), acc(a->ndof), f(a->ndof) {

    uint32_t nv = a->num_verts;
    auto x_of = [&](uint32_t n) { return a->nodes(n, 0); };
    auto y_of = [&](uint32_t n) { return a->nodes(n, 1); };
    auto z_of = [&](uint32_t n) { return a->nodes(n, 2); };
    auto r_of = [&](uint32_t n) { return std::hypot(x_of(n), y_of(n)); };

    // fixed nodes: on the bottom face, near the axis
    double z_max = 0, r_base = 0;
    for (uint32_t n = 0; n < nv; n++) { z_max = std::max(z_max, z_of(n)); }
    for (uint32_t n = 0; n < nv; n++) {
      if (z_of(n) < 1e-9 * z_max) { r_base = std::max(r_base, r_of(n)); }
    }
    std::vector<int> bc_dofs;
    for (uint32_t n = 0; n < nv; n++) {
      if (z_of(n) < 1e-9 * z_max && r_of(n) < 0.35 * r_base) {
        bc_dofs.push_back(3 * n + 0);
        bc_dofs.push_back(3 * n + 1);
        bc_dofs.push_back(3 * n + 2);
      }
    }

    // the loaded face is found from the boundary geometry alone. The outer
    // wall radius is linear in z from (z = thickness, base radius) to
    // (z_max, rim radius); the rim's top ring gives both the rim radii and
    // the wall thickness, and the base's bottom ring gives the base radius.
    double r_rim_out = 0, r_rim_in = 1e30, r_base_out = 0;
    for (uint32_t n = 0; n < nv; n++) {
      if (z_of(n) > (1.0 - 1e-9) * z_max) {
        r_rim_out = std::max(r_rim_out, r_of(n));
        r_rim_in = std::min(r_rim_in, r_of(n));
      }
      if (z_of(n) < 1e-9 * z_max) { r_base_out = std::max(r_base_out, r_of(n)); }
    }
    double t_wall = r_rim_out - r_rim_in;
    auto r_wall = [&](double z) {
      double s = std::clamp((z - t_wall) / (z_max - t_wall), 0.0, 1.0);
      return r_base_out + (r_rim_out - r_base_out) * s;
    };

    // top of the handle: highest boundary vertex clearly outside the wall
    // (handles bulge out by at least two wall thicknesses)
    double z_handle_top = 0;
    for (uint32_t n = 0; n < nv; n++) {
      if (r_of(n) > r_wall(z_of(n)) + 0.5 * t_wall) { z_handle_top = std::max(z_handle_top, z_of(n)); }
    }
    if (z_handle_top == 0) { z_handle_top = 0.5 * z_max; }  // no handles: load mid-wall

    // candidate faces: boundary quads of the outer wall (radial-ish normal,
    // centroid radius near the wall cone) above the handle, at the handle's
    // azimuth (handle 0 is always at theta = 0), centered in the gap
    femto::SubMesh<> bdr = femto::boundary_of(mesh);
    double z_mid = 0.5 * (z_handle_top + z_max);
    uint32_t face[4] = {};
    double best = 1e30;
    for (uint32_t q = 0; q < bdr.quad.shape[0]; q++) {
      uint32_t vids[4];
      vec3 x[4];
      for (int j = 0; j < 4; j++) { vids[j] = mesh.quad(bdr.quad(q), j).index; x[j] = femto::load<vec3>(a->nodes, vids[j]); }
      vec3 c = 0.25 * (x[0] + x[1] + x[2] + x[3]);
      double z_c = c[2], r_c = std::hypot(c[0], c[1]);
      if (z_c < z_handle_top || r_c < r_wall(z_c) - 0.5 * t_wall || r_c > r_wall(z_c) + 0.5 * t_wall) { continue; }

      // unit normal (from the diagonals) and the radial direction at the centroid
      vec3 nrm = fm::cross(x[2] - x[0], x[3] - x[1]);
      double radial = fm::dot(fm::xy(nrm), fm::xy(c)) / (fm::norm(nrm) * r_c);
      if (std::fabs(radial) < 0.5) { continue; }  // rim/base annulus, not wall

      double theta_c = std::fabs(std::atan2(c[1], c[0]));
      double score = 4.0 * theta_c + std::fabs(z_c - z_mid) / z_max;
      if (score < best) {
        best = score;
        for (int j = 0; j < 4; j++) { face[j] = vids[j]; }
      }
    }
    if (best == 1e30) { femto::error("transient: no outer wall face found above the handle"); }

    // consistent nodal loads for a uniform (inward) pressure on the bilinear
    // face: 2x2 Gauss, with the normal oriented outward at each point
    for (uint32_t i = 0; i < a->ndof; i++) { f[i] = 0.0; }
    double g = 1.0 / std::sqrt(3.0);
    for (int gp = 0; gp < 4; gp++) {
      double xi = (gp % 2 ? g : -g), eta = (gp / 2 ? g : -g);
      double N[4] = {0.25 * (1 - xi) * (1 - eta), 0.25 * (1 + xi) * (1 - eta),
                     0.25 * (1 + xi) * (1 + eta), 0.25 * (1 - xi) * (1 + eta)};
      double dNdxi[4] = {-0.25 * (1 - eta), 0.25 * (1 - eta), 0.25 * (1 + eta), -0.25 * (1 + eta)};
      double dNdeta[4] = {-0.25 * (1 - xi), -0.25 * (1 + xi), 0.25 * (1 + xi), 0.25 * (1 - xi)};
      vec3 xg{}, tang1{}, tang2{};
      for (int j = 0; j < 4; j++) {
        vec3 xj = femto::load<vec3>(a->nodes, face[j]);
        xg += N[j] * xj; tang1 += dNdxi[j] * xj; tang2 += dNdeta[j] * xj;
      }
      vec3 nrm = fm::cross(tang1, tang2);
      if (fm::dot(fm::xy(nrm), fm::xy(xg)) < 0) { nrm = -nrm; }   // orient outward
      for (int j = 0; j < 4; j++) {
        for (int d = 0; d < 3; d++) { f[3 * face[j] + d] += N[j] * (-pressure) * nrm[d]; }
      }
    }

    auto t0 = std::chrono::steady_clock::now();

    // impose the boundary conditions by rewriting the fixed rows and columns
    // of the stiffness matrix with rows of the identity. The mass matrix is
    // lumped (row sums), the standard choice for explicit integration: its
    // inverse is diagonal, so the steps need no factorization or solves
    Kmod({}, bc_dofs) = [](int i, int j) { return double(i == j); };
    Kmod(bc_dofs, {}) = [](int i, int j) { return double(i == j); };
    Ml = a->M(femto::ones(int(a->ndof)));
    for (int i : bc_dofs) { Ml[uint32_t(i)] = 1.0; }

    // stable timestep for central differences is dt_crit = 2 / omega_max;
    // run well below it (about 27% of critical) for extra accuracy
    dt = (1.6 / 3.0) / std::sqrt(estimate_lambda_max(*a));

    // at rest; the pressure is already on at t = 0, so the initial
    // acceleration solves M a = f. The identity rows keep the fixed dofs
    // exactly zero: f is zero there, so u, v, and acc stay zero too.
    for (uint32_t i = 0; i < a->ndof; i++) { u[i] = v[i] = 0.0; acc[i] = f[i] / Ml[i]; }

    setup_ms = milliseconds_since(t0);
  }

  // central differences (velocity Verlet form), with the face pressure
  // active until force_duration and off afterwards
  void advance(uint32_t nsteps) {
    auto t0 = std::chrono::steady_clock::now();
    uint32_t n = a->ndof;
    for (uint32_t step = 0; step < nsteps; step++) {
      for (uint32_t i = 0; i < n; i++) { u[i] += dt * v[i] + 0.5 * dt * dt * acc[i]; }
      vector rhs = Kmod(u);
      bool force_on = (step_count + step + 1) * dt <= force_duration;
      for (uint32_t i = 0; i < n; i++) { rhs[i] = ((force_on ? f[i] : 0.0) - rhs[i]) / Ml[i]; }
      for (uint32_t i = 0; i < n; i++) { v[i] += 0.5 * dt * (acc[i] + rhs[i]); }
      acc = std::move(rhs);
    }
    // running peak, used to normalize the displayed deformation
    for (uint32_t i = 0; i < n; i++) { max_displacement = std::max(max_displacement, std::fabs(u[i])); }
    step_count += nsteps;
    step_ms += milliseconds_since(t0);
  }

};

// dense symmetric eigensolver (cyclic Jacobi), for the small projected
// problems that arise in the Lanczos iteration below
inline void jacobi_eigensolver(std::vector<double> & A, std::vector<double> & V, int m) {
  V.assign(size_t(m) * m, 0.0);
  for (int i = 0; i < m; i++) { V[size_t(i) * m + i] = 1.0; }

  for (int sweep = 0; sweep < 60; sweep++) {
    double off = 0;
    for (int i = 0; i < m; i++) {
      for (int j = i + 1; j < m; j++) { off += A[size_t(i) * m + j] * A[size_t(i) * m + j]; }
    }
    if (off < 1e-24) { break; }

    for (int q = 1; q < m; q++) {
      for (int pp = 0; pp < q; pp++) {
        double apq = A[size_t(pp) * m + q];
        if (std::fabs(apq) < 1e-300) { continue; }
        double app = A[size_t(pp) * m + pp], aqq = A[size_t(q) * m + q];
        double tau = (aqq - app) / (2 * apq);
        double t = (tau >= 0 ? 1.0 : -1.0) / (std::fabs(tau) + std::sqrt(1 + tau * tau));
        double c = 1 / std::sqrt(1 + t * t), s = t * c;
        for (int k = 0; k < m; k++) {
          double akp = A[size_t(k) * m + pp], akq = A[size_t(k) * m + q];
          A[size_t(k) * m + pp] = c * akp - s * akq;
          A[size_t(k) * m + q] = s * akp + c * akq;
        }
        for (int k = 0; k < m; k++) {
          double apk = A[size_t(pp) * m + k], aqk = A[size_t(q) * m + k];
          A[size_t(pp) * m + k] = c * apk - s * aqk;
          A[size_t(q) * m + k] = s * apk + c * aqk;
        }
        for (int k = 0; k < m; k++) {
          double vkp = V[size_t(k) * m + pp], vkq = V[size_t(k) * m + q];
          V[size_t(k) * m + pp] = c * vkp - s * vkq;
          V[size_t(k) * m + q] = s * vkp + c * vkq;
        }
      }
    }
  }
}

struct ModalResult {
  std::vector<double> eigenvalues;  // (rad/s)^2, ascending, rigid modes excluded
  std::vector<float> modes;         // num_modes x num_verts x 3
  uint32_t num_modes;
  uint32_t iterations;
  double solve_ms;    // everything below
  double factor_ms;   // Cholesky of K + sigma M
  double apply_ms;    // operator applications (M matvec + triangular solves)
  double mgs_ms;      // M-orthonormalization
  double ritz_ms;     // projection, dense eigensolve, and subspace rotation
};

// smallest eigenpairs of K x = lambda M x for the unconstrained (free-free)
// mug, by shift-inverted Lanczos in the M-inner product, with full
// reorthogonalization (one operator application per Krylov dimension,
// instead of one per subspace vector per iteration). The first six modes are
// rigid translations/rotations (lambda ~ 0) and are skipped; the next
// `num_modes` elastic pairs are returned.
inline ModalResult modal(Analysis & a, uint32_t num_modes) {

  ModalResult out;
  out.num_modes = num_modes;

  uint32_t n = a.ndof;
  int wanted = int(num_modes) + 6;  // requested + rigid
  int max_dim = std::max(120, 4 * wanted);

  auto t0 = std::chrono::steady_clock::now();

  // shift-invert operator (K + sigma M)^{-1} M: sigma > 0 regularizes the
  // rigid modes, chosen far below the elastic eigenvalues
  double trK = 0, trM = 0;
  for (uint32_t i = 0; i < n; i++) { trK += a.K.at(i, i); trM += a.M.at(i, i); }
  double sigma = 1e-6 * trK / trM;

  femto::sparse_matrix<> Ks = a.K;
  for (size_t i = 0; i < Ks.nnz; i++) { Ks.values(i) += sigma * a.M.values(i); }
  Ks.symmetry = femto::Symmetry::Symmetric;
  Ks.definiteness = femto::Definiteness::PositiveDefinite;
  auto invKs = femto::inv(Ks);

  out.factor_ms = milliseconds_since(t0);
  out.apply_ms = out.mgs_ms = out.ritz_ms = 0;

  // M-orthonormal Krylov basis of (K + sigma M)^{-1} M; MV[j] caches M V[j]
  // so reorthogonalization and the next operator application are dot products
  // and a solve, with one fresh M matvec per step
  std::vector<vector> V, MV;
  std::vector<double> h(size_t(max_dim) * max_dim, 0.0);  // projected matrix, by column
  double beta = 0.0;

  {
    vector v(n);
    for (uint32_t k = 0; k < n; k++) { v[k] = std::sin(0.37 * (k + 1)); }
    vector Mv = a.M(v);
    double nrm = 0;
    for (uint32_t k = 0; k < n; k++) { nrm += v[k] * Mv[k]; }
    nrm = std::sqrt(nrm);
    for (uint32_t k = 0; k < n; k++) { v[k] /= nrm; Mv[k] /= nrm; }
    V.push_back(std::move(v));
    MV.push_back(std::move(Mv));
  }

  std::vector<double> theta, Y;
  std::vector<int> order;
  int rdim = 0;  // dimension of the last Rayleigh-Ritz solve

  // Rayleigh-Ritz over the current basis: eigenvalues theta of the projected
  // shift-inverted operator, sorted descending (largest theta = smallest
  // lambda). Convergence via the Lanczos residual bound beta * |y_last|.
  auto ritz = [&](int dim) -> bool {
    auto t_ritz = std::chrono::steady_clock::now();
    std::vector<double> A_r(size_t(dim) * dim, 0.0);
    for (int c = 0; c < dim; c++) {
      for (int i = 0; i <= c; i++) {
        A_r[size_t(i) * dim + c] = A_r[size_t(c) * dim + i] = h[size_t(i) + size_t(c) * max_dim];
      }
    }
    jacobi_eigensolver(A_r, Y, dim);

    order.resize(dim);
    for (int i = 0; i < dim; i++) { order[i] = i; }
    std::sort(order.begin(), order.end(), [&](int i, int j) {
      return A_r[size_t(i) * dim + i] > A_r[size_t(j) * dim + j];
    });
    theta.resize(dim);
    for (int i = 0; i < dim; i++) { theta[i] = A_r[size_t(order[i]) * dim + order[i]]; }
    rdim = dim;

    bool done = (dim >= wanted + 2);
    for (int i = 0; i < wanted && done; i++) {
      double tail = std::fabs(Y[size_t(dim - 1) * dim + order[i]]);
      if (beta * tail > 1e-7 * std::fabs(theta[i])) { done = false; }
    }
    out.ritz_ms += milliseconds_since(t_ritz);
    return done;
  };

  int d = 1;
  for (; d < max_dim; d++) {
    int j = d - 1;  // the column being expanded

    auto t_apply = std::chrono::steady_clock::now();
    vector w = femto::dot(invKs, MV[j]);
    out.apply_ms += milliseconds_since(t_apply);

    // full reorthogonalization (two passes), accumulating the projection
    // coefficients that form column j of the projected matrix
    auto t_mgs = std::chrono::steady_clock::now();
    for (int rep = 0; rep < 2; rep++) {
      for (int i = 0; i <= j; i++) {
        double proj = 0;
        for (uint32_t k = 0; k < n; k++) { proj += w[k] * MV[i][k]; }
        h[size_t(i) + size_t(j) * max_dim] += proj;
        for (uint32_t k = 0; k < n; k++) { w[k] -= proj * V[i][k]; }
      }
    }
    vector Mw = a.M(w);
    double nrm = 0;
    for (uint32_t k = 0; k < n; k++) { nrm += w[k] * Mw[k]; }
    beta = std::sqrt(std::max(nrm, 0.0));
    out.mgs_ms += milliseconds_since(t_mgs);

    if (beta < 1e-300) { d++; break; }  // Krylov space exhausted

    // the convergence check runs a dense eigensolve, so only every few steps
    if ((d >= wanted + 2) && ((d % 4 == 0) || (d == max_dim - 1))) {
      if (ritz(d)) { break; }
    }

    for (uint32_t k = 0; k < n; k++) { w[k] /= beta; Mw[k] /= beta; }
    V.push_back(std::move(w));
    MV.push_back(std::move(Mw));
  }
  if (rdim < std::min(d, max_dim) - 1) { ritz(std::min<int>(d, V.size())); }

  out.iterations = uint32_t(rdim);
  out.eigenvalues.resize(num_modes);
  out.modes.assign(size_t(num_modes) * a.num_verts * 3, 0.0f);
  for (uint32_t i = 0; i < num_modes; i++) {
    int c = order[6 + i];
    out.eigenvalues[i] = 1.0 / theta[6 + i] - sigma;

    // Ritz vector, rotated out of the Krylov basis (M-orthonormal already)
    vector x(n);
    for (uint32_t k = 0; k < n; k++) { x[k] = 0.0; }
    for (int j = 0; j < rdim; j++) {
      double y = Y[size_t(j) * rdim + c];
      for (uint32_t k = 0; k < n; k++) { x[k] += y * V[j][k]; }
    }
    for (uint32_t v = 0; v < a.num_verts; v++) {
      for (uint32_t dcomp = 0; dcomp < 3; dcomp++) {
        out.modes[(size_t(i) * a.num_verts + v) * 3 + dcomp] = float(x[3 * v + dcomp]);
      }
    }
  }

  out.solve_ms = milliseconds_since(t0);
  return out;

}

}
