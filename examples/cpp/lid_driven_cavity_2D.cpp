// 2D incompressible Navier-Stokes: the lid-driven cavity.  A square box of
// fluid is driven by its top wall sliding at the lid speed; the other walls
// are no-slip.  Taylor-Hood elements (Q2 velocity, Q1 pressure) and the
// steady equations solved directly by Newton's method on the coupled
// velocity-pressure system, rather than marching in time until the transient
// dies out.  The first Newton step from rest is the Stokes solution; from
// there the Reynolds number is raised in stages so each Newton solve starts
// close to its answer.  Re = U L / nu sets the cavity vortex's shape.

#include "example_common.hpp"

#include "femto/field.hpp"
#include "forall.hpp"

#include <cmath>
#include <iostream>

using namespace femto;

struct LidDrivenCavity {

  struct Params {
    double L = 1.0;
    int n = 32;                    // cells per side
    double lid_speed = 1.0;
    double viscosity = 0.01;       // kinematic
    double tolerance = 1e-8;       // on |R| relative to the Stokes residual
  };

  Params params;
  Mesh<> mesh;
  Field<Family::H1> u;             // Q2, 2 components
  Field<Family::H1> p;             // Q1
  BasisFunction<Family::H1> phi, psi;
  Domain<> domain;
  nd::cpu_array<double, 3> u_q, du_q, p_q, stress_q;     // assemble()'s work arrays, allocated once
  nd::cpu_array<double, 2> adv_q, div_q;
  Residual<Family::H1> r_stress, r_adv, r_div;
  nd::cpu_array<double, 5> nu_q, conv_q, dconv_q, gradp_q, divu_q;
  uint32_t nu = 0, np = 0;         // dof counts
  std::vector<char> fixed;         // per dof of [u; p]: walls, lid and the pressure pin
  sparse_matrix<> J;               // the Jacobian over [u; p]; its pattern is built once
  std::vector<int> slots[5];       // where each block's nonzeros land in J.values (-1: a fixed row or column)
  sparse_factorization invJ;       // kept so later steps refactorize numerically, on the same pattern
  bool factorized = false;
  std::vector<double> history;     // |R| after each Newton step, relative to the first
  double residual = 0, stokes_residual = 0;
  uint32_t iterations = 0;

  // the residual's integrands at a quadrature point: the viscous and
  // pressure stress, the advection (u . grad) u, and div u
  struct Integrands {
    double viscosity;
    void operator()(const vec2 & u, const mat2 & du, const double & p, mat2 & stress, vec2 & adv, double & div) const {
      stress = viscosity * du - p * Identity<2>();
      adv = dot(du, u);
      div = tr(du);
    }
  };

  static uint32_t cells(Params prm) { return uint32_t(std::max(2, prm.n)); }

  LidDrivenCavity(Params prm) : params(prm), mesh(Mesh<>::cuboid({cells(prm), cells(prm)}, vec2{prm.L, prm.L})),
                                u(create_field<Family::H1>(mesh, 2, 2)), p(create_field<Family::H1>(mesh, 1, 1)),
                                phi(u), psi(p), domain(mesh, MeshQuadratureRule(3)) {
    nu = 2 * u.data.shape[0];
    np = p.data.shape[0];
    fixed.assign(nu + np, 0);

    // the lid slides, every other wall is no-slip (the lid wins at its
    // corners); the initial state already has these values, and Newton's
    // corrections are zero there.  The pressure floats, so one node pins it
    nd::cpu_array<double, 2> nodes = nodes_for(u, mesh);
    double eps = 1e-9 * prm.L;
    for (uint32_t i = 0; i < nodes.shape[0]; i++) {
      double x = nodes(i, 0), y = nodes(i, 1);
      bool lid = y > prm.L - eps, wall = x < eps || x > prm.L - eps || y < eps;
      u.data(i, 0) = lid ? prm.lid_speed : 0.0;
      u.data(i, 1) = 0.0;
      fixed[2 * i] = fixed[2 * i + 1] = lid || wall;
    }
    for (uint32_t i = 0; i < np; i++) { p.data(i, 0) = 0.0; }
    fixed[nu] = 1;
  }

  double reynolds() const { return params.lid_speed * params.L / params.viscosity; }

  // the steady residual R(u, p) = [ int nu grad u : grad phi + phi . (u . grad) u - p div phi ; int psi div u ]
  // and its Jacobian, at the current state and the given viscosity.  The
  // Jacobian is one unsymmetric matrix over [u; p] with identity rows at
  // the fixed dofs, so the Newton update is a single LU solve
  void assemble(double viscosity, vector & R) {
    uint32_t nq = total(domain.num_qpts);

    u_q = evaluate(u, domain);
    du_q = evaluate(grad(u), domain);
    p_q = evaluate(p, domain);

    // residual: viscous and pressure parts as one stress against grad(phi),
    // the advection against phi, the divergence against psi
    forall(Integrands{viscosity}, u_q, du_q, p_q, stress_q, adv_q, div_q);
    r_stress = integrate(dot(stress_q, grad(phi)), domain);
    r_adv = integrate(dot(adv_q, phi), domain);
    r_div = integrate(dot(div_q, psi), domain);
    R = vector(nu + np);
    for (uint32_t i = 0; i < nu; i++) { R[i] = fixed[i] ? 0.0 : r_stress.data.data()[i] + r_adv.data.data()[i]; }
    for (uint32_t i = 0; i < np; i++) { R[nu + i] = fixed[nu + i] ? 0.0 : r_div.data.data()[i]; }

    // Jacobian blocks: viscosity, the two linearizations of (u . grad) u,
    // and the pressure gradient / divergence pair
    nu_q.resize({nq, 2, 2, 2, 2});
    conv_q.resize({nq, 2, 1, 2, 2});   // phi_i u_k d_k dphi_j   (rows u, cols u)
    dconv_q.resize({nq, 2, 1, 2, 1});  // phi_i (d_j u_i) dphi_j  (rows u, cols u)
    gradp_q.resize({nq, 2, 2, 1, 1});  // -div(phi) dpsi          (rows u, cols p)
    divu_q.resize({nq, 1, 1, 2, 2});   // psi div(dphi)           (rows p, cols u)
    for (uint32_t q = 0; q < nq; q++) {
      for (int i = 0; i < 2; i++) for (int k = 0; k < 2; k++) {
        for (int j = 0; j < 2; j++) for (int l = 0; l < 2; l++) { nu_q(q, i, k, j, l) = (i == j && k == l) ? viscosity : 0.0; }
        conv_q(q, i, 0, i, k) = u_q(q, k, 0);
        conv_q(q, i, 0, 1 - i, k) = 0.0;
        dconv_q(q, i, 0, k, 0) = du_q(q, i, k);
        gradp_q(q, i, k, 0, 0) = -double(i == k);
        divu_q(q, 0, 0, i, k) = double(i == k);
      }
    }
    // dot(trial, C, test): the rows come from the last argument, and C is
    // indexed (q, test component, test shape, trial component, trial shape)
    sparse_matrix<> Kuu = integrate(dot(grad(phi), nu_q, grad(phi)), domain);
    sparse_matrix<> Cuu = integrate(dot(grad(phi), conv_q, phi), domain);
    sparse_matrix<> Duu = integrate(dot(phi, dconv_q, phi), domain);
    sparse_matrix<> Gup = integrate(dot(psi, gradp_q, grad(phi)), domain);
    sparse_matrix<> Dpu = integrate(dot(grad(phi), divu_q, psi), domain);

    const sparse_matrix<> * blocks[5] = {&Kuu, &Cuu, &Duu, &Gup, &Dpu};
    const uint32_t offsets[5][2] = {{0, 0}, {0, 0}, {0, 0}, {0, nu}, {nu, 0}};
    FEMTO_ASSERT(Gup.nrows == int(nu) && Gup.ncols == int(np) && Dpu.nrows == int(np) && Dpu.ncols == int(nu), "block shapes");

    // the first time through, the pattern: the union of the blocks, less the
    // fixed rows and columns, plus the diagonal.  The entries themselves are
    // scattered in through slots, so the pattern (and the factorization's
    // analysis of it) is reused by every later step
    if (J.nnz == 0) {
      std::vector<triplet> entries;
      for (int b = 0; b < 5; b++) {
        const sparse_matrix<> & A = *blocks[b];
        uint32_t row0 = offsets[b][0], col0 = offsets[b][1];
        for (int i = 0; i < A.nrows; i++) {
          if (fixed[row0 + i]) continue;
          for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; k++) {
            if (!fixed[col0 + A.col_ind[k]]) { entries.push_back({int(row0 + i), int(col0 + A.col_ind[k]), 0.0}); }
          }
        }
      }
      for (uint32_t i = 0; i < nu + np; i++) { entries.push_back({int(i), int(i), 0.0}); }
      J = sparse_matrix<>::from_triplets(entries, nu + np, nu + np);
      J.symmetry = Symmetry::Unsymmetric;
      J.definiteness = Definiteness::Indefinite;
      for (int b = 0; b < 5; b++) {
        const sparse_matrix<> & A = *blocks[b];
        uint32_t row0 = offsets[b][0], col0 = offsets[b][1];
        slots[b].assign(A.nnz, -1);
        for (int i = 0; i < A.nrows; i++) {
          if (fixed[row0 + i]) continue;
          for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; k++) {
            if (!fixed[col0 + A.col_ind[k]]) { slots[b][k] = J.search_row_for_given_column(row0 + i, col0 + A.col_ind[k]); }
          }
        }
      }
    }

    for (size_t k = 0; k < J.nnz; k++) { J.values[k] = 0.0; }
    for (int b = 0; b < 5; b++) {
      const sparse_matrix<> & A = *blocks[b];
      for (size_t k = 0; k < A.nnz; k++) { if (slots[b][k] >= 0) { J.values[slots[b][k]] += A.values[k]; } }
    }
    for (uint32_t i = 0; i < nu + np; i++) { if (fixed[i]) { J.at(i, i) = 1.0; } }
  }

  void apply(const vector & dx, double alpha) {
    double * uv = u.data.data(), * pv = p.data.data();
    for (uint32_t i = 0; i < nu; i++) { uv[i] += alpha * dx[i]; }
    for (uint32_t i = 0; i < np; i++) { pv[i] += alpha * dx[nu + i]; }
  }

  // Newton at one viscosity, with a backtracking line search on |R|
  void newton(double viscosity, double tolerance, int max_iterations = 30) {
    vector R, dx;
    assemble(viscosity, R);
    double rnorm = norm(R);
    if (stokes_residual == 0) { stokes_residual = rnorm; }
    for (int it = 0; it < max_iterations && rnorm > tolerance * stokes_residual; it++) {
      if (factorized) { invJ.update(J); } else { invJ = inv(J); factorized = true; }
      dx = dot(invJ, -1.0 * R);
      double alpha = 1.0, trial = 0;
      for (int cut = 0; cut < 8; cut++) {
        apply(dx, alpha);
        assemble(viscosity, R);
        trial = norm(R);
        if (trial < rnorm) break;
        apply(dx, -alpha);
        alpha *= 0.5;
      }
      rnorm = trial;
      history.push_back(rnorm / stokes_residual);
      iterations++;
    }
    residual = rnorm / stokes_residual;
  }

  // the Stokes solution first, then Newton at Reynolds numbers rising by
  // 4x up to the requested one, each started from the previous state.  The
  // intermediate stages only need to land near enough for the next Newton
  // to converge, so they stop early; only the last one goes to tolerance
  void solve() {
    double Re_target = reynolds();
    history.clear();
    iterations = 0;
    stokes_residual = 0;
    for (double Re = std::min(Re_target, 100.0); ; Re = std::min(Re_target, 4 * Re)) {
      bool last = Re >= Re_target;
      newton(params.lid_speed * params.L / Re, last ? params.tolerance : 1e-4);
      if (last) break;
    }
  }

  // the horizontal velocity along the vertical centerline, at the Q2 nodes
  // nearest x = L/2, as (y, u_x) pairs sorted by y
  std::vector<double> centerline() const {
    nd::cpu_array<double, 2> nodes = nodes_for(u, mesh);
    std::vector<std::pair<double, double>> samples;
    for (uint32_t i = 0; i < nodes.shape[0]; i++) {
      if (std::fabs(nodes(i, 0) - 0.5 * params.L) < 1e-9 * params.L) { samples.push_back({nodes(i, 1), u.data(i, 0)}); }
    }
    std::sort(samples.begin(), samples.end());
    std::vector<double> out;
    for (auto [y, ux] : samples) { out.push_back(y); out.push_back(ux); }
    return out;
  }

  // vertex values: the Q2 nodes are ordered vertices first
  std::vector<double> velocity() const { return std::vector<double>(u.data.data(), u.data.data() + 2 * mesh.vert.shape[0]); }
  std::vector<double> pressure_field() const { return std::vector<double>(p.data.data(), p.data.data() + mesh.vert.shape[0]); }
  std::vector<double> nodes() const { return vertex_coordinates(mesh); }
  std::vector<uint32_t> cells() const { return cell_vertices(mesh, Geometry::Quadrilateral); }

};

#ifdef __EMSCRIPTEN__

using namespace emscripten;

EMSCRIPTEN_BINDINGS(lid_driven_cavity_2D) {
  value_object<LidDrivenCavity::Params>("CavityParams")
    .field("L", &LidDrivenCavity::Params::L)
    .field("n", &LidDrivenCavity::Params::n)
    .field("lidSpeed", &LidDrivenCavity::Params::lid_speed)
    .field("viscosity", &LidDrivenCavity::Params::viscosity)
    .field("tolerance", &LidDrivenCavity::Params::tolerance);

  class_<LidDrivenCavity>("LidDrivenCavity")
    .constructor<LidDrivenCavity::Params>()
    .function("solve", &LidDrivenCavity::solve)
    .function("reynolds", &LidDrivenCavity::reynolds)
    .function("iterations", +[](const LidDrivenCavity & c) { return c.iterations; })
    .function("residual", +[](const LidDrivenCavity & c) { return c.residual; })
    .function("history", +[](const LidDrivenCavity & c) { return to_f64(c.history); })
    .function("velocity", +[](const LidDrivenCavity & c) { return to_f64(c.velocity()); })
    .function("pressure", +[](const LidDrivenCavity & c) { return to_f64(c.pressure_field()); })
    .function("centerline", +[](const LidDrivenCavity & c) { return to_f64(c.centerline()); })
    .function("nodes", +[](const LidDrivenCavity & c) { return to_f64(c.nodes()); })
    .function("cells", +[](const LidDrivenCavity & c) { return to_u32(c.cells()); });
}

#else

int main() {
  // Ghia, Ghia & Shin (1982): the centerline minimum of u_x is -0.2109 at
  // y = 0.4531 for Re = 100 and -0.3829 at y = 0.1719 for Re = 1000
  for (double Re : {100.0, 1000.0}) {
    LidDrivenCavity::Params p;
    p.viscosity = 1.0 / Re;
    LidDrivenCavity cavity(p);
    cavity.solve();
    std::cout << "Re = " << cavity.reynolds() << ": " << cavity.iterations << " Newton steps, |R| / |R_0| = " << cavity.residual << std::endl;
    for (size_t k = 0; k < cavity.history.size(); k++) { std::cout << "  step " << k + 1 << ": " << cavity.history[k] << std::endl; }
    std::vector<double> line = cavity.centerline();
    double u_min = 0, y_min = 0, u_mid = 0;
    for (size_t k = 0; k < line.size(); k += 2) {
      if (line[k + 1] < u_min) { u_min = line[k + 1]; y_min = line[k]; }
      if (std::fabs(line[k] - 0.5) < 1e-9) { u_mid = line[k + 1]; }
    }
    std::cout << "  centerline: u_x(0.5, 0.5) = " << u_mid << ", minimum " << u_min << " at y = " << y_min << std::endl;
  }
  return 0;
}

#endif
