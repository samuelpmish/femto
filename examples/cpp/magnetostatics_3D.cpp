// 3D magnetostatics: an induction-heating work coil -- six turns of thick
// copper -- wound around a cylindrical ferrite core, in a box of air.  The
// vector potential A is an H(curl) field on tetrahedra solving
// curl(nu curl A) = J with A x n = 0 on the box, nu = 1/mu the reluctivity per
// element and J the current running along the wire.  The helix is open at both
// ends, so its current is not divergence free on the mesh: J is projected,
// J - grad(psi) with a scalar Poisson solve for psi, and a small mass term
// eps A fixes the gauge of the (otherwise singular) curl-curl operator.
// Neither changes B = curl A.
//
// The mesh is data/meshes/induction_coil.msh, built by
// tools/generate_example_meshes.py: the coil and core surfaces are swept there
// and fTetWild fills them and the air around them with tetrahedra.  Elements
// are sorted into copper, ferrite and air by the same analytic tests that
// generated the surfaces, so the constants below have to match it.
//
// SI units: meters, amperes, teslas.  The mesh is in millimeters.

#include "example_common.hpp"

#include "femto/field.hpp"
#include "forall.hpp"

#include <cmath>
#include <iostream>

using namespace femto;

struct Magnetostatics {

  static constexpr double mu0 = 4.0e-7 * M_PI;
  static constexpr double mm = 1.0e-3;

  // the coil and core, matching tools/generate_example_meshes.py
  static constexpr double CORE_R = 8.0 * mm, CORE_LEN = 60.0 * mm;
  static constexpr double COIL_R = 14.0 * mm, WIRE_R = 2.5 * mm, COIL_PITCH = 8.0 * mm;
  static constexpr int COIL_TURNS = 6;
  static constexpr double AIR_XY = 40.0 * mm, AIR_Z = 50.0 * mm;

  struct Params {
    double current = 60.0;              // A through the coil
    double mu_r = 1000.0;               // of the ferrite core
    double gauge = 1.0e-6;              // eps, relative to nu_air / L^2
  };

  enum Region : int { AIR = 0, CORE = 1, COIL = 2 };

  Params params;
  Mesh<> mesh;
  Field<Family::Hcurl> A;
  std::vector<int> region;            // per tetrahedron
  std::vector<double> B;              // per tetrahedron, 3 components
  double B_center = 0, B_max = 0, B_core = 0;

  // the coil's centerline and the direction the current runs along it
  static vec3 helix_point(double phi) {
    return vec3{COIL_R * std::cos(phi), COIL_R * std::sin(phi), COIL_PITCH * phi / (2 * M_PI)};
  }
  static vec3 helix_tangent(double phi) {
    vec3 t{-COIL_R * std::sin(phi), COIL_R * std::cos(phi), COIL_PITCH / (2 * M_PI)};
    return t / norm(t);
  }

  // the wire passes a given azimuth once per turn, so the nearest point of the
  // centerline is the best of a handful of candidates
  static double nearest_angle(vec3 c, double & distance) {
    double phi = std::atan2(c[1], c[0]), r = std::hypot(c[0], c[1]);
    double best = 1e300, best_angle = 0;
    for (int k = -COIL_TURNS; k <= COIL_TURNS; k++) {
      double angle = phi + 2 * M_PI * k;
      if (std::fabs(angle) > M_PI * COIL_TURNS) continue;
      double d = std::hypot(r - COIL_R, c[2] - COIL_PITCH * angle / (2 * M_PI));
      if (d < best) { best = d; best_angle = angle; }
    }
    distance = best;
    return best_angle;
  }

  // reluctivity, the gauge term and the current density at a quadrature
  // point, from its cell's region and its position
  struct Materials {
    double nu_air, mu_r, eps, J0;
    void operator()(const int & region, const vec3 & x, mat3 & nu, mat3 & gauge, vec3 & J) const {
      nu = mat3((region == CORE ? nu_air / mu_r : nu_air) * Identity<3>());
      gauge = mat3(eps * Identity<3>());
      double distance;
      J = (region == COIL) ? J0 * helix_tangent(nearest_angle(x, distance)) : vec3{0.0, 0.0, 0.0};
    }
  };

  static Region classify(vec3 c) {
    if (std::hypot(c[0], c[1]) < CORE_R && std::fabs(c[2]) < 0.5 * CORE_LEN) { return CORE; }
    double distance;
    nearest_angle(c, distance);
    return (distance < WIRE_R) ? COIL : AIR;
  }

  Magnetostatics(Params p) : params(p) {
#ifdef __EMSCRIPTEN__
    // the page fetches the mesh and writes it here before constructing us
    mesh = Mesh<>::load("/induction_coil.msh");
#else
    mesh = Mesh<>::load(FEMTO_MESH_DIR "induction_coil.msh");
#endif
  mesh.X.v() *= mm;

    region.resize(mesh.tet.shape[0]);
    for (uint32_t e = 0; e < mesh.tet.shape[0]; e++) {
      region[e] = classify(centroid<3>(mesh, Geometry::Tetrahedron, e));
    }

    A = create_field<Family::Hcurl>(mesh, 1, 1);
  }

  void solve() {
    BasisFunction phi(A);
    Domain domain(mesh, MeshQuadratureRule(2));

    // per quadrature point: reluctivity, the gauge term, and the current
    // density, which runs along the wire at I / (its cross section)
    uint32_t nq = total(domain.num_qpts), qpe = nq / mesh.tet.shape[0];
    double nu_air = 1.0 / mu0;
    double eps = params.gauge * nu_air / (AIR_Z * AIR_Z);
    double J0 = params.current / (M_PI * WIRE_R * WIRE_R);

    nd::cpu_array<double, 3> x_q = evaluate(mesh.X, domain);
    nd::cpu_array<int, 2> region_q = at_quadrature_points(region, qpe);
    nd::cpu_array<double, 3> nu_q, eps_q;
    nd::cpu_array<double, 2> J_q;
    forall(Materials{nu_air, params.mu_r, eps, J0}, region_q, x_q, nu_q, eps_q, J_q);

    // the divergence-free part of J: int grad(chi).grad(psi) = int J.grad(chi)
    // for every nodal chi, then J -= grad(psi), so the load is orthogonal to
    // the discrete gradients that the curl-curl operator cannot see
    {
      Field<Family::H1> psi = create_field<Family::H1>(mesh, 1, 1);
      BasisFunction chi(psi);
    nd::cpu_array<double, 3> I_q = forall(+[](const vec3 &) { return mat3(Identity<3>()); }, J_q);
      sparse_matrix Kpsi = integrate(dot(grad(chi), I_q, grad(chi)), domain);
      Residual<Family::H1> rhs = integrate(dot(J_q, grad(chi)), domain);
      ConstrainedSystem poisson;
      poisson.dofs = {0};
      poisson.set_matrix(Kpsi);
      psi.v() = poisson.solve(rhs.v(), zeros(1));
      nd::cpu_array<double, 3> gradpsi_q = evaluate(grad(psi), domain);
    forall(+[](const vec3 & g, vec3 & J) { J -= g; }, gradpsi_q, J_q);
    }

    sparse_matrix K = integrate(dot(curl(phi), nu_q, curl(phi)), domain);
    sparse_matrix M = integrate(dot(phi, eps_q, phi), domain);
    for (size_t i = 0; i < K.nnz; i++) { K.values[i] += M.values[i]; }

    Residual<Family::Hcurl> f = integrate(dot(J_q, phi), domain);

    // A x n = 0: every edge on the box carries a zero tangential dof
    ConstrainedSystem system;
    SubMesh<> bdr = boundary_of(mesh);
    for (uint32_t k = 0; k < bdr.edge.shape[0]; k++) { system.dofs.push_back(int(A.offsets.edge + bdr.edge(k))); }
    system.set_matrix(K);
    A.v() = system.solve(f.v(), zeros(int(system.dofs.size())));

    // B = curl A at the cell centers
    Domain centers(mesh, MeshQuadratureRule(1));
    nd::cpu_array<double, 3> B_q = evaluate(curl(A), centers);
    B.resize(size_t(mesh.tet.shape[0]) * 3);
    B_max = 0;
    double core_sum = 0, core_count = 0, center_sum = 0, center_count = 0;
    for (uint32_t e = 0; e < mesh.tet.shape[0]; e++) {
    vec3 b = load<vec3>(B_q, e);
    for (int c = 0; c < 3; c++) { B[3 * e + c] = b[c]; }
    B_max = std::max(B_max, norm(b));
      if (region[e] == CORE) {
        vec3 c = centroid<3>(mesh, Geometry::Tetrahedron, e);
        core_sum += B[3 * e + 2];
        core_count++;
        if (std::fabs(c[2]) < 0.1 * CORE_LEN) { center_sum += B[3 * e + 2]; center_count++; }
      }
    }
    B_center = center_sum / std::max(center_count, 1.0);      // on the axis, mid core
    B_core = core_sum / std::max(core_count, 1.0);            // averaged over the ferrite
  }

  // what a solenoid of the coil's turns and length would make at its center in
  // air, for comparison: the core concentrates a good deal more than this
  double solenoid_estimate() const {
    double length = COIL_TURNS * COIL_PITCH, half = 0.5 * length;
    return mu0 * (COIL_TURNS / length) * params.current * half / std::sqrt(half * half + COIL_R * COIL_R);
  }

  std::vector<double> nodes() const { return vertex_coordinates(mesh); }
  std::vector<uint32_t> cells() const { return cell_vertices(mesh, Geometry::Tetrahedron); }

};

#ifdef __EMSCRIPTEN__

using namespace emscripten;

EMSCRIPTEN_BINDINGS(magnetostatics_3D) {
  value_object<Magnetostatics::Params>("MagnetostaticsParams")
    .field("current", &Magnetostatics::Params::current)
    .field("muR", &Magnetostatics::Params::mu_r)
    .field("gauge", &Magnetostatics::Params::gauge);

  class_<Magnetostatics>("Magnetostatics")
    .constructor<Magnetostatics::Params>()
    .function("solve", &Magnetostatics::solve)
    .function("centerField", +[](const Magnetostatics & m) { return m.B_center; })
    .function("coreField", +[](const Magnetostatics & m) { return m.B_core; })
    .function("maxField", +[](const Magnetostatics & m) { return m.B_max; })
    .function("solenoidEstimate", &Magnetostatics::solenoid_estimate)
    .function("B", +[](const Magnetostatics & m) { return to_f64(m.B); })
    .function("region", +[](const Magnetostatics & m) { return to_i32(m.region); })
    .function("nodes", +[](const Magnetostatics & m) { return to_f64(m.nodes()); })
    .function("cells", +[](const Magnetostatics & m) { return to_u32(m.cells()); });
}

#else

int main() {
  for (double mu_r : {1.0, 1000.0}) {
    Magnetostatics::Params p;
    p.mu_r = mu_r;
    Magnetostatics m(p);
    std::cout << "mu_r = " << mu_r << ": " << m.mesh.tet.shape[0] << " tetrahedra, " << m.A.size() << " edge dofs" << std::endl;
    m.solve();
    std::cout << "   B on the axis " << m.B_center * 1e3 << " mT, averaged over the core " << m.B_core * 1e3
              << " mT, peak |B| " << m.B_max * 1e3 << " mT (air-core solenoid estimate "
              << m.solenoid_estimate() * 1e3 << " mT)" << std::endl;
  }
  return 0;
}

#endif
