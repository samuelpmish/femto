// 2D fracture specimen: a square coupon with a sharp notch cut in from its
// left edge and a pin hole above and below the notch, meshed with triangles
// graded toward the notch tip (data/meshes/fracture_specimen.msh, converted
// from the MFEM mesh of the same name).  The pins are pulled apart in load
// steps, opening the notch; the material is isotropic linear elastic or
// neo-Hookean, in plane strain or plane stress, and every step is solved by
// Newton's method with the consistent tangent.  The reaction force at the
// upper pin is recorded per step.
//
// The displacement is quadratic on the straight-sided triangles: constant
// strain elements are far too stiff to say anything useful about a notch tip.
//
// Lengths in mm, stresses in MPa, forces in N per mm of thickness.

#include "example_common.hpp"

#include "femto/field.hpp"
#include "forall.hpp"
#include "materials/linear_elasticity.hpp"
#include "materials/neohookean.hpp"
#include "materials/plane_strain.hpp"
#include "materials/plane_stress.hpp"

#include <cmath>
#include <iostream>
#include <variant>

using namespace femto;

struct FractureSpecimen {

  enum class Material : int { LINEAR_ELASTIC = 0, NEOHOOKEAN = 1 };
  enum class State : int { PLANE_STRAIN = 0, PLANE_STRESS = 1 };

  // the mesh is 2 x 2 in its own units, with the notch tip at the origin and
  // the pin holes (radius 0.125) centered half a unit above and below it
  static constexpr double width = 50.0;             // mm, the side of the coupon
  static constexpr double scale = 0.5 * width;      // mesh units -> mm
  static constexpr double hole_radius = 0.125 * scale;
  static constexpr double hole_height = 0.5 * scale;

#ifdef __EMSCRIPTEN__
  // the wasm build embeds the mesh in its virtual filesystem, see
  // examples/html/CMakeLists.txt
  static constexpr const char * mesh_file = "/fracture_specimen.msh";
#else
  static constexpr const char * mesh_file = FEMTO_MESH_DIR "fracture_specimen.msh";
#endif

  struct Params {
    int material = 0;              // Material
    int state = 0;                 // State
    double E = 1000.0, nu = 0.3;
    double opening = 2.0;          // total separation of the two pins
    int load_steps = 10;
  };

  using Model = std::variant<
    PlaneStrain<LinearElasticModel>, PlaneStress<LinearElasticModel>,
    PlaneStrain<NeoHookeanModel>, PlaneStress<NeoHookeanModel>>;

  Params params;
  Mesh<> mesh;
  Field<Family::H1> u;
  BasisFunction<Family::H1> phi;
  Domain<> domain, centers;                // the assembly rule, and one point per cell for the stress plot
  nd::cpu_array<double, 3> du_dX_q, P_q;   // work arrays, allocated once and refilled in place
  nd::cpu_array<double, 5> dP_dF_q;
  Residual<Family::H1> r_u;
  nd::cpu_array<double, 3> du_dX_c;
  nd::cpu_array<double, 2> vm_c;
  Model model;
  ConstrainedSystem system;
  std::vector<double> pin_motion;   // per constrained dof, its share of the opening
  std::vector<int> upper_pin_dofs;  // the upper pin's vertical dofs, for the reaction
  std::vector<double> curve;        // (opening, reaction force) pairs
  int step_count = 0;
  int newton_iterations = 0;
  double residual_norm = 0;

  static Mesh<> load_mesh() {
    Mesh<> m = Mesh<>::load(mesh_file);
    m.X.v() *= scale;
    return m;
  }

  FractureSpecimen(Params p) : params(p), mesh(load_mesh()), u(create_field<Family::H1>(mesh, 2, 2)), phi(u),
                               domain(mesh, MeshQuadratureRule(3)), centers(mesh, MeshQuadratureRule(1)) {

    double lambda = p.E * p.nu / ((1 + p.nu) * (1 - 2 * p.nu)), mu = p.E / (2 * (1 + p.nu));
    bool neo = Material(p.material) == Material::NEOHOOKEAN, stress = State(p.state) == State::PLANE_STRESS;
    if (!neo && !stress) { model = PlaneStrain<LinearElasticModel>{{lambda, mu}}; }
    if (!neo &&  stress) { model = PlaneStress<LinearElasticModel>{{lambda, mu}}; }
    if ( neo && !stress) { model = PlaneStrain<NeoHookeanModel>{{lambda, mu}}; }
    if ( neo &&  stress) { model = PlaneStress<NeoHookeanModel>{{lambda, mu}}; }

    nd::zero(u.data);

    // the pins hold every node on their hole: the upper one moves up by half
    // the opening, the lower one down by half, and neither slides sideways.
    // The next ring of nodes is at 1.38 hole radii, so a plain distance test
    // picks out the hole exactly.
    nd::cpu_array<double, 2> X = nodes_for(u, mesh);
    for (uint32_t i = 0; i < X.shape[0]; i++) {
      for (int upper = 0; upper < 2; upper++) {
        vec2 d = load<vec2>(X, i) - vec2{-0.5 * scale, upper ? hole_height : -hole_height};
        if (norm_squared(d) > 1.2 * 1.2 * hole_radius * hole_radius) continue;
        system.dofs.push_back(int(2 * i));     pin_motion.push_back(0.0);
        system.dofs.push_back(int(2 * i + 1)); pin_motion.push_back(upper ? +0.5 : -0.5);
        if (upper) { upper_pin_dofs.push_back(int(2 * i + 1)); }
      }
    }

    curve = {0.0, 0.0};
  }

  const Residual<Family::H1> & residual(const Field<Family::H1> & u_) {
    du_dX_q = evaluate(grad(u_), domain);
    P_q = forall(std::function<mat2(const mat2 &)>([&](const mat2 & du_dX) {
      return std::visit([&](auto & m) { return m(du_dX); }, model);
    }), du_dX_q);
    r_u = integrate(dot(P_q, grad(phi)), domain);
    return r_u;
  }

  sparse_matrix<> stiffness(const Field<Family::H1> & u_) {
    du_dX_q = evaluate(grad(u_), domain);
    dP_dF_q = forall(std::function<mat<2,2,mat2>(const mat2 &)>([&](const mat2 & du_dX) {
      return std::visit([&](auto & m) { return m.jac(du_dX); }, model);
    }), du_dX_q);
    return integrate(dot(grad(phi), dP_dF_q, grad(phi)), domain);
  }

  // one load step: the pins move to the next opening, Newton iterates the rest
  // of the coupon into equilibrium with a fresh tangent each iteration
  void step() {
    if (step_count >= params.load_steps) return;
    step_count++;
    double opening = params.opening * step_count / params.load_steps;

    vector target(uint32_t(system.dofs.size()));
    for (size_t k = 0; k < system.dofs.size(); k++) { target[k] = opening * pin_motion[k]; }

    for (newton_iterations = 0; newton_iterations < 20; newton_iterations++) {
      vector r = residual(u).v();
      vector error = u.v()[system.dofs] - target;
      vector r_free = r;
      r_free[system.dofs] = 0.0;
      residual_norm = norm(r_free);
      if (residual_norm < 1e-10 * params.E && norm(error) < 1e-12) break;

      system.set_matrix(stiffness(u));
      vector du = system.solve(r, error);
      u.v() -= du;
    }

    vector r = residual(u).v();
    curve.push_back(opening);
    curve.push_back(total(r[upper_pin_dofs]));
  }

  // von Mises stress at the cell centers (Cauchy stress from P F^T / J)
  std::vector<double> von_mises() {
    du_dX_c = evaluate(grad(u), centers);
    vm_c = forall(std::function<double(const mat2 &)>([&](const mat2 & H) {
      mat2 P = std::visit([&](auto & m) { return m(H); }, model);
      mat2 F = Identity<2>() + H;
      mat2 s = dot(P, transpose(F)) / det(F);
      return std::sqrt(tr(s) * tr(s) - 3 * det(s));
    }), du_dX_c);
    return std::vector<double>(vm_c.data(), vm_c.data() + vm_c.sz);
  }

  // the vertex values: the quadratic nodes are ordered vertices first
  std::vector<double> displacement() const { return std::vector<double>(u.data.data(), u.data.data() + 2 * mesh.vert.shape[0]); }
  std::vector<double> nodes() const { return vertex_coordinates(mesh); }
  std::vector<uint32_t> cells() const { return cell_vertices(mesh, Geometry::Triangle); }

};

#ifdef __EMSCRIPTEN__

using namespace emscripten;

EMSCRIPTEN_BINDINGS(fracture_2D) {
  value_object<FractureSpecimen::Params>("FractureParams")
    .field("material", &FractureSpecimen::Params::material)
    .field("state", &FractureSpecimen::Params::state)
    .field("E", &FractureSpecimen::Params::E)
    .field("nu", &FractureSpecimen::Params::nu)
    .field("opening", &FractureSpecimen::Params::opening)
    .field("loadSteps", &FractureSpecimen::Params::load_steps);

  class_<FractureSpecimen>("FractureSpecimen")
    .constructor<FractureSpecimen::Params>()
    .function("step", &FractureSpecimen::step)
    .function("stepCount", +[](const FractureSpecimen & s) { return s.step_count; })
    .function("newtonIterations", +[](const FractureSpecimen & s) { return s.newton_iterations; })
    .function("residualNorm", +[](const FractureSpecimen & s) { return s.residual_norm; })
    .function("curve", +[](const FractureSpecimen & s) { return to_f64(s.curve); })
    .function("displacement", +[](const FractureSpecimen & s) { return to_f64(s.displacement()); })
    .function("vonMises", +[](FractureSpecimen & s) { return to_f64(s.von_mises()); })
    .function("nodes", +[](const FractureSpecimen & s) { return to_f64(s.nodes()); })
    .function("cells", +[](const FractureSpecimen & s) { return to_u32(s.cells()); });
}

#else

int main() {
  const char * materials[] = {"linear elastic", "neo-Hookean"};
  const char * states[] = {"plane strain", "plane stress"};
  for (int material = 0; material < 2; material++) {
    for (int state = 0; state < 2; state++) {
      FractureSpecimen::Params p;
      p.material = material;
      p.state = state;
      FractureSpecimen specimen(p);
      while (specimen.step_count < p.load_steps) { specimen.step(); }
      std::vector<double> vm = specimen.von_mises();
      double peak = 0;
      for (double s : vm) { peak = std::max(peak, s); }
      std::cout << materials[material] << ", " << states[state] << ": reaction force per step (N/mm):";
      for (int k = 1; k <= p.load_steps; k++) { std::cout << " " << specimen.curve[2 * k + 1]; }
      std::cout << "\n   final Newton iterations " << specimen.newton_iterations << ", residual " << specimen.residual_norm
                << ", peak von Mises " << peak << " MPa" << std::endl;
    }
  }
  return 0;
}

#endif
