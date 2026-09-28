// 3D solid mechanics: a coffee mug is generated procedurally and its lowest
// elastic vibration modes are computed.  The assembly of the mass and
// stiffness matrices and the shift-inverted Lanczos eigensolver live in
// mug_analysis.hpp; this example wraps them in a simulation object with a
// native main and a browser binding.
//
// SI units: meters, kilograms, pascals; frequencies in Hz.

#include "example_common.hpp"
#include "mug_analysis.hpp"

#include <cmath>
#include <iostream>

using namespace femto;

struct CoffeeCupModal {

  struct Params {
    double base_radius = 0.035, rim_radius = 0.040, height = 0.095, thickness = 0.005;
    int handles = 1;
    int p = 1;                          // polynomial order of the mesh
    double rho = 2400.0;                // ceramic
    double E = 70.0e9;
    double nu = 0.2;
    int num_modes = 6;
  };

  Params params;
  Mesh<> mesh;
  mug::Analysis analysis;
  mug::ModalResult result;

  CoffeeCupModal(Params p) : params(p) {
    mesh = Mesh<>::coffee_mug(p.base_radius, p.rim_radius, p.height, p.thickness, p.handles, p.p);
  }

  void solve() {
    analysis = mug::analyze(mesh, params.rho, params.E, params.nu);
    result = mug::modal(analysis, uint32_t(params.num_modes));
  }

  std::vector<double> frequencies() const {
    std::vector<double> f(result.eigenvalues.size());
    for (size_t i = 0; i < f.size(); i++) { f[i] = std::sqrt(std::max(0.0, result.eigenvalues[i])) / (2 * M_PI); }
    return f;
  }

  // mode i, 3 components per vertex
  std::vector<double> mode(int i) const {
    size_t n = size_t(mesh.vert.shape[0]) * 3;
    std::vector<double> out(n);
    for (size_t k = 0; k < n; k++) { out[k] = result.modes[size_t(i) * n + k]; }
    return out;
  }

  std::vector<double> nodes() const { return vertex_coordinates(mesh); }
  std::vector<uint32_t> cells() const { return cell_vertices(mesh, Geometry::Hexahedron); }
  std::vector<uint32_t> surface() const { return surface_facets(mesh); }

};

#ifdef __EMSCRIPTEN__

using namespace emscripten;

EMSCRIPTEN_BINDINGS(coffee_cup_modal) {
  value_object<CoffeeCupModal::Params>("CoffeeCupParams")
    .field("baseRadius", &CoffeeCupModal::Params::base_radius)
    .field("rimRadius", &CoffeeCupModal::Params::rim_radius)
    .field("height", &CoffeeCupModal::Params::height)
    .field("thickness", &CoffeeCupModal::Params::thickness)
    .field("handles", &CoffeeCupModal::Params::handles)
    .field("p", &CoffeeCupModal::Params::p)
    .field("rho", &CoffeeCupModal::Params::rho)
    .field("E", &CoffeeCupModal::Params::E)
    .field("nu", &CoffeeCupModal::Params::nu)
    .field("numModes", &CoffeeCupModal::Params::num_modes);

  class_<CoffeeCupModal>("CoffeeCupModal")
    .constructor<CoffeeCupModal::Params>()
    .function("solve", &CoffeeCupModal::solve)
    .function("frequencies", +[](const CoffeeCupModal & c) { return to_f64(c.frequencies()); })
    .function("mode", +[](const CoffeeCupModal & c, int i) { return to_f64(c.mode(i)); })
    .function("iterations", +[](const CoffeeCupModal & c) { return c.result.iterations; })
    .function("solveMilliseconds", +[](const CoffeeCupModal & c) { return c.result.solve_ms; })
    .function("nodes", +[](const CoffeeCupModal & c) { return to_f64(c.nodes()); })
    .function("cells", +[](const CoffeeCupModal & c) { return to_u32(c.cells()); })
    .function("surface", +[](const CoffeeCupModal & c) { return to_u32(c.surface()); });
}

#else

int main() {
  CoffeeCupModal cup(CoffeeCupModal::Params{});
  std::cout << cup.mesh.hex.shape[0] << " hexahedra, " << cup.mesh.vert.shape[0] << " vertices" << std::endl;
  cup.solve();
  std::vector<double> f = cup.frequencies();
  std::cout << "lowest elastic modes (Hz):";
  for (double fi : f) { std::cout << " " << fi; }
  std::cout << "\n" << cup.result.iterations << " Lanczos iterations, " << cup.result.solve_ms << " ms" << std::endl;
  return 0;
}

#endif
