// 2D parallel-plate capacitor: two conducting rectangles a gap apart, a
// dielectric block filling part of the gap, everything in a box of air.
// Each plate is pinned to its voltage at a single node; the plates are
// modeled as a material of enormous permittivity, so the rest of the plate
// floats to that voltage.  The electrostatic energy of the solution gives the
// capacitance, C = 2 W / V^2, fringing fields included.
//
// Lengths are in mm, permittivities relative to eps0, so the capacitance
// comes out in units of eps0 per unit depth.

#include "example_common.hpp"

#include "femto/field.hpp"
#include "forall.hpp"

#include <cmath>
#include <iostream>

using namespace femto;

struct Capacitor {

  struct Params {
    double Lx = 40.0, Ly = 30.0;        // the box of air
    double plate_width = 16.0;
    double plate_thickness = 1.0;
    double gap = 4.0;
    double dielectric_width = 8.0;      // of the block centered in the gap, 0 for none
    double eps_r = 4.0;                 // of that block
    double voltage = 1.0;               // between the plates
    double h = 0.5;                     // element size
  };

  enum Region : int { AIR = 0, PLATE_TOP = 1, PLATE_BOTTOM = 2, DIELECTRIC = 3 };

  // the permittivity at a quadrature point: the plates conduct, the block is the dielectric
  struct Permittivity {
    double eps_r;
    mat2 operator()(const int & region) const {
      double eps = (region == AIR) ? 1.0 : (region == DIELECTRIC) ? eps_r : 1.0e6;
      return mat2(eps * Identity<2>());
    }
  };

  Params params;
  Mesh<> mesh;
  Field<Family::H1> V;
  std::vector<int> region;          // per element
  std::vector<int> pinned;          // the two dof ids holding the voltage
  double energy = 0, capacitance = 0;

  // The plates only exist as whole elements, so an element size that does not
  // divide the gap would silently change the one dimension the capacitance is
  // most sensitive to.  Instead the gap picks the (square) cell size: an even
  // number of cells spans it, and the box of air is trimmed to a whole number
  // of those cells.  Everything else is snapped to them, and params holds the
  // geometry that was actually built -- what ideal_capacitance compares
  // against and what the browser panel reports.
  Capacitor(Params p) : params(p) {
    uint32_t cells_per_gap = 2 * std::max<uint32_t>(1, uint32_t(std::round(0.5 * p.gap / p.h)));
    double d = p.gap / cells_per_gap;
    uint32_t nx = 2 * std::max<uint32_t>(1, uint32_t(std::round(0.5 * p.Lx / d)));
    uint32_t ny = 2 * std::max<uint32_t>(1, uint32_t(std::round(0.5 * p.Ly / d)));
    params.Lx = nx * d;
    params.Ly = ny * d;
    params.h = d;
    mesh = Mesh<>::cuboid({nx, ny}, vec2{params.Lx, params.Ly});

    auto centered = [&](double v) { return 2 * std::max(1.0, std::round(0.5 * v / d)) * d; };
    params.plate_width = centered(p.plate_width);
    params.plate_thickness = std::max(1.0, std::round(p.plate_thickness / d)) * d;
    params.dielectric_width = (p.dielectric_width > 0) ? centered(p.dielectric_width) : 0.0;

    double cx = 0.5 * params.Lx, cy = 0.5 * params.Ly;
    double y_top = cy + 0.5 * params.gap, y_bot = cy - 0.5 * params.gap;
    auto inside = [](double v, double lo, double hi) { return v > lo && v < hi; };

    region.assign(mesh.quad.shape[0], AIR);
    for (uint32_t e = 0; e < mesh.quad.shape[0]; e++) {
      vec2 c = centroid<2>(mesh, Geometry::Quadrilateral, e);
      bool in_x = inside(c[0], cx - 0.5 * params.plate_width, cx + 0.5 * params.plate_width);
      if (in_x && inside(c[1], y_top, y_top + params.plate_thickness)) { region[e] = PLATE_TOP; }
      if (in_x && inside(c[1], y_bot - params.plate_thickness, y_bot)) { region[e] = PLATE_BOTTOM; }
      if (inside(c[0], cx - 0.5 * params.dielectric_width, cx + 0.5 * params.dielectric_width) && inside(c[1], y_bot, y_top)) { region[e] = DIELECTRIC; }
    }

    V = create_field<Family::H1>(mesh, 1, 1);
    pinned = {
      int(nearest_vertex<2>(mesh, vec2{cx, y_top + 0.5 * params.plate_thickness})),
      int(nearest_vertex<2>(mesh, vec2{cx, y_bot - 0.5 * params.plate_thickness}))
    };
  }

  void solve() {
    BasisFunction phi(V);
    Domain domain(mesh, MeshQuadratureRule(2));

    uint32_t nq = total(domain.num_qpts), qpe = nq / mesh.quad.shape[0];
    nd::cpu_array<double, 3> eps_q = forall(Permittivity{params.eps_r}, at_quadrature_points(region, qpe));

    sparse_matrix K = integrate(dot(grad(phi), eps_q, grad(phi)), domain);

    ConstrainedSystem system;
    system.dofs = pinned;
    system.set_matrix(K);
    vector values(2);
    values[0] = +0.5 * params.voltage;
    values[1] = -0.5 * params.voltage;
    V.v() = system.solve(zeros(int(V.size())), values);

    // W = 1/2 V^T K V, in units of eps0 (per unit depth); C = 2 W / V^2
    vector v = V.v();
    vector Kv = K(v);
    energy = 0.5 * dot(v, Kv);
    capacitance = 2.0 * energy / (params.voltage * params.voltage);
  }

  // the ideal parallel-plate value, for comparison: no fringing, the
  // dielectric block in parallel with the air on either side of it
  double ideal_capacitance() const {
    double air_width = params.plate_width - params.dielectric_width;
    return (air_width + params.eps_r * params.dielectric_width) / params.gap;
  }

  std::vector<double> voltage() const { return std::vector<double>(V.data.data(), V.data.data() + V.data.shape[0]); }
  std::vector<double> nodes() const { return vertex_coordinates(mesh); }
  std::vector<uint32_t> cells() const { return cell_vertices(mesh, Geometry::Quadrilateral); }

};

#ifdef __EMSCRIPTEN__

using namespace emscripten;

EMSCRIPTEN_BINDINGS(capacitor_2D) {
  value_object<Capacitor::Params>("CapacitorParams")
    .field("Lx", &Capacitor::Params::Lx)
    .field("Ly", &Capacitor::Params::Ly)
    .field("plateWidth", &Capacitor::Params::plate_width)
    .field("plateThickness", &Capacitor::Params::plate_thickness)
    .field("gap", &Capacitor::Params::gap)
    .field("dielectricWidth", &Capacitor::Params::dielectric_width)
    .field("epsR", &Capacitor::Params::eps_r)
    .field("voltage", &Capacitor::Params::voltage)
    .field("h", &Capacitor::Params::h);

  class_<Capacitor>("Capacitor")
    .constructor<Capacitor::Params>()
    .function("solve", &Capacitor::solve)
    .function("capacitance", +[](const Capacitor & c) { return c.capacitance; })
    .function("idealCapacitance", &Capacitor::ideal_capacitance)
    .function("geometry", +[](const Capacitor & c) { return c.params; })   // snapped to the mesh
    .function("energy", +[](const Capacitor & c) { return c.energy; })
    .function("voltage", +[](const Capacitor & c) { return to_f64(c.voltage()); })
    .function("nodes", +[](const Capacitor & c) { return to_f64(c.nodes()); })
    .function("cells", +[](const Capacitor & c) { return to_u32(c.cells()); })
    .function("region", +[](const Capacitor & c) { return to_i32(c.region); });
}

#else

int main() {
  Capacitor::Params p;
  for (double wd : {0.0, 8.0, 16.0}) {
    p.dielectric_width = wd;
    Capacitor cap(p);
    cap.solve();
    std::cout << "dielectric width " << wd << " mm: C = " << cap.capacitance << " eps0 per unit depth"
              << " (ideal parallel-plate value " << cap.ideal_capacitance() << ")" << std::endl;
  }

  // the fringing field is what the ideal formula misses, and it should not
  // depend on the element size once the geometry is snapped to the mesh
  p.dielectric_width = 8.0;
  std::cout << "element size sweep (dielectric width 8 mm):" << std::endl;
  for (double h : {2.0, 1.0, 0.5, 0.25}) {
    p.h = h;
    Capacitor cap(p);
    cap.solve();
    std::cout << "   h = " << h << " mm: C = " << cap.capacitance << " eps0, fringing "
              << cap.capacitance - cap.ideal_capacitance() << " eps0" << std::endl;
  }
  return 0;
}

#endif
