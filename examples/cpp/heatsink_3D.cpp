// 3D heat conduction: a TO-220 voltage regulator on a heatsink.  The package
// is modeled the way it is built -- a copper tab with its mounting hole, the
// epoxy body molded around it, the silicon die soldered to the tab, and three
// copper leads -- and the die dissipates the regulator's power.  Every exposed
// surface loses heat to the air by Newton's law of cooling, q = h (T - T_air),
// which is a Robin term integrated over the boundary facets.
//
// The three meshes are the bare package and two of the heatsinks commonly
// clipped or bolted to its tab.  They are tetrahedral meshes built by
// tools/generate_example_meshes.py (surfaces from that script, filled by
// fTetWild) and loaded from data/meshes/to220_*.msh; elements are sorted into
// copper, epoxy, silicon and aluminum by the same box tests the generator
// used, so the constants below have to match it.
//
// Lengths in mm, power in W, temperature in C: conductivities are W/(mm K),
// the film coefficient W/(mm^2 K).

#include "example_common.hpp"

#include "femto/field.hpp"
#include "forall.hpp"

#include <cmath>
#include <iostream>

using namespace femto;

struct HeatSink {

  enum class Design : int { BARE = 0, CLIP = 1, CHANNEL = 2, MULTIWATT = 3 };

  // h_air is the lumped surface coefficient of Newton's law of cooling,
  // q = h (T - T_air), applied to every exposed face: natural or forced
  // convection and radiation together, in W/(mm^2 K).  Still air alone gives a
  // small hot surface about 10 W/(m^2 K), radiation from black anodizing at
  // these temperatures adds 8 or so, and manufacturers rate these heatsinks
  // at a 75 K rise, where the sum is about 25 W/(m^2 K): that value
  // reproduces the published junction-to-ambient figures of all four
  // designs.  A fan blowing across the fins is 50 to 150.  Between closely
  // spaced fins convection is weaker than on a free surface, so a uniform h
  // flatters the finned designs a little.
  struct Params {
    int design = 1;                     // Design
    double power = 2.0;                 // W dissipated in the die
    double h_air = 2.5e-5;              // W/(mm^2 K): still air, black anodized, radiation included
    double T_air = 25.0;
  };

  static constexpr double T_junction_max = 125.0;   // what silicon is typically rated to

  enum Region : int { SINK = 0, TAB = 1, DIE = 2, LEAD = 3, PLASTIC = 4 };

  // TO-220AB, matching tools/generate_example_meshes.py (mm)
  static constexpr double BODY_W = 10.16, BODY_T = 4.6, TAB_T = 1.3;
  static constexpr double DIE_W = 4.0, DIE_T = 0.6, DIE_H = 4.0, DIE_Z0 = 3.0;

  // W/(mm K): aluminum, copper, silicon, copper, epoxy molding compound
  static constexpr double conductivity[5] = {0.2, 0.39, 0.15, 0.39, 8.0e-4};

  static const char * mesh_file(int design) {
    switch (Design(design)) {
      case Design::BARE:      return "to220_bare.msh";
      case Design::CHANNEL:   return "to220_channel.msh";
      case Design::MULTIWATT: return "to220_multiwatt.msh";
      default:                return "to220_clip.msh";
    }
  }

  // where an element sits in the package, from its centroid.  Anything behind
  // the tab or wider than the package is heatsink: the multiwatt extrusion's
  // fins stand proud of the mounting face, on either side of the package
  static Region classify(vec3 c) {
    if (c[1] < 0.0 || std::fabs(c[0]) > 0.6 * BODY_W) { return SINK; }
    if (c[1] < TAB_T) { return TAB; }
    bool die = std::fabs(c[0]) < 0.5 * DIE_W && c[1] < TAB_T + DIE_T &&
               c[2] > DIE_Z0 && c[2] < DIE_Z0 + DIE_H;
    if (die) { return DIE; }
    if (c[2] < 0.0) { return LEAD; }
    return PLASTIC;
  }

  // the conductivity and heat source at a quadrature point, from its cell's region
  struct Materials {
    double source;
    void operator()(const int & region, mat3 & k, double & q) const {
      k = mat3(conductivity[region] * Identity<3>());
      q = (region == DIE) ? source : 0.0;
    }
  };

  Params params;
  Mesh<> mesh;
  Field<Family::H1> T;
  std::vector<int> region;            // per tetrahedron
  std::vector<double> volume;         // per tetrahedron, mm^3
  double die_volume = 0, surface_area = 0;
  double T_max = 0, T_die = 0, thermal_resistance = 0;

  HeatSink(Params p) : params(p) {
#ifdef __EMSCRIPTEN__
    // the page fetches the mesh and writes it here before constructing us
    mesh = Mesh<>::load(std::string("/") + mesh_file(p.design));
#else
    mesh = Mesh<>::load(std::string(FEMTO_MESH_DIR) + mesh_file(p.design));
#endif

    uint32_t n = mesh.tet.shape[0];
    region.resize(n);
    volume.resize(n);
    for (uint32_t e = 0; e < n; e++) {
      region[e] = classify(centroid<3>(mesh, Geometry::Tetrahedron, e));
      vec3 x[4];
      for (int j = 0; j < 4; j++) { x[j] = load<vec3>(mesh.X.data, mesh.tet(e, j).index); }
      volume[e] = std::fabs(dot(x[1] - x[0], cross(x[2] - x[0], x[3] - x[0]))) / 6.0;
      if (region[e] == DIE) { die_volume += volume[e]; }
    }

    T = create_field<Family::H1>(mesh, 1, 1);
  }

  void solve() {
    BasisFunction phi(T);
    Domain domain(mesh, MeshQuadratureRule(2));
    Domain bdr(boundary_of(mesh), MeshQuadratureRule(2));

    // conductivity and the die's heat source at the volume quadrature points
    uint32_t nq = total(domain.num_qpts), qpe = nq / mesh.tet.shape[0];
    nd::cpu_array<int, 2> region_q = at_quadrature_points(region, qpe);
    nd::cpu_array<double, 3> k_q;
    nd::cpu_array<double, 2> q_q;
    forall(Materials{params.power / die_volume}, region_q, k_q, q_q);

    // the film coefficient and its ambient load on the boundary facets
    uint32_t nb = total(bdr.num_qpts);
    nd::cpu_array<double, 1> h_q({nb}), hT_q({nb}), ones_q({nb});
    for (uint32_t q = 0; q < nb; q++) { h_q(q) = params.h_air; hT_q(q) = params.h_air * params.T_air; ones_q(q) = 1.0; }
    Residual<Family::H1> area = integrate(dot(ones_q, phi), bdr);
    surface_area = total(area.v());

    // K T = f with K = int k grad(phi).grad(phi) + int_bdr h phi phi and
    // f = int q phi + int_bdr h T_air phi.  The boundary matrix is assembled
    // on the volume matrix's sparsity pattern so the two can be added
    sparse_matrix K = integrate(dot(grad(phi), k_q, grad(phi)), domain);
    sparse_matrix Kb = K;
    Kb = integrate(dot(phi, h_q, phi), bdr);
    for (size_t i = 0; i < K.nnz; i++) { K.values[i] += Kb.values[i]; }
    K.symmetry = Symmetry::Symmetric;
    K.definiteness = Definiteness::PositiveDefinite;

    Residual<Family::H1> f = integrate(dot(q_q, phi), domain);
    Residual<Family::H1> fb = integrate(dot(hT_q, phi), bdr);

    auto invK = inv(K);
    T.v() = dot(invK, f.v() + fb.v());

    T_max = -1e300;
    for (uint32_t i = 0; i < T.data.shape[0]; i++) { T_max = std::max(T_max, T.data(i, 0)); }

    // the junction temperature: the die's average
    double sum = 0, weight = 0;
    for (uint32_t e = 0; e < mesh.tet.shape[0]; e++) {
      if (region[e] != DIE) continue;
      double mean = 0;
      for (int j = 0; j < 4; j++) { mean += 0.25 * T.data(mesh.tet(e, j).index, 0); }
      sum += mean * volume[e];
      weight += volume[e];
    }
    T_die = sum / weight;
    thermal_resistance = (T_die - params.T_air) / params.power;
  }

  std::vector<double> temperature() const { return std::vector<double>(T.data.data(), T.data.data() + T.data.shape[0]); }
  std::vector<double> nodes() const { return vertex_coordinates(mesh); }
  std::vector<uint32_t> cells() const { return cell_vertices(mesh, Geometry::Tetrahedron); }
  std::vector<uint32_t> surface() const { return surface_facets(mesh); }

};

#ifdef __EMSCRIPTEN__

using namespace emscripten;

EMSCRIPTEN_BINDINGS(heatsink_3D) {
  value_object<HeatSink::Params>("HeatSinkParams")
    .field("design", &HeatSink::Params::design)
    .field("power", &HeatSink::Params::power)
    .field("hAir", &HeatSink::Params::h_air)
    .field("TAir", &HeatSink::Params::T_air);

  class_<HeatSink>("HeatSink")
    .constructor<HeatSink::Params>()
    .class_function("meshFile", +[](int design) { return std::string(HeatSink::mesh_file(design)); })
    .function("solve", &HeatSink::solve)
    .function("maxTemperature", +[](const HeatSink & s) { return s.T_max; })
    .function("junctionTemperature", +[](const HeatSink & s) { return s.T_die; })
    .function("thermalResistance", +[](const HeatSink & s) { return s.thermal_resistance; })
    .function("surfaceArea", +[](const HeatSink & s) { return s.surface_area; })
    .function("temperature", +[](const HeatSink & s) { return to_f64(s.temperature()); })
    .function("nodes", +[](const HeatSink & s) { return to_f64(s.nodes()); })
    .function("cells", +[](const HeatSink & s) { return to_u32(s.cells()); })
    .function("surface", +[](const HeatSink & s) { return to_u32(s.surface()); })
    .function("region", +[](const HeatSink & s) { return to_i32(s.region); });
}

#else

int main() {
  const char * names[] = {"bare package", "clip-on heatsink", "bolt-on channel heatsink", "multiwatt extrusion"};
  for (int design = 0; design < 4; design++) {
    HeatSink::Params p;
    p.design = design;
    HeatSink sink(p);
    sink.solve();
    std::cout << names[design] << ": " << sink.mesh.tet.shape[0] << " tetrahedra, "
              << sink.surface_area << " mm^2 of surface, junction at " << sink.T_die
              << " C at " << p.power << " W, " << sink.thermal_resistance << " K/W, so "
              << (HeatSink::T_junction_max - p.T_air) / sink.thermal_resistance << " W would reach "
              << HeatSink::T_junction_max << " C" << std::endl;
  }
  return 0;
}

#endif
