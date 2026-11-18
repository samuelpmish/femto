// SPDX-FileCopyrightText: Copyright (c) 2023-2026 Sam Mish
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT

// 2D magnetostatics of a Hall effect current sensor: the wire being measured
// passes through the window of a ferrite core, which gathers the field
// circling the wire and drives it across a narrow gap, where a Hall element
// reads it.  By Ampere's law the core makes the reading nearly independent of
// where the wire sits in the window, and nearly blind to a second wire outside
// the core; without the core neither holds.
//
// The out-of-plane vector potential A is a quadratic (P2) field solving
// -div(nu grad A) = J in a disk of air, and B = curl(A e_z).  The rim of the
// disk absorbs rather than reflects: dA/dr + A / R = 0 there, which a line
// current's field meets up to a constant and its dipole part exactly, so the
// disk can stay small.  Each wire is a line current, a point source
// mu0 I phi_i(x) in the plane, so wires move without remeshing: the matrix is
// factorized once per core and every new wire position is two
// back-substitutions, one per wire, which also splits the reading into the
// signal from the wire being measured and the interference from the other
// one.  Without a core the core's cells are simply air, so both cases share
// one mesh.
//
// The mesh is generated on the fly by Triangle (cmake/triangle.cmake), from
// the outlines of the core, the sensor and the rim, refined to quality
// triangles no larger than a size function allows.  Its regional attributes
// tag each triangle as air, core or sensor.
//
// The Hall element is modeled on a real part, and its output drives an analog
// meter (see meter_current).
//
// Lengths are in mm.  With a line source the potential does not depend on the
// length unit (both sides of the weak form are dimensionless in it), so A
// comes out in T m and only B = grad A picks up the 1/mm.

#include "example_common.hpp"

#include "femto/field.hpp"
#include "forall.hpp"

#include <cmath>
#include <iostream>

#define REAL double
#define VOID void
extern "C" {
#include "triangle.h"
}

using namespace femto;

struct HallSensor {

  static constexpr double mu0 = 4.0e-7 * M_PI;
  static constexpr double mm = 1.0e-3;

  // the geometry
  static constexpr double CORE_W = 40.0, CORE_R = 6.0;     // outer square, corner radius
  static constexpr double HOLE_W = 18.0, HOLE_R = 1.0;     // the window
  static constexpr double GAP = 4.0;                       // the slot through the bottom of the core
  static constexpr double SENSOR_W = 1.1, SENSOR_H = 2.9;  // the Hall element's package, on edge across the gap
  static constexpr double SENSOR_Y = -0.25 * (CORE_W + HOLE_W);
  static constexpr double AIR_R = 120.0;
  static constexpr double WIRE_R = 2.5;                    // as drawn and kept clear of the core
  static constexpr double REACH = 70.0;                    // how far out the external wire may go

  // The Hall element is an AKM HW-101A: InSb, in a SOT143 package 2.9 x 1.55
  // x 1.1 mm, standing in the gap with its face across the field.  At 1 V
  // constant-voltage drive the datasheet's typical curve gives about
  // 5.8 mV/mT at low field (its ranks span 122-370 mV at 50 mT), and the
  // output resistance is 240-550 ohm (350 in its example).  The output drives
  // a 50-0-50 uA center-zero meter with a 1 kohm movement.  The element's
  // offset voltage (up to 7 mV) is taken as nulled
  static constexpr double HALL_DRIVE = 1.0;               // V
  static constexpr double HALL_SENSITIVITY = 5.8;         // V per T per volt of drive
  static constexpr double HALL_R_OUT = 350.0;             // ohm
  static constexpr double METER_R = 1000.0;               // ohm
  static constexpr double METER_FULL_SCALE = 50.0e-6;     // A, either side of zero

  static constexpr uint32_t DEGREE = 2;
  static constexpr uint32_t SUBDIVISIONS = 4;             // per element edge, for drawing


  enum Region : int { AIR = 0, CORE = 1, SENSOR = 2 };

  // a straight wire through the plane, at (x, y) in mm, carrying current A out of the page
  struct Wire { double x, y, current; };

  // the reluctivity relative to air at a quadrature point
  struct Reluctivity {
    double mu_r;
    mat2 operator()(const int & region) const { return mat2(((region == CORE) ? 1.0 / mu_r : 1.0) * Identity<2>()); }
  };

  double mu_r;
  std::vector<int> region;               // per triangle
  Mesh<> mesh;
  Domain<> domain;                       // for the matrix
  Domain<> centers;                      // one point per triangle: B is linear, so its centroid value is its mean
  Domain<> drawing;                      // a grid of SUBDIVISIONS^2 small triangles per element
  Field<Family::H1> A, A_primary, A_external;
  std::vector<uint32_t> sensor_cells;
  std::vector<double> sensor_weights;    // their share of the sensor's area
  sparse_factorization invK;
  double signal = 0, interference = 0;   // B across the gap at the sensor, from each wire

  // the target element size: 1.2 mm through the core and window, 0.4 mm
  // across the gap where the sensor reads the field, growing out into the air
  static double size(vec2 x) {
    bool gap = std::fabs(x[0]) < 0.5 * GAP + 1.0 && x[1] > -0.5 * CORE_W - 1.0 && x[1] < -0.5 * HOLE_W + 1.0;
    return gap ? 0.4 : std::max(1.2, 0.08 * (norm(x) - 12.0));
  }

  // a square of half-side a with its corners rounded to radius r, as a
  // polyline running counterclockwise from (x0, -a) on its bottom side round
  // to (-x0, -a)
  static std::vector<vec2> rounded_square(double a, double r, double x0) {
    std::vector<vec2> out{vec2{x0, -a}};
    double s = a - r;
    for (vec2 c : {vec2{s, -s}, vec2{s, s}, vec2{-s, s}, vec2{-s, -s}}) {
      double start = std::atan2(c[1], c[0]) - 0.25 * M_PI;   // the bottom right corner sweeps -90 to 0 degrees
      int n = std::max(2, int(std::ceil(0.5 * M_PI * r / size(c))));
      for (int k = 0; k <= n; k++) {
        double t = start + 0.5 * M_PI * k / n;
        out.push_back(c + r * vec2{std::cos(t), std::sin(t)});
      }
    }
    out.push_back(vec2{-x0, -a});
    return out;
  }

  // Triangle's input is the outlines as closed loops: the core is one C, its
  // outside then its window walked backward, joined by the faces of the slot.
  // A point inside the core and one inside the sensor seed their regions; the
  // air is what is left (attribute 0).  Fills region on the way
  Mesh<> build_mesh() {
    std::vector<vec2> core = rounded_square(0.5 * CORE_W, CORE_R, 0.5 * GAP);
    std::vector<vec2> window = rounded_square(0.5 * HOLE_W, HOLE_R, 0.5 * GAP);
    core.insert(core.end(), window.rbegin(), window.rend());
    double sx = 0.5 * SENSOR_W, sy = 0.5 * SENSOR_H;
    std::vector<vec2> sensor{vec2{-sx, SENSOR_Y - sy}, vec2{sx, SENSOR_Y - sy}, vec2{sx, SENSOR_Y + sy}, vec2{-sx, SENSOR_Y + sy}};
    std::vector<vec2> rim(std::max(48, int(std::ceil(2 * M_PI * AIR_R / size(vec2{AIR_R, 0.0})))));
    for (size_t k = 0; k < rim.size(); k++) { rim[k] = AIR_R * vec2{std::cos(2 * M_PI * k / rim.size()), std::sin(2 * M_PI * k / rim.size())}; }

    std::vector<double> points;
    std::vector<int> segments;
    for (const std::vector<vec2> & loop : {core, sensor, rim}) {
      int start = int(points.size() / 2), n = int(loop.size());
      for (int i = 0; i < n; i++) {
        points.insert(points.end(), {loop[i][0], loop[i][1]});
        segments.insert(segments.end(), {start + i, start + (i + 1) % n});
      }
    }
    std::vector<double> regions{0.0, 0.25 * (CORE_W + HOLE_W), double(CORE), 0.0,
                                0.0, SENSOR_Y, double(SENSOR), 0.0};

    // p: the outlines, z: zero-based, q30: no angle under 30 degrees, u: refine
    // while triunsuitable() says so, A: regional attributes, Q: quietly, B/P:
    // no boundary markers or segments back
    triangulateio in{}, out{};
    in.pointlist = points.data();
    in.numberofpoints = int(points.size() / 2);
    in.segmentlist = segments.data();
    in.numberofsegments = int(segments.size() / 2);
    in.regionlist = regions.data();
    in.numberofregions = int(regions.size() / 4);
    char switches[] = "pzq30uAQBP";
    triangulate(switches, &in, &out, nullptr);

    uint32_t nv = uint32_t(out.numberofpoints), nt = uint32_t(out.numberoftriangles);
    nd::array< double, 2, memory::space::cpu > nodes({nv, 2});
    nd::array< uint32_t, 2, memory::space::cpu > tris({nt, 3}), no_quads({0, 4});
    for (uint32_t i = 0; i < 2 * nv; i++) { nodes.data()[i] = out.pointlist[i]; }
    for (uint32_t i = 0; i < 3 * nt; i++) { tris.data()[i] = uint32_t(out.trianglelist[i]); }   // counterclockwise, as femto wants
    region.resize(nt);
    for (uint32_t t = 0; t < nt; t++) { region[t] = int(out.triangleattributelist[t]); }
    for (void * p : {(void *)out.pointlist, (void *)out.pointattributelist, (void *)out.trianglelist, (void *)out.triangleattributelist}) { trifree(p); }
    return Mesh<>::create_2D(nodes, 1, tris, no_quads);
  }

  // the drawing points of a reference triangle, a grid with SUBDIVISIONS
  // intervals along each edge, numbered row by row from the edge xi_1 = 0
  static MeshQuadratureRule drawing_points() {
    MeshQuadratureRule rule(1);
    uint32_t n = SUBDIVISIONS, k = 0;
    rule.tri.points = nd::array< double, 2, memory::space::cpu >({(n + 1) * (n + 2) / 2, 2});
    rule.tri.weights = nd::array< double, 1, memory::space::cpu >({(n + 1) * (n + 2) / 2});   // unused
    for (uint32_t j = 0; j <= n; j++) {
      for (uint32_t i = 0; i + j <= n; i++, k++) {
        rule.tri.points(k, 0) = double(i) / n;
        rule.tri.points(k, 1) = double(j) / n;
        rule.tri.weights(k) = 0.0;
      }
    }
    return rule;
  }

  // the allowed position nearest x: the primary wire (0) stays in the window,
  // the external one (1) outside the core and within reach.  WIRE_R exceeds
  // the window's corner radius, so the window inset by it is a sharp square;
  // the core's outline offset by it is a square with corners rounded to
  // CORE_R + WIRE_R, which a point inside is pushed out to the nearest side
  // or corner of
  static vec2 constrain(int which, vec2 x) {
    if (which == 0) { return clamp(x, -(0.5 * HOLE_W - WIRE_R), 0.5 * HOLE_W - WIRE_R); }
    double s = 0.5 * CORE_W - CORE_R, R = CORE_R + WIRE_R;
    vec2 p = abs(x), q = p - vec2{s, s};
    if (q[0] > 0 && q[1] > 0) {
      if (norm(q) < R) { p = vec2{s, s} + R * normalize(q); }
    } else if (q[0] >= q[1]) {
      p[0] = std::max(p[0], s + R);
    } else {
      p[1] = std::max(p[1], s + R);
    }
    x = vec2{std::copysign(p[0], x[0]), std::copysign(p[1], x[1])};
    return (norm(x) > REACH) ? x * (REACH / norm(x)) : x;
  }

  // what an ideal core (mu_r -> infinity) puts across the gap: all of
  // Ampere's mu0 I drops there
  static double ideal_reading(double current) { return mu0 * current / (GAP * mm); }

  vec2 X(uint32_t i) const { return load<vec2>(mesh.X.data, i); }
  vec2 X(uint32_t e, int j) const { return X(mesh.tri(e, j).index); }

  HallSensor(double mu_r) : mu_r(mu_r), mesh(build_mesh()), domain(mesh, MeshQuadratureRule(2)),
                            centers(mesh, MeshQuadratureRule(1)), drawing(mesh, drawing_points()) {
    uint32_t nt = mesh.tri.shape[0];
    double sensor_area = 0;
    for (uint32_t e = 0; e < nt; e++) {
      if (region[e] != SENSOR) continue;
      double area = 0.5 * std::fabs(dot(cross(X(e, 1) - X(e, 0)), X(e, 2) - X(e, 0)));
      sensor_cells.push_back(e);
      sensor_weights.push_back(area);
      sensor_area += area;
    }
    for (double & w : sensor_weights) { w /= sensor_area; }

    A = create_field<Family::H1>(mesh, DEGREE, 1);
    A_primary = create_field<Family::H1>(mesh, DEGREE, 1);
    A_external = create_field<Family::H1>(mesh, DEGREE, 1);
    BasisFunction phi(A);
    nd::cpu_array<double, 3> nu_q = forall(Reluctivity{mu_r}, at_quadrature_points(region, total(domain.num_qpts) / nt));
    sparse_matrix K = integrate(dot(grad(phi), nu_q, grad(phi)), domain);

    // the rim's int phi phi / R, assembled on K's pattern so the two add
    Domain rim(boundary_of(mesh), MeshQuadratureRule(3));
    nd::cpu_array<double, 1> robin_q({total(rim.num_qpts)});
    for (uint32_t q = 0; q < robin_q.shape[0]; q++) { robin_q(q) = 1.0 / AIR_R; }
    sparse_matrix Kb = K;
    Kb = integrate(dot(phi, robin_q, phi), rim);
    for (size_t i = 0; i < K.nnz; i++) { K.values[i] += Kb.values[i]; }
    K.symmetry = Symmetry::Symmetric;
    K.definiteness = Definiteness::PositiveDefinite;
    invK = inv(K);
  }

  // the potential of one wire, in T m.  Its load is mu0 I phi_i(x) on the
  // nodes of the triangle holding x, at the reference coordinates the
  // barycentric ones give; the triangle with the largest smallest barycentric
  // coordinate is the one holding x, even when x lies on an edge
  vector potential(Wire w) {
    vec2 x{w.x, w.y};
    uint32_t best = 0;
    vec3 best_l{-1.0e300, -1.0e300, -1.0e300};
    for (uint32_t e = 0; e < mesh.tri.shape[0]; e++) {
      vec2 a = X(e, 0), b = X(e, 1), c = X(e, 2);
      vec3 l = vec3{dot(cross(c - b), x - b), dot(cross(a - c), x - c), dot(cross(b - a), x - a)} / dot(cross(b - a), c - a);
      if (min(l) > min(best_l)) { best = e; best_l = l; }
    }
    FiniteElement<Geometry::Triangle, Family::H1> element{DEGREE};
    uint32_t dofs[6];
    element.indices(A.offsets, &mesh.tri(best, 0), dofs);
    vector f = zeros(int(A.size()));
    for (uint32_t k = 0; k < element.num_nodes(); k++) { f[dofs[k]] = mu0 * w.current * element.shape_function(vec2{best_l[1], best_l[2]}, k); }
    return dot(invK, f);
  }

  // B = curl(A e_z), in T, at a domain's points
  nd::cpu_array<double, 2> flux_density(const Field<Family::H1> & a, const Domain<> & where) const {
    nd::cpu_array<double, 3> dA_q = evaluate(grad(a), where);
    return forall(+[](const vec2 & g) { return vec2(-cross(g) / mm); }, dA_q);
  }

  // what the Hall element reads: B across the gap, averaged over the element
  double sensor_reading(const Field<Family::H1> & a) const {
    nd::cpu_array<double, 2> B_c = flux_density(a, centers);
    double sum = 0;
    for (size_t k = 0; k < sensor_cells.size(); k++) { sum += sensor_weights[k] * B_c(sensor_cells[k], 0); }
    return sum;
  }

  void solve(Wire primary, Wire external) {
    A_primary.v() = potential(primary);
    A_external.v() = potential(external);
    A.v() = vector(A_primary.v() + A_external.v());
    signal = sensor_reading(A_primary);
    interference = sensor_reading(A_external);
  }

  // the current the meter sees: the Hall voltage, divided between the
  // element's output resistance and the meter's
  double meter_current() const { return HALL_SENSITIVITY * HALL_DRIVE * (signal + interference) / (HALL_R_OUT + METER_R); }

  // the drawing: its points, its small triangles (counterclockwise, into the
  // points), and the total field at the points
  nd::cpu_array<double, 3> drawing_positions() const { return evaluate(mesh.X, drawing); }
  std::vector<uint32_t> drawing_triangles() const {
    uint32_t n = SUBDIVISIONS, np = (n + 1) * (n + 2) / 2;
    auto id = [n](uint32_t i, uint32_t j) { return j * (n + 1) - j * (j - 1) / 2 + i; };
    std::vector<uint32_t> out;
    for (uint32_t e = 0; e < mesh.tri.shape[0]; e++) {
      for (uint32_t j = 0; j < n; j++) {
        for (uint32_t i = 0; i + j < n; i++) {
          out.insert(out.end(), {e * np + id(i, j), e * np + id(i + 1, j), e * np + id(i, j + 1)});
          if (i + j + 1 < n) { out.insert(out.end(), {e * np + id(i + 1, j), e * np + id(i + 1, j + 1), e * np + id(i, j + 1)}); }
        }
      }
    }
    return out;
  }
  nd::cpu_array<double, 3> drawing_potential() const { return evaluate(A, drawing); }
  nd::cpu_array<double, 2> drawing_flux_density() const { return flux_density(A, drawing); }

  std::vector<double> nodes() const { return vertex_coordinates(mesh); }
  std::vector<uint32_t> cells() const { return cell_vertices(mesh, Geometry::Triangle); }

};

// Triangle's -u test (see cmake/triangle.cmake): a triangle is split while it
// is half again as large as an equilateral one at the target size where it
// sits, which leaves the typical triangle about that size
extern "C" int triunsuitable(double * a, double * b, double * c, double area) {
  double s = HallSensor::size(vec2{a[0] + b[0] + c[0], a[1] + b[1] + c[1]} / 3.0);
  return area > 1.5 * 0.433 * s * s;
}

#ifdef __EMSCRIPTEN__

using namespace emscripten;

EMSCRIPTEN_BINDINGS(hall_sensor_2D) {
  value_object<HallSensor::Wire>("HallWire")
    .field("x", &HallSensor::Wire::x)
    .field("y", &HallSensor::Wire::y)
    .field("current", &HallSensor::Wire::current);

  class_<HallSensor>("HallSensor")
    .constructor<double>()
    .class_property("wireRadius", &HallSensor::WIRE_R)
    .class_property("holeWidth", &HallSensor::HOLE_W)
    .class_property("coreWidth", &HallSensor::CORE_W)
    .class_property("sensorY", &HallSensor::SENSOR_Y)
    .class_property("sensorWidth", &HallSensor::SENSOR_W)
    .class_property("sensorHeight", &HallSensor::SENSOR_H)
    .class_property("meterFullScale", &HallSensor::METER_FULL_SCALE)
    .class_function("constrain", +[](int which, double x, double y) {
      vec2 p = HallSensor::constrain(which, vec2{x, y});
      return to_f64(std::vector<double>{p[0], p[1]});
    })
    .function("solve", &HallSensor::solve)
    .function("signal", +[](const HallSensor & h) { return h.signal; })
    .function("interference", +[](const HallSensor & h) { return h.interference; })
    .function("meterCurrent", &HallSensor::meter_current)
    .function("drawingPositions", +[](const HallSensor & h) { return to_f64(h.drawing_positions()); })
    .function("drawingTriangles", +[](const HallSensor & h) { return to_u32(h.drawing_triangles()); })
    .function("drawingPotential", +[](const HallSensor & h) { return to_f64(h.drawing_potential()); })
    .function("drawingB", +[](const HallSensor & h) { return to_f64(h.drawing_flux_density()); })
    .function("nodes", +[](const HallSensor & h) { return to_f64(h.nodes()); })
    .function("cells", +[](const HallSensor & h) { return to_u32(h.cells()); })
    .function("region", +[](const HallSensor & h) { return to_i32(h.region); });
}

#else

int main() {
  using Wire = HallSensor::Wire;
  const double I = 10.0, a = 0.5 * HallSensor::HOLE_W - HallSensor::WIRE_R;
  const Wire none{0.0, 0.0, 0.0}, centered{0.0, 0.0, I}, external{45.0, 0.0, I};
  double spread[2], crosstalk[2];
  bool ok = true;

  // a line current's B across the gap at the sensor, in free space
  auto line_current = [](Wire w) {
    vec2 r = vec2{0.0, HallSensor::SENSOR_Y} - vec2{w.x, w.y};
    return -HallSensor::mu0 * w.current * r[1] / (2 * M_PI * dot(r, r) * HallSensor::mm);
  };

  for (int with_core : {0, 1}) {
    HallSensor h(with_core ? 1000.0 : 1.0);
    if (!with_core) {
      std::cout << h.mesh.tri.shape[0] << " triangles, " << h.A.size() << " P" << HallSensor::DEGREE << " dofs, "
                << h.sensor_cells.size() << " triangles in the sensor" << std::endl;
    }

    // the primary wire at the center of the window and in each corner of it
    double lo = 1e300, hi = -1e300;
    for (vec2 x : {vec2{a, a}, vec2{-a, a}, vec2{-a, -a}, vec2{a, -a}}) {
      h.solve({x[0], x[1], I}, none);
      lo = std::min(lo, h.signal);
      hi = std::max(hi, h.signal);
    }
    h.solve(centered, external);
    spread[with_core] = (hi - lo) / h.signal;
    crosstalk[with_core] = h.interference / h.signal;
    std::cout << (with_core ? "core, mu_r 1000: " : "no core:         ")
              << "centered wire " << h.signal * 1e3 << " mT, "
              << 100 * spread[with_core] << "% spread over the window, "
              << "external wire at 45 mm " << h.interference * 1e3 << " mT (" << 100 * crosstalk[with_core] << "%), "
              << "meter " << h.meter_current() * 1e6 << " uA" << std::endl;

    // without the core both wires are line currents in free space, which
    // checks the source, the units and the absorbing rim
    if (!with_core) {
      std::cout << "   line currents at the sensor: " << line_current(centered) * 1e3 << " mT and "
                << line_current(external) * 1e3 << " mT" << std::endl;
      ok = ok && std::fabs(h.signal / line_current(centered) - 1) < 0.02;
      ok = ok && std::fabs(h.interference / line_current(external) - 1) < 0.02;
    } else {
      std::cout << "   ideal core (mu0 I / gap):    " << HallSensor::ideal_reading(I) * 1e3 << " mT" << std::endl;
    }
  }

  // the point of the core: it evens out the reading and screens out the neighbor
  ok = ok && spread[1] < 0.2 * spread[0] && std::fabs(crosstalk[1]) < 0.2 * std::fabs(crosstalk[0]);
  std::cout << (ok ? "self-check passed" : "SELF-CHECK FAILED") << std::endl;
  return ok ? 0 : 1;
}

#endif
