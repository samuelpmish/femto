#include "femto/mesh.hpp"

#include "femto/assert.hpp"

#include <array>
#include <cmath>
#include <algorithm>
#include <vector>

namespace femto {

namespace {

// the bottle's outline and the grading of the air around it, shared by the
// axisymmetric (hex) and planar (quad) generators. The bottle sits with its
// base on z = 0 and opens upward; the air extends air_below beneath it,
// air_above past the rim and air_margin past the body radially. Elements are
// about h wide near the bottle and grow geometrically away from it.
struct BottleProfile {
  double body_radius, body_height, shoulder_height, neck_radius, neck_length, t, h;
  static constexpr double growth = 1.25;  // outer column / level growth ratio

  BottleProfile(double body_radius, double body_height, double shoulder_height,
                double neck_radius, double neck_length, double thickness,
                double air_margin, double air_below, double air_above, double h)
    : body_radius(body_radius), body_height(body_height), shoulder_height(shoulder_height),
      neck_radius(neck_radius), neck_length(neck_length), t(thickness), h(h) {
    FEMTO_ASSERT(body_radius > 0 && body_height > 0 && shoulder_height > 0 && neck_radius > 0 &&
                 neck_length > 0 && thickness > 0 && air_margin > 0 && air_below > 0 && air_above > 0 && h > 0,
                 "Mesh::bottle() requires positive dimensions");
    FEMTO_ASSERT(neck_radius <= body_radius, "Mesh::bottle() requires neck_radius <= body_radius");
    FEMTO_ASSERT(neck_radius > 1.5 * thickness, "Mesh::bottle() requires neck_radius > 1.5x the wall thickness");
    FEMTO_ASSERT(body_height > 2.0 * thickness, "Mesh::bottle() requires body_height > 2x the wall thickness");
  }

  double z_top() const { return body_height + shoulder_height + neck_length; }

  // outer wall radius as a function of height (clamped outside the bottle, so
  // the same rings continue straight up/down through the air above and below)
  double r_outer(double z) const {
    z = std::clamp(z, 0.0, z_top());
    if (z <= body_height) { return body_radius; }
    if (z >= body_height + shoulder_height) { return neck_radius; }
    double s = (z - body_height) / shoulder_height;
    return neck_radius + (body_radius - neck_radius) * 0.5 * (1.0 + std::cos(M_PI * s));
  }

  // number of geometrically growing cells that span `length`, starting at h
  uint32_t graded_count(double length) const {
    return std::max(2, (int)std::ceil(std::log(1.0 + length * (growth - 1.0) / h) / std::log(growth)));
  }

  // fraction of the way through `count` geometrically growing cells after cell k
  static double graded_fraction(uint32_t k, uint32_t count) {
    return (std::pow(growth, double(k)) - 1.0) / (std::pow(growth, double(count)) - 1.0);
  }

  // z levels: graded air below, the base slab, uniform levels through the
  // body, enough shoulder levels that the wall slope stays moderate, the neck,
  // then graded air above
  std::vector<double> levels(double air_below, double air_above) const {
    std::vector<double> zs;
    uint32_t below = graded_count(air_below);
    for (uint32_t k = below; k-- > 1;) { zs.push_back(-air_below * graded_fraction(k, below)); }
    zs.push_back(0.0);
    zs.push_back(t);
    auto uniform = [&](double z0, double z1, uint32_t count) {
      for (uint32_t k = 1; k <= count; k++) { zs.push_back(z0 + (z1 - z0) * k / count); }
    };
    uniform(t, body_height, std::max(1, (int)std::round((body_height - t) / h)));
    uint32_t n_shoulder = std::max(std::max(1, (int)std::ceil(shoulder_height / h)),
                                   (int)std::ceil((body_radius - neck_radius) / h));
    uniform(body_height, body_height + shoulder_height, n_shoulder);
    uniform(body_height + shoulder_height, z_top(), std::max(2, (int)std::round(neck_length / h)));
    uint32_t above = graded_count(air_above);
    for (uint32_t k = 1; k <= above; k++) { zs.push_back(z_top() + air_above * graded_fraction(k, above)); }
    return zs;
  }
};

// drop the nodes no element refers to and build the mesh
template < size_t verts_per_elem >
nd::array< double, 2, memory::space::cpu > compact_nodes(std::vector< std::array<double, 3> > & nodes, uint32_t dim,
                                                        std::vector< std::array<uint32_t, verts_per_elem> > & elems) {
  std::vector<uint32_t> new_id(nodes.size(), uint32_t(-1));
  uint32_t num_used = 0;
  for (auto & elem : elems) {
    for (uint32_t & v : elem) {
      if (new_id[v] == uint32_t(-1)) { new_id[v] = num_used++; }
      v = new_id[v];
    }
  }
  nd::array< double, 2, memory::space::cpu > node_array({num_used, dim});
  for (uint32_t i = 0; i < nodes.size(); i++) {
    if (new_id[i] == uint32_t(-1)) { continue; }
    for (uint32_t d = 0; d < dim; d++) { node_array(new_id[i], d) = nodes[i][d]; }
  }
  return node_array;
}

}  // namespace

// The air around (and inside) a cylindrically symmetric bottle, as a single
// conforming hexahedral mesh: a quadrilateral "butterfly" disk (square core,
// blend rings, then circular rings) extruded through a list of z levels, with
// the ring radii at each level chosen so that two of the rings coincide with
// the inner and outer surfaces of the bottle wall. The hexes that fall inside
// the wall or the base slab are simply not emitted, so the bottle appears as
// a cavity whose surface is made of element faces.
template <>
Mesh<> Mesh<>::bottle(double body_radius, double body_height, double shoulder_height,
                      double neck_radius, double neck_length, double thickness,
                      double air_margin, double air_below, double air_above, double h) {

  BottleProfile P(body_radius, body_height, shoulder_height, neck_radius, neck_length, thickness,
                  air_margin, air_below, air_above, h);
  double t = thickness;
  double z_top = P.z_top();

  ////////////////////////////////////////////////////////////////////////////
  // discretization

  double r_in_body = body_radius - t;
  uint32_t n = 2 * std::max(1, (int)std::round(0.5 * r_in_body / h));  // core divisions (even: y = 0 is a grid plane)
  uint32_t m1 = std::max(1, (int)std::round(0.5 * r_in_body / h));     // blend rings, core -> inner wall
  uint32_t nt = 1;                                                     // rings across the wall
  uint32_t m2 = P.graded_count(air_margin);                            // rings out to the far field
  uint32_t M = m1 + nt + m2;                                           // rings total
  uint32_t ring_nodes = 4 * n;

  std::vector<double> zs = P.levels(air_below, air_above);
  uint32_t num_levels = uint32_t(zs.size());

  ////////////////////////////////////////////////////////////////////////////
  // node ids

  uint32_t core_nodes = (n + 1) * (n + 1);
  uint32_t disk2d = core_nodes + M * ring_nodes;

  // walk the perimeter of the core lattice counterclockwise from (+a, -a)
  auto perimeter_coords = [n](uint32_t q) -> std::array<uint32_t, 2> {
    if (q <     n) { return {        n,         q}; }
    if (q < 2 * n) { return {2 * n - q,         n}; }
    if (q < 3 * n) { return {        0, 3 * n - q}; }
    return {q - 3 * n, 0};
  };

  auto ring2d = [&](uint32_t ring, uint32_t q) -> uint32_t {
    q = q % ring_nodes;
    if (ring == 0) {
      auto [x, y] = perimeter_coords(q);
      return x + y * (n + 1);
    }
    return core_nodes + (ring - 1) * ring_nodes + q;
  };

  auto node_id = [&](uint32_t level, uint32_t id2) { return level * disk2d + id2; };

  ////////////////////////////////////////////////////////////////////////////
  // node positions

  std::vector< std::array<double, 3> > nodes(size_t(disk2d) * num_levels);

  auto theta = [&](uint32_t q) { return -0.25 * M_PI + (2.0 * M_PI * q) / ring_nodes; };

  for (uint32_t level = 0; level < num_levels; level++) {
    double z = zs[level];
    double r_out = P.r_outer(z);
    double r_in = r_out - t;
    double a = 0.5 * r_in;  // core half-width

    for (uint32_t y = 0; y <= n; y++) {
      for (uint32_t x = 0; x <= n; x++) {
        nodes[node_id(level, x + y * (n + 1))] = {-a + (2 * a * x) / n, -a + (2 * a * y) / n, z};
      }
    }

    for (uint32_t ring = 1; ring <= M; ring++) {
      for (uint32_t q = 0; q < ring_nodes; q++) {
        double c = std::cos(theta(q)), s = std::sin(theta(q));
        std::array<double, 3> p;
        if (ring <= m1) {
          // blend from the core perimeter to the inner wall circle
          auto [x, y] = perimeter_coords(q);
          double f = double(ring) / m1;
          double qx = -a + (2 * a * x) / n, qy = -a + (2 * a * y) / n;
          p = {qx * (1 - f) + r_in * c * f, qy * (1 - f) + r_in * s * f, z};
        } else if (ring <= m1 + nt) {
          // circles across the wall thickness
          double r = r_in + t * double(ring - m1) / nt;
          p = {r * c, r * s, z};
        } else {
          // circles growing geometrically out to the far field
          double f = BottleProfile::graded_fraction(ring - m1 - nt, m2);
          double r = r_out + (body_radius + air_margin - r_out) * f;
          p = {r * c, r * s, z};
        }
        nodes[node_id(level, ring2d(ring, q))] = p;
      }
    }
  }

  ////////////////////////////////////////////////////////////////////////////
  // hexahedra (skipping the ones inside the bottle)

  std::vector< std::array<uint32_t, 8> > hexes;
  hexes.reserve(size_t(n * n + M * ring_nodes) * (num_levels - 1));

  auto add_hex = [&](uint32_t level, std::array<uint32_t, 4> quad) {
    hexes.push_back({node_id(level, quad[0]), node_id(level, quad[1]), node_id(level, quad[2]), node_id(level, quad[3]),
                     node_id(level + 1, quad[0]), node_id(level + 1, quad[1]), node_id(level + 1, quad[2]), node_id(level + 1, quad[3])});
  };

  for (uint32_t level = 0; level + 1 < num_levels; level++) {
    double z_mid = 0.5 * (zs[level] + zs[level + 1]);
    bool in_base = (0.0 < z_mid && z_mid < t);
    bool in_wall = (t < z_mid && z_mid < z_top);

    if (!in_base) {
      for (uint32_t y = 0; y < n; y++) {
        for (uint32_t x = 0; x < n; x++) {
          uint32_t b = x + y * (n + 1);
          add_hex(level, {b, b + 1, b + 1 + (n + 1), b + (n + 1)});
        }
      }
    }
    for (uint32_t ring = 0; ring < M; ring++) {
      bool solid = (in_base && ring < m1 + nt) || (in_wall && m1 <= ring && ring < m1 + nt);
      if (solid) { continue; }
      for (uint32_t q = 0; q < ring_nodes; q++) {
        add_hex(level, {ring2d(ring, q), ring2d(ring + 1, q), ring2d(ring + 1, q + 1), ring2d(ring, q + 1)});
      }
    }
  }

  nd::array< double, 2, memory::space::cpu > node_array = compact_nodes(nodes, 3, hexes);
  nd::array< uint32_t, 2, memory::space::cpu > no_tets({0, 0});
  nd::array< uint32_t, 2, memory::space::cpu > hex_array({uint32_t(hexes.size()), 8});
  for (uint32_t e = 0; e < hexes.size(); e++) {
    for (uint32_t v = 0; v < 8; v++) { hex_array(e, v) = hexes[e][v]; }
  }

  return Mesh<>::create_3D(node_array, 1, no_tets, hex_array);

}

// The planar version: the same profile, extruded in y instead of z, meshed
// with a structured grid whose columns of nodes follow the walls the way the
// rings above do (uniform columns across the interior, one across each wall,
// geometrically graded ones out to the far field on both sides). Cells in
// the wall or the base slab are not emitted.
template <>
Mesh<> Mesh<>::bottle_2d(double body_half_width, double body_height, double shoulder_height,
                         double neck_half_width, double neck_length, double thickness,
                         double air_margin, double air_below, double air_above, double h) {

  BottleProfile P(body_half_width, body_height, shoulder_height, neck_half_width, neck_length, thickness,
                  air_margin, air_below, air_above, h);
  double t = thickness;
  double y_top = P.z_top();

  uint32_t n = 2 * std::max(1, (int)std::round((body_half_width - t) / h));  // interior cells (even: x = 0 is a grid line)
  uint32_t m2 = P.graded_count(air_margin);                                  // graded cells out to the far field
  uint32_t half = n / 2 + 1 + m2;                                            // node columns on either side of x = 0
  uint32_t num_cols = 2 * half + 1;

  std::vector<double> ys = P.levels(air_below, air_above);
  uint32_t num_levels = uint32_t(ys.size());

  // column c -> signed index k from the center: |k| <= n/2 interior,
  // |k| = n/2 + 1 the outer wall surface, beyond that the graded air
  auto signed_index = [&](uint32_t c) { return int(c) - int(half); };
  auto x_of = [&](double y, int k) {
    double r_out = P.r_outer(y), r_in = r_out - t;
    int a = std::abs(k);
    double s = (k < 0) ? -1.0 : 1.0;
    if (a <= int(n / 2)) { return (2.0 * r_in / n) * k; }
    if (a == int(n / 2) + 1) { return s * r_out; }
    double f = BottleProfile::graded_fraction(uint32_t(a - int(n / 2) - 1), m2);
    return s * (r_out + (body_half_width + air_margin - r_out) * f);
  };

  std::vector< std::array<double, 3> > nodes(size_t(num_cols) * num_levels);
  auto node_id = [&](uint32_t level, uint32_t c) { return level * num_cols + c; };
  for (uint32_t level = 0; level < num_levels; level++) {
    for (uint32_t c = 0; c < num_cols; c++) {
      nodes[node_id(level, c)] = {x_of(ys[level], signed_index(c)), ys[level], 0.0};
    }
  }

  std::vector< std::array<uint32_t, 4> > quads;
  quads.reserve(size_t(num_cols - 1) * (num_levels - 1));
  for (uint32_t level = 0; level + 1 < num_levels; level++) {
    double y_mid = 0.5 * (ys[level] + ys[level + 1]);
    bool in_base = (0.0 < y_mid && y_mid < t);
    bool in_wall = (t < y_mid && y_mid < y_top);
    for (uint32_t c = 0; c + 1 < num_cols; c++) {
      int outer = std::max(std::abs(signed_index(c)), std::abs(signed_index(c + 1)));
      bool wall_cell = (outer == int(n / 2) + 1);
      bool solid = (in_base && outer <= int(n / 2) + 1) || (in_wall && wall_cell);
      if (solid) { continue; }
      quads.push_back({node_id(level, c), node_id(level, c + 1), node_id(level + 1, c + 1), node_id(level + 1, c)});
    }
  }

  nd::array< double, 2, memory::space::cpu > node_array = compact_nodes(nodes, 2, quads);
  nd::array< uint32_t, 2, memory::space::cpu > no_tris({0, 0});
  nd::array< uint32_t, 2, memory::space::cpu > quad_array({uint32_t(quads.size()), 4});
  for (uint32_t e = 0; e < quads.size(); e++) {
    for (uint32_t v = 0; v < 4; v++) { quad_array(e, v) = quads[e][v]; }
  }

  return Mesh<>::create_2D(node_array, 1, no_tris, quad_array);

}

}
