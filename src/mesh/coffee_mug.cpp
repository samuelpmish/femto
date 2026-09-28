#include "femto/mesh.hpp"

#include "femto/assert.hpp"

#include <cmath>
#include <vector>

namespace femto {

// The mug is a single conforming hexahedral mesh, assembled from three kinds
// of structured blocks that share nodes at their interfaces:
//
//  1. the base: a quadrilateral "butterfly" disk (square core + blend rings),
//     whose outermost rings are exact circles spanning the wall thickness,
//     extruded vertically from z = 0 to z = thickness
//
//  2. the wall: a cylindrical shell of (sector x radial x vertical) hexes
//     sitting on the base's outer rings, following the linear flare from
//     base_radius at the base to rim_radius at the rim
//
//  3. the handles: swept blocks connecting two patches of the outer wall
//     surface. Each grid line of a handle is a half-ellipse from a node of
//     the lower patch to the matching node of the upper patch, so the end
//     faces coincide with wall surface quads node-for-node.
//
// All blocks are indexed into one node array, so conformity is exact by
// construction (no tolerance-based merging).
template <>
Mesh<> Mesh<>::coffee_mug(double base_radius, double rim_radius, double height,
                          double thickness, int handles, int p) {

  FEMTO_ASSERT(base_radius > 0 && rim_radius > 0 && height > 0 && thickness > 0,
               "Mesh::coffee_mug() requires positive dimensions");
  FEMTO_ASSERT(0 <= handles && handles <= 6,
               "Mesh::coffee_mug() supports between 0 and 6 handles");
  FEMTO_ASSERT(p >= 1, "Mesh::coffee_mug() requires p >= 1");
  FEMTO_ASSERT(base_radius >= 2 * thickness && rim_radius >= 2 * thickness,
               "Mesh::coffee_mug() requires radii of at least twice the wall thickness");
  FEMTO_ASSERT(height >= 2.5 * thickness,
               "Mesh::coffee_mug() requires height of at least 2.5x the wall thickness");

  double t = thickness;
  double h = 0.75 * t;              // target element size
  double r_in = base_radius - t;    // inner wall radius at the bottom

  // discretization parameters
  uint32_t n = std::max(1, (int)std::round(r_in / h));        // core divisions
  uint32_t mb = std::max(1, (int)std::round(0.5 * r_in / h)); // blend rings, core -> inner circle
  uint32_t nt = 2;                                            // layers across the wall thickness
  uint32_t nzb = 2;                                           // layers through the base slab
  uint32_t nz = std::max(2, (int)std::round((height - t) / h)); // wall levels
  uint32_t ring_nodes = 4 * n;
  double a = 0.5 * r_in;            // core half-width
  double hz = (height - t) / nz;

  // handle discretization: each handle occupies na x nb quads of the outer
  // wall surface at each end (na vertical, nb tangential), sized so the
  // cross-section is roughly square and about 1.5 wall thicknesses across
  double sector_w = 2.0 * M_PI * (0.5 * (base_radius + rim_radius)) / ring_nodes;
  uint32_t nb = std::max(1, (int)std::round(1.5 * t / sector_w));
  uint32_t na = std::min(6, std::max(2, (int)std::round(nb * sector_w / hz)));
  uint32_t l_lo = 0, l_hi = 0, nu = 0;
  double b_base = 0;

  if (handles > 0) {
    FEMTO_ASSERT(ring_nodes >= uint32_t(handles) * (nb + 2),
                 "Mesh::coffee_mug() radius is too small for this many handles");

    auto level_near = [&](double z) { return (int)std::round((z - t) / hz - 0.5 * na); };
    int lo = std::max(1, level_near(0.25 * height));
    int hi = std::min(int(nz) - int(na) - 1, level_near(0.80 * height));
    FEMTO_ASSERT(lo + int(na) + 1 <= hi,
                 "Mesh::coffee_mug() height is too small for handles (needs about 7x the wall thickness)");
    l_lo = uint32_t(lo);
    l_hi = uint32_t(hi);

    double dr = (rim_radius - base_radius) * (l_hi - l_lo) * hz / (height - t);
    double dz = (l_hi - l_lo) * hz;
    b_base = std::max(0.35 * std::sqrt(dr * dr + dz * dz), 2.0 * t);
    double arc = M_PI * std::sqrt(0.5 * (0.25 * (dr * dr + dz * dz) + b_base * b_base));
    nu = std::max(4, (int)std::round(arc / h));
  }

  ////////////////////////////////////////////////////////////////////////////
  // node ids

  // 2D butterfly disk: a square core with (n+1)^2 nodes, surrounded by rings
  // 1..(mb + nt). Rings 1..mb blend from the core perimeter to the circle of
  // radius r_in; rings mb+1..mb+nt are circles spanning the wall thickness.
  uint32_t M = mb + nt;
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

  // base slab: disk2d nodes at each of the nzb+1 levels, z in [0, t]
  uint32_t base_total = disk2d * (nzb + 1);
  auto base_id = [&](uint32_t level, uint32_t id2) { return level * disk2d + id2; };

  // wall shell: levels 1..nz add their own (nt+1) x ring_nodes nodes;
  // level 0 reuses the top of the base's outer rings
  uint32_t wall_stride = (nt + 1) * ring_nodes;
  uint32_t wall_total = nz * wall_stride;
  auto wall_id = [&](uint32_t level, uint32_t j, uint32_t q) -> uint32_t {
    q = q % ring_nodes;
    if (level == 0) { return base_id(nzb, ring2d(mb + j, q)); }
    return base_total + (level - 1) * wall_stride + j * ring_nodes + q;
  };

  // handles: sections i = 1..nu-1 add their own (na+1) x (nb+1) nodes;
  // sections 0 and nu are patches of the outer wall surface (j reversed at
  // the top, because the section frame flips as the sweep turns around)
  uint32_t handle_stride = (na + 1) * (nb + 1);
  uint32_t handle_total = (nu > 0) ? (nu - 1) * handle_stride : 0;
  std::vector<uint32_t> k0(handles);
  for (int hh = 0; hh < handles; hh++) {
    double theta_h = (2.0 * M_PI * hh) / handles;
    int k = (int)std::round((theta_h + 0.25 * M_PI) * ring_nodes / (2.0 * M_PI) - 0.5 * nb) % (int)ring_nodes;
    k0[hh] = uint32_t(k + ((k < 0) ? (int)ring_nodes : 0));
  }
  auto handle_id = [&](uint32_t hh, uint32_t i, uint32_t j, uint32_t k) -> uint32_t {
    if (i == 0)  { return wall_id(l_lo + j, nt, k0[hh] + k); }
    if (i == nu) { return wall_id(l_hi + (na - j), nt, k0[hh] + k); }
    return base_total + wall_total + hh * handle_total + (i - 1) * handle_stride + j * (nb + 1) + k;
  };

  uint32_t num_nodes = base_total + wall_total + handles * handle_total;

  ////////////////////////////////////////////////////////////////////////////
  // node positions

  nd::array< double, 2, memory::space::cpu > nodes({num_nodes, 3});

  auto theta = [&](uint32_t q) { return -0.25 * M_PI + (2.0 * M_PI * q) / ring_nodes; };

  // radius of wall layer j (0 = inner, nt = outer) at wall level l
  auto wall_r = [&](uint32_t level, uint32_t j) {
    double outer = base_radius + (rim_radius - base_radius) * double(level) / nz;
    return outer - t * (1.0 - double(j) / nt);
  };
  auto wall_z = [&](uint32_t level) { return t + hz * level; };

  // in-plane positions of the 2D disk
  std::vector< std::array<double, 2> > disk_xy(disk2d);
  for (uint32_t y = 0; y <= n; y++) {
    for (uint32_t x = 0; x <= n; x++) {
      disk_xy[x + y * (n + 1)] = {-a + (2 * a * x) / n, -a + (2 * a * y) / n};
    }
  }
  for (uint32_t ring = 1; ring <= M; ring++) {
    for (uint32_t q = 0; q < ring_nodes; q++) {
      double c = std::cos(theta(q)), s = std::sin(theta(q));
      if (ring <= mb) {
        // blend from the core perimeter to the circle of radius r_in
        auto [x, y] = perimeter_coords(q);
        double f = double(ring) / mb;
        double qx = -a + (2 * a * x) / n, qy = -a + (2 * a * y) / n;
        disk_xy[ring2d(ring, q)] = {qx * (1 - f) + r_in * c * f, qy * (1 - f) + r_in * s * f};
      } else {
        // circular rings across the wall thickness (shared with the wall shell)
        double r = wall_r(0, ring - mb);
        disk_xy[ring2d(ring, q)] = {r * c, r * s};
      }
    }
  }

  for (uint32_t level = 0; level <= nzb; level++) {
    double z = (t * level) / nzb;
    for (uint32_t i = 0; i < disk2d; i++) {
      uint32_t id = base_id(level, i);
      nodes(id, 0) = disk_xy[i][0];
      nodes(id, 1) = disk_xy[i][1];
      nodes(id, 2) = z;
    }
  }

  for (uint32_t level = 1; level <= nz; level++) {
    for (uint32_t j = 0; j <= nt; j++) {
      double r = wall_r(level, j);
      for (uint32_t q = 0; q < ring_nodes; q++) {
        uint32_t id = wall_id(level, j, q);
        nodes(id, 0) = r * std::cos(theta(q));
        nodes(id, 1) = r * std::sin(theta(q));
        nodes(id, 2) = wall_z(level);
      }
    }
  }

  for (int hh = 0; hh < handles; hh++) {
    for (uint32_t j = 0; j <= na; j++) {
      // fiber (j, k) is a half-ellipse in the vertical plane of its sector,
      // from wall node (l_lo + j) up to wall node (l_hi + na - j), bulging
      // radially outward. Outer fibers (small j) bulge more, which is what
      // gives the handle its thickness away from the wall.
      double r_a = wall_r(l_lo + j, nt),      z_a = wall_z(l_lo + j);
      double r_b = wall_r(l_hi + (na - j), nt), z_b = wall_z(l_hi + (na - j));
      double mid_r = 0.5 * (r_a + r_b), mid_z = 0.5 * (z_a + z_b);
      double ax_r = 0.5 * (r_b - r_a), ax_z = 0.5 * (z_b - z_a);
      double bulge = b_base + (0.5 - double(j) / na) * (na * hz);

      for (uint32_t i = 1; i < nu; i++) {
        double u = double(i) / nu;
        double cu = std::cos(M_PI * u), su = std::sin(M_PI * u);
        double r = mid_r - ax_r * cu + bulge * su;
        double z = mid_z - ax_z * cu;
        for (uint32_t k = 0; k <= nb; k++) {
          double th = theta(k0[hh] + k);
          uint32_t id = handle_id(hh, i, j, k);
          nodes(id, 0) = r * std::cos(th);
          nodes(id, 1) = r * std::sin(th);
          nodes(id, 2) = z;
        }
      }
    }
  }

  ////////////////////////////////////////////////////////////////////////////
  // hexahedra

  uint32_t num_hexes = (n * n + M * ring_nodes) * nzb   // base slab
                     + nz * nt * ring_nodes             // wall shell
                     + handles * nu * na * nb;          // handles

  nd::array< uint32_t, 2, memory::space::cpu > no_tets({0, 0});
  nd::array< uint32_t, 2, memory::space::cpu > hexes({num_hexes, 8});
  uint32_t e = 0;

  auto add_hex = [&](std::array<uint32_t, 4> bottom, std::array<uint32_t, 4> top) {
    for (uint32_t v = 0; v < 4; v++) {
      hexes(e, v) = bottom[v];
      hexes(e, v + 4) = top[v];
    }
    e++;
  };

  // base slab: extrude the disk quads (all counterclockwise as seen from +z)
  for (uint32_t level = 0; level < nzb; level++) {
    auto quad = [&](std::array<uint32_t, 4> ids) {
      add_hex({base_id(level, ids[0]), base_id(level, ids[1]), base_id(level, ids[2]), base_id(level, ids[3])},
              {base_id(level + 1, ids[0]), base_id(level + 1, ids[1]), base_id(level + 1, ids[2]), base_id(level + 1, ids[3])});
    };
    for (uint32_t y = 0; y < n; y++) {
      for (uint32_t x = 0; x < n; x++) {
        uint32_t b = x + y * (n + 1);
        quad({b, b + 1, b + 1 + (n + 1), b + (n + 1)});
      }
    }
    for (uint32_t ring = 0; ring < M; ring++) {
      for (uint32_t q = 0; q < ring_nodes; q++) {
        quad({ring2d(ring, q), ring2d(ring + 1, q), ring2d(ring + 1, q + 1), ring2d(ring, q + 1)});
      }
    }
  }

  // wall shell
  for (uint32_t level = 0; level < nz; level++) {
    for (uint32_t j = 0; j < nt; j++) {
      for (uint32_t q = 0; q < ring_nodes; q++) {
        add_hex({wall_id(level, j, q), wall_id(level, j + 1, q), wall_id(level, j + 1, q + 1), wall_id(level, j, q + 1)},
                {wall_id(level + 1, j, q), wall_id(level + 1, j + 1, q), wall_id(level + 1, j + 1, q + 1), wall_id(level + 1, j, q + 1)});
      }
    }
  }

  // handles: section quads are oriented so their normal follows the sweep
  // (outward at the bottom patch, inward at the top, where j is reversed)
  for (int hh = 0; hh < handles; hh++) {
    for (uint32_t i = 0; i < nu; i++) {
      for (uint32_t j = 0; j < na; j++) {
        for (uint32_t k = 0; k < nb; k++) {
          add_hex({handle_id(hh, i, j, k), handle_id(hh, i, j, k + 1), handle_id(hh, i, j + 1, k + 1), handle_id(hh, i, j + 1, k)},
                  {handle_id(hh, i + 1, j, k), handle_id(hh, i + 1, j, k + 1), handle_id(hh, i + 1, j + 1, k + 1), handle_id(hh, i + 1, j + 1, k)});
        }
      }
    }
  }

  FEMTO_ASSERT(e == num_hexes, "Mesh::coffee_mug() internal error: hex count mismatch");

  Mesh<> mesh = Mesh<>::create_3D(nodes, 1, no_tets, hexes);

  // higher order meshes get more geometry nodes (placed by interpolation
  // within each trilinear element -- the surfaces are not re-curved)
  if (p > 1) {
    Field X_p = create_field<Family::H1>(mesh, uint32_t(p), 3);
    X_p.data = nodes_for(X_p, mesh);
    mesh.X = X_p;
  }

  return mesh;

}

}
