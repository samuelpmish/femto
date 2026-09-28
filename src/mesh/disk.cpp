#include "femto/mesh.hpp"

#include "femto/assert.hpp"

#include <map>
#include <set>
#include <array>
#include <cmath>
#include <vector>

namespace femto {

// project every node on the boundary of `mesh` radially onto the circle/sphere
// of radius `r` centered at `center`. Interior nodes are left alone, so meshes
// of degree > 1 get curved elements only in the outermost layer.
template < uint32_t dim >
static void snap_boundary_nodes(Mesh<> & mesh, vec<dim, double> center, double r) {

  SubMesh<> bdr = boundary_of(mesh);
  Field<Family::H1> & X = mesh.X;
  GeometryInfo npe = interior_nodes_per_geom(FunctionSpace{X.family, X.degree}, mesh.geometry_dimension);

  auto snap = [&](uint32_t row) {
    vec<dim, double> v;
    for (uint32_t d = 0; d < dim; d++) { v[d] = X.data(row, d) - center[d]; }
    v = normalize(v) * r;
    for (uint32_t d = 0; d < dim; d++) { X.data(row, d) = center[d] + v[d]; }
  };

  for (uint32_t i = 0; i < bdr.vert.shape[0]; i++) {
    snap(X.offsets.vert + bdr.vert(i));
  }

  for (uint32_t i = 0; i < bdr.edge.shape[0]; i++) {
    for (uint32_t k = 0; k < npe.edge; k++) {
      snap(X.offsets.edge + bdr.edge(i) * npe.edge + k);
    }
  }

  for (uint32_t i = 0; i < bdr.tri.shape[0]; i++) {
    for (uint32_t k = 0; k < npe.tri; k++) {
      snap(X.offsets.tri + bdr.tri(i) * npe.tri + k);
    }
  }

  for (uint32_t i = 0; i < bdr.quad.shape[0]; i++) {
    for (uint32_t k = 0; k < npe.quad; k++) {
      snap(X.offsets.quad + bdr.quad(i) * npe.quad + k);
    }
  }

}

template <>
Mesh<> Mesh<>::disk(vec2 center, double r, double h, int p, Geometry element_type) {

  FEMTO_ASSERT(r > 0 && h > 0, "Mesh::disk() requires positive radius and element size");

  constexpr uint32_t dim = 2;

  nd::array< uint32_t, 2, memory::space::cpu > no_tris({0, 0});
  nd::array< uint32_t, 2, memory::space::cpu > no_quads({0, 0});

  Mesh<> mesh;

  if (element_type == Geometry::Triangle) {

    // start from a regular hexagon (6 equilateral triangles, edge length == r)
    // inscribed in the circle, uniformly refine until edges are about h long,
    // then push the boundary vertices out to sit exactly on the circle
    std::vector< vec2 > verts;
    std::vector< std::array<uint32_t, 3> > tris;

    verts.push_back(center);
    for (uint32_t i = 0; i < 6; i++) {
      double theta = (M_PI / 3.0) * i;
      verts.push_back(center + vec2{r * std::cos(theta), r * std::sin(theta)});
    }
    for (uint32_t i = 0; i < 6; i++) {
      tris.push_back({0, 1 + i, 1 + (i + 1) % 6});
    }

    int num_refinements = std::max(0, (int)std::ceil(std::log2(r / h)));
    for (int level = 0; level < num_refinements; level++) {

      std::map< std::pair<uint32_t, uint32_t>, uint32_t > midpoint_ids;
      auto midpoint = [&](uint32_t i, uint32_t j) {
        std::pair<uint32_t, uint32_t> key = std::minmax(i, j);
        auto it = midpoint_ids.find(key);
        if (it != midpoint_ids.end()) { return it->second; }
        uint32_t id = uint32_t(verts.size());
        verts.push_back((verts[i] + verts[j]) * 0.5);
        midpoint_ids[key] = id;
        return id;
      };

      std::vector< std::array<uint32_t, 3> > refined;
      refined.reserve(4 * tris.size());
      for (auto [a, b, c] : tris) {
        uint32_t ab = midpoint(a, b);
        uint32_t bc = midpoint(b, c);
        uint32_t ca = midpoint(c, a);
        refined.push_back({ a, ab, ca});
        refined.push_back({ab,  b, bc});
        refined.push_back({ca, bc,  c});
        refined.push_back({ab, bc, ca});
      }
      tris = std::move(refined);

    }

    // map the hexagon onto the disk by scaling each vertex radially by the
    // ratio of the circle's radius to the hexagon's in its direction: the
    // boundary lands exactly on the circle and the stretch (at most 15%) is
    // spread over every ring of triangles rather than piled onto the last,
    // which snapping only the boundary vertices would do (the outermost
    // triangles then have aspect ratios of 5 or so and reflect waves)
    for (auto & v : verts) {
      vec2 d = v - center;
      double theta = std::atan2(d[1], d[0]);
      double t = std::fmod(theta + 2 * M_PI, M_PI / 3) - M_PI / 6;   // angle from the nearest edge normal (normals at 30, 90, ... degrees)
      v = center + d * (std::cos(t) / std::cos(M_PI / 6));
    }

    nd::array< double, 2, memory::space::cpu > nodes({uint32_t(verts.size()), dim});
    for (uint32_t i = 0; i < verts.size(); i++) {
      nodes(i, 0) = verts[i][0];
      nodes(i, 1) = verts[i][1];
    }

    nd::array< uint32_t, 2, memory::space::cpu > tri_ids({uint32_t(tris.size()), 3});
    for (uint32_t i = 0; i < tris.size(); i++) {
      tri_ids(i, 0) = tris[i][0];
      tri_ids(i, 1) = tris[i][1];
      tri_ids(i, 2) = tris[i][2];
    }

    mesh = Mesh<>::create_2D(nodes, 1, tri_ids, no_quads);

  } else if (element_type == Geometry::Quadrilateral) {

    // "butterfly" topology: a square core with half-width r/2, surrounded by
    // rings of quadrilaterals that blend from the core boundary to the circle
    double a = 0.5 * r;                                    // core half-width
    uint32_t n = std::max(1, (int)std::round(2 * a / h));  // elements across the core
    uint32_t m = std::max(1, (int)std::round((r - a) / h));// radial layers, core -> circle

    uint32_t core_nodes = (n + 1) * (n + 1);
    uint32_t ring_nodes = 4 * n;
    uint32_t num_nodes = core_nodes + m * ring_nodes;
    uint32_t num_quads = n * n + m * ring_nodes;

    // walk the perimeter of the core lattice counterclockwise,
    // starting from the corner at (+a, -a)
    auto perimeter_coords = [n](uint32_t p) -> std::array<uint32_t, 2> {
      if (p <     n) { return {        n,         p}; }
      if (p < 2 * n) { return {2 * n - p,         n}; }
      if (p < 3 * n) { return {        0, 3 * n - p}; }
      return {p - 3 * n, 0};
    };

    // ring 0 is the core perimeter; rings 1..m have their own nodes
    auto ring_id = [&](uint32_t j, uint32_t p) -> uint32_t {
      p = p % ring_nodes;
      if (j == 0) {
        auto [x, y] = perimeter_coords(p);
        return x + y * (n + 1);
      }
      return core_nodes + (j - 1) * ring_nodes + p;
    };

    nd::array< double, 2, memory::space::cpu > nodes({num_nodes, dim});
    for (uint32_t y = 0; y <= n; y++) {
      for (uint32_t x = 0; x <= n; x++) {
        uint32_t id = x + y * (n + 1);
        nodes(id, 0) = center[0] - a + (2 * a * x) / n;
        nodes(id, 1) = center[1] - a + (2 * a * y) / n;
      }
    }

    for (uint32_t j = 1; j <= m; j++) {
      double t = double(j) / m;
      for (uint32_t p = 0; p < ring_nodes; p++) {
        auto [x, y] = perimeter_coords(p);
        vec2 q{-a + (2 * a * x) / n, -a + (2 * a * y) / n};
        double theta = -0.25 * M_PI + (2 * M_PI * p) / ring_nodes;
        vec2 c{r * std::cos(theta), r * std::sin(theta)};
        vec2 position = center + q * (1.0 - t) + c * t;
        nodes(ring_id(j, p), 0) = position[0];
        nodes(ring_id(j, p), 1) = position[1];
      }
    }

    nd::array< uint32_t, 2, memory::space::cpu > quad_ids({num_quads, 4});
    uint32_t q_id = 0;
    for (uint32_t y = 0; y < n; y++) {
      for (uint32_t x = 0; x < n; x++) {
        uint32_t base = x + y * (n + 1);
        quad_ids(q_id, 0) = base;
        quad_ids(q_id, 1) = base + 1;
        quad_ids(q_id, 2) = base + 1 + (n + 1);
        quad_ids(q_id, 3) = base     + (n + 1);
        q_id++;
      }
    }
    for (uint32_t j = 0; j < m; j++) {
      for (uint32_t p = 0; p < ring_nodes; p++) {
        quad_ids(q_id, 0) = ring_id(j,     p    );
        quad_ids(q_id, 1) = ring_id(j + 1, p    );
        quad_ids(q_id, 2) = ring_id(j + 1, p + 1);
        quad_ids(q_id, 3) = ring_id(j,     p + 1);
        q_id++;
      }
    }

    mesh = Mesh<>::create_2D(nodes, 1, no_tris, quad_ids);

  } else {

    FEMTO_ERROR(std::string("Mesh::disk() only supports Triangle and Quadrilateral elements, got ") + to_string(element_type));

  }

  if (p > 1) {
    Field X_p = create_field<Family::H1>(mesh, uint32_t(p), dim);
    X_p.data = nodes_for(X_p, mesh);
    mesh.X = X_p;
    snap_boundary_nodes(mesh, center, r);
  }

  return mesh;

}

template <>
Mesh<> Mesh<>::ball(vec3 center, double r, double h, int p, Geometry element_type) {

  FEMTO_ASSERT(r > 0 && h > 0, "Mesh::ball() requires positive radius and element size");

  constexpr uint32_t dim = 3;

  nd::array< uint32_t, 2, memory::space::cpu > no_tets({0, 0});
  nd::array< uint32_t, 2, memory::space::cpu > no_hexes({0, 0});

  Mesh<> mesh;

  if (element_type == Geometry::Tetrahedron) {

    // start from an icosahedron with vertices on the sphere, connected to the
    // center to form 20 tetrahedra. Uniformly refine (1 tet -> 8 tets) until
    // edges are about h long, then push boundary vertices out onto the sphere.
    constexpr double phi = 1.618033988749894848; // golden ratio
    constexpr double ico_verts[12][3] = {
      {  -1,  phi,    0}, {   1,  phi,    0}, {  -1, -phi,    0}, {   1, -phi,    0},
      {   0,   -1,  phi}, {   0,    1,  phi}, {   0,   -1, -phi}, {   0,    1, -phi},
      { phi,    0,   -1}, { phi,    0,    1}, {-phi,    0,   -1}, {-phi,    0,    1}
    };
    constexpr uint32_t ico_faces[20][3] = {
      {0, 11,  5}, {0,  5,  1}, {0,  1,  7}, {0,  7, 10}, {0, 10, 11},
      {1,  5,  9}, {5, 11,  4}, {11, 10, 2}, {10, 7,  6}, {7,  1,  8},
      {3,  9,  4}, {3,  4,  2}, {3,  2,  6}, {3,  6,  8}, {3,  8,  9},
      {4,  9,  5}, {2,  4, 11}, {6,  2, 10}, {8,  6,  7}, {9,  8,  1}
    };

    std::vector< vec3 > verts;
    double scale = r / std::sqrt(1.0 + phi * phi);
    for (uint32_t i = 0; i < 12; i++) {
      verts.push_back(center + vec3{ico_verts[i][0], ico_verts[i][1], ico_verts[i][2]} * scale);
    }
    uint32_t center_id = 12;
    verts.push_back(center);

    auto signed_volume = [&](const std::array<uint32_t, 4> & t) {
      vec3 e1 = verts[t[1]] - verts[t[0]];
      vec3 e2 = verts[t[2]] - verts[t[0]];
      vec3 e3 = verts[t[3]] - verts[t[0]];
      return dot(cross(e1, e2), e3);
    };

    auto push_oriented = [&](std::vector< std::array<uint32_t, 4> > & tets, std::array<uint32_t, 4> t) {
      if (signed_volume(t) < 0) { std::swap(t[1], t[2]); }
      tets.push_back(t);
    };

    std::vector< std::array<uint32_t, 4> > tets;
    for (uint32_t i = 0; i < 20; i++) {
      push_oriented(tets, {ico_faces[i][0], ico_faces[i][1], ico_faces[i][2], center_id});
    }

    // the longest edges are the ones on the sphere surface, with length ~1.05 r
    double edge_length = norm(verts[ico_faces[0][0]] - verts[ico_faces[0][1]]);
    int num_refinements = std::max(0, (int)std::ceil(std::log2(edge_length / h)));

    for (int level = 0; level < num_refinements; level++) {

      std::map< std::pair<uint32_t, uint32_t>, uint32_t > midpoint_ids;
      auto midpoint = [&](uint32_t i, uint32_t j) {
        std::pair<uint32_t, uint32_t> key = std::minmax(i, j);
        auto it = midpoint_ids.find(key);
        if (it != midpoint_ids.end()) { return it->second; }
        uint32_t id = uint32_t(verts.size());
        verts.push_back((verts[i] + verts[j]) * 0.5);
        midpoint_ids[key] = id;
        return id;
      };

      std::vector< std::array<uint32_t, 4> > refined;
      refined.reserve(8 * tets.size());
      for (auto [v0, v1, v2, v3] : tets) {
        uint32_t m01 = midpoint(v0, v1);
        uint32_t m02 = midpoint(v0, v2);
        uint32_t m03 = midpoint(v0, v3);
        uint32_t m12 = midpoint(v1, v2);
        uint32_t m13 = midpoint(v1, v3);
        uint32_t m23 = midpoint(v2, v3);

        // one tet at each corner
        push_oriented(refined, { v0, m01, m02, m03});
        push_oriented(refined, {m01,  v1, m12, m13});
        push_oriented(refined, {m02, m12,  v2, m23});
        push_oriented(refined, {m03, m13, m23,  v3});

        // split the interior octahedron into 4 tets along its shortest diagonal
        uint32_t diagonals[3][2] = {{m01, m23}, {m02, m13}, {m03, m12}};
        uint32_t rings[3][4] = {
          {m02, m03, m13, m12},
          {m01, m03, m23, m12},
          {m01, m02, m23, m13}
        };
        uint32_t shortest = 0;
        for (uint32_t k = 1; k < 3; k++) {
          if (norm(verts[diagonals[k][0]] - verts[diagonals[k][1]]) <
              norm(verts[diagonals[shortest][0]] - verts[diagonals[shortest][1]])) {
            shortest = k;
          }
        }
        for (uint32_t k = 0; k < 4; k++) {
          push_oriented(refined, {
            diagonals[shortest][0],
            rings[shortest][k],
            rings[shortest][(k + 1) % 4],
            diagonals[shortest][1]
          });
        }
      }
      tets = std::move(refined);

    }

    // boundary faces appear in exactly one tet; put their vertices on the sphere
    std::map< std::array<uint32_t, 3>, int > face_counts;
    for (auto [v0, v1, v2, v3] : tets) {
      uint32_t faces[4][3] = {{v0, v1, v2}, {v0, v1, v3}, {v0, v2, v3}, {v1, v2, v3}};
      for (auto & face : faces) {
        std::array<uint32_t, 3> key = {face[0], face[1], face[2]};
        std::sort(key.begin(), key.end());
        face_counts[key]++;
      }
    }
    std::set< uint32_t > boundary_verts;
    for (auto [face, count] : face_counts) {
      if (count == 1) {
        boundary_verts.insert(face[0]);
        boundary_verts.insert(face[1]);
        boundary_verts.insert(face[2]);
      }
    }
    for (auto v : boundary_verts) {
      verts[v] = center + normalize(verts[v] - center) * r;
    }

    nd::array< double, 2, memory::space::cpu > nodes({uint32_t(verts.size()), dim});
    for (uint32_t i = 0; i < verts.size(); i++) {
      nodes(i, 0) = verts[i][0];
      nodes(i, 1) = verts[i][1];
      nodes(i, 2) = verts[i][2];
    }

    nd::array< uint32_t, 2, memory::space::cpu > tet_ids({uint32_t(tets.size()), 4});
    for (uint32_t i = 0; i < tets.size(); i++) {
      tet_ids(i, 0) = tets[i][0];
      tet_ids(i, 1) = tets[i][1];
      tet_ids(i, 2) = tets[i][2];
      tet_ids(i, 3) = tets[i][3];
    }

    mesh = Mesh<>::create_3D(nodes, 1, tet_ids, no_hexes);

  } else if (element_type == Geometry::Hexahedron) {

    // "butterfly" topology: a cube core with half-width r/2, surrounded by
    // shells of hexahedra that blend from the core surface to the sphere
    double a = 0.5 * r;                                    // core half-width
    uint32_t n = std::max(1, (int)std::round(2 * a / h));  // elements across the core
    uint32_t m = std::max(1, (int)std::round((r - a) / h));// radial layers, core -> sphere

    uint32_t np = n + 1;
    uint32_t core_nodes = np * np * np;

    auto grid_id = [np](uint32_t x, uint32_t y, uint32_t z) { return x + y * np + z * np * np; };

    // enumerate the lattice nodes on the surface of the core cube
    std::vector< uint32_t > surf_index(core_nodes, uint32_t(-1));
    std::vector< std::array<uint32_t, 3> > surf_coords;
    for (uint32_t z = 0; z <= n; z++) {
      for (uint32_t y = 0; y <= n; y++) {
        for (uint32_t x = 0; x <= n; x++) {
          if (x == 0 || x == n || y == 0 || y == n || z == 0 || z == n) {
            surf_index[grid_id(x, y, z)] = uint32_t(surf_coords.size());
            surf_coords.push_back({x, y, z});
          }
        }
      }
    }
    uint32_t shell_nodes = uint32_t(surf_coords.size()); // 6n^2 + 2

    uint32_t num_nodes = core_nodes + m * shell_nodes;
    uint32_t num_hexes = n * n * n + m * 6 * n * n;

    // shell 0 is the core surface; shells 1..m have their own nodes
    auto shell_id = [&](uint32_t j, uint32_t gid) -> uint32_t {
      if (j == 0) { return gid; }
      return core_nodes + (j - 1) * shell_nodes + surf_index[gid];
    };

    nd::array< double, 2, memory::space::cpu > nodes({num_nodes, dim});
    for (uint32_t z = 0; z <= n; z++) {
      for (uint32_t y = 0; y <= n; y++) {
        for (uint32_t x = 0; x <= n; x++) {
          uint32_t id = grid_id(x, y, z);
          nodes(id, 0) = center[0] - a + (2 * a * x) / n;
          nodes(id, 1) = center[1] - a + (2 * a * y) / n;
          nodes(id, 2) = center[2] - a + (2 * a * z) / n;
        }
      }
    }

    for (uint32_t j = 1; j <= m; j++) {
      double t = double(j) / m;
      for (uint32_t s = 0; s < shell_nodes; s++) {
        auto [x, y, z] = surf_coords[s];
        vec3 q{-a + (2 * a * x) / n, -a + (2 * a * y) / n, -a + (2 * a * z) / n};
        vec3 c = normalize(q) * r;
        vec3 position = center + q * (1.0 - t) + c * t;
        uint32_t id = core_nodes + (j - 1) * shell_nodes + s;
        nodes(id, 0) = position[0];
        nodes(id, 1) = position[1];
        nodes(id, 2) = position[2];
      }
    }

    // quadrilaterals on the core surface, oriented with outward normals
    std::vector< std::array<uint32_t, 4> > surf_quads;
    surf_quads.reserve(6 * n * n);
    for (uint32_t v = 0; v < n; v++) {
      for (uint32_t u = 0; u < n; u++) {
        surf_quads.push_back({grid_id(u, v, 0), grid_id(u, v + 1, 0), grid_id(u + 1, v + 1, 0), grid_id(u + 1, v, 0)}); // -z
        surf_quads.push_back({grid_id(u, v, n), grid_id(u + 1, v, n), grid_id(u + 1, v + 1, n), grid_id(u, v + 1, n)}); // +z
        surf_quads.push_back({grid_id(u, 0, v), grid_id(u + 1, 0, v), grid_id(u + 1, 0, v + 1), grid_id(u, 0, v + 1)}); // -y
        surf_quads.push_back({grid_id(u, n, v), grid_id(u, n, v + 1), grid_id(u + 1, n, v + 1), grid_id(u + 1, n, v)}); // +y
        surf_quads.push_back({grid_id(0, u, v), grid_id(0, u, v + 1), grid_id(0, u + 1, v + 1), grid_id(0, u + 1, v)}); // -x
        surf_quads.push_back({grid_id(n, u, v), grid_id(n, u + 1, v), grid_id(n, u + 1, v + 1), grid_id(n, u, v + 1)}); // +x
      }
    }

    nd::array< uint32_t, 2, memory::space::cpu > hex_ids({num_hexes, 8});
    uint32_t h_id = 0;
    for (uint32_t z = 0; z < n; z++) {
      for (uint32_t y = 0; y < n; y++) {
        for (uint32_t x = 0; x < n; x++) {
          uint32_t base = grid_id(x, y, z);
          hex_ids(h_id, 0) = base;
          hex_ids(h_id, 1) = base + 1;
          hex_ids(h_id, 2) = base + 1 + np;
          hex_ids(h_id, 3) = base     + np;
          hex_ids(h_id, 4) = base          + np * np;
          hex_ids(h_id, 5) = base + 1      + np * np;
          hex_ids(h_id, 6) = base + 1 + np + np * np;
          hex_ids(h_id, 7) = base     + np + np * np;
          h_id++;
        }
      }
    }
    for (uint32_t j = 0; j < m; j++) {
      for (auto & quad : surf_quads) {
        for (uint32_t k = 0; k < 4; k++) {
          hex_ids(h_id, k    ) = shell_id(j,     quad[k]);
          hex_ids(h_id, k + 4) = shell_id(j + 1, quad[k]);
        }
        h_id++;
      }
    }

    mesh = Mesh<>::create_3D(nodes, 1, no_tets, hex_ids);

  } else {

    FEMTO_ERROR(std::string("Mesh::ball() only supports Tetrahedron and Hexahedron elements, got ") + to_string(element_type));

  }

  if (p > 1) {
    Field X_p = create_field<Family::H1>(mesh, uint32_t(p), dim);
    X_p.data = nodes_for(X_p, mesh);
    mesh.X = X_p;
    snap_boundary_nodes(mesh, center, r);
  }

  return mesh;

}

}
