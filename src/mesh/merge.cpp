#include "femto/mesh.hpp"

#include "femto/assert.hpp"

#include "bvh.hpp"

#include <cmath>
#include <vector>
#include <algorithm>

namespace femto {

namespace {

// a conservative float bounding box around the ball |x - p| <= half_r,
// rounding outward so that no coincident pair is missed by the float BVH
cuBQL::box3f vertex_box(vec3 p, double half_r) {
  cuBQL::box3f box;
  box.lower.x = std::nextafterf(float(p[0] - half_r), -INFINITY);
  box.lower.y = std::nextafterf(float(p[1] - half_r), -INFINITY);
  box.lower.z = std::nextafterf(float(p[2] - half_r), -INFINITY);
  box.upper.x = std::nextafterf(float(p[0] + half_r), +INFINITY);
  box.upper.y = std::nextafterf(float(p[1] + half_r), +INFINITY);
  box.upper.z = std::nextafterf(float(p[2] + half_r), +INFINITY);
  return box;
}

}

Mesh<> merge(std::vector< Mesh<> > meshes, double r) {

  FEMTO_ASSERT(meshes.size() > 0, "merge() requires at least one mesh");

  uint32_t sdim = meshes[0].spatial_dimension;
  uint32_t gdim = meshes[0].geometry_dimension;
  GeometryInfo counts{};
  for (const auto & mesh : meshes) {
    counts += mesh.geometry_counts();
    FEMTO_ASSERT(mesh.X.degree == 1, "can't merge high order meshes");
    FEMTO_ASSERT(sdim == mesh.spatial_dimension, "can't merge meshes with different dimensions");
    FEMTO_ASSERT(gdim == mesh.geometry_dimension, "can't merge meshes with different dimensions");
  }

  uint32_t num_vertices = counts.vert;

  // gather the vertices from each mesh (2D meshes are embedded at z = 0)
  std::vector< vec3 > points(num_vertices);
  {
    uint32_t offset = 0;
    for (const auto & mesh : meshes) {
      for (uint32_t i = 0; i < mesh.vert.shape[0]; i++) {
        points[offset + i] = vec3{
          mesh.X.data(i, 0),
          mesh.X.data(i, 1),
          (sdim == 3) ? mesh.X.data(i, 2) : 0.0
        };
      }
      offset += mesh.vert.shape[0];
    }
  }

  // build a BVH over the vertices with cuBQL's host builder, and traverse
  // it from each vertex to find its representative: the lowest-index vertex
  // within a distance r of it. Boxes of half-width r/2 overlap only if the
  // componentwise distance is <= r (plus float rounding), so the BVH
  // traversal visits a superset of the true candidates, and the exact
  // double precision distance check rejects the rest
  std::vector< uint32_t > rep(num_vertices);
  {
    std::vector< cuBQL::box3f > boxes(num_vertices);
    for (uint32_t i = 0; i < num_vertices; i++) {
      boxes[i] = vertex_box(points[i], 0.5 * r);
    }

    cuBQL::bvh3f bvh;
    cuBQL::cpu::spatialMedian(bvh, boxes.data(), num_vertices, cuBQL::BuildConfig{});

    for (uint32_t i = 0; i < num_vertices; i++) {
      vec3 p = points[i];
      uint32_t best = i;
      cuBQL::fixedBoxQuery::forEachPrim([&](uint32_t j) -> int {
        vec3 d = p - points[j];
        if (dot(d, d) <= r * r) { best = std::min(best, j); }
        return CUBQL_CONTINUE_TRAVERSAL;
      }, bvh, boxes[i]);
      rep[i] = best;
    }

    cuBQL::cpu::freeBVH(bvh);
  }

  // rep[i] <= i, so following representatives always terminates at a
  // vertex that is its own representative. Collapse chains (a within r
  // of b, b within r of c) in increasing index order, then assign each
  // representative a new (deduplicated) vertex id
  std::vector< uint32_t > new_ids(num_vertices);
  uint32_t num_merged = 0;
  for (uint32_t i = 0; i < num_vertices; i++) {
    if (rep[i] == i) {
      new_ids[i] = num_merged++;
    } else {
      rep[i] = rep[rep[i]]; // rep[i] < i, so rep[rep[i]] is already a root
      new_ids[i] = new_ids[rep[i]];
    }
  }

  nd::array< double, 2, memory::space::cpu > merged_nodes({num_merged, sdim});
  for (uint32_t i = 0; i < num_vertices; i++) {
    if (rep[i] == i) {
      for (uint32_t d = 0; d < sdim; d++) {
        merged_nodes(new_ids[i], d) = points[i][d];
      }
    }
  }

  if (gdim == 2) {
    uint32_t vert_offset = 0;

    uint32_t tri_offset = 0;
    nd::array< uint32_t, 2, memory::space::cpu > merged_tris({counts.tri, 3});

    uint32_t quad_offset = 0;
    nd::array< uint32_t, 2, memory::space::cpu > merged_quads({counts.quad, 4});

    for (const auto & mesh : meshes) {
      for (uint32_t i = 0; i < mesh.tri.shape[0]; i++) {
        for (uint32_t j = 0; j < 3; j++) {
          merged_tris(i + tri_offset, j) = new_ids[mesh.tri(i, j).index + vert_offset];
        }
      }
      tri_offset += mesh.tri.shape[0];

      for (uint32_t i = 0; i < mesh.quad.shape[0]; i++) {
        for (uint32_t j = 0; j < 4; j++) {
          merged_quads(i + quad_offset, j) = new_ids[mesh.quad(i, j).index + vert_offset];
        }
      }
      quad_offset += mesh.quad.shape[0];

      vert_offset += mesh.vert.shape[0];
    }

    uint32_t degree = 1;
    return Mesh<>::create_2D(merged_nodes, degree, merged_tris, merged_quads);
  }

  if (gdim == 3) {
    uint32_t vert_offset = 0;

    uint32_t tet_offset = 0;
    nd::array< uint32_t, 2, memory::space::cpu > merged_tets({counts.tet, 4});

    uint32_t hex_offset = 0;
    nd::array< uint32_t, 2, memory::space::cpu > merged_hexes({counts.hex, 8});

    for (const auto & mesh : meshes) {
      for (uint32_t i = 0; i < mesh.tet.shape[0]; i++) {
        for (uint32_t j = 0; j < 4; j++) {
          merged_tets(i + tet_offset, j) = new_ids[mesh.tet(i, j).index + vert_offset];
        }
      }
      tet_offset += mesh.tet.shape[0];

      for (uint32_t i = 0; i < mesh.hex.shape[0]; i++) {
        for (uint32_t j = 0; j < 8; j++) {
          merged_hexes(i + hex_offset, j) = new_ids[mesh.hex(i, j).index + vert_offset];
        }
      }
      hex_offset += mesh.hex.shape[0];

      vert_offset += mesh.vert.shape[0];
    }

    uint32_t degree = 1;
    return Mesh<>::create_3D(merged_nodes, degree, merged_tets, merged_hexes);
  }

  return Mesh<>{};

}

}
