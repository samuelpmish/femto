#include "femto/mesh.hpp"

namespace femto {

struct connection_hasher {
  std::size_t operator()(const Connection & c) const {
    return c.index + (uint64_t(c.sign()) << 32);
  }
};

bool is_manifold_2D(const Mesh<> & mesh) {

  // first count the number of elements that include a given edge
  std::vector< uint32_t > edge_count(mesh.edge.shape[0], 0);
  foreach_constexpr< Geometry::Triangle, Geometry::Quadrilateral >([&](auto geom){
    const auto & elements = mesh[geom];
    using gtype = GeometryType< geom >;
    for (int elem = 0; elem < elements.shape[0]; elem++) {
      for (int edge = 0; edge < gtype::num_edges; edge++) {
        edge_count[elements(elem, gtype::edge_offset + edge).index]++;
      }
    }
  });

  // for each of those edges on the boundary, 
  // increment a counter for each of their vertices
  std::vector< uint32_t > vertex_count(mesh.vert.shape[0], 0);
  for (int i = 0; i < mesh.edge.shape[0]; i++) {
    if (edge_count[i] == 1) { 
      vertex_count[mesh.edge(i, 0).index]++;
      vertex_count[mesh.edge(i, 1).index]++;
    }
  }

  // manifold vertices belong to 2 boundary edges 
  for (int i = 0; i < mesh.vert.shape[0]; i++) {
    if (vertex_count[i] != 2 && vertex_count[i] != 0) return false;
  }

  return true;

}

bool is_manifold_3D(const Mesh<> & mesh) {

  //////////////////////////////////
  // check for non-manifold edges //
  //////////////////////////////////

  // first count the number of elements that include a given face
  std::vector< uint32_t > tri_count(mesh.tri.shape[0], 0);
  const auto & tets = mesh[Geometry::Tetrahedron];
  for (int i = 0; i < tets.shape[0]; i++) {
    for (int j = 0; j < Tetrahedron::num_triangles; j++) {
      tri_count[tets(i, Tetrahedron::tri_offset + j).index]++;
    }
  }

  std::vector< uint32_t > quad_count(mesh.quad.shape[0], 0);
  const auto & hexes = mesh[Geometry::Hexahedron];
  for (int i = 0; i < hexes.shape[0]; i++) {
    for (int j = 0; j < Hexahedron::num_quadrilaterals; j++) {
      quad_count[hexes(i, Hexahedron::quad_offset + j).index]++;
    }
  }

  // for each of those edges on the boundary, 
  // increment a counter for each of their vertices
  std::vector< uint32_t > edge_count(mesh.edge.shape[0], 0);
  for (int i = 0; i < mesh.tri.shape[0]; i++) {
    // if boundary tri
    if (tri_count[i] == 1) { 
      edge_count[mesh.tri(i, Triangle::edge_offset + 0).index]++;
      edge_count[mesh.tri(i, Triangle::edge_offset + 1).index]++;
      edge_count[mesh.tri(i, Triangle::edge_offset + 2).index]++;
    }
  }

  for (int i = 0; i < mesh.quad.shape[0]; i++) {
    // if boundary quad
    if (quad_count[i] == 1) { 
      edge_count[mesh.quad(i, Quadrilateral::edge_offset + 0).index]++;
      edge_count[mesh.quad(i, Quadrilateral::edge_offset + 1).index]++;
      edge_count[mesh.quad(i, Quadrilateral::edge_offset + 2).index]++;
      edge_count[mesh.quad(i, Quadrilateral::edge_offset + 3).index]++;
    }
  }

  // manifold boundary edges will belong to exactly 2 boundary faces
  for (int i = 0; i < mesh.edge.shape[0]; i++) {
    if (edge_count[i] != 2 && edge_count[i] != 0) {
      return false;
    }
  }

  /////////////////////////////////////
  // check for non-manifold vertices //
  /////////////////////////////////////

  auto flip_sign = [](Connection c) {
    Connection flipped = c;
    flipped.set_sign(!c.sign());
    return flipped;
  };
 
  std::vector< uint32_t > vertex_count(mesh.vert.shape[0], 0);
  std::vector< Connection > first_vertex_edge_id(mesh.vert.shape[0]);
  std::unordered_map< Connection, Connection, connection_hasher > bdr_edge_connectivity;
  for (int i = 0; i < mesh.tri.shape[0]; i++) {
    // if boundary triangle
    if (tri_count[i] == 1) { 
      std::array< uint32_t, 3 > vert_ids = {
        mesh.tri(i, 0).index, mesh.tri(i, 1).index, mesh.tri(i, 2).index
      };
      std::array< Connection, 3 > edges = {
        mesh.tri(i, Triangle::edge_offset + 2), 
        mesh.tri(i, Triangle::edge_offset + 0), 
        mesh.tri(i, Triangle::edge_offset + 1)
      };

      for (int j = 0; j < Triangle::num_vertices; j++) {
        uint32_t v_id = vert_ids[j];
        vertex_count[v_id]++;
        if (vertex_count[v_id] == 1) {
          first_vertex_edge_id[v_id] = edges[j];
        }
      }

      for (int j = 0; j < Triangle::num_edges; j++) {
        bdr_edge_connectivity[edges[j]] = flip_sign(edges[(j+1) % Triangle::num_edges]);
      }
    }
  }

  for (int i = 0; i < mesh.quad.shape[0]; i++) {
    // if boundary Quadrilateral
    if (quad_count[i] == 1) { 
      std::array< uint32_t, Quadrilateral::num_vertices > vert_ids = {
        mesh.quad(i, 0).index, mesh.quad(i, 1).index, mesh.quad(i, 2).index, mesh.quad(i, 3).index 
      };

      std::array< Connection, Quadrilateral::num_edges > edges = {
        mesh.quad(i, Quadrilateral::edge_offset + 3), 
        mesh.quad(i, Quadrilateral::edge_offset + 0), 
        mesh.quad(i, Quadrilateral::edge_offset + 1),
        mesh.quad(i, Quadrilateral::edge_offset + 2)
      };

      for (int j = 0; j < Quadrilateral::num_vertices; j++) {
        uint32_t v_id = vert_ids[j];
        vertex_count[v_id]++;
        if (vertex_count[v_id] == 1) {
          first_vertex_edge_id[v_id] = edges[j];
        }
      }

      for (int j = 0; j < Quadrilateral::num_edges; j++) {
        bdr_edge_connectivity[edges[j]] = flip_sign(edges[(j+1) % Quadrilateral::num_edges]);
      }
    }
  }

  for (int i = 0; i < mesh.vert.shape[0]; i++) {

    int vcount = vertex_count[i];

    // ignore interior vertices
    if (vcount == 0) continue;

    // boundary vertices that belong to fewer than 3 boundary elements
    // cannot be manifold, so we have an early exit condition
    if (vcount <= 2) {

      return false;

    } else {

      // boundary vertices with at least 3 boundary elements are manifold
      // iff the boundary edges containing that vertex are all part of the
      // same edge loop
      //
      // example: consider the central vertex (*)
      //
      //   |   10  |   11  |
      // --o---←---o---←---o--
      //   |       |       |
      //  7↓      8↑      9↓
      //   |       |       |
      // --o---←---*---←---o-- 
      //   |   5   |   6   |
      //  2↓      3↑      4↓
      //   |       |       |
      // --o---→---o---→---o-- 
      //   |   0   |   1   |
      // 
      // its `vcount` is 4, and if we start with edge 3 and iterate through
      // the edges connected to (*) in clockwise order, 
      // we'll visit edges 5, 8, and 6 before coming back to 3. 
      // So, the cycle took 4 steps, which is equal to `vcount`. 
      // This means we can conclude that the connectivity around (*) is 
      // well-formed.

      // start with an arbitrary bdr edge that contains our vertex
      Connection bdr_edge = first_vertex_edge_id[i];
      Connection original_edge = bdr_edge;
      for (int j = 0; j < vcount; j++) {

        // get the next (clockwise when viewed from outside) bdr edge
        bdr_edge = bdr_edge_connectivity[bdr_edge];

        // if we get back to where we started but haven't visited all
        // the bdr edges for this vertex, then they must not be part
        // of the same edge loop, so the vertex is non-manifold
        if (bdr_edge == original_edge && (j + 1 < vcount)) {
          return false;
        }

      }

      // if we didn't make it back to the original edge, 
      // then the mesh connectivity is ill-formed
      if (bdr_edge != original_edge) {
        return false;
      }

    }

  }

  return true;

}

bool is_manifold(const Mesh<> & mesh, bool print_report) {
  if (mesh.geometry_dimension == 2) { return is_manifold_2D(mesh); }
  if (mesh.geometry_dimension == 3) { return is_manifold_3D(mesh); }
  return false;
}

} // namespace femto