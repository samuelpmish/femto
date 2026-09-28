#include <set>
#include <array>
#include <tuple>
#include <vector>
#include <cstring>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <unordered_map>

#include "containers/ndarray.hpp"
#include "femto/geometry.hpp"
#include "femto/connectivity.hpp"
#include "misc/nvtx.hpp"

namespace femto {

template < size_t n >
auto sort(const std::array< uint32_t, n > & values) { 
  auto copy = values;
  std::sort(copy.begin(), copy.end()); 
  return copy;
}

template < typename GeometryType > 
nd::array<Connection, 2, memory::space::cpu> process_edges_2D(
  std::set<uint32_t> & vertex_set,
  unordered_map_of_arrays<2> & sorted_edges,
  std::vector < std::array< uint32_t, 2 > > & edges,
  const nd::view<const uint32_t, 2> & geom_to_vertex) {

  std::string nvtx_label = std::string("process_edges_2D ") + to_string(GeometryType::geometry);
  nvtx::scoped_range nvtx_scope(nvtx_label.c_str());

  uint32_t num_geometries = geom_to_vertex.shape[0];
  if (num_geometries == 0) {
    return nd::array<Connection, 2, memory::space::cpu>{};
  }

  uint32_t entries_per_geometry = (GeometryType::num_vertices + GeometryType::num_edges + 1); 
  nd::array<Connection, 2, memory::space::cpu> output({num_geometries, entries_per_geometry});

  for (uint32_t i = 0; i < num_geometries; i++) {

    // load the vertices for this geometry
    std::array< uint32_t, GeometryType::num_vertices > geometry_vertices;
    for (int j = 0; j < GeometryType::num_vertices; j++) {
      geometry_vertices[j] = geom_to_vertex(i,j);
      output(i, j) = Connection(geometry_vertices[j]);
    }

    // iterate through the edges of this geometry
    for (size_t j = 0; j < GeometryType::num_edges; j++) {

      // get the vertices that correspond to each edge of the triangle
      std::array< uint32_t, Edge::num_vertices > edge_vertices;
      for (int k = 0; k < Edge::num_vertices; k++) {
        edge_vertices[k] = geometry_vertices[GeometryType::local_edge_ids[j][k]];
        vertex_set.insert(edge_vertices[k]);
      }

      output(i, j + GeometryType::edge_offset) = geom_lookup<Edge>(edge_vertices, sorted_edges, edges);

    }

    Connection cell(i);
    cell.set_geometry(GeometryType::geometry);
    output(i, GeometryType::cell_offset) = cell;
    
  }

  return output;

}

template < typename GeometryType > 
nd::array<Connection, 2, memory::space::cpu> process_geometries_3D(
  std::set<uint32_t> & vertex_set,
  unordered_map_of_arrays<2> & sorted_edges,
  std::vector < std::array< uint32_t, 2 > > & edges,
  unordered_map_of_arrays<3> & sorted_tris,
  std::vector < std::array< uint32_t, 3 > > & tris,
  unordered_map_of_arrays<4> & sorted_quads,
  std::vector < std::array< uint32_t, 4 > > & quads,
  const nd::view<const uint32_t, 2> & geom_to_vertex) {

  std::string nvtx_label = std::string("process_geometries_3D ") + to_string(GeometryType::geometry);
  nvtx::scoped_range nvtx_scope(nvtx_label.c_str());

  uint32_t num_geometries = geom_to_vertex.shape[0];
  uint32_t entries_per_geometry = GeometryType::cell_offset + 1; 
  nd::array<Connection, 2, memory::space::cpu> output({num_geometries, entries_per_geometry});

  for (size_t i = 0; i < num_geometries; i++) {

    // load the vertices for this geometry
    std::array< uint32_t, GeometryType::num_vertices > geometry_vertices;
    for (int j = 0; j < GeometryType::num_vertices; j++) {
      geometry_vertices[j] = geom_to_vertex(i,j);
      output(i, j) = Connection(geometry_vertices[j]);
    }

    // iterate through the edges of this geometry
    for (size_t j = 0; j < GeometryType::num_edges; j++) {

      // get the vertices that correspond to each edge of the triangle
      std::array< uint32_t, Edge::num_vertices > edge_vertices;
      for (int k = 0; k < Edge::num_vertices; k++) {
        edge_vertices[k] = geometry_vertices[GeometryType::local_edge_ids[j][k]];
        vertex_set.insert(edge_vertices[k]);
      }

      output(i, j + GeometryType::edge_offset) = geom_lookup<Edge>(edge_vertices, sorted_edges, edges);

    }

    if constexpr (GeometryType::num_triangles > 0) {

      // iterate through the edges of this geometry
      for (size_t j = 0; j < GeometryType::num_triangles; j++) {

        // get the vertices that correspond to each triangle of the geometry
        std::array< uint32_t, Triangle::num_vertices > tri_vertices;
        for (int k = 0; k < Triangle::num_vertices; k++) {
          tri_vertices[k] = geometry_vertices[GeometryType::local_triangle_ids[j][k]];
          vertex_set.insert(tri_vertices[k]);
        }

        output(i, j + GeometryType::tri_offset) = geom_lookup<Triangle>(tri_vertices, sorted_tris, tris);
    
      }

    }

    if constexpr (GeometryType::num_quadrilaterals > 0) {

      // iterate through the edges of this geometry
      for (size_t j = 0; j < GeometryType::num_quadrilaterals; j++) {

        // get the vertices that correspond to each quadrilateral of the geometry
        std::array< uint32_t, Quadrilateral::num_vertices > quad_vertices;
        for (int k = 0; k < Quadrilateral::num_vertices; k++) {
          quad_vertices[k] = geometry_vertices[GeometryType::local_quadrilateral_ids[j][k]];
          vertex_set.insert(quad_vertices[k]);
        }

        output(i, j + GeometryType::quad_offset) = geom_lookup<Quadrilateral>(quad_vertices, sorted_quads, quads);
    
      }

    }

    Connection cell(i);
    cell.set_geometry(GeometryType::geometry);
    output(i, GeometryType::cell_offset) = cell;
    
  }

  return output;

}

template < typename GeometryType > 
Connection geom_lookup(std::array< uint32_t, GeometryType::num_vertices > & geom_vertices,
                       unordered_map_of_arrays< GeometryType::num_vertices > & sorted_geoms,
                       std::vector < std::array< uint32_t, GeometryType::num_vertices > > & geoms) {

  auto key = sort(geom_vertices);

  if (sorted_geoms.count(key)) {

    // if we have visited this geom before, then
    // we note its index in the mesh
    auto geom_index = sorted_geoms[key];

    // but we also need to determine if the sign of the
    // local geom agrees with its "official" orientation.
    //
    // to do this, we load the "official" geom and compare
    // its vertex ordering to determine if it needs to be reoriented or not
    std::array< uint32_t, GeometryType::num_vertices > official_geom_vertices;
    for (int k = 0; k < GeometryType::num_vertices; k++) {
      official_geom_vertices[k] = geoms[geom_index][k];
    }

    return Connection(geom_index, official_geom_vertices, geom_vertices);

  // but if this is the first time this geom has been
  // visited, we append it to our list of geoms first
  } else {

    sorted_geoms[key] = geoms.size();
    geoms.push_back(geom_vertices);

    // the first element to visit an geom determines the official geom
    // orientation in the mesh, so its sign is always positive
    Connection output(geoms.size() - 1);
    output.set_geometry(GeometryType::geometry);
    return output;

  }

}

template Connection geom_lookup<Edge>(std::array< uint32_t, Edge::num_vertices > & geom_vertices,
                     unordered_map_of_arrays< Edge::num_vertices > & sorted_geoms,
                     std::vector < std::array< uint32_t, Edge::num_vertices > > & geoms);

template Connection geom_lookup<Triangle>(std::array< uint32_t, Triangle::num_vertices > & geom_vertices,
                     unordered_map_of_arrays< Triangle::num_vertices > & sorted_geoms,
                     std::vector < std::array< uint32_t, Triangle::num_vertices > > & geoms);

template Connection geom_lookup<Quadrilateral>(std::array< uint32_t, Quadrilateral::num_vertices > & geom_vertices,
                     unordered_map_of_arrays< Quadrilateral::num_vertices > & sorted_geoms,
                     std::vector < std::array< uint32_t, Quadrilateral::num_vertices > > & geoms);

template nd::array<Connection, 2, memory::space::cpu> process_edges_2D<Triangle>(
  std::set<uint32_t> & vertex_set,
  unordered_map_of_arrays<2> & sorted_edges,
  std::vector < std::array< uint32_t, 2 > > & edges,
  const nd::view<const uint32_t, 2> & geom_to_vertex);

template nd::array<Connection, 2, memory::space::cpu> process_edges_2D<Quadrilateral>(
  std::set<uint32_t> & vertex_set,
  unordered_map_of_arrays<2> & sorted_edges,
  std::vector < std::array< uint32_t, 2 > > & edges,
  const nd::view<const uint32_t, 2> & geom_to_vertex);

template nd::array<Connection, 2, memory::space::cpu> process_geometries_3D<Tetrahedron>(
  std::set<uint32_t> & vertex_set,
  unordered_map_of_arrays<2> & sorted_edges,
  std::vector < std::array< uint32_t, 2 > > & edges,
  unordered_map_of_arrays<3> & sorted_tris,
  std::vector < std::array< uint32_t, 3 > > & tris,
  unordered_map_of_arrays<4> & sorted_quads,
  std::vector < std::array< uint32_t, 4 > > & quads,
  const nd::view<const uint32_t, 2> & geom_to_vertex);

template nd::array<Connection, 2, memory::space::cpu> process_geometries_3D<Hexahedron>(
  std::set<uint32_t> & vertex_set,
  unordered_map_of_arrays<2> & sorted_edges,
  std::vector < std::array< uint32_t, 2 > > & edges,
  unordered_map_of_arrays<3> & sorted_tris,
  std::vector < std::array< uint32_t, 3 > > & tris,
  unordered_map_of_arrays<4> & sorted_quads,
  std::vector < std::array< uint32_t, 4 > > & quads,
  const nd::view<const uint32_t, 2> & geom_to_vertex);

} // namespace femto
