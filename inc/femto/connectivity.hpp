#pragma once

#include <set>
#include <array>
#include <cstring>
#include <unordered_map>

#include "containers/ndarray.hpp"
#include "femto/connection.hpp"

namespace femto {

struct array_hasher {
  template<size_t n>
  std::size_t operator()(const std::array< uint32_t, n > & arr) const {
    uint32_t seed = 0;
    for(const auto elem : arr) {
      seed ^= std::hash<uint32_t>()(elem) + 0x9e3779b9 + (seed<<6) + (seed>>2);
    }
    return seed;
  }
};

template < size_t n >
using unordered_map_of_arrays = std::unordered_map< std::array< uint32_t, n >, uint32_t, array_hasher >;

template < typename GeometryType > 
Connection geom_lookup(std::array< uint32_t, GeometryType::num_vertices > & geom_vertices,
                     unordered_map_of_arrays< GeometryType::num_vertices > & sorted_geoms,
                     std::vector < std::array< uint32_t, GeometryType::num_vertices > > & geoms);

template < typename GeometryType > 
nd::array<Connection, 2, memory::space::cpu> process_edges_2D(
  std::set<uint32_t> & vertex_set,
  unordered_map_of_arrays<2> & sorted_edges,
  std::vector < std::array< uint32_t, 2 > > & edges,
  const nd::view<const uint32_t, 2> & geom_to_vertex);

template < typename GeometryType > 
nd::array<Connection, 2, memory::space::cpu> process_geometries_3D(
  std::set<uint32_t> & vertex_set,
  unordered_map_of_arrays<2> & sorted_edges,
  std::vector < std::array< uint32_t, 2 > > & edges,
  unordered_map_of_arrays<3> & sorted_tris,
  std::vector < std::array< uint32_t, 3 > > & tris,
  unordered_map_of_arrays<4> & sorted_quads,
  std::vector < std::array< uint32_t, 4 > > & quads,
  const nd::view<const uint32_t, 2> & geom_to_vertex);

} // namespace femto
