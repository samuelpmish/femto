#pragma once

#include <array>
#include <inttypes.h>

#include "fm/macros.hpp"
#include "containers/ndarray.hpp"
#include "misc/for_constexpr.hpp"

namespace femto {

enum class Geometry : uint8_t {
  Vertex,
  Edge,
  Triangle,
  Quadrilateral,
  Tetrahedron,
  Hexahedron
};

template < typename T >
void foreach_geometry(T && function) {
  foreach_constexpr< 
    Geometry::Edge,
    Geometry::Triangle,
    Geometry::Quadrilateral,
    Geometry::Tetrahedron,
    Geometry::Hexahedron
  >(function);
}

inline const char * to_string(Geometry g) {
  switch (g) {
    case Geometry::Vertex:        return "Geometry_Vertex";
    case Geometry::Edge:          return "Geometry_Edge";
    case Geometry::Triangle:      return "Geometry_Triangle";
    case Geometry::Quadrilateral: return "Geometry_Quadrilateral";
    case Geometry::Tetrahedron:   return "Geometry_Tetrahedron";
    case Geometry::Hexahedron:    return "Geometry_Hexahedron";
  }
  return "";
}

constexpr Geometry all_geometries[6] = {
  Geometry::Vertex,
  Geometry::Edge,
  Geometry::Triangle,
  Geometry::Quadrilateral,
  Geometry::Tetrahedron,
  Geometry::Hexahedron
};

constexpr nd::range<const Geometry *> geometries_by_dim[4] = {
  {all_geometries+0, all_geometries+1},
  {all_geometries+1, all_geometries+2},
  {all_geometries+2, all_geometries+4},
  {all_geometries+4, all_geometries+6}
};

__host__ __device__ constexpr uint32_t dimension(Geometry g) {
  switch (g) {
    case Geometry::Vertex:        return 0;
    case Geometry::Edge:          return 1;
    case Geometry::Triangle:      return 2;
    case Geometry::Quadrilateral: return 2;
    case Geometry::Tetrahedron:   return 3;
    case Geometry::Hexahedron:    return 3;
  }
  return 1u<<30;
}

template < typename T >
struct GeometryData {
  T vert;
  T edge;
  T tri;
  T quad;
  T tet;
  T hex;

  T & operator[](Geometry g) {
    switch (g) {
      case Geometry::Vertex:        return vert;
      case Geometry::Edge:          return edge;
      case Geometry::Triangle:      return tri;
      case Geometry::Quadrilateral: return quad;
      case Geometry::Tetrahedron:   return tet;
      case Geometry::Hexahedron:    return hex;
    }
    return vert; // unreachable code, to silence compiler warnings
  }

  const T & operator[](Geometry g) const {
    switch (g) {
      case Geometry::Vertex:        return vert;
      case Geometry::Edge:          return edge;
      case Geometry::Triangle:      return tri;
      case Geometry::Quadrilateral: return quad;
      case Geometry::Tetrahedron:   return tet;
      case Geometry::Hexahedron:    return hex;
    }
    return vert; // unreachable code, to silence compiler warnings
  }

  bool operator==(const GeometryData<T> & other) const {
    return (vert == other.vert) && 
           (edge == other.edge) && 
           ( tri == other.tri ) && 
           (quad == other.quad) && 
           ( tet == other.tet ) && 
           ( hex == other.hex );
  }

};

struct GeometryInfo : public GeometryData<uint32_t> {
  static GeometryInfo from_array(uint32_t * data) {
    return GeometryInfo { data[0], data[1], data[2], data[3], data[4], data[5] };
  }
};

inline void operator+=(GeometryInfo & a, const GeometryInfo & b) {
  a.vert += b.vert;
  a.edge += b.edge;
  a.tri += b.tri;
  a.quad += b.quad;
  a.tet += b.tet;
  a.hex += b.hex;
};

inline GeometryInfo operator*(GeometryInfo a, GeometryInfo b) {
  return GeometryInfo{
    a.vert * b.vert,
    a.edge * b.edge,
    a.tri * b.tri,
    a.quad * b.quad,
    a.tet * b.tet,
    a.hex * b.hex
  };
};

inline GeometryInfo operator*(GeometryInfo a, uint32_t scale) {
  return GeometryInfo{
    a.vert * scale,
    a.edge * scale,
    a.tri * scale,
    a.quad * scale,
    a.tet * scale,
    a.hex * scale
  };
};

inline uint32_t total(GeometryInfo input){
  return input.vert + input.edge + input.tri + input.quad + input.tet + input.hex;
};

inline GeometryInfo scan(const GeometryInfo & input){
  GeometryInfo output;
  output.vert  = 0;
  output.edge  = output.vert + input.vert;
  output.tri   = output.edge + input.edge;
  output.quad  = output.tri  + input.tri;
  output.tet   = output.quad + input.quad;
  output.hex   = output.tet  + input.tet;
  return output;
};

template < uint32_t n >
std::array< uint32_t, n + 1 > scan(const uint32_t (&input)[n]){
  std::array< uint32_t, n + 1 > output{};
  for (uint32_t i = 0; i < n; i++) {
    output[i+1] = output[i] + input[i];
  }
  return output;
};

inline GeometryData< nd::range<uint32_t> > ranges(GeometryInfo a) {
  GeometryData< nd::range<uint32_t> > output;

  uint32_t total = 0;
  output.vert = nd::range{total, total + a.vert}; total += a.vert;
  output.edge = nd::range{total, total + a.edge}; total += a.edge;
  output.tri  = nd::range{total, total + a.tri};  total += a.tri;
  output.quad = nd::range{total, total + a.quad}; total += a.quad;
  output.tet  = nd::range{total, total + a.tet};  total += a.tet;
  output.hex  = nd::range{total, total + a.hex};  total += a.hex;

  return output;
};

template < Geometry g >
struct GeometryType;

template <>
struct GeometryType< Geometry::Vertex > {
  static constexpr Geometry geometry = Geometry::Vertex;
  static constexpr int offset = 0;
  static constexpr int dim = 0; 
};
using Vertex = GeometryType< Geometry::Vertex >;

template <>
struct GeometryType< Geometry::Edge > {
  static constexpr Geometry geometry = Geometry::Edge;
  static constexpr int offset = 1;

  static constexpr int dim = 1; 
  static constexpr int num_vertices = 2;

  static constexpr int cell_offset = num_vertices;
};
using Edge = GeometryType< Geometry::Edge >;

template <>
struct GeometryType< Geometry::Triangle > {
  static constexpr Geometry geometry = Geometry::Triangle;
  static constexpr int offset = 2;

  static constexpr int dim = 2; 
  static constexpr int num_vertices = 3;
  static constexpr int num_edges = 3;

  static constexpr int edge_offset = num_vertices;
  static constexpr int cell_offset = num_vertices + num_edges;

  static constexpr int local_edge_ids[3][2] = {{0, 1},{1, 2},{2, 0}};

  __host__ __device__ static constexpr uint32_t number(int n) { return (n * (n + 1)) / 2; };
};
using Triangle = GeometryType< Geometry::Triangle >;

template <>
struct GeometryType< Geometry::Quadrilateral > {
  static constexpr Geometry geometry = Geometry::Quadrilateral;
  static constexpr int offset = 3;

  static constexpr int dim = 2; 
  static constexpr int num_vertices = 4;
  static constexpr int num_edges = 4;

  static constexpr int edge_offset = num_vertices;
  static constexpr int cell_offset = edge_offset + num_edges;

  static constexpr int local_edge_ids[4][4] = {{0, 1},{1, 2},{2, 3},{3,0}};
};
using Quadrilateral = GeometryType< Geometry::Quadrilateral >;

template <>
struct GeometryType< Geometry::Tetrahedron > {
  static constexpr Geometry geometry = Geometry::Tetrahedron;
  static constexpr int offset = 4;

  static constexpr int dim = 3; 
  static constexpr int num_vertices = 4;
  static constexpr int num_edges = 6;
  static constexpr int num_triangles = 4;
  static constexpr int num_quadrilaterals = 0;

  static constexpr int edge_offset = num_vertices;
  static constexpr int tri_offset = edge_offset + num_edges;
  static constexpr int quad_offset = tri_offset + num_triangles;
  static constexpr int cell_offset = quad_offset + num_quadrilaterals;

  static constexpr double vertices[4][3] = {{0,0,0}, {1,0,0}, {0,1,0}, {0,0,1}};
  static constexpr int local_edge_ids[6][2] = {{0, 1}, {1, 2}, {2, 0}, {0, 3}, {1, 3}, {2, 3}};
  static constexpr int local_triangle_ids[4][3] = {{2, 1, 0}, {0, 1, 3}, {1, 2, 3}, {2, 0, 3}};

  __host__ __device__ static constexpr int number(int n) { return (n * (n + 1) * (n + 2)) / 6; }
};
using Tetrahedron = GeometryType< Geometry::Tetrahedron >;

template <>
struct GeometryType< Geometry::Hexahedron > {
  static constexpr Geometry geometry = Geometry::Hexahedron;
  static constexpr int offset = 5;

  static constexpr int dim = 3; 
  static constexpr int num_vertices = 8;
  static constexpr int num_edges = 12;
  static constexpr int num_triangles = 0;
  static constexpr int num_quadrilaterals = 6;

  static constexpr int edge_offset = num_vertices;
  static constexpr int tri_offset = edge_offset + num_edges;
  static constexpr int quad_offset = tri_offset + num_triangles;
  static constexpr int cell_offset = quad_offset + num_quadrilaterals;

  //  mathematica code for visualizing these edge/quad numberings
  /*
    localEdgeIds = 1 +  {{0, 1},{1, 2},{3, 2},{0, 3},{0, 4},{1, 5},{2, 6},{3, 7},{4, 5},{5, 6},{7, 6},{4, 7}};
    localQuadIds = 1 + {{1, 0, 3, 2}, {0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}, {4, 5, 6, 7}};
    vertices = {{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1}, {-1, -1, 1}, {1, -1, 1}, {1, 1, 1}, {-1, 1, 1}};
    Graphics3D[{
      Thickness[0.01], JoinForm["Round"], Black, PointSize[0.03], 
      Point /@ vertices, Table[Text[Style[i - 1, Large], 1.2 vertices[[i]]], {i, 1, 8}],
      Red, Arrow[( { {0.9, 0.1}, {0.1, 0.9} } ) . vertices[[#]]] & /@ localEdgeIds,
      Table[Text[Style[i - 1, Large], 1.3 Mean[vertices[[localEdgeIds[[i]]]]]], {i, 1, 12}],
      Blue, Arrow[( {
            {0.7, 0.1, 0.1, 0.1},
            {0.1, 0.7, 0.1, 0.1},
            {0.1, 0.1, 0.7, 0.1},
            {0.1, 0.1, 0.1, 0.7}
           } ) . vertices[[#]]] & /@ localQuadIds,
      Table[Text[Style[i - 1, Large], Mean[vertices[[localQuadIds[[i]]]]]], {i, 1, 6}]
    }, Boxed -> False]
  */
  static constexpr int local_edge_ids[12][2] = {{0, 1},{1, 2},{3, 2},{0, 3},{0, 4},{1, 5},{2, 6},{3, 7},{4, 5},{5, 6},{7, 6},{4, 7}};
  static constexpr int local_quadrilateral_ids[6][4] = {{1, 0, 3, 2},{0, 1, 5, 4},{1, 2, 6, 5},{2, 3, 7, 6},{3, 0, 4, 7},{4, 5, 6, 7}};
};
using Hexahedron = GeometryType< Geometry::Hexahedron >;

}