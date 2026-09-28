#include "femto/mesh.hpp"

#include "femto/connectivity.hpp"
#include "femto/interpolation.hpp"

#include "misc/json.hpp"
#include "misc/nvtx.hpp"
#include "misc/timer.hpp"

#include <fstream>
#include <iostream>

namespace nd {

template < typename T, memory::space mem_space >
void from_json(const nlohmann::json& j, nd::array< T, 2, mem_space > & arr) {
  uint32_t n1 = j.size();
  uint32_t n2 = j[0].size();

  arr = nd::array< T, 2, mem_space >({n1, n2});

  for (int i1 = 0; i1 < n1; i1++) {
    for (int i2 = 0; i2 < n2; i2++) {
      arr(i1,i2) = j[i1][i2];
    }
  }
}

}

namespace femto {

template <>
Mesh<memory::space::cpu> Mesh<memory::space::cpu>::create_1D(const nd::array< double, 2, memory::space::cpu > & nodes, uint32_t degree,
                                               const nd::array< uint32_t, 2, memory::space::cpu > & edges) {

  nvtx::scoped_range nvtx_scope("Mesh::create_1D");

  constexpr uint32_t gdim = 1;

  nd::array<Connection,2,memory::space::cpu> empty({0,0});

  Mesh<memory::space::cpu> mesh;

  mesh.tet = empty;
  mesh.hex = empty;
  mesh.tri = empty;
  mesh.quad = empty;

  std::set< uint32_t > vertex_set{};

  uint32_t num_edges = edges.shape[0];
  mesh.edge = nd::array<Connection, 2, memory::space::cpu>({num_edges, Edge::num_vertices + 1});
  {
    nvtx::scoped_range nvtx_edge_loop("Mesh::create_1D edge loop");
    for (size_t i = 0; i < num_edges; i++) {
      for (size_t j = 0; j < Edge::num_vertices; j++) {
        Connection vertex(edges(i,j));
        vertex.set_geometry(Geometry::Vertex);
        vertex.set_subindex(j);
        mesh.edge(i,j) = vertex;
        vertex_set.insert(vertex.index);
      }

      Connection edge(i);
      edge.set_geometry(Geometry::Edge);
      mesh.edge(i, Edge::num_vertices) = edge;
    }
  }

  uint32_t num_vertices = vertex_set.size();
  mesh.vert = nd::array<Connection, 2, memory::space::cpu>({num_vertices, 1});
  for (int i = 0; i < num_vertices; i++) {
    Connection vertex(i);
    vertex.set_geometry(Geometry::Vertex);
    mesh.vert(i, 0) = vertex;
  }

  GeometryInfo nodes_per = interior_nodes_per_geom(FunctionSpace{Family::H1, degree}, gdim);
  GeometryInfo counts = mesh.geometry_counts();

  mesh.X.degree = degree;
  mesh.X.data = nodes;
  mesh.X.offsets = scan(nodes_per * counts);

  mesh.geometry_dimension = 1;
  mesh.spatial_dimension = mesh.X.data.shape[1];

  return mesh;

}

template <>
Mesh<memory::space::cpu> Mesh<memory::space::cpu>::create_2D(const nd::array< double, 2, memory::space::cpu > & nodes, uint32_t degree,
                                               const nd::array< uint32_t, 2, memory::space::cpu > & tris, const nd::array< uint32_t, 2, memory::space::cpu > & quads) {

  nvtx::scoped_range nvtx_scope("Mesh::create_2D");

  constexpr uint32_t gdim = 2;

  nd::array<Connection,2,memory::space::cpu> empty({0,0});

  Mesh<memory::space::cpu> mesh;

  mesh.tet = empty;
  mesh.hex = empty;

  // keep track of the vertices visited, as a sanity check
  std::set< uint32_t > vertex_set{};

  unordered_map_of_arrays< 2 > sorted_edges{};
  std::vector < std::array< uint32_t, 2 > > edges{};

  // iterate over the triangles and quads, and figure out missing connectivities (edges)
  mesh.tri = process_edges_2D<Triangle>(vertex_set, sorted_edges, edges, tris);
  mesh.quad = process_edges_2D<Quadrilateral>(vertex_set, sorted_edges, edges, quads);

  // now that we know how many edges there are in the mesh
  // we size the container appropriately and copy the data into place
  uint32_t num_edges = edges.size();
  mesh.edge = nd::array<Connection, 2, memory::space::cpu>({num_edges, 3});
  {
    nvtx::scoped_range nvtx_edge_loop("Mesh::create_2D edge loop");
    for (size_t i = 0; i < num_edges; i++) {
      for (size_t j = 0; j < Edge::num_vertices; j++) {
        Connection vertex(edges[i][j]);
        vertex.set_geometry(Geometry::Vertex);
        vertex.set_subindex(j);
        mesh.edge(i,j) = vertex;
      }

      Connection edge(i);
      edge.set_geometry(Geometry::Edge);
      mesh.edge(i, Edge::num_vertices) = edge;
    }
  }

  // verify that there are no gaps in the vertex numbering
  if ((vertex_set.size()-1) != *vertex_set.rbegin()) {
    std::cout << "invalid mesh file: vertex numbering is not contiguous" << std::endl;
    exit(1);
  }

  uint32_t num_vertices = vertex_set.size();
  mesh.vert = nd::array<Connection, 2, memory::space::cpu>({num_vertices, 1});
  for (int i = 0; i < num_vertices; i++) {
    Connection vertex(i);
    vertex.set_geometry(Geometry::Vertex);
    mesh.vert(i, 0) = vertex;
  }

  GeometryInfo nodes_per = interior_nodes_per_geom(FunctionSpace{Family::H1, degree}, gdim);
  GeometryInfo counts = mesh.geometry_counts();

  mesh.X.degree = degree;
  mesh.X.data = nodes;
  mesh.X.offsets = scan(nodes_per * counts);

  mesh.geometry_dimension = 2;
  mesh.spatial_dimension = mesh.X.data.shape[1];

  return mesh;
}

template <>
Mesh<memory::space::cpu> Mesh<memory::space::cpu>::create_3D(const nd::array< double, 2, memory::space::cpu > & nodes, uint32_t degree,
                                               const nd::array< uint32_t, 2, memory::space::cpu > & tets, const nd::array< uint32_t, 2, memory::space::cpu > & hexes) {

  nvtx::scoped_range nvtx_scope("Mesh::create_3D");

  constexpr uint32_t gdim = 3;

  Mesh<memory::space::cpu> mesh;

  // keep track of the vertices visited, as a sanity check
  std::set< uint32_t > vertex_set{};

  unordered_map_of_arrays< 2 > sorted_edges{};
  std::vector < std::array< uint32_t, 2 > > edges{};

  unordered_map_of_arrays< 3 > sorted_tris{};
  std::vector < std::array< uint32_t, 3 > > tris{};

  unordered_map_of_arrays< 4 > sorted_quads{};
  std::vector < std::array< uint32_t, 4 > > quads{};

  // iterate over the triangles and quads, and figure out missing connectivities (edges)
  mesh.tet = process_geometries_3D<Tetrahedron>(vertex_set, sorted_edges, edges, sorted_tris, tris, sorted_quads, quads, tets);
  mesh.hex = process_geometries_3D<Hexahedron>(vertex_set, sorted_edges, edges, sorted_tris, tris, sorted_quads, quads, hexes);

  uint32_t num_quads = quads.size();
  mesh.quad = nd::array<Connection, 2, memory::space::cpu>({num_quads, 9});
  {
    nvtx::scoped_range nvtx_quad_loop("Mesh::create_3D quad loop");
    for (uint32_t i = 0; i < num_quads; i++) {
      for (uint32_t j = 0; j < Quadrilateral::num_vertices; j++) {
        Connection vertex(quads[i][j]);
        vertex.set_geometry(Geometry::Vertex);
        vertex.set_subindex(j);
        mesh.quad(i,j) = vertex;
      }

      for (uint32_t j = 0; j < Quadrilateral::num_edges; j++) {
        std::array< uint32_t, Edge::num_vertices > edge_vertices;
        for (uint32_t k = 0; k < Edge::num_vertices; k++) {
          edge_vertices[k] = quads[i][Quadrilateral::local_edge_ids[j][k]];
        }
        mesh.quad(i,j+Quadrilateral::edge_offset) = geom_lookup<Edge>(edge_vertices, sorted_edges, edges);
      }

      Connection quad(i);
      quad.set_geometry(Geometry::Quadrilateral);
      mesh.quad(i,Quadrilateral::cell_offset) = quad;
    }
  }

  uint32_t num_tris = tris.size();
  mesh.tri = nd::array<Connection, 2, memory::space::cpu>({num_tris, 7});
  {
    nvtx::scoped_range nvtx_tri_loop("Mesh::create_3D tri loop");
    for (uint32_t i = 0; i < num_tris; i++) {
      for (uint32_t j = 0; j < Triangle::num_vertices; j++) {
        Connection vertex(tris[i][j]);
        vertex.set_geometry(Geometry::Vertex);
        vertex.set_subindex(j);
        mesh.tri(i,j) = vertex;
      }

      for (int j = 0; j < Triangle::num_edges; j++) {
        std::array< uint32_t, Edge::num_vertices > edge_vertices;
        for (int k = 0; k < Edge::num_vertices; k++) {
          edge_vertices[k] = tris[i][Triangle::local_edge_ids[j][k]];
        }
        mesh.tri(i,j+Triangle::edge_offset) = geom_lookup<Edge>(edge_vertices, sorted_edges, edges);
      }

      Connection tri(i);
      tri.set_geometry(Geometry::Triangle);
      mesh.tri(i,Triangle::cell_offset) = tri;
    }
  }

  uint32_t num_edges = edges.size();
  mesh.edge = nd::array<Connection, 2, memory::space::cpu>({num_edges, 3});
  {
    nvtx::scoped_range nvtx_edge_loop("Mesh::create_3D edge loop");
    for (uint32_t i = 0; i < num_edges; i++) {
      for (size_t j = 0; j < Edge::num_vertices; j++) {
        Connection vertex(edges[i][j]);
        vertex.set_geometry(Geometry::Vertex);
        vertex.set_subindex(j);
        mesh.edge(i,j) = vertex;
      }

      Connection edge(i);
      edge.set_geometry(Geometry::Edge);
      mesh.edge(i,Edge::cell_offset) = edge;
    }
  }

  // verify that there are no gaps in the vertex numbering
  if ((vertex_set.size()-1) != *vertex_set.rbegin()) {
    std::cout << "invalid mesh file: vertex numbering is not contiguous" << std::endl;
    exit(1);
  }

  uint32_t num_vertices = vertex_set.size();
  mesh.vert = nd::array<Connection, 2, memory::space::cpu>({num_vertices, 1});
  for (uint32_t i = 0; i < num_vertices; i++) {
    Connection vertex(i);
    vertex.set_geometry(Geometry::Vertex);
    mesh.vert(i, 0) = vertex;
  }

  GeometryInfo nodes_per = nodes_per_geom(FunctionSpace{Family::H1, degree}, gdim);
  GeometryInfo counts = mesh.geometry_counts();

  mesh.X.degree = degree;
  mesh.X.data = nodes;
  mesh.X.offsets = scan(nodes_per * counts);

  mesh.geometry_dimension = 3;
  mesh.spatial_dimension = mesh.X.data.shape[1];

  return mesh;

}

template <>
Mesh<memory::space::cpu> Mesh<memory::space::cpu>::import_from_json_string(std::string str) {

  // parse the string as json
  auto j = nlohmann::json::parse(str);

  nd::array<uint32_t,2,memory::space::cpu> empty({0,0});

  auto nodes = j.at("nodes");
  int degree = nodes["degree"];
  nd::array<double,2,memory::space::cpu> nodes_arr = nodes["data"].get< nd::array<double, 2, memory::space::cpu> >();
 
  if (j.contains("tet") || j.contains("hex")) {
    auto tet_to_vertex = j.contains("tet") ? j["tet"].get< nd::array<uint32_t, 2, memory::space::cpu > >() : empty;
    auto hex_to_vertex = j.contains("hex") ? j["hex"].get< nd::array<uint32_t, 2, memory::space::cpu > >() : empty;
    return Mesh<memory::space::cpu>::create_3D(nodes_arr, degree, tet_to_vertex, hex_to_vertex);
  } 

  if (j.contains("tri") || j.contains("quad")) {
    auto tri_to_vertex = j.contains("tri") ? j["tri"].get< nd::array<uint32_t, 2, memory::space::cpu > >() : empty;
    auto quad_to_vertex = j.contains("quad") ? j["quad"].get< nd::array<uint32_t, 2, memory::space::cpu > >() : empty;
    return Mesh<memory::space::cpu>::create_2D(nodes_arr, degree, tri_to_vertex, quad_to_vertex);
  }

  if (j.contains("edge")) {
    auto edges = j["edge"].get< nd::array<uint32_t, 2, memory::space::cpu > >();
    return Mesh<memory::space::cpu>::create_1D(nodes_arr, degree, edges);
  }

  std::cout << "invalid mesh format" << std::endl;
  exit(1);

}

// defined in connectivity.cpp
uint32_t index(uint32_t x);

nd::array<uint32_t, 1, memory::space::cpu> to_ndarray(const std::set<uint32_t> & s) {
  nd::array<uint32_t, 1, memory::space::cpu> output({uint32_t(s.size())});
  uint32_t i = 0;
  for (auto value : s) {
    output(i) = value;
    i++;
  }
  return output;
}

// boundary cells are (d-1)-dimensional cells that
// appear exactly once in the mesh. Find the boundary cells
// by iterating over each facet of the d-dimensional cells,
// and counting how many times each of those facets appears.
SubMesh<> boundary_of(const Mesh<> & mesh) {

  nd::array<uint32_t, 1, memory::space::cpu> empty;  // default-constructs to shape {0}

  SubMesh<> bdr{&mesh};

  bdr.geometry_dimension = mesh.geometry_dimension - 1;
  bdr.spatial_dimension = mesh.spatial_dimension;

  auto there_can_be_only_one = [](std::set<uint32_t> & set, auto value) {
    if (set.count(value)) {
      set.erase(value);
    } else {
      set.insert(value);
    }
  };

  if (mesh.geometry_dimension == 3) {

    // find the tris and quads that appear exactly once in the mesh
    std::set<uint32_t> bdr_tris{};

    for (int i = 0; i < mesh.tet.shape[0]; i++) {
      for (int j = 0; j < Tetrahedron::num_triangles; j++) {
        auto tri_id = mesh.tet(i,j+Tetrahedron::tri_offset).index;
        there_can_be_only_one(bdr_tris, tri_id);
      }
    }

    std::set<uint32_t> bdr_quads{};
    for (int i = 0; i < mesh.hex.shape[0]; i++) {
      for (int j = 0; j < Hexahedron::num_quadrilaterals; j++) {
        auto quad_id = mesh.hex(i,j+Hexahedron::quad_offset).index;
        there_can_be_only_one(bdr_quads, quad_id);
      }
    }

    // then get the edges that belong to those boundary tris and quads
    std::set<uint32_t> bdr_edges;
    for (auto tri_id : bdr_tris) {
      bdr_edges.insert(mesh.tri(tri_id, Triangle::edge_offset + 0).index);
      bdr_edges.insert(mesh.tri(tri_id, Triangle::edge_offset + 1).index);
      bdr_edges.insert(mesh.tri(tri_id, Triangle::edge_offset + 2).index);
    }

    for (auto quad_id : bdr_quads) {
      bdr_edges.insert(mesh.quad(quad_id, Quadrilateral::edge_offset + 0).index);
      bdr_edges.insert(mesh.quad(quad_id, Quadrilateral::edge_offset + 1).index);
      bdr_edges.insert(mesh.quad(quad_id, Quadrilateral::edge_offset + 2).index);
      bdr_edges.insert(mesh.quad(quad_id, Quadrilateral::edge_offset + 3).index);
    }

    // finally, get the vertices that belong to the boundary edges
    std::set<uint32_t> bdr_verts;
    for (auto edge_id : bdr_edges) {
      bdr_verts.insert(mesh.edge(edge_id, 0).index);
      bdr_verts.insert(mesh.edge(edge_id, 1).index);
    }

    bdr.vert = to_ndarray(bdr_verts);
    bdr.edge = to_ndarray(bdr_edges);
    bdr.tri = to_ndarray(bdr_tris);
    bdr.quad = to_ndarray(bdr_quads);
    bdr.tet = empty;
    bdr.hex = empty;
  
  }

  if (mesh.geometry_dimension == 2) {

    // find the tris and quads that appear exactly once in the mesh
    std::set<uint32_t> bdr_edges;
    for (int i = 0; i < mesh.tri.shape[0]; i++) {
      for (int j = 0; j < Triangle::num_edges; j++) {
        auto edge_id = mesh.tri(i,j+Triangle::edge_offset).index;
        there_can_be_only_one(bdr_edges, edge_id);
      }
    }

    for (int i = 0; i < mesh.quad.shape[0]; i++) {
      for (int j = 0; j < Quadrilateral::num_edges; j++) {
        auto edge_id = mesh.quad(i,j+Quadrilateral::edge_offset).index;
        there_can_be_only_one(bdr_edges, edge_id);
      }
    }

    // then the vertices that belong to the boundary edges
    std::set<uint32_t> bdr_verts;
    for (auto edge_id : bdr_edges) {
      bdr_verts.insert(mesh.edge(edge_id, 0).index);
      bdr_verts.insert(mesh.edge(edge_id, 1).index);
    }

    bdr.vert = to_ndarray(bdr_verts);
    bdr.edge = to_ndarray(bdr_edges);
    bdr.tri = empty;
    bdr.quad = empty;
    bdr.tet = empty;
    bdr.hex = empty;
  
  }

  if (mesh.geometry_dimension == 1) {

    std::cout << "unimplemented" << std::endl;
    exit(1);

  }

  return bdr;

}

} // namespace femto
