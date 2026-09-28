#include "femto/mesh.hpp"

#include <fstream>
#include <iostream>
#include <filesystem>

#include "femto/mesh/io.hpp"

#include "misc/timer.hpp"

namespace femto {

static bool is_supported(io::Element::Type type) {
  return (type == io::Element::Type::Tri3) ||
         (type == io::Element::Type::Tri6) ||
         (type == io::Element::Type::Quad4) ||
         (type == io::Element::Type::Quad9) ||
         (type == io::Element::Type::Tet4) ||
         (type == io::Element::Type::Tet10) ||
         (type == io::Element::Type::Hex8) ||
         (type == io::Element::Type::Hex27);
}

static Geometry element_geometry(io::Element::Type type) {
  switch (type) {
    case io::Element::Type::Tri3: return Geometry::Triangle;
    case io::Element::Type::Tri6: return Geometry::Triangle;
    case io::Element::Type::Quad4: return Geometry::Quadrilateral;
    case io::Element::Type::Quad9: return Geometry::Quadrilateral;
    case io::Element::Type::Tet4: return Geometry::Tetrahedron;
    case io::Element::Type::Tet10: return Geometry::Tetrahedron;
    case io::Element::Type::Hex8: return Geometry::Hexahedron;
    case io::Element::Type::Hex27: return Geometry::Hexahedron;
    default: return Geometry::Vertex;
  }
}

// returns file extension in lower case characters
static std::string file_extension(std::string filename) {
  std::string ext = std::filesystem::path(filename).extension().string();
  for(auto& c : ext) { c = tolower(c); }
  return ext;
}

Mesh<> from_io_mesh(const io::Mesh & iomesh) {
   
//  static Mesh create_1D(const nd::cpu_array<double, 2> & nodes, int degree, 
//                        const nd::cpu_array<uint32_t, 2> & edges);
//
//  static Mesh create_2D(const nd::cpu_array<double, 2> & nodes, int degree, 
//                        const nd::cpu_array<uint32_t, 2> & tris, const nd::cpu_array<uint32_t, 2> & quads);
//
//  static Mesh create_3D(const nd::cpu_array<double, 2> & nodes, int degree, 
//                        const nd::cpu_array<uint32_t, 2> & tets, const nd::cpu_array<uint32_t, 2> & hexes);
//

  // preprocessing: figure out the spatial and geometric dimensions
  // of the mesh, and check that it doesn't contain any unsupported
  // element types
  uint32_t spatial_dim = 0;
  for (auto node : iomesh.nodes) {
    if (node[2] != 0) spatial_dim = std::max(spatial_dim, 3u);
    if (node[1] != 0) spatial_dim = std::max(spatial_dim, 2u);
  }

  uint32_t degree = 0;
  uint32_t geometry_dim = 0;
  GeometryInfo counts{};
  for (auto elem : iomesh.elements) {
    if (is_supported(elem.type)) {
      if (geometry_dim == 0) { 
        geometry_dim = io::dimension(elem.type); 
        degree = io::degree(elem.type); 
      }

      if (degree != io::degree(elem.type)) {
        std::cout << "error: meshes must contain elements with consistent polynomial order" << std::endl;
        exit(1);
      }

      if (geometry_dim != io::dimension(elem.type)) {
        std::cout << "error: meshes must only contain cells with consistent dimension" << std::endl;
        exit(1);
      }

      counts[element_geometry(elem.type)]++;
    } else {
      std::cout << "error: file contains an unsupported element type" << std::endl;
      exit(1);
    }
  }

  nd::array< double, 2, memory::space::cpu > nodes({uint32_t(iomesh.nodes.size()), spatial_dim});
  for (uint32_t i = 0; i < iomesh.nodes.size(); i++) {
    for (uint32_t j = 0; j < spatial_dim; j++) {
      nodes(i, j) = iomesh.nodes[i][j];
    }
  }

  if (geometry_dim == 2) {

    uint32_t nodes_per_tri = FiniteElement<Geometry::Triangle, Family::H1>{degree}.num_nodes();
    nd::array< uint32_t, 2, memory::space::cpu > tris({counts.tri, nodes_per_tri});
    uint32_t tri_id = 0;

    uint32_t nodes_per_quad = FiniteElement<Geometry::Quadrilateral, Family::H1>{degree}.num_nodes();
    nd::array< uint32_t, 2, memory::space::cpu > quads({counts.quad, nodes_per_quad});
    uint32_t quad_id = 0;

    for (auto elem : iomesh.elements) {
      if (element_geometry(elem.type) == Geometry::Triangle) {
        for (int i = 0; i < nodes_per_tri; i++) {
          tris(tri_id, i) = elem.node_ids[i];
        }
        tri_id++;
      }
      if (element_geometry(elem.type) == Geometry::Quadrilateral) {
        for (int i = 0; i < nodes_per_quad; i++) {
          quads(quad_id, i) = elem.node_ids[i];
        }
        quad_id++;
      }
    }

    return Mesh<>::create_2D(nodes, degree, tris, quads);
  }

  if (geometry_dim == 3) {
    uint32_t nodes_per_tet = FiniteElement<Geometry::Tetrahedron, Family::H1>{degree}.num_nodes();
    nd::array< uint32_t, 2, memory::space::cpu > tets({counts.tet, nodes_per_tet});
    uint32_t tet_id = 0;

    uint32_t nodes_per_hex = FiniteElement<Geometry::Hexahedron, Family::H1>{degree}.num_nodes();
    nd::array< uint32_t, 2, memory::space::cpu > hexes({counts.hex, nodes_per_hex});
    uint32_t hex_id = 0;

    for (auto elem : iomesh.elements) {
      if (element_geometry(elem.type) == Geometry::Tetrahedron) {
        for (int i = 0; i < nodes_per_tet; i++) {
          tets(tet_id, i) = elem.node_ids[i];
        }
        tet_id++;
      }
      if (element_geometry(elem.type) == Geometry::Hexahedron) {
        for (int i = 0; i < nodes_per_hex; i++) {
          hexes(hex_id, i) = elem.node_ids[i];
        }
        hex_id++;
      }
    }

    return Mesh<>::create_3D(nodes, degree, tets, hexes);
  }

  std::cout << "only 2D and 3D meshes are supported" << std::endl; 
  exit(1);
}

template <>
Mesh<> Mesh<>::load(std::string filename) {
  std::string ext = file_extension(filename);
  femto::timer stopwatch;
  if (ext == ".msh") {
    stopwatch.start();
    auto iomsh = io::import_gmsh_v22(filename);    
    stopwatch.stop();
    if (print_timings) {
      std::cout << "import io::mesh: " << stopwatch.elapsed() * 1000.0 << "ms" << std::endl;
    }

    return from_io_mesh(iomsh);
  } else if (ext == ".vtk") {
    // TODO
    std::cout << "loading from vtk unsupported" << std::endl;
    exit(1);
  } else if (ext == ".json") {
    std::ifstream infile(filename);
    std::string str;
    if (infile) {
      stopwatch.start();
      infile.seekg(0, std::ios::end);   
      str.reserve(infile.tellg());
      infile.seekg(0, std::ios::beg);
      str.assign((std::istreambuf_iterator<char>(infile)), std::istreambuf_iterator<char>());
      stopwatch.stop();

      if (print_timings) {
        std::cout << "read JSON file into string: " << stopwatch.elapsed() * 1000.0 << "ms" << std::endl;
      }
    }

    return Mesh<>::import_from_json_string(str);
  } else if (ext == ".vtu") {
    // TODO
    std::cout << "loading from vtu unsupported" << std::endl;
    exit(1);
  } else {
    std::cout << "unrecognized extension '" << ext << "', exiting..." << std::endl;
    exit(1);
  }

  // unreachable
  Mesh<> m{};
  return m;
}

} // namespace femto
