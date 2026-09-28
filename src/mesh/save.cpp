#include "femto/mesh.hpp"

#include <fstream>
#include <iostream>
#include <filesystem>

#include "femto/mesh/io.hpp"

namespace femto {

inline std::vector<int> identity_permutation(int n) {
  std::vector<int> P(n);
  for (int i = 0; i < n; i++) { P[i] = i; }
  return P;
}

// io::Mesh expects gmsh node ordering, so these are permutations to map from femto -> gmsh
std::vector<int> permutation(io::Element::Type type) {
  switch (type) {

    // many of the elements share the same node ordering
    case io::Element::Type::Unsupported:
    case io::Element::Type::Line2:
    case io::Element::Type::Line3:
    case io::Element::Type::Tri3:
    case io::Element::Type::Tri6:
    case io::Element::Type::Tet4: {
      return identity_permutation(nodes_per_elem(type));
    }

    // but some of the elements assign numbers
    // to edge/face nodes in a different order
    // see below for the differences and corresponding permutations

    ///////////////////////////////////////////////////////////////////////////////
    //                                                                           //
    //         femto:                                 gmsh:                      //
    //                                                                           //
    //            v                                     v                        //
    //            ^                                     ^                        //
    //            |                                     |                        //
    //      2-----------3                         3-----------2                  //
    //      |     |     |                         |     |     |                  //
    //      |     |     |                         |     |     |                  //
    //      |     +---- | --> u                   |     +---- | -->              //
    //      |           |                         |           |                  //
    //      |           |                         |           |                  //
    //      0-----------1                         0-----------1                  //
    //                                                                           //
    ///////////////////////////////////////////////////////////////////////////////
    case io::Element::Type::Quad4:
      return {0, 1, 3, 2};

    ///////////////////////////////////////////////////////////////////////////////
    //                                                                           //
    //         femto:                                 gmsh:                      //
    //                                                                           //
    //            v                                     v                        //
    //            ^                                     ^                        //
    //            |                                     |                        //
    //      6-----7-----8                         3-----6-----2                  //
    //      |     |     |                         |           |                  //
    //      |     |     |                         |           |                  //
    //      3     4---- 5 --> u                   7     8     5 --> u            //
    //      |           |                         |           |                  //
    //      |           |                         |           |                  //
    //      0-----1-----2                         0-----4-----1                  //
    //                                                                           //
    ///////////////////////////////////////////////////////////////////////////////
    case io::Element::Type::Quad9:
      return {0, 4, 1, 7, 8, 5, 3, 6, 2};
 
    ///////////////////////////////////////////////////////////////////////////////
    //                                                                           //
    // Quad4:                 Quad8:                  Quad9:                     //
    //                                                                           //
    //       v                                                                   //
    //       ^                                                                   //
    //       |                                                                   //
    // 3-----------2          3-----6-----2           3-----6-----2              //
    // |     |     |          |           |           |           |              //
    // |     |     |          |           |           |           |              //
    // |     +---- | --> u    7           5           7     8     5              //
    // |           |          |           |           |           |              //
    // |           |          |           |           |           |              //
    // 0-----------1          0-----4-----1           0-----4-----1              //
    //                                                                           //
    ///////////////////////////////////////////////////////////////////////////////




    ///////////////////////////////////////////////////////////////////////////////
    //                                                                           //
    //          femto:                           gmsh:                           //
    //                                                                           //
    //              2                              2                             //
    //            ,/|`\                          ,/|`\                           //
    //          ,/  |  `\                      ,/  |  `\                         //
    //        ,6    '.   `5                  ,6    '.   `5                       //
    //      ,/       9     `\              ,/       8     `\                     //
    //    ,/         |       `\          ,/         |       `\                   //
    //   0--------4--'.--------1        0--------4--'.--------1                  //
    //    `\.         |      ,/          `\.         |      ,/                   //
    //       `\.      |    ,8               `\.      |    ,9                     //
    //          `7.   '. ,/                    `7.   '. ,/                       //
    //             `\. |/                         `\. |/                         //
    //                `3                             `3                          //
    //                                                                           //
    ///////////////////////////////////////////////////////////////////////////////
    case io::Element::Type::Tet10:
      return {0, 1, 2, 3, 4, 5, 6, 7, 9, 8};

    ///////////////////////////////////////////////////////////////////////////////
    //                                                                           //
    //         femto:                                   gmsh:                    //
    //                                                                           //
    //                                                                           //
    //      2----------3                            3----------2                 //
    //      |\         |\                           |\         |\                //
    //      | \        | \                          | \        | \               //
    //      |  \       |  \                         |  \       |  \              //
    //      |   6------+---7                        |   7------+---6             //
    //      |   |      |   |                        |   |      |   |             //
    //      0---+------1   |                        0---+------1   |             //
    //       \  |       \  |                         \  |       \  |             //
    //        \ |        \ |                          \ |        \ |             //
    //         \|         \|                           \|         \|             //
    //          4----------5                            4----------5             //
    //                                                                           //
    ///////////////////////////////////////////////////////////////////////////////
    case io::Element::Type::Hex8:
      return {0, 1, 3, 2, 4, 5, 7, 6};

    ///////////////////////////////////////////////////////////////////////////////
    //                                                                           //
    //         femto:                                   gmsh:                    //
    //                                                                           //
    //                                                                           //
    //      6-----7----8                          3----13----2                   //
    //      |\         |\                         |\         |\                  //
    //      | 15   16  | 17                       | 15   24  | 14                //
    //      3  \  4    5  \                       9  \ 20    11 \                //
    //      |  24----25+---26                     |   7----19+---6               //
    //      |12 |  13  | 14|                      |22 |  26  | 23|               //
    //      0---+-1----2   |                      0---+-8----1   |               //
    //       \ 21    22 \  23                      \ 17    25 \  18              //
    //        9 |  10    11|                       10 |  21    12|               //
    //         \|         \|                         \|         \|               //
    //         18----18----20                         4----16----5               //
    //                                                                           //
    ///////////////////////////////////////////////////////////////////////////////
    case io::Element::Type::Hex27:
      return {0, 8, 1, 9, 20, 11, 3, 13, 2, 10, 21, 12, 22, 26, 23, 15, 24, 14, 4, 16, 5, 17, 25, 18, 7, 19, 6};

    default:
      return {};

  }

}

io::Element::Type io_element_type(Geometry g, uint32_t degree) {
  if (g == Geometry::Triangle && degree == 1)      return io::Element::Type::Tri3;
  if (g == Geometry::Triangle && degree == 2)      return io::Element::Type::Tri6;
  if (g == Geometry::Quadrilateral && degree == 1) return io::Element::Type::Quad4;
  if (g == Geometry::Quadrilateral && degree == 2) return io::Element::Type::Quad9;
  if (g == Geometry::Tetrahedron && degree == 1)   return io::Element::Type::Tet4;
  if (g == Geometry::Tetrahedron && degree == 2)   return io::Element::Type::Tet10;
  if (g == Geometry::Hexahedron && degree == 1)    return io::Element::Type::Hex8;
  if (g == Geometry::Hexahedron && degree == 2)    return io::Element::Type::Hex27;
  return io::Element::Type::Unsupported;
}

// returns file extension in lower case characters
static std::string file_extension(std::string filename) {
  std::string ext = std::filesystem::path(filename).extension().string();
  for(auto& c : ext) { c = tolower(c); }
  return ext;
}

io::Mesh to_iomesh(const Mesh<> & mesh) {

  io::Mesh output;

  uint32_t sdim = mesh.spatial_dimension;
  uint32_t num_nodes = mesh.X.data.shape[0];
  output.nodes.resize(num_nodes);
  for (uint32_t i = 0; i < num_nodes; i++) {
    fm::vec3 node;
    node[0] = mesh.X.data(i, 0);
    node[1] = mesh.X.data(i, 1);
    if (sdim == 3) {
      node[2] = mesh.X.data(i, 2);
    }
    output.nodes[i] = node;
  }

  uint32_t degree = mesh.X.degree;
  FEMTO_ASSERT(degree <= 2, "can only export meshes with linear/quadratic elements");

  GeometryInfo offsets = mesh.X.offsets;
  std::vector< uint32_t > node_ids;

  uint32_t gdim = mesh.geometry_dimension;
  FEMTO_ASSERT(gdim == 2 || gdim == 3, "can only export 2D / 3D meshes");

  if (gdim == 2) {
    uint32_t num_tris = mesh.tri.shape[0];
    uint32_t num_quads = mesh.quad.shape[0];
    output.elements.resize(num_tris + num_quads);

    {
      FiniteElement<Geometry::Triangle, Family::H1> el{degree};
      uint32_t nodes_per_tri = el.num_nodes();
      node_ids.resize(nodes_per_tri);
      auto type = io_element_type(Geometry::Triangle, degree);
      std::vector<int> p = permutation(type);
      for (uint32_t i = 0; i < num_tris; i++) {
        el.indices(offsets, &mesh.tri(i, 0), node_ids.data());
  
        output.elements[i].type = type;
        output.elements[i].node_ids.resize(nodes_per_tri);
        for (int j = 0; j < nodes_per_tri; j++) {
          output.elements[i].node_ids[p[j]] = node_ids[j];
        }
      }
    }

    {
      FiniteElement<Geometry::Quadrilateral, Family::H1> el{degree};
      uint32_t nodes_per_quad = el.num_nodes();
      node_ids.resize(nodes_per_quad);
      auto type = io_element_type(Geometry::Quadrilateral, degree);
      std::vector<int> p = permutation(type);
      for (uint32_t i = 0; i < num_quads; i++) {
        el.indices(offsets, &mesh.quad(i, 0), node_ids.data());
  
        output.elements[i].type = type;
        output.elements[i].node_ids.resize(nodes_per_quad);
        for (int j = 0; j < nodes_per_quad; j++) {
          output.elements[i].node_ids[p[j]] = node_ids[j];
        }
      }
    }
  }

  if (gdim == 3) {
    uint32_t num_tets = mesh.tet.shape[0];
    uint32_t num_hexes = mesh.hex.shape[0];
    output.elements.resize(num_tets + num_hexes);

    {
      FiniteElement<Geometry::Tetrahedron, Family::H1> el{degree};
      uint32_t nodes_per_tet = el.num_nodes();
      node_ids.resize(nodes_per_tet);
      auto type = io_element_type(Geometry::Tetrahedron, degree);
      std::vector<int> p = permutation(type);
      for (uint32_t i = 0; i < num_tets; i++) {
        el.indices(offsets, &mesh.tet(i, 0), node_ids.data());
  
        output.elements[i].type = type;
        output.elements[i].node_ids.resize(nodes_per_tet);
        for (int j = 0; j < nodes_per_tet; j++) {
          output.elements[i].node_ids[p[j]] = node_ids[j];
        }
      }
    }

    {
      FiniteElement<Geometry::Hexahedron, Family::H1> el{degree};
      uint32_t nodes_per_quad = el.num_nodes();
      node_ids.resize(nodes_per_quad);
      auto type = io_element_type(Geometry::Hexahedron, degree);
      std::vector<int> p = permutation(type);
      for (uint32_t i = 0; i < num_hexes; i++) {
        el.indices(offsets, &mesh.hex(i, 0), node_ids.data());
  
        output.elements[i].type = type;
        output.elements[i].node_ids.resize(nodes_per_quad);
        for (int j = 0; j < nodes_per_quad; j++) {
          output.elements[i].node_ids[p[j]] = node_ids[j];
        }
      }
    }
  }

  return output;
}

void save(const Mesh<> & mesh, std::string filename) {

  io::Mesh iomsh = to_iomesh(mesh);
  std::string ext = file_extension(filename);
  if (ext == ".msh") {
    export_gmsh_v22(iomsh, filename, io::FileEncoding::Binary);
  } else if (ext == ".vtk") {
    export_vtk(iomsh, filename, io::FileEncoding::Binary);
  } else if (ext == ".vtu") {
    export_vtu(iomsh, filename);
  }
}

} // namespace femto