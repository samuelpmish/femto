#include "femto/mesh.hpp"

#include "fm/types/vec.hpp"

namespace femto {

namespace impl {

  static Mesh<> tet_to_hex(const Mesh<> & mesh) {

#if 0
    uint32_t num_output_vertices = mesh.vert.shape[0] + 
                                   mesh.edge.shape[0] + 
                                   mesh.tri.shape[0] + 
                                   mesh.tet.shape[0];

    vec3 * x_in = reinterpret_cast< vec3 * >(mesh.X.data.begin());

    std::vector < vec3 > x_out(num_output_vertices);

    int count = 0;
    for (int i = 0; i < mesh.vert.shape[0]; i++) {
      x_out[count++] = x_in[i];
    }

    for (int i = 0; i < mesh.edge.shape[0]; i++) {
      vec3 avg{};
      for (int j = 0; j < 2; j++) {
        avg += x_in[index(mesh.edge(i,j))] / 2.0;
      }
      x_out[count++] = avg;
    }

    for (int i = 0; i < mesh.tri.shape[0]; i++) {
      vec3 avg{};
      for (int j = 0; j < 3; j++) {
        avg += x_in[index(mesh.tri(i,j))] / 3.0;
      }
      x_out[count++] = avg;
    }

    for (int i = 0; i < mesh.tet.shape[0]; i++) {
      vec3 avg{};
      for (int j = 0; j < 4; j++) {
        avg += x_in[index(mesh.tet(i,j))] / 4.0;
      }
      x_out[count++] = avg;
    }

    uint32_t vert_offset = 0;
    uint32_t edge_offset = mesh.vert.shape[0];
    uint32_t tri_offset = edge_offset + mesh.edge.shape[0];
    uint32_t tet_offset = tri_offset + mesh.tri.shape[0];

    #define VERTEX_ID(j) index(mesh.tet(i, j))
    #define EDGE_ID(j) edge_offset + index(mesh.tet(i, j + 4))
    #define TRI_ID(j) tri_offset + index(mesh.tet(i, j + 10))
    #define TET_ID tet_offset + index(mesh.tet(i, 14))

    nd::cpu_array<uint32_t, 2> hexes({4 * mesh.tet.shape[0], 8}); 
    for (int i = 0; i < mesh.tet.shape[0]; i++) {

      // each tet is subdivided into 4 hexes
      hexes(4*i+0,0) = VERTEX_ID(0);
      hexes(4*i+0,1) = EDGE_ID(0);
      hexes(4*i+0,2) = TRI_ID(0);
      hexes(4*i+0,3) = EDGE_ID(2);
      hexes(4*i+0,4) = EDGE_ID(3);
      hexes(4*i+0,5) = TRI_ID(1);
      hexes(4*i+0,6) = TET_ID;
      hexes(4*i+0,7) = TRI_ID(3);

      hexes(4*i+1,0) = EDGE_ID(0);
      hexes(4*i+1,1) = VERTEX_ID(1);
      hexes(4*i+1,2) = EDGE_ID(1);
      hexes(4*i+1,3) = TRI_ID(0);
      hexes(4*i+1,4) = TRI_ID(1);
      hexes(4*i+1,5) = EDGE_ID(4);
      hexes(4*i+1,6) = TRI_ID(2);
      hexes(4*i+1,7) = TET_ID;

      hexes(4*i+2,0) = TRI_ID(0);
      hexes(4*i+2,1) = EDGE_ID(1);
      hexes(4*i+2,2) = VERTEX_ID(2);
      hexes(4*i+2,3) = EDGE_ID(2);
      hexes(4*i+2,4) = TET_ID;
      hexes(4*i+2,5) = TRI_ID(2);
      hexes(4*i+2,6) = EDGE_ID(5);
      hexes(4*i+2,7) = TRI_ID(3);

      hexes(4*i+3,0) = TRI_ID(1);
      hexes(4*i+3,1) = EDGE_ID(4);
      hexes(4*i+3,2) = TRI_ID(2);
      hexes(4*i+3,3) = TET_ID;
      hexes(4*i+3,4) = EDGE_ID(3);
      hexes(4*i+3,5) = VERTEX_ID(3);
      hexes(4*i+3,6) = EDGE_ID(5);
      hexes(4*i+3,7) = TRI_ID(3);

    }

    return Mesh<>::create_3D(nodes, );
#endif
    return mesh;
  }

  static Mesh<> hex_to_tet(const Mesh<> & mesh) {

    return mesh;
  }

  template < int dim >
  static Mesh<> tri_to_quad(const Mesh<> & mesh) {

    return mesh;
  }

  template < int dim >
  static Mesh<> quad_to_tri(const Mesh<> & mesh) {

    return mesh;
  }


}

Mesh<> convert_to_simplices(const Mesh<> & mesh) {
  if (mesh.geometry_dimension == 2) {
    if (mesh.spatial_dimension == 2) { return impl::quad_to_tri<2>(mesh); }
    if (mesh.spatial_dimension == 3) { return impl::quad_to_tri<3>(mesh); }
  } 

  if (mesh.geometry_dimension == 3) { return impl::hex_to_tet(mesh); }

  return mesh;
}

Mesh<> convert_to_tensor_product(const Mesh<> & mesh) {
  if (mesh.geometry_dimension == 2) {
    if (mesh.spatial_dimension == 2) { return impl::tri_to_quad<2>(mesh); }
    if (mesh.spatial_dimension == 3) { return impl::tri_to_quad<3>(mesh); }
  } 

  if (mesh.geometry_dimension == 3) { return impl::tet_to_hex(mesh); }

  return mesh;
}

} // namespace femto