#include "femto/mesh.hpp"
#include "fm/types/vec.hpp"

#include "misc/json.hpp"

#include <fstream>

using namespace femto;

#if 0
/*
  for converting a mesh of 4-node tets into a mesh of 8-node hexes
*/

namespace femto {
  void to_json(nlohmann::json& j, const vec3 & v) {
      j[0] = v[0]; j[1] = v[1]; j[2] = v[2];
  }
}

void to_json(nlohmann::json& j, const nd::cpu_array<uint32_t, 2> & v) {
  for (int i = 0; i < v.shape[0]; i++) {
    for (int k = 0; k < v.shape[1]; k++) {
      j[i][k] = v(i,k);
    }
  }
}

int main(int argc, char *argv[]) {

  if (argc > 2) {
    Mesh mesh = Mesh<>::load(std::string(argv[1]));

    int num_output_vertices = mesh.vert.shape[0] + 
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
        avg += x_in[mesh.edge(i,j).index] / 2.0;
      }
      x_out[count++] = avg;
    }

    for (int i = 0; i < mesh.tri.shape[0]; i++) {
      vec3 avg{};
      for (int j = 0; j < 3; j++) {
        avg += x_in[mesh.tri(i,j).index] / 3.0;
      }
      x_out[count++] = avg;
    }

    for (int i = 0; i < mesh.tet.shape[0]; i++) {
      vec3 avg{};
      for (int j = 0; j < 4; j++) {
        avg += x_in[mesh.tet(i,j).index] / 4.0;
      }
      x_out[count++] = avg;
    }

    uint32_t vert_offset = 0;
    uint32_t edge_offset = mesh.vert.shape[0];
    uint32_t tri_offset = edge_offset + mesh.edge.shape[0];
    uint32_t tet_offset = tri_offset + mesh.tri.shape[0];

    #define VERTEX_ID(j) mesh.tet(i, j).index
    #define EDGE_ID(j) edge_offset + mesh.tet(i, j + 4).index
    #define TRI_ID(j) tri_offset + mesh.tet(i, j + 10).index
    #define TET_ID tet_offset + mesh.tet(i, 14).index

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

    nlohmann::json output = {
      {"nodes", {{"degree", 1}, {"data", x_out}}},
      {"hexes", hexes}
    };

    std::ofstream outfile{std::string(argv[2])};
    outfile << output.dump();
    outfile.close();

  } else {
    std::cout << "please specify input and outfile file names" << std::endl;
  }
}
#endif
