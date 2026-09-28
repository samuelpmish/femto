#include "femto/mesh/io.hpp"

#include "util.hpp"

#include <fstream>

using io::Element;

// tessellation pattern for tri6 (1 tri6 -> 16 tri3)
//
// o
// * *
// *   *
// *     *
// * * * * *
// * *     * *
// *   *   *   *
// *     * *     *
// o * * * * * * * o
// * *     * *     * *
// *   *   *   *   *   *
// *     * *     * *     *
// * * * * * * * * * * * * *         
// * *     * *     * *     * *
// *   *   *   *   *   *   *   *
// *     * *     * *     * *     *
// o * * * * * * * o * * * * * * * o
//
//
// tessellation pattern for Quad8, Quad9 (1 Quad8/9 -> 32 tri3)
//
// o * * * * * * * o * * * * * * * o
// * *     * *     *     * *     * *
// *   *   *   *   *   *   *   *   *
// *     * *     * * *     * *     *
// * * * * * * * * * * * * * * * * *
// * *     * *     *     * *     * *
// *   *   *   *   *   *   *   *   *
// *     * *     * * *     * *     *
// o * * * * * * *(o)* * * * * * * o
// *     * *     * * *     * *     *
// *   *   *   *   *   *   *   *   *
// * *     * *     *     * *     * *
// * * * * * * * * * * * * * * * * *
// *     * *     * * *     * *     *
// *   *   *   *   *   *   *   *   *
// * *     * *     *     * *     * *
// o * * * * * * * o * * * * * * * o

int triangles_per_element(Element::Type type){
  switch (type) {
    case Element::Type::Line2:       return 0;
    case Element::Type::Line3:       return 0;
    case Element::Type::Tri3:        return 1;
    case Element::Type::Tri6:        return 16;
    case Element::Type::Quad4:       return 4;
    case Element::Type::Quad8:       return 32;
    case Element::Type::Quad9:       return 32;
    case Element::Type::Tet4:        return 0;
    case Element::Type::Tet10:       return 0;
    case Element::Type::Pyr5:        return 0;
    case Element::Type::Pyr13:       return 0;
    case Element::Type::Pyr14:       return 0;
    case Element::Type::Prism6:      return 0;
    case Element::Type::Prism15:     return 0;
    case Element::Type::Prism18:     return 0;
    case Element::Type::Hex8:        return 0;
    case Element::Type::Hex20:       return 0;
    case Element::Type::Hex27:       return 0;
    case Element::Type::Unsupported: return -1;
  }
  return -1;
}

using Triangle = std::array<vec3f, 3>;

std::vector< Triangle > tessellate_quad(const vec3f p[5][5]) {

  std::vector< Triangle > output;
  output.reserve(32);
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      vec3f q[4] = {p[i][j], p[i+1][j], p[i+1][j+1], p[i][j+1]};

      // j       
      // ^       
      // |       
      // 1 1 0 0 
      // 1 1 0 0 
      // 0 0 1 1
      // 0 0 1 1 --> i
      int offset = (i / 2) ^ (j / 2);

      // offset = 0   offset = 1
      //  3 * * 2      3 * * 2
      //  *   * *      * *   * 
      //  * *   *      *   * *
      //  0 * * 1      0 * * 1 
      output.push_back({q[0], q[1], q[2+offset]});
      output.push_back({q[2], q[3], q[0+offset]});
    }
  }
  return output;

}

std::vector< Triangle > tessellate(Element::Type type, const vec3f * p) {

  switch (type) {

    case Element::Type::Tri3: return {{p[0], p[1], p[2]}};

    case Element::Type::Tri6: {
      float xi[2];
      vec3f n[5][5];
      for (int j = 0; j < 5; j++) {
        xi[1] = j * 0.25f;
        for (int i = 0; i < 5; i++) {
          xi[0] = i * 0.25f;
          n[i][j] = p[0] * (1 - xi[0] - xi[1]) * (1 - 2 * xi[0] - 2 * xi[1]) +
                    p[1] * xi[0] * (2 * xi[0] - 1)                           +
                    p[2] * xi[1] * (2 * xi[1] - 1)                           +
                    p[3] * 4 * xi[0] * (1 - xi[0] - xi[1])                   +
                    p[4] * 4 * xi[0] * xi[1]                                 +
                    p[5] * 4 * xi[1] * (1 - xi[0] - xi[1]);
        }
      }

      std::vector< Triangle > output;
      output.reserve(16);
      for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 4 - j; i++) {
          vec3f q[4] = {n[i][j], n[i+1][j], n[i+1][j+1], n[i][j+1]};
          output.push_back({q[0], q[1], q[3]});
          if ((i + j) < 3) { output.push_back({q[1], q[2], q[3]}); }
        }
      }
      return output;
    }

    case Element::Type::Quad4: {
      vec3f c = (p[0] + p[1] + p[2] + p[3]) * 0.25f;
      return {{p[0], p[1], c}, {p[1], p[2], c}, {p[2], p[3], c}, {p[3], p[0], c}};
    }

    case Element::Type::Quad8: {

      float xi[2];
      vec3f n[5][5];
      for (int i = 0; i < 5; i++) {
        xi[0] = -1.0f + i * 0.5f;
        for (int j = 0; j < 5; j++) {
          xi[1] = -1.0f + j * 0.5f;
          n[i][j] = p[0] * (1 - xi[0]) * (1 - xi[1]) * (1 + xi[0] + xi[1]) * (-0.25f) +
                    p[1] * (1 + xi[0]) * (1 - xi[1]) * (1 - xi[0] + xi[1]) * (-0.25f) +
                    p[2] * (1 + xi[0]) * (1 + xi[1]) * (1 - xi[0] - xi[1]) * (-0.25f) +
                    p[3] * (1 - xi[0]) * (1 + xi[1]) * (1 + xi[0] - xi[1]) * (-0.25f) +
                    p[4] * (1 - xi[0] * xi[0]) * (1 - xi[1]) * (0.5f) +
                    p[5] * (1 + xi[0]) * (1 - xi[1] * xi[1]) * (0.5f) +
                    p[6] * (1 - xi[0] * xi[0]) * (1 + xi[1]) * (0.5f) +
                    p[7] * (1 - xi[0]) * (1 - xi[1] * xi[1]) * (0.5f);
        }
      }

      return tessellate_quad(n);

    }

    case Element::Type::Quad9: {

      constexpr float phi[5][3] = {{ 1.000, 0.000,  0.000}, 
                                   { 0.375, 0.750, -0.125}, 
                                   { 0.000, 1.000,  0.000}, 
                                   {-0.125, 0.750,  0.375}, 
                                   { 0.000, 0.000,  1.000}};

      vec3f n[5][5];
      for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
          n[i][j] = p[0] * phi[i][0] * phi[j][0] +
                    p[1] * phi[i][2] * phi[j][0] +
                    p[2] * phi[i][2] * phi[j][2] +
                    p[3] * phi[i][0] * phi[j][2] +
                    p[4] * phi[i][1] * phi[j][0] +
                    p[5] * phi[i][2] * phi[j][1] +
                    p[6] * phi[i][1] * phi[j][2] +
                    p[7] * phi[i][0] * phi[j][1] +
                    p[8] * phi[i][1] * phi[j][1];
        }
      }

      return tessellate_quad(n);

    } 

    default: return {};
  }
}

namespace io {

Mesh import_stl(std::string filename) {

  Mesh mesh;

  std::ifstream infile(filename);

  if (!infile) {
    std::cout << "error: " << filename << " not found" << std::endl;
    exit(1);
  }

  std::string line;

  /////////////////
  // read header //
  /////////////////
  char header[80];
  infile.read(header, 5);

  if (std::string(header, header+5) == "solid") {
    // ASCII format
    getline(infile, line); // skip the rest of the first line

    auto expect = [](const std::string & tok, const std::string & str) {
      if (tok != str) {
        std::cout << "expected '" << str << "', but got '" << tok << "'" << std::endl;
      }
    };

    double unused;
    int count = 0;
    fm::vec3 coords;

    std::string word;
    infile >> word;

    while(word != "endsolid") {
      expect(word, "facet"); 
      infile >> word; expect(word, "normal");
      infile >> unused >> unused >> unused; // discard normal values
        infile >> word; expect(word, "outer"); 
        infile >> word; expect(word, "loop");
          infile >> word; expect(word, "vertex");
          infile >> coords[0] >> coords[1] >> coords[2];
          mesh.nodes.push_back(coords);
          infile >> word; expect(word, "vertex");
          infile >> coords[0] >> coords[1] >> coords[2];
          mesh.nodes.push_back(coords);
          infile >> word; expect(word, "vertex");
          infile >> coords[0] >> coords[1] >> coords[2];
          mesh.nodes.push_back(coords);
          mesh.elements.push_back({Element::Type::Tri3, {count, count+1, count+2}, {}});
          count += 3;
        infile >> word; expect(word, "endloop");
      infile >> word; expect(word, "endfacet");
      infile >> word;
    }

  } else {
    // binary format
    infile.read(header+5, 75); // skip the rest of the 80-byte header

    int num_triangles;
    infile.read((char*)&num_triangles, sizeof(int));

    vec3f normal;
    vec3f v[3];
    uint16_t attributes;
    for (int i = 0; i < num_triangles; i++) {
      infile.read((char *)&normal, sizeof(vec3f)); // unused
      infile.read((char *)&v[0],   sizeof(vec3f));
      infile.read((char *)&v[1],   sizeof(vec3f));
      infile.read((char *)&v[2],   sizeof(vec3f));
      infile.read((char *)&attributes, 2); // unused

      mesh.nodes.push_back({v[0][0], v[0][1], v[0][2]});
      mesh.nodes.push_back({v[1][0], v[1][1], v[1][2]});
      mesh.nodes.push_back({v[2][0], v[2][1], v[2][2]});
      mesh.elements.push_back({Element::Type::Tri3, {3*i, 3*i+1, 3*i+2}, {}});
    }
  }

  return mesh;

}

bool export_stl(const Mesh & mesh, std::string filename) {

  std::ofstream outfile(filename, std::ios::binary);

  // header: 80 bytes
  uint8_t header[80];
  outfile.write((char*)&header[0], 80);

  // 4 bytes to encode number of triangles
  uint32_t num_triangles = 0;
  for (auto & e : mesh.elements) {
    num_triangles += triangles_per_element(e.type);
  }
  outfile.write((char*)&num_triangles, 4);

  // unused here, but part of the STL file specification
  uint16_t attributes; 

  constexpr int max_nodes = 27;
  vec3f nodes[max_nodes]; 

  for (auto & e : mesh.elements) {

    // load the nodes for this element
    for (int i = 0; i < nodes_per_elem(e.type); i++) {
      for (int j = 0; j < 3; j++) {
        nodes[i][j] = mesh.nodes[e.node_ids[i]][j];
      }
    }

    auto triangles = tessellate(e.type, nodes);

    for (auto & [p0, p1, p2] : triangles) {
      vec3f n = normalize(cross(p1 - p0, p2 - p0));

      outfile.write((char *)&n,  sizeof(vec3f));
      outfile.write((char *)&p0, sizeof(vec3f));
      outfile.write((char *)&p1, sizeof(vec3f));
      outfile.write((char *)&p2, sizeof(vec3f));
      outfile.write((char *)&attributes, 2);
    }

  }

  outfile.close();

  return false;

}

} // namespace io
