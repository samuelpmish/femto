#include "femto/mesh.hpp"

#include "gtest/gtest.h"

#include <fstream>

using namespace femto;

TEST(conversions, quad_to_tri) {
  std::string filename = FEMTO_MESH_DIR"patch_test_quads.json"; 
  Mesh quad_mesh = Mesh<>::load(filename);
  Mesh tri_mesh = convert_to_simplices(quad_mesh);
}
