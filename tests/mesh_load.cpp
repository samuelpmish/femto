#include <gtest/gtest.h>

#include "femto/mesh.hpp"
#include "misc/timer.hpp"

#include <iostream>

using namespace femto;

TEST(mesh_load, gmsh_tets) {
  print_timings = true;
  Mesh mesh = Mesh<>::load(FEMTO_MESH_DIR"icosahedron.msh");
  std::cout << mesh.tet.shape[0] << std::endl; 
}

TEST(mesh_load, json_tris) {
  print_timings = true;
  Mesh mesh = Mesh<>::load(FEMTO_MESH_DIR"patch_test_tris.json");
  std::cout << mesh.tri.shape[0] << std::endl; 
}
