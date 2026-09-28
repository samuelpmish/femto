#include <gtest/gtest.h>

#include "femto/mesh.hpp"
#include "misc/timer.hpp"

#include <iostream>

using namespace femto;

TEST(manifold, tri_mesh) {
  EXPECT_TRUE(is_manifold(Mesh<>::load(FEMTO_MESH_DIR"patch_test_tris.json")));
}

TEST(manifold, quad_mesh) {
  EXPECT_TRUE(is_manifold(Mesh<>::load(FEMTO_MESH_DIR"patch_test_quads.json")));
}

TEST(manifold, tri_and_quad_mesh) {
  EXPECT_TRUE(is_manifold(Mesh<>::load(FEMTO_MESH_DIR"patch_test_tris_and_quads.json")));
}

TEST(manifold, tet_mesh) {
  EXPECT_TRUE(is_manifold(Mesh<>::load(FEMTO_MESH_DIR"patch_test_tets.json")));
}

TEST(manifold, hex_mesh) {
  EXPECT_TRUE(is_manifold(Mesh<>::load(FEMTO_MESH_DIR"patch_test_hexes.json")));
}

TEST(manifold, tet_and_hex_mesh) {
  EXPECT_TRUE(is_manifold(Mesh<>::load(FEMTO_MESH_DIR"patch_test_tets_and_hexes.json")));
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

TEST(nonmanifold, tri_mesh) {

  /*
    0       4
    | \   / | 
    |   2   |
    | /   \ |
    1       3
  */
  double coordinates[5][2] = {{-1, +1}, {-1, -1}, {0, 0}, {1, -1}, {1, 1}};
  nd::array< double, 2, memory::space::cpu > nodes({5, 2});
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 2; j++) {
      nodes(i, j) = coordinates[i][j];
    }
  }

  uint32_t elements[2][3] = {{0, 1, 2}, {2, 3, 4}};
  nd::array< uint32_t, 2, memory::space::cpu > tris({2, 3});
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      tris(i, j) = elements[i][j];
    }
  }

  nd::array< uint32_t, 2, memory::space::cpu > quads{};

  Mesh mesh = Mesh<>::create_2D(nodes, 1, tris, quads);

  EXPECT_FALSE(is_manifold(mesh));

}

TEST(nonmanifold, quad_mesh) {

  /*
        5---6
        |   |
    2---3---4
    |   |
    0---1
  */
  double coordinates[7][2] = {{-1, -1}, {0, -1}, {-1, 0}, {0, 0}, {1, 0}, {0, 1}, {1, 1}};
  nd::array< double, 2, memory::space::cpu > nodes({7, 2});
  for (int i = 0; i < 7; i++) {
    for (int j = 0; j < 2; j++) {
      nodes(i, j) = coordinates[i][j];
    }
  }

  uint32_t elements[2][4] = {{0, 1, 3, 2}, {3, 4, 6, 5}};
  nd::array< uint32_t, 2, memory::space::cpu > quads({2, 4});
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 4; j++) {
      quads(i, j) = elements[i][j];
    }
  }

  nd::array< uint32_t, 2, memory::space::cpu > tris{};

  Mesh mesh = Mesh<>::create_2D(nodes, 1, tris, quads);

  EXPECT_FALSE(is_manifold(mesh));

}

TEST(nonmanifold_edge, tet_mesh) {

  double coordinates[6][3] = {
    {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 0}, {1, 1, 1}
  };
  nd::array< double, 2, memory::space::cpu > nodes({6, 3});
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < 3; j++) {
      nodes(i, j) = coordinates[i][j];
    }
  }

  uint32_t elements[2][4] = {{0, 1, 2, 3}, {4, 2, 1, 5}};
  nd::array< uint32_t, 2, memory::space::cpu > tets({2, 4});
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 4; j++) {
      tets(i, j) = elements[i][j];
    }
  }

  nd::array< uint32_t, 2, memory::space::cpu > hexes{};

  Mesh mesh = Mesh<>::create_3D(nodes, 1, tets, hexes);

  EXPECT_FALSE(is_manifold(mesh));

}

TEST(nonmanifold_vertex, tet_mesh) {

  /*
    4 --------- 6
     \ '.   .' /
       \  5  /
         \|/
          3
        / | \
      /   |   \
    0     |     2
      '*. | .*'
          1
   */
  double coordinates[7][3] = {
    {-1, -1, -1}, {+1, -1, -1}, {0, 1, -1}, 
                  { 0,  0,  0}, 
    {-1, -1, +1}, {+1, -1, +1}, {0, 1, +1}
  };
  nd::array< double, 2, memory::space::cpu > nodes({7, 3});
  for (int i = 0; i < 7; i++) {
    for (int j = 0; j < 3; j++) {
      nodes(i, j) = coordinates[i][j];
    }
  }

  uint32_t elements[2][4] = {{0, 1, 2, 3}, {4, 6, 5, 3}};
  nd::array< uint32_t, 2, memory::space::cpu > tets({2, 4});
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 4; j++) {
      tets(i, j) = elements[i][j];
    }
  }

  nd::array< uint32_t, 2, memory::space::cpu > hexes{};

  Mesh mesh = Mesh<>::create_3D(nodes, 1, tets, hexes);

  EXPECT_FALSE(is_manifold(mesh));

}
