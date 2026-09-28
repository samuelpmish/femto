#include <gtest/gtest.h>

#include "femto/mesh.hpp"

#include "forall.hpp"

#include "fm/types/vec.hpp"
#include "fm/types/matrix.hpp"
#include "containers/ndarray_conversions.hpp"

using namespace femto;

TEST(nodes_per_geom, linear_H1) { 
  uint32_t gdim = 3;
  auto nodes_per = nodes_per_geom(FunctionSpace{Family::H1, 1}, gdim);
  EXPECT_EQ(nodes_per.vert, 1);
  EXPECT_EQ(nodes_per.edge, 2);
  EXPECT_EQ(nodes_per.tri, 3);
  EXPECT_EQ(nodes_per.quad, 4);
  EXPECT_EQ(nodes_per.tet, 4);
  EXPECT_EQ(nodes_per.hex, 8);
}

TEST(nodes_per_geom, quadratic_H1) { 
  uint32_t gdim = 3;
  auto nodes_per = nodes_per_geom(FunctionSpace{Family::H1, 2}, gdim);
  EXPECT_EQ(nodes_per.vert, 1);
  EXPECT_EQ(nodes_per.edge, 3);
  EXPECT_EQ(nodes_per.tri, 6);
  EXPECT_EQ(nodes_per.quad, 9);
  EXPECT_EQ(nodes_per.tet, 10);
  EXPECT_EQ(nodes_per.hex, 27);
}

TEST(nodes_per_geom, cubic_H1) { 
  uint32_t gdim = 3;
  auto nodes_per = nodes_per_geom(FunctionSpace{Family::H1, 3}, gdim);
  EXPECT_EQ(nodes_per.vert, 1);
  EXPECT_EQ(nodes_per.edge, 4);
  EXPECT_EQ(nodes_per.tri, 10);
  EXPECT_EQ(nodes_per.quad, 16);
  EXPECT_EQ(nodes_per.tet, 20);
  EXPECT_EQ(nodes_per.hex, 64);
}

TEST(interior_nodes_per_geom, linear_H1) { 
  uint32_t gdim = 3;
  auto interior_nodes_per = interior_nodes_per_geom(FunctionSpace{Family::H1, 1}, gdim);
  EXPECT_EQ(interior_nodes_per.vert, 1);
  EXPECT_EQ(interior_nodes_per.edge, 0);
  EXPECT_EQ(interior_nodes_per.tri, 0);
  EXPECT_EQ(interior_nodes_per.quad, 0);
  EXPECT_EQ(interior_nodes_per.tet, 0);
  EXPECT_EQ(interior_nodes_per.hex, 0);
}

TEST(interior_nodes_per_geom, quadratic_H1) { 
  uint32_t gdim = 3;
  auto interior_nodes_per = interior_nodes_per_geom(FunctionSpace{Family::H1, 2}, gdim);
  EXPECT_EQ(interior_nodes_per.vert, 1);
  EXPECT_EQ(interior_nodes_per.edge, 1);
  EXPECT_EQ(interior_nodes_per.tri, 0);
  EXPECT_EQ(interior_nodes_per.quad, 1);
  EXPECT_EQ(interior_nodes_per.tet, 0);
  EXPECT_EQ(interior_nodes_per.hex, 1);
}

TEST(interior_nodes_per_geom, cubic_H1) { 
  uint32_t gdim = 3;
  auto interior_nodes_per = interior_nodes_per_geom(FunctionSpace{Family::H1, 3}, gdim);
  EXPECT_EQ(interior_nodes_per.vert, 1);
  EXPECT_EQ(interior_nodes_per.edge, 2);
  EXPECT_EQ(interior_nodes_per.tri, 1);
  EXPECT_EQ(interior_nodes_per.quad, 4);
  EXPECT_EQ(interior_nodes_per.tet, 0);
  EXPECT_EQ(interior_nodes_per.hex, 8);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

TEST(nodes_per_geom, linear_Hcurl) { 
  uint32_t gdim = 3;
  auto nodes_per = nodes_per_geom(FunctionSpace{Family::Hcurl, 1}, gdim);
  EXPECT_EQ(nodes_per.vert, 0);
  EXPECT_EQ(nodes_per.edge, 1);
  EXPECT_EQ(nodes_per.tri, 3);
  EXPECT_EQ(nodes_per.quad, 4);
  EXPECT_EQ(nodes_per.tet, 6);
  EXPECT_EQ(nodes_per.hex, 12);
}

TEST(nodes_per_geom, quadratic_Hcurl) { 
  uint32_t gdim = 3;
  auto nodes_per = nodes_per_geom(FunctionSpace{Family::Hcurl, 2}, gdim);
  EXPECT_EQ(nodes_per.vert, 0);
  EXPECT_EQ(nodes_per.edge, 2);
  EXPECT_EQ(nodes_per.tri, 8);
  EXPECT_EQ(nodes_per.quad, 12);
  EXPECT_EQ(nodes_per.tet, 20);
  EXPECT_EQ(nodes_per.hex, 54);
}

TEST(nodes_per_geom, cubic_Hcurl) { 
  uint32_t gdim = 3;
  auto nodes_per = nodes_per_geom(FunctionSpace{Family::Hcurl, 3}, gdim);
  EXPECT_EQ(nodes_per.vert, 0);
  EXPECT_EQ(nodes_per.edge, 3);
  EXPECT_EQ(nodes_per.tri, 15);
  EXPECT_EQ(nodes_per.quad, 24);
  EXPECT_EQ(nodes_per.tet, 45);
  EXPECT_EQ(nodes_per.hex, 144);
}

TEST(interior_nodes_per_geom, linear_Hcurl) { 
  uint32_t gdim = 3;
  auto interior_nodes_per = interior_nodes_per_geom(FunctionSpace{Family::Hcurl, 1}, gdim);
  EXPECT_EQ(interior_nodes_per.vert, 0);
  EXPECT_EQ(interior_nodes_per.edge, 1);
  EXPECT_EQ(interior_nodes_per.tri, 0);
  EXPECT_EQ(interior_nodes_per.quad, 0);
  EXPECT_EQ(interior_nodes_per.tet, 0);
  EXPECT_EQ(interior_nodes_per.hex, 0);
}

TEST(interior_nodes_per_geom, quadratic_Hcurl) { 
  uint32_t gdim = 3;
  auto interior_nodes_per = interior_nodes_per_geom(FunctionSpace{Family::Hcurl, 2}, gdim);
  EXPECT_EQ(interior_nodes_per.vert, 0);
  EXPECT_EQ(interior_nodes_per.edge, 2);
  EXPECT_EQ(interior_nodes_per.tri, 2);
  EXPECT_EQ(interior_nodes_per.quad, 4);
  EXPECT_EQ(interior_nodes_per.tet, 0);
  EXPECT_EQ(interior_nodes_per.hex, 6);
}

TEST(interior_nodes_per_geom, cubic_Hcurl) { 
  uint32_t gdim = 3;
  auto interior_nodes_per = interior_nodes_per_geom(FunctionSpace{Family::Hcurl, 3}, gdim);
  EXPECT_EQ(interior_nodes_per.vert, 0);
  EXPECT_EQ(interior_nodes_per.edge, 3);
  EXPECT_EQ(interior_nodes_per.tri, 6);
  EXPECT_EQ(interior_nodes_per.quad, 12);
  EXPECT_EQ(interior_nodes_per.tet, 3);
  EXPECT_EQ(interior_nodes_per.hex, 36);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

TEST(nodes_per_geom, linear_DG) { 
  {
    uint32_t gdim = 1;
    auto nodes_per = nodes_per_geom(FunctionSpace{Family::DG, 1}, gdim);
    EXPECT_EQ(nodes_per.vert, 0);
    EXPECT_EQ(nodes_per.edge, 2);
    EXPECT_EQ(nodes_per.tri, 0);
    EXPECT_EQ(nodes_per.quad, 0);
    EXPECT_EQ(nodes_per.tet, 0);
    EXPECT_EQ(nodes_per.hex, 0);
  }

  {
    uint32_t gdim = 2;
    auto nodes_per = nodes_per_geom(FunctionSpace{Family::DG, 1}, gdim);
    EXPECT_EQ(nodes_per.vert, 0);
    EXPECT_EQ(nodes_per.edge, 0);
    EXPECT_EQ(nodes_per.tri, 3);
    EXPECT_EQ(nodes_per.quad, 4);
    EXPECT_EQ(nodes_per.tet, 0);
    EXPECT_EQ(nodes_per.hex, 0);
  }

  {
    uint32_t gdim = 3;
    auto nodes_per = nodes_per_geom(FunctionSpace{Family::DG, 1}, gdim);
    EXPECT_EQ(nodes_per.vert, 0);
    EXPECT_EQ(nodes_per.edge, 0);
    EXPECT_EQ(nodes_per.tri, 0);
    EXPECT_EQ(nodes_per.quad, 0);
    EXPECT_EQ(nodes_per.tet, 4);
    EXPECT_EQ(nodes_per.hex, 8);
  }
}

TEST(nodes_per_geom, quadratic_DG) { 
  {
    uint32_t gdim = 1;
    auto nodes_per = nodes_per_geom(FunctionSpace{Family::DG, 2}, gdim);
    EXPECT_EQ(nodes_per.vert, 0);
    EXPECT_EQ(nodes_per.edge, 3);
    EXPECT_EQ(nodes_per.tri, 0);
    EXPECT_EQ(nodes_per.quad, 0);
    EXPECT_EQ(nodes_per.tet, 0);
    EXPECT_EQ(nodes_per.hex, 0);
  }

  {
    uint32_t gdim = 2;
    auto nodes_per = nodes_per_geom(FunctionSpace{Family::DG, 2}, gdim);
    EXPECT_EQ(nodes_per.vert, 0);
    EXPECT_EQ(nodes_per.edge, 0);
    EXPECT_EQ(nodes_per.tri, 6);
    EXPECT_EQ(nodes_per.quad, 9);
    EXPECT_EQ(nodes_per.tet, 0);
    EXPECT_EQ(nodes_per.hex, 0);
  }

  {
    uint32_t gdim = 3;
    auto nodes_per = nodes_per_geom(FunctionSpace{Family::DG, 2}, gdim);
    EXPECT_EQ(nodes_per.vert, 0);
    EXPECT_EQ(nodes_per.edge, 0);
    EXPECT_EQ(nodes_per.tri, 0);
    EXPECT_EQ(nodes_per.quad, 0);
    EXPECT_EQ(nodes_per.tet, 10);
    EXPECT_EQ(nodes_per.hex, 27);
  }
}

TEST(nodes_per_geom, cubic_DG) { 
  {
    uint32_t gdim = 1;
    auto nodes_per = nodes_per_geom(FunctionSpace{Family::DG, 3}, gdim);
    EXPECT_EQ(nodes_per.vert, 0);
    EXPECT_EQ(nodes_per.edge, 4);
    EXPECT_EQ(nodes_per.tri, 0);
    EXPECT_EQ(nodes_per.quad, 0);
    EXPECT_EQ(nodes_per.tet, 0);
    EXPECT_EQ(nodes_per.hex, 0);
  }

  {
    uint32_t gdim = 2;
    auto nodes_per = nodes_per_geom(FunctionSpace{Family::DG, 3}, gdim);
    EXPECT_EQ(nodes_per.vert, 0);
    EXPECT_EQ(nodes_per.edge, 0);
    EXPECT_EQ(nodes_per.tri, 10);
    EXPECT_EQ(nodes_per.quad, 16);
    EXPECT_EQ(nodes_per.tet, 0);
    EXPECT_EQ(nodes_per.hex, 0);
  }

  {
    uint32_t gdim = 3;
    auto nodes_per = nodes_per_geom(FunctionSpace{Family::DG, 3}, gdim);
    EXPECT_EQ(nodes_per.vert, 0);
    EXPECT_EQ(nodes_per.edge, 0);
    EXPECT_EQ(nodes_per.tri, 0);
    EXPECT_EQ(nodes_per.quad, 0);
    EXPECT_EQ(nodes_per.tet, 20);
    EXPECT_EQ(nodes_per.hex, 64);
  }
}


TEST(interior_nodes_per_geom, linear_DG) { 
  {
    uint32_t gdim = 1;
    auto nodes_per = interior_nodes_per_geom(FunctionSpace{Family::DG, 1}, gdim);
    EXPECT_EQ(nodes_per.vert, 0);
    EXPECT_EQ(nodes_per.edge, 2);
    EXPECT_EQ(nodes_per.tri, 0);
    EXPECT_EQ(nodes_per.quad, 0);
    EXPECT_EQ(nodes_per.tet, 0);
    EXPECT_EQ(nodes_per.hex, 0);
  }

  {
    uint32_t gdim = 2;
    auto nodes_per = interior_nodes_per_geom(FunctionSpace{Family::DG, 1}, gdim);
    EXPECT_EQ(nodes_per.vert, 0);
    EXPECT_EQ(nodes_per.edge, 0);
    EXPECT_EQ(nodes_per.tri, 3);
    EXPECT_EQ(nodes_per.quad, 4);
    EXPECT_EQ(nodes_per.tet, 0);
    EXPECT_EQ(nodes_per.hex, 0);
  }

  {
    uint32_t gdim = 3;
    auto nodes_per = interior_nodes_per_geom(FunctionSpace{Family::DG, 1}, gdim);
    EXPECT_EQ(nodes_per.vert, 0);
    EXPECT_EQ(nodes_per.edge, 0);
    EXPECT_EQ(nodes_per.tri, 0);
    EXPECT_EQ(nodes_per.quad, 0);
    EXPECT_EQ(nodes_per.tet, 4);
    EXPECT_EQ(nodes_per.hex, 8);
  }
}

TEST(interior_nodes_per_geom, quadratic_DG) { 
  {
    uint32_t gdim = 1;
    auto nodes_per = interior_nodes_per_geom(FunctionSpace{Family::DG, 2}, gdim);
    EXPECT_EQ(nodes_per.vert, 0);
    EXPECT_EQ(nodes_per.edge, 3);
    EXPECT_EQ(nodes_per.tri, 0);
    EXPECT_EQ(nodes_per.quad, 0);
    EXPECT_EQ(nodes_per.tet, 0);
    EXPECT_EQ(nodes_per.hex, 0);
  }

  {
    uint32_t gdim = 2;
    auto nodes_per = interior_nodes_per_geom(FunctionSpace{Family::DG, 2}, gdim);
    EXPECT_EQ(nodes_per.vert, 0);
    EXPECT_EQ(nodes_per.edge, 0);
    EXPECT_EQ(nodes_per.tri, 6);
    EXPECT_EQ(nodes_per.quad, 9);
    EXPECT_EQ(nodes_per.tet, 0);
    EXPECT_EQ(nodes_per.hex, 0);
  }

  {
    uint32_t gdim = 3;
    auto nodes_per = interior_nodes_per_geom(FunctionSpace{Family::DG, 2}, gdim);
    EXPECT_EQ(nodes_per.vert, 0);
    EXPECT_EQ(nodes_per.edge, 0);
    EXPECT_EQ(nodes_per.tri, 0);
    EXPECT_EQ(nodes_per.quad, 0);
    EXPECT_EQ(nodes_per.tet, 10);
    EXPECT_EQ(nodes_per.hex, 27);
  }
}

TEST(interior_nodes_per_geom, cubic_DG) { 
  {
    uint32_t gdim = 1;
    auto nodes_per = interior_nodes_per_geom(FunctionSpace{Family::DG, 3}, gdim);
    EXPECT_EQ(nodes_per.vert, 0);
    EXPECT_EQ(nodes_per.edge, 4);
    EXPECT_EQ(nodes_per.tri, 0);
    EXPECT_EQ(nodes_per.quad, 0);
    EXPECT_EQ(nodes_per.tet, 0);
    EXPECT_EQ(nodes_per.hex, 0);
  }

  {
    uint32_t gdim = 2;
    auto nodes_per = interior_nodes_per_geom(FunctionSpace{Family::DG, 3}, gdim);
    EXPECT_EQ(nodes_per.vert, 0);
    EXPECT_EQ(nodes_per.edge, 0);
    EXPECT_EQ(nodes_per.tri, 10);
    EXPECT_EQ(nodes_per.quad, 16);
    EXPECT_EQ(nodes_per.tet, 0);
    EXPECT_EQ(nodes_per.hex, 0);
  }

  {
    uint32_t gdim = 3;
    auto nodes_per = interior_nodes_per_geom(FunctionSpace{Family::DG, 3}, gdim);
    EXPECT_EQ(nodes_per.vert, 0);
    EXPECT_EQ(nodes_per.edge, 0);
    EXPECT_EQ(nodes_per.tri, 0);
    EXPECT_EQ(nodes_per.quad, 0);
    EXPECT_EQ(nodes_per.tet, 20);
    EXPECT_EQ(nodes_per.hex, 64);
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

TEST(Field2D, node_counts) { 

  Mesh mesh = Mesh<>::load(FEMTO_MESH_DIR"patch_test_tris_and_quads.json");
  EXPECT_EQ(mesh.X.data.shape[0], 8);

  Field f1 = create_field<Family::H1>(mesh, 1);
  EXPECT_EQ(f1.data.shape[0], 8);

  Field f2 = create_field<Family::DG>(mesh, 1);
  EXPECT_EQ(f2.offsets.edge, 0);
  EXPECT_EQ(f2.offsets.tri, 0);
  EXPECT_EQ(f2.offsets.quad, 2 * 3);
  EXPECT_EQ(f2.offsets.tet, 2 * 3 + 4 * 4);
  EXPECT_EQ(f2.offsets.hex, 2 * 3 + 4 * 4);
  EXPECT_EQ(f2.data.shape[0], 2 * 3 + 4 * 4);

  Field f3 = create_field<Family::DG>(mesh, 2);
  EXPECT_EQ(f3.offsets.edge, 0);
  EXPECT_EQ(f3.offsets.tri, 0);
  EXPECT_EQ(f3.offsets.quad, 2 * 6);
  EXPECT_EQ(f3.offsets.tet, 2 * 6 + 4 * 9);
  EXPECT_EQ(f3.offsets.hex, 2 * 6 + 4 * 9);
  EXPECT_EQ(f3.data.shape[0], 4 * 9 + 2 * 6);

  Mesh mesh3D = Mesh<>::load(FEMTO_MESH_DIR"patch_test_tets_and_hexes.json");
  Field f4 = create_field<Family::DG>(mesh3D, 1);
  EXPECT_EQ(f4.offsets.edge, 0);
  EXPECT_EQ(f4.offsets.tri, 0);
  EXPECT_EQ(f4.offsets.quad, 0);
  EXPECT_EQ(f4.offsets.tet, 0);
  EXPECT_EQ(f4.offsets.hex, 12 * 4);
  EXPECT_EQ(f4.data.shape[0], 12 * 4 + 7 * 8);

  //Field f2 = create_field<Family::H1>(mesh, 2);
  //EXPECT_EQ(f2.data.shape[0], 8);

  //Field f3 = create_field<Family::H1>(mesh, 3);
  //EXPECT_EQ(f3.data.shape[0], 8);

}