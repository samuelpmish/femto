#include <gtest/gtest.h>

#include "forall.hpp"
#include "femto/mesh.hpp"
#include "fm/types/AABB.hpp"

#include <iostream>

using namespace femto;

TEST(mesh_merge, quads) {
  Mesh mesh1 = Mesh<>::cuboid({2, 2}, vec2{1.0, 1.0});
  Mesh mesh2 = Mesh<>::cuboid({2, 2}, vec2{1.0, 1.0});

  mesh2.X.data = forall(+[](const vec2 & X) {
    return X + vec2{1.0, 0.0};
  }, mesh2.X.data);

  Mesh merged = merge({mesh1, mesh2}, 0.0001);

  EXPECT_EQ(merged.vert.shape[0], 15);
  EXPECT_EQ(merged.edge.shape[0], 22);
  EXPECT_EQ(merged.tri.shape[0], 0);
  EXPECT_EQ(merged.quad.shape[0], 8);

  //save(merged, "merged_quads.msh");
}

TEST(mesh_merge, hexes) {
  Mesh mesh1 = Mesh<>::cuboid({2, 2, 2}, vec3{1.0, 1.0, 1.0});
  Mesh mesh2 = Mesh<>::cuboid({2, 2, 2}, vec3{1.0, 1.0, 1.0});

  mesh2.X.data = forall(+[](const vec3 & X) {
    return X + vec3{1.0, 0.0, 0.0};
  }, mesh2.X.data);

  Mesh merged = merge({mesh1, mesh2}, 0.0001);

  EXPECT_EQ(merged.vert.shape[0], 45);
  EXPECT_EQ(merged.tri.shape[0], 0);
  EXPECT_EQ(merged.tet.shape[0], 0);
  EXPECT_EQ(merged.hex.shape[0], 16);

  //save(merged, "merged_hexes.msh");
}

TEST(mesh_merge, disjoint_hexes) {
  Mesh mesh1 = Mesh<>::cuboid({2, 2, 2}, vec3{1.0, 1.0, 1.0});
  Mesh mesh2 = Mesh<>::cuboid({2, 2, 2}, vec3{1.0, 1.0, 1.0});

  mesh2.X.data = forall(+[](const vec3 & X) {
    return X + vec3{2.0, 0.0, 0.0};
  }, mesh2.X.data);

  Mesh merged = merge({mesh1, mesh2}, 0.0001);

  EXPECT_EQ(merged.vert.shape[0], 54);
  EXPECT_EQ(merged.tri.shape[0], 0);
  EXPECT_EQ(merged.tet.shape[0], 0);
  EXPECT_EQ(merged.hex.shape[0], 16);

  //save(merged, "merged_hexes.msh");
}

femto::Mesh<> create_diagonal_member(fm::AABB<3> box, float thickness, stack::array<int32_t, 3> divisions) {
  vec3 dimensions = {box.max[0] - box.min[0], thickness, box.max[2] - box.min[2]};
  stack::array<uint32_t, 3> uint_divisions;
  uint_divisions[0] = divisions[0];
  uint_divisions[1] = divisions[1];
  uint_divisions[2] = divisions[2];

  femto::Mesh<> box_mesh = femto::Mesh<>::cuboid(uint_divisions, dimensions);

  float dy = box.max[1] - box.min[1];
  if (dy > 0) {
    float dydx = (dy - thickness) / dimensions[0];
    box_mesh.X.data = femto::forall(std::function<vec3(vec3)>([&](vec3 x){
      return x + box.min + vec3{0, dydx * x[0], 0.0};
    }), box_mesh.X.data);
  } else {
    float dydx = (dy + thickness) / dimensions[0];
    box_mesh.X.data = femto::forall(std::function<vec3(vec3)>([&](vec3 x){
      return x + box.min + vec3{0, dydx * x[0] - thickness, 0.0};
    }), box_mesh.X.data);
  }

  return box_mesh;
}

femto::Mesh<> create_mesh(AABB<3> box, stack::array<int32_t, 3> divisions) {
  vec3 dimensions = {box.max[0] - box.min[0], box.max[1] - box.min[1], box.max[2] - box.min[2]};
  stack::array<uint32_t, 3> uint_divisions;
  uint_divisions[0] = divisions[0];
  uint_divisions[1] = divisions[1];
  uint_divisions[2] = divisions[2];

  femto::Mesh<> box_mesh = femto::Mesh<>::cuboid(uint_divisions, dimensions);

  box_mesh.X.data = femto::forall(std::function<vec3(vec3)>([&](vec3 x){
    return x + box.min;
  }), box_mesh.X.data);

  return box_mesh;
}

TEST(mesh_merge, from_logic_gate) {

  std::array<float, 10> x = {0.0, 1.0, 1.6, 3.12569992161755, 3.6, 4.35, 5.05710678118654, 6.84710678118654, 7.55421356237309, 8.80421356237309};
  std::array<float, 10> y = {0.0, 0.4, 1.85753787975413, 2.6353553390593, 3.28, 3.92, 4.56464, 5.34246, 6.8, 7.2};
  std::array<float, 4> z = {0.0, 0.625, 1.25, 1.875};
  std::array<float, 2> t = {0.1, 0.070710678118655};
  int dx78 = 1;
  int dx89 = 1;
  int dt1 = 1;
  int dz3 = 3;

  std::vector< femto::Mesh<> > parts;
  parts.push_back(create_diagonal_member(AABB<3>{{x[7], y[2], z[2]}, {x[8], y[3], z[3]}}, t[1], {dx78, dt1, dz3/3}));
  parts.push_back(create_diagonal_member(AABB<3>{{x[7], y[3], z[0]}, {x[8], y[2], z[1]}}, t[1], {dx78, dt1, dz3/3}));
  parts.push_back(create_mesh(AABB<3>{{x[8], y[2], z[0]}, {x[9], y[2]+t[1], z[3]}}, {dx89, dt1, dz3}));
  parts.push_back(create_mesh(AABB<3>{{x[8], y[3]-t[1], z[0]}, {x[9], y[3], z[3]}}, {dx89, dt1, dz3}));
  femto::Mesh<> merged_mesh = merge(parts, 0.0001);

  std::cout << "mesh has: " << std::endl;
  std::cout << "  " << merged_mesh.vert.shape[0] << " vertices" << std::endl;
  std::cout << "  " << merged_mesh.edge.shape[0] << " edges" << std::endl;
  std::cout << "  " << merged_mesh.quad.shape[0] << " quads" << std::endl;
  std::cout << "  " << merged_mesh.hex.shape[0] << " hexes" << std::endl;

  EXPECT_TRUE(is_manifold(merged_mesh));

}
