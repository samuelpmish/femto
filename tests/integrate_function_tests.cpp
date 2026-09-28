#include <gtest/gtest.h>

#include <iomanip>
#include <iostream>
#include <functional>

#include "femto/domain.hpp"

#include <gtest/gtest.h>

#include "femto/mesh.hpp"

#include "forall.hpp"

#include "fm/types/vec.hpp"
#include "fm/types/matrix.hpp"
#include "containers/ndarray_conversions.hpp"

using namespace femto;

TEST(IntegrateTest, ScalarFunction2D) {
  Mesh m = Mesh<>::load(FEMTO_MESH_DIR"patch_test_tris_and_quads.json");
  Domain domain(m, MeshQuadratureRule{3});
  double measure = integrate(+[](vec2 X){ return 1.0; }, domain);
  EXPECT_NEAR(measure, 1.0, 1.0e-14);

  measure = integrate(std::function<double(vec2)>([](vec2 X){ return 1.0; }), domain);
  EXPECT_NEAR(measure, 1.0, 1.0e-14);
}

TEST(IntegrateTest, ScalarFunction3D) {
  Mesh m = Mesh<>::load(FEMTO_MESH_DIR"patch_test_tets_and_hexes.json");
  Domain domain(m, MeshQuadratureRule{3});
  double measure = integrate(+[](vec3 X){ return 1.0; }, domain);
  EXPECT_NEAR(measure, 2.0, 1.0e-14);

  measure = integrate(std::function<double(vec3)>([](vec3 X){ return 1.0; }), domain);
  EXPECT_NEAR(measure, 2.0, 1.0e-14);
}

TEST(IntegrateTest, VectorFunction2D) {
  Mesh m = Mesh<>::load(FEMTO_MESH_DIR"patch_test_tris_and_quads.json");
  Domain domain(m, MeshQuadratureRule{3});
/*
  Integrate[{x, y}, {x, y} \[Element] Rectangle[{0, 0}, {1, 1}]
--------------------------------------------------------------------------------
  {1/2, 1/2}
*/
  vec2 first_moments = integrate(+[](vec2 X){ return X; }, domain);
  EXPECT_NEAR(first_moments[0], 0.5, 1.0e-14);
  EXPECT_NEAR(first_moments[1], 0.5, 1.0e-14);

  first_moments = integrate(std::function<vec2(vec2)>([](vec2 X){ return X; }), domain);
  EXPECT_NEAR(first_moments[0], 0.5, 1.0e-14);
  EXPECT_NEAR(first_moments[1], 0.5, 1.0e-14);
}

TEST(IntegrateTest, VectorFunction3D) {
  Mesh m = Mesh<>::load(FEMTO_MESH_DIR"patch_test_tets_and_hexes.json");
  Domain domain(m, MeshQuadratureRule{2});
/*
  Integrate[{x, y, z}, {x, y, z} \[Element] Cuboid[{0, 0, 0}, {2, 1, 1}]
--------------------------------------------------------------------------------
  {2, 1, 1}
*/
  vec3 first_moments = integrate(+[](vec3 X){ return X; }, domain);
  EXPECT_NEAR(first_moments[0], 2.0, 1.0e-14);
  EXPECT_NEAR(first_moments[1], 1.0, 1.0e-14);
  EXPECT_NEAR(first_moments[2], 1.0, 1.0e-14);

  first_moments = integrate(std::function<vec3(vec3)>([](vec3 X){ return X; }), domain);
  EXPECT_NEAR(first_moments[0], 2.0, 1.0e-14);
  EXPECT_NEAR(first_moments[1], 1.0, 1.0e-14);
  EXPECT_NEAR(first_moments[2], 1.0, 1.0e-14);
}

TEST(IntegrateTest, MatrixFunction2D) {
  Mesh m = Mesh<>::load(FEMTO_MESH_DIR"patch_test_tris_and_quads.json");
  Domain domain(m, MeshQuadratureRule{3});
/*
  Integrate[Outer[Times, {x, y}, {x, y}], {x, y} \[Element] Rectangle[{0, 0}, {1, 1}]
--------------------------------------------------------------------------------
  {{1/3, 1/4}, {1/4, 1/3}}
*/
  mat2 second_moments = integrate(+[](vec2 X){ return outer(X, X); }, domain);
  EXPECT_NEAR(second_moments[0][0], 1.0 / 3.0, 1.0e-14);
  EXPECT_NEAR(second_moments[0][1], 1.0 / 4.0, 1.0e-14);

  EXPECT_NEAR(second_moments[1][0], 1.0 / 4.0, 1.0e-14);
  EXPECT_NEAR(second_moments[1][1], 1.0 / 3.0, 1.0e-14);

  second_moments = integrate(std::function<mat2(vec2)>([](vec2 X){ return outer(X, X); }), domain);
  EXPECT_NEAR(second_moments[0][0], 1.0 / 3.0, 1.0e-14);
  EXPECT_NEAR(second_moments[0][1], 1.0 / 4.0, 1.0e-14);

  EXPECT_NEAR(second_moments[1][0], 1.0 / 4.0, 1.0e-14);
  EXPECT_NEAR(second_moments[1][1], 1.0 / 3.0, 1.0e-14);

}

TEST(IntegrateTest, MatrixFunction3D) {
  Mesh m = Mesh<>::load(FEMTO_MESH_DIR"patch_test_tets_and_hexes.json");
  Domain domain(m, MeshQuadratureRule{3});
/*
  Integrate[Outer[Times, {x, y, z}, {x, y, z}], {x, y, z} \[Element] Cuboid[{0, 0, 0}, {2, 1, 1}]
--------------------------------------------------------------------------------
  {{8/3, 1, 1}, {1, 2/3, 1/2}, {1, 1/2, 2/3}}
*/
  mat3 second_moments = integrate(+[](vec3 X){ return outer(X, X); }, domain);
  EXPECT_NEAR(second_moments[0][0], 8.0 / 3.0, 1.0e-14);
  EXPECT_NEAR(second_moments[0][1],       1.0, 1.0e-14);
  EXPECT_NEAR(second_moments[0][2],       1.0, 1.0e-14);

  EXPECT_NEAR(second_moments[1][0],       1.0, 1.0e-14);
  EXPECT_NEAR(second_moments[1][1], 2.0 / 3.0, 1.0e-14);
  EXPECT_NEAR(second_moments[1][2], 1.0 / 2.0, 1.0e-14);

  EXPECT_NEAR(second_moments[2][0],       1.0, 1.0e-14);
  EXPECT_NEAR(second_moments[2][1], 1.0 / 2.0, 1.0e-14);
  EXPECT_NEAR(second_moments[2][2], 2.0 / 3.0, 1.0e-14);

  second_moments = integrate(std::function<mat3(vec3)>([](vec3 X){ return outer(X, X); }), domain);
  EXPECT_NEAR(second_moments[0][0], 8.0 / 3.0, 1.0e-14);
  EXPECT_NEAR(second_moments[0][1],       1.0, 1.0e-14);
  EXPECT_NEAR(second_moments[0][2],       1.0, 1.0e-14);

  EXPECT_NEAR(second_moments[1][0],       1.0, 1.0e-14);
  EXPECT_NEAR(second_moments[1][1], 2.0 / 3.0, 1.0e-14);
  EXPECT_NEAR(second_moments[1][2], 1.0 / 2.0, 1.0e-14);

  EXPECT_NEAR(second_moments[2][0],       1.0, 1.0e-14);
  EXPECT_NEAR(second_moments[2][1], 1.0 / 2.0, 1.0e-14);
  EXPECT_NEAR(second_moments[2][2], 2.0 / 3.0, 1.0e-14);

}
