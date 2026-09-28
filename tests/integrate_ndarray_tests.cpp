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

TEST(IntegrateTest, ScalarNDArray2D) {
  Mesh mesh = Mesh<>::load(FEMTO_MESH_DIR"patch_test_tris_and_quads.json");
  Domain domain(mesh, MeshQuadratureRule{3});
  nd::cpu_array<double, 1> integrand_q({total(domain.num_qpts)});
  for (uint32_t i = 0; i < integrand_q.shape[0]; i++) {
    integrand_q[i] = 1.0;
  }
  double measure = integrate(integrand_q, domain);
  EXPECT_NEAR(measure, 1.0, 1.0e-14);
}

TEST(IntegrateTest, ScalarNDArray3D) {
  Mesh mesh = Mesh<>::load(FEMTO_MESH_DIR"patch_test_tets_and_hexes.json");
  Domain domain(mesh, MeshQuadratureRule{3});
  nd::cpu_array<double, 1> integrand_q({total(domain.num_qpts)});
  for (uint32_t i = 0; i < integrand_q.shape[0]; i++) {
    integrand_q[i] = 1.0;
  }
  double measure = integrate(integrand_q, domain);
  EXPECT_NEAR(measure, 2.0, 1.0e-14);
}

TEST(IntegrateTest, VectorNDArray2D) {
  Mesh mesh = Mesh<>::load(FEMTO_MESH_DIR"patch_test_tris_and_quads.json");
  Domain domain(mesh, MeshQuadratureRule{3});
  nd::cpu_array<double, 3> X_q = evaluate(mesh.X, domain);

  nd::cpu_array<vec2, 1> integrand_q({total(domain.num_qpts)});
  for (uint32_t i = 0; i < integrand_q.shape[0]; i++) {
    integrand_q[i] = vec2{X_q(i, 0, 0), X_q(i, 1, 0)};
  }

/*
  Integrate[{x, y}, {x, y} \[Element] Rectangle[{0, 0}, {1, 1}]
--------------------------------------------------------------------------------
  {1/2, 1/2}
*/
  vec2 first_moments = integrate(integrand_q, domain);
  EXPECT_NEAR(first_moments[0], 0.5, 1.0e-14);
  EXPECT_NEAR(first_moments[1], 0.5, 1.0e-14);
}

TEST(IntegrateTest, VectorNDArray3D) {
  Mesh mesh = Mesh<>::load(FEMTO_MESH_DIR"patch_test_tets_and_hexes.json");
  Domain domain(mesh, MeshQuadratureRule{2});
  nd::cpu_array<double, 3> X_q = evaluate(mesh.X, domain);

  nd::cpu_array<vec3, 1> integrand_q({total(domain.num_qpts)});
  for (uint32_t i = 0; i < integrand_q.shape[0]; i++) {
    integrand_q[i] = vec3{X_q(i, 0, 0), X_q(i, 1, 0), X_q(i, 2, 0)};
  }

/*
  Integrate[{x, y, z}, {x, y, z} \[Element] Cuboid[{0, 0, 0}, {2, 1, 1}]
--------------------------------------------------------------------------------
  {2, 1, 1}
*/
  vec3 first_moments = integrate(integrand_q, domain);
  EXPECT_NEAR(first_moments[0], 2.0, 1.0e-14);
  EXPECT_NEAR(first_moments[1], 1.0, 1.0e-14);
  EXPECT_NEAR(first_moments[2], 1.0, 1.0e-14);
}

TEST(IntegrateTest, MatrixNDArray2D) {
  Mesh mesh = Mesh<>::load(FEMTO_MESH_DIR"patch_test_tris_and_quads.json");
  Domain domain(mesh, MeshQuadratureRule{3});
  nd::cpu_array<double, 3> X_q = evaluate(mesh.X, domain);

  nd::cpu_array<mat2, 1> integrand_q({total(domain.num_qpts)});
  for (uint32_t i = 0; i < integrand_q.shape[0]; i++) {
    vec2 X{X_q(i, 0, 0), X_q(i, 1, 0)};
    integrand_q[i] = outer(X, X);
  }

/*
  Integrate[Outer[Times, {x, y}, {x, y}], {x, y} \[Element] Rectangle[{0, 0}, {1, 1}]
--------------------------------------------------------------------------------
  {{1/3, 1/4}, {1/4, 1/3}}
*/

  mat2 second_moments = integrate(integrand_q, domain);
  EXPECT_NEAR(second_moments[0][0], 1.0 / 3.0, 1.0e-14);
  EXPECT_NEAR(second_moments[0][1], 1.0 / 4.0, 1.0e-14);

  EXPECT_NEAR(second_moments[1][0], 1.0 / 4.0, 1.0e-14);
  EXPECT_NEAR(second_moments[1][1], 1.0 / 3.0, 1.0e-14);
}

TEST(IntegrateTest, MatrixNDArray3D) {
  Mesh mesh = Mesh<>::load(FEMTO_MESH_DIR"patch_test_tets_and_hexes.json");
  Domain domain(mesh, MeshQuadratureRule{3});
  nd::cpu_array<double, 3> X_q = evaluate(mesh.X, domain);

  nd::cpu_array<mat3, 1> integrand_q({total(domain.num_qpts)});
  for (uint32_t i = 0; i < integrand_q.shape[0]; i++) {
    vec3 X{X_q(i, 0, 0), X_q(i, 1, 0), X_q(i, 2, 0)};
    integrand_q[i] = outer(X, X);
  }

/*
  Integrate[Outer[Times, {x, y, z}, {x, y, z}], {x, y, z} \[Element] Cuboid[{0, 0, 0}, {2, 1, 1}]
--------------------------------------------------------------------------------
  {{8/3, 1, 1}, {1, 2/3, 1/2}, {1, 1/2, 2/3}}
*/
  mat3 second_moments = integrate(integrand_q, domain);
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
