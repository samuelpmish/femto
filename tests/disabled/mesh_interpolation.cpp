#include <gtest/gtest.h>

#include <iostream>
#include <functional>

#include "containers/ndarray_conversions.hpp"
#include "femto/mesh.hpp"
#include "forall.hpp"
#include "fm/types/matrix.hpp"
#include "fm/types/vec.hpp"

template <int dim>
void interpolation_test(std::string filename, std::function<double(vec<dim>)> f,
                 std::function<vec<dim>(vec<dim>)> df_dx, int p, double tolerance) {

  std::function<vec<dim>(vec<dim>, mat<dim,dim>)> dxi_to_dx = [](vec<dim> du_dxi, mat<dim,dim> dx_dxi) {
    return dot(du_dxi, inv(dx_dxi));
  };

  auto mesh = Mesh<>::load(FEMTO_MESH_DIR + filename);

  Field u = create_field(mesh, Family::H1, p);
  nd::cpu_array<double, 2> nodes = nodes_for(u, mesh);

  // evaluate f at each node
  u = forall(f, nodes);

  for (int q = 1; q <= 6; q++) {

    QuadratureRule rule = gauss_legendre_rule(q);

    auto u_q = interpolate(u, mesh, rule);
    auto du_dxi_q = gradient_wrt_xi(u, mesh, rule);
    auto dx_dxi_q = gradient_wrt_xi(mesh.X, mesh, rule);
    auto du_dx_q = forall(dxi_to_dx, du_dxi_q, dx_dxi_q);

    // evaluate f, df_dx directly at each quadrature point
    auto x_q = interpolate(mesh.X, mesh, rule);

    auto f_q = forall(f, x_q);

    auto df_dx_q = forall(df_dx, x_q);

    EXPECT_LT(relative_error(u_q, f_q), tolerance);
    EXPECT_LT(relative_error(du_dx_q, df_dx_q), tolerance);

  }

}

#define GENERATE_TEST_2D(NAME, mesh, p, tolerance)                                        \
TEST(InterpolationTest2D, NAME) {                                                         \
  interpolation_test(                                                                     \
    mesh,                                                                                 \
    std::function< double(vec2) >([](vec2 x) { return pow(x[0], p) - x[1] + 3.0; }),      \
    std::function< vec2(vec2) >([](vec2 x) { return vec2{p * pow(x[0], p - 1), -1.0}; }), \
    p,                                                                                    \
    tolerance                                                                             \
  );                                                                                      \
}

GENERATE_TEST_2D(PatchTestTriP1, "patch_test_tris.json", 1, 1.0e-15);
GENERATE_TEST_2D(PatchTestTriP2, "patch_test_tris.json", 2, 1.0e-14);
GENERATE_TEST_2D(PatchTestTriP3, "patch_test_tris.json", 3, 1.5e-14);

GENERATE_TEST_2D(PatchTestQuadP1, "patch_test_quads.json", 1, 1.0e-15);
GENERATE_TEST_2D(PatchTestQuadP2, "patch_test_quads.json", 2, 1.0e-14);
GENERATE_TEST_2D(PatchTestQuadP3, "patch_test_quads.json", 3, 1.0e-14);

GENERATE_TEST_2D(PatchTestTriAndQuadP1, "patch_test_tris_and_quads.json", 1, 1.0e-15);
GENERATE_TEST_2D(PatchTestTriAndQuadP2, "patch_test_tris_and_quads.json", 2, 1.0e-14);
GENERATE_TEST_2D(PatchTestTriAndQuadP3, "patch_test_tris_and_quads.json", 3, 2.0e-14);

GENERATE_TEST_2D(WrenchTestP1, "wrench.json", 1, 1.0e-12);
GENERATE_TEST_2D(WrenchTestP2, "wrench.json", 2, 1.0e-12);
GENERATE_TEST_2D(WrenchTestP3, "wrench.json", 3, 1.0e-12);

/////////////////////////////////////////////////////////////////////////////////////////////////

#define GENERATE_TEST_3D(NAME, mesh, p, tolerance)                                              \
TEST(InterpolationTest3D, NAME) {                                                               \
  interpolation_test(                                                                           \
    mesh,                                                                                       \
    std::function< double(vec3) >([](vec3 x) { return pow(x[0], p) - x[1] - x[2] + 3.0; }),     \
    std::function< vec3(vec3) >([](vec3 x) { return vec3{p * pow(x[0], p - 1), -1.0, -1.0}; }), \
    p,                                                                                          \
    tolerance                                                                                   \
  );                                                                                            \
}

GENERATE_TEST_3D(PatchTestHexesP1, "patch_test_hexes.json", 1, 2.0e-15);
GENERATE_TEST_3D(PatchTestHexesP2, "patch_test_hexes.json", 2, 2.0e-14);
GENERATE_TEST_3D(PatchTestHexesP3, "patch_test_hexes.json", 3, 2.5e-14);

GENERATE_TEST_3D(PatchTestTetsP1, "patch_test_tets.json", 1, 1.5e-16);
GENERATE_TEST_3D(PatchTestTetsP2, "patch_test_tets.json", 2, 1.5e-15);
GENERATE_TEST_3D(PatchTestTetsP3, "patch_test_tets.json", 3, 8.0e-15);
