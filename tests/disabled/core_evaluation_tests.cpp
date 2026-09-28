#include <gtest/gtest.h>

#include <iostream>
#include <functional>

#include "femto/mesh.hpp"
#include "femto/field.hpp"

#include "femto/domain.hpp"

#include "misc/for_constexpr.hpp"
#include "forall.hpp"

#include "fm/types/matrix.hpp"
#include "fm/types/vec.hpp"

template <int dim>
void interpolation_test(std::string filename, std::function<double(vec<dim>)> f,
                 std::function<vec<dim>(vec<dim>)> df_dX, int p, double tolerance) {

  std::function<vec<dim>(vec<dim>, mat<dim,dim>)> dxi_to_dX = [](vec<dim> du_dxi, mat<dim,dim> dX_dxi) {
    return dot(du_dxi, inv(dX_dxi));
  };

  auto mesh = Mesh<>::load(FEMTO_MESH_DIR + filename);

  Field u = create_field(mesh, Family::H1, p);
  nd::cpu_array<double, 2> nodes = nodes_for(u, mesh);

  // evaluate f at each node
  u = forall(f, nodes);

  for (int q = 1; q <= 3; q++) {

    Domain domain(mesh, MeshQuadratureRule(q));

    auto & X = mesh.X;

    nd::cpu_array<double, 3> u_q = evaluate(u, domain);

    // evaluate f, df_dx directly at each quadrature point
    auto X_q      = evaluate(X, domain);
    auto du_dxi_q = evaluate(grad_wrt_xi(u), domain);
    auto dX_dxi_q = evaluate(grad_wrt_xi(X), domain);

    auto du_dX_q = forall(dxi_to_dX, du_dxi_q, dX_dxi_q);

    auto f_q = forall(f, X_q);
    auto df_dX_q = forall(df_dX, X_q);

    EXPECT_LT(relative_error(flatten(u_q), flatten(f_q)), tolerance);
    EXPECT_LT(relative_error(du_dX_q, df_dX_q), tolerance);

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

#if 1
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
#endif

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

#if 1
GENERATE_TEST_3D(PatchTestHexesP1, "one_hex.json", 1, 2.0e-15);

GENERATE_TEST_3D(PatchTestHexP1, "patch_test_hexes.json", 1, 2.0e-15);
GENERATE_TEST_3D(PatchTestHexP2, "patch_test_hexes.json", 2, 2.0e-14);
GENERATE_TEST_3D(PatchTestHexP3, "patch_test_hexes.json", 3, 2.5e-14);

GENERATE_TEST_3D(PatchTestTetP1, "patch_test_tets.json", 1, 1.5e-16);
GENERATE_TEST_3D(PatchTestTetP2, "patch_test_tets.json", 2, 1.5e-15);
GENERATE_TEST_3D(PatchTestTetP3, "patch_test_tets.json", 3, 8.0e-15);

GENERATE_TEST_3D(PatchTestTetAndHexP1, "patch_test_tets_and_hexes.json", 1, 1.5e-15);
GENERATE_TEST_3D(PatchTestTetAndHexP2, "patch_test_tets_and_hexes.json", 2, 1.5e-14);
GENERATE_TEST_3D(PatchTestTetAndHexP3, "patch_test_tets_and_hexes.json", 3, 2.5e-14);
#endif

//GENERATE_TEST_3D(PatchTestHexP3, "patch_test_hexes.json", 3, 2.5e-14);