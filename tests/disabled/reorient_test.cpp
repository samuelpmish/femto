#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <utility>
#include <iostream>
#include <inttypes.h>

#include "containers/ndarray_conversions.hpp"
#include "femto/mesh.hpp"
#include "forall.hpp"
#include "fm/types/matrix.hpp"
#include "fm/types/vec.hpp"

/* Mathematica code for finding the 24 admissible permutations for a Hex8 element

  nodes = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
  isValidPermutation[p_] := Module[{
     A = {nodes[[p[[2]]]] - nodes[[p[[1]]]], nodes[[p[[4]]]] - nodes[[p[[1]]]], nodes[[p[[5]]]] - nodes[[p[[1]]]]},
     B = # - nodes[[p[[1]]]] & /@ nodes[[p]]
     }, (Det[A] == 1) && (B.Inverse[A] == nodes) ]
  Table[If[isValidPermutation[p], p, Nothing], {p, Permutations[Range[8]]}] - 1
*/
std::array<uint64_t, 8> hex8_permutations[24] = {
    {0, 1, 2, 3, 4, 5, 6, 7}, {0, 3, 7, 4, 1, 2, 6, 5},
    {0, 4, 5, 1, 3, 7, 6, 2}, {1, 0, 4, 5, 2, 3, 7, 6},
    {1, 2, 3, 0, 5, 6, 7, 4}, {1, 5, 6, 2, 0, 4, 7, 3},
    {2, 1, 5, 6, 3, 0, 4, 7}, {2, 3, 0, 1, 6, 7, 4, 5},
    {2, 6, 7, 3, 1, 5, 4, 0}, {3, 0, 1, 2, 7, 4, 5, 6},
    {3, 2, 6, 7, 0, 1, 5, 4}, {3, 7, 4, 0, 2, 6, 5, 1},
    {4, 0, 3, 7, 5, 1, 2, 6}, {4, 5, 1, 0, 7, 6, 2, 3},
    {4, 7, 6, 5, 0, 3, 2, 1}, {5, 1, 0, 4, 6, 2, 3, 7},
    {5, 4, 7, 6, 1, 0, 3, 2}, {5, 6, 2, 1, 4, 7, 3, 0},
    {6, 2, 1, 5, 7, 3, 0, 4}, {6, 5, 4, 7, 2, 1, 0, 3},
    {6, 7, 3, 2, 5, 4, 0, 1}, {7, 3, 2, 6, 4, 0, 1, 5},
    {7, 4, 0, 3, 6, 5, 1, 2}, {7, 6, 5, 4, 3, 2, 1, 0}};


std::string two_hex_mesh(int i) {
  char buffer[512];
  char mesh_template[353] = 
R"({
  "nodes": {
    "degree": 1,
    "data": [
      [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], 
      [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
      [0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [1.0, 1.0, 2.0], [0.0, 1.0, 2.0]
    ]
  },
  "hex": [[%d, %d, %d, %d, %d, %d, %d, %d], [4, 5, 6, 7, 8, 9, 10, 11]]
})";

  auto p = hex8_permutations[i];

  snprintf(buffer, 512, mesh_template, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);

  return std::string(buffer);
}

std::string patch_test_hex_mesh(int i) {
  char buffer[1024];
  char mesh_template[636] = 
R"({
  "nodes": {
    "degree": 1,
    "data": [
      [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], 
      [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
      [0.2, 0.3, 0.3], [0.7, 0.5, 0.3], [0.7, 0.7, 0.3], [0.3, 0.8, 0.3],
      [0.3, 0.4, 0.7], [0.7, 0.2, 0.6], [0.7, 0.6, 0.7], [0.2, 0.7, 0.6]
    ]
  },

  "hex": [
    [ 0,  1,  2,  3,  8,  9, 10, 11],
    [ 4,  5,  1,  0, 12, 13,  9,  8],
    [ 5,  6,  2,  1, 13, 14, 10,  9],
    [ 6,  7,  3,  2, 14, 15, 11, 10],
    [ 7,  4,  0,  3, 15, 12,  8, 11],
    [12, 13, 14, 15,  4,  5,  6,  7],
    [%d, %d, %d, %d, %d, %d, %d, %d]
  ]
})";

  auto p = hex8_permutations[i];
  for (int j = 0; j < 8; j++) {
    p[j] += 8;
  }

  snprintf(buffer, 1024, mesh_template, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);

  return std::string(buffer);
}

void interpolation_test(std::string str) {

  int p = 3;
  int q = 3;
  double tolerance = 2.0e-14;

  std::function<vec3(vec3, mat3)> dxi_to_dx = [](vec3 du_dxi, mat3 dx_dxi) {
    return dot(du_dxi, inv(dx_dxi));
  };

  std::function<double(vec3)> f = [](vec3 x) { return x[0] + 2.0 * x[1] + 3.0 * x[2]; };
  std::function<vec3(vec3)> df_dx = [](vec3 x) { return vec3{1.0, 2.0, 3.0}; };

  auto mesh = Mesh<>::import_from_json_string(str);

  Field u = create_field(mesh, Family::H1, p);
  nd::cpu_array<double, 2> nodes = nodes_for(u, mesh);

  // evaluate f at each node
  u = forall(f, nodes);

  QuadratureRule rule = gauss_legendre_rule(q);

  auto u_q = interpolate(u, mesh, rule);
  auto x_q = interpolate(mesh.X, mesh, rule);
  auto f_q = forall(f, x_q);
  EXPECT_LT(relative_error(u_q, f_q), tolerance);

  auto du_dxi_q = gradient_wrt_xi(u, mesh, rule);
  auto dx_dxi_q = gradient_wrt_xi(mesh.X, mesh, rule);
  auto du_dx_q = forall(dxi_to_dx, du_dxi_q, dx_dxi_q);
  auto df_dx_q = forall(df_dx, x_q);
  EXPECT_LT(relative_error(du_dx_q, df_dx_q), tolerance);

}

TEST(Reorient, CubicHex0) { interpolation_test(two_hex_mesh(0)); }
TEST(Reorient, CubicHex1) { interpolation_test(two_hex_mesh(1)); }
TEST(Reorient, CubicHex2) { interpolation_test(two_hex_mesh(2)); }
TEST(Reorient, CubicHex3) { interpolation_test(two_hex_mesh(3)); }
TEST(Reorient, CubicHex4) { interpolation_test(two_hex_mesh(4)); }
TEST(Reorient, CubicHex5) { interpolation_test(two_hex_mesh(5)); }
TEST(Reorient, CubicHex6) { interpolation_test(two_hex_mesh(6)); }
TEST(Reorient, CubicHex7) { interpolation_test(two_hex_mesh(7)); }
TEST(Reorient, CubicHex8) { interpolation_test(two_hex_mesh(8)); }
TEST(Reorient, CubicHex9) { interpolation_test(two_hex_mesh(9)); }
TEST(Reorient, CubicHex10) { interpolation_test(two_hex_mesh(10)); }
TEST(Reorient, CubicHex11) { interpolation_test(two_hex_mesh(11)); }
TEST(Reorient, CubicHex12) { interpolation_test(two_hex_mesh(12)); }
TEST(Reorient, CubicHex13) { interpolation_test(two_hex_mesh(13)); }
TEST(Reorient, CubicHex14) { interpolation_test(two_hex_mesh(14)); }
TEST(Reorient, CubicHex15) { interpolation_test(two_hex_mesh(15)); }
TEST(Reorient, CubicHex16) { interpolation_test(two_hex_mesh(16)); }
TEST(Reorient, CubicHex17) { interpolation_test(two_hex_mesh(17)); }
TEST(Reorient, CubicHex18) { interpolation_test(two_hex_mesh(18)); }
TEST(Reorient, CubicHex19) { interpolation_test(two_hex_mesh(19)); }
TEST(Reorient, CubicHex20) { interpolation_test(two_hex_mesh(20)); }
TEST(Reorient, CubicHex21) { interpolation_test(two_hex_mesh(21)); }
TEST(Reorient, CubicHex22) { interpolation_test(two_hex_mesh(22)); }
TEST(Reorient, CubicHex23) { interpolation_test(two_hex_mesh(23)); }

TEST(Reorient, CubicPatchTestHex0)  { interpolation_test(patch_test_hex_mesh(0)); }
TEST(Reorient, CubicPatchTestHex1)  { interpolation_test(patch_test_hex_mesh(1)); }
TEST(Reorient, CubicPatchTestHex2)  { interpolation_test(patch_test_hex_mesh(2)); }
TEST(Reorient, CubicPatchTestHex3)  { interpolation_test(patch_test_hex_mesh(3)); }
TEST(Reorient, CubicPatchTestHex4)  { interpolation_test(patch_test_hex_mesh(4)); }
TEST(Reorient, CubicPatchTestHex5)  { interpolation_test(patch_test_hex_mesh(5)); }
TEST(Reorient, CubicPatchTestHex6)  { interpolation_test(patch_test_hex_mesh(6)); }
TEST(Reorient, CubicPatchTestHex7)  { interpolation_test(patch_test_hex_mesh(7)); }
TEST(Reorient, CubicPatchTestHex8)  { interpolation_test(patch_test_hex_mesh(8)); }
TEST(Reorient, CubicPatchTestHex9)  { interpolation_test(patch_test_hex_mesh(9)); }
TEST(Reorient, CubicPatchTestHex10) { interpolation_test(patch_test_hex_mesh(10)); }
TEST(Reorient, CubicPatchTestHex11) { interpolation_test(patch_test_hex_mesh(11)); }
TEST(Reorient, CubicPatchTestHex12) { interpolation_test(patch_test_hex_mesh(12)); }
TEST(Reorient, CubicPatchTestHex13) { interpolation_test(patch_test_hex_mesh(13)); }
TEST(Reorient, CubicPatchTestHex14) { interpolation_test(patch_test_hex_mesh(14)); }
TEST(Reorient, CubicPatchTestHex15) { interpolation_test(patch_test_hex_mesh(15)); }
TEST(Reorient, CubicPatchTestHex16) { interpolation_test(patch_test_hex_mesh(16)); }
TEST(Reorient, CubicPatchTestHex17) { interpolation_test(patch_test_hex_mesh(17)); }
TEST(Reorient, CubicPatchTestHex18) { interpolation_test(patch_test_hex_mesh(18)); }
TEST(Reorient, CubicPatchTestHex19) { interpolation_test(patch_test_hex_mesh(19)); }
TEST(Reorient, CubicPatchTestHex20) { interpolation_test(patch_test_hex_mesh(20)); }
TEST(Reorient, CubicPatchTestHex21) { interpolation_test(patch_test_hex_mesh(21)); }
TEST(Reorient, CubicPatchTestHex22) { interpolation_test(patch_test_hex_mesh(22)); }
TEST(Reorient, CubicPatchTestHex23) { interpolation_test(patch_test_hex_mesh(23)); }