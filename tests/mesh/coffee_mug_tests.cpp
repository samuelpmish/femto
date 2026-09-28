#include "gtest/gtest.h"

#include "femto/mesh.hpp"

#include <map>
#include <array>

using namespace femto;

// determinant of the corner Jacobian at each of the 8 corners of each hex;
// all positive means no inverted or tangled elements
static double min_corner_jacobian(const Mesh<> & mesh) {

  constexpr int corner_coords[8][3] = {
    {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
    {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}
  };

  auto corner_id = [&](int i, int j, int k) {
    for (int c = 0; c < 8; c++) {
      if (corner_coords[c][0] == i && corner_coords[c][1] == j && corner_coords[c][2] == k) {
        return c;
      }
    }
    return -1;
  };

  double min_det = 1.0e30;
  for (uint32_t e = 0; e < mesh.hex.shape[0]; e++) {

    vec3 v[8];
    for (int c = 0; c < 8; c++) {
      uint32_t vid = mesh.hex(e, c).index;
      v[c] = vec3{mesh.X.data(vid, 0), mesh.X.data(vid, 1), mesh.X.data(vid, 2)};
    }

    for (int c = 0; c < 8; c++) {
      auto [i, j, k] = corner_coords[c];
      vec3 ex = (v[corner_id(1 - i, j, k)] - v[c]) * (i == 0 ? 1.0 : -1.0);
      vec3 ey = (v[corner_id(i, 1 - j, k)] - v[c]) * (j == 0 ? 1.0 : -1.0);
      vec3 ez = (v[corner_id(i, j, 1 - k)] - v[c]) * (k == 0 ? 1.0 : -1.0);
      min_det = std::min(min_det, dot(cross(ex, ey), ez));
    }

  }
  return min_det;

}

// a conforming single-block construction should never emit two vertices at
// (nearly) the same location -- coincident interfaces must share node ids
static bool has_duplicate_vertices(const Mesh<> & mesh, double tolerance) {
  std::map< std::array<int64_t, 3>, int > bins;
  for (uint32_t i = 0; i < mesh.vert.shape[0]; i++) {
    std::array<int64_t, 3> key;
    for (uint32_t d = 0; d < 3; d++) {
      key[d] = (int64_t)std::llround(mesh.X.data(i, d) / tolerance);
    }
    if (bins.count(key)) { return true; }
    bins[key] = 1;
  }
  return false;
}

TEST(coffee_mug, body_only) {

  double base_radius = 3.0;
  double rim_radius = 4.0;
  double height = 9.0;
  double thickness = 0.5;

  Mesh<> mug = Mesh<>::coffee_mug(base_radius, rim_radius, height, thickness, 0, 1);

  EXPECT_GT(mug.hex.shape[0], 0u);
  EXPECT_EQ(mug.tet.shape[0], 0u);
  EXPECT_TRUE(is_manifold(mug));
  EXPECT_GT(min_corner_jacobian(mug), 0.0);
  EXPECT_FALSE(has_duplicate_vertices(mug, 0.01 * thickness));

  // the mug should fit exactly in a cylinder of the larger radius
  double max_r = 0.0, min_z = 1.0e30, max_z = -1.0e30;
  for (uint32_t i = 0; i < mug.vert.shape[0]; i++) {
    double x = mug.X.data(i, 0);
    double y = mug.X.data(i, 1);
    double z = mug.X.data(i, 2);
    max_r = std::max(max_r, std::sqrt(x * x + y * y));
    min_z = std::min(min_z, z);
    max_z = std::max(max_z, z);
  }
  EXPECT_NEAR(max_r, std::max(base_radius, rim_radius), 1e-12);
  EXPECT_NEAR(min_z, 0.0, 1e-12);
  EXPECT_NEAR(max_z, height, 1e-12);

}

TEST(coffee_mug, with_handles) {

  for (int handles : {1, 2, 3, 6}) {
    Mesh<> mug = Mesh<>::coffee_mug(3.0, 4.0, 9.0, 0.5, handles, 1);
    EXPECT_GT(mug.hex.shape[0], 0u);

    // the handles are attached conformally, so the whole mesh is one
    // manifold solid with no duplicated nodes at the interfaces
    EXPECT_TRUE(is_manifold(mug));
    EXPECT_FALSE(has_duplicate_vertices(mug, 0.005));
    EXPECT_GT(min_corner_jacobian(mug), 0.0);

    // handles stick out past the wall of the cup
    double max_r = 0.0;
    for (uint32_t i = 0; i < mug.vert.shape[0]; i++) {
      double x = mug.X.data(i, 0);
      double y = mug.X.data(i, 1);
      max_r = std::max(max_r, std::sqrt(x * x + y * y));
    }
    EXPECT_GT(max_r, 4.0);
  }

}

TEST(coffee_mug, tapered) {

  // rim narrower than base
  Mesh<> mug = Mesh<>::coffee_mug(4.0, 3.0, 8.0, 0.5, 2, 1);
  EXPECT_TRUE(is_manifold(mug));
  EXPECT_GT(min_corner_jacobian(mug), 0.0);

}

TEST(coffee_mug, high_order) {

  Mesh<> p1 = Mesh<>::coffee_mug(3.0, 4.0, 9.0, 0.5, 1, 1);
  Mesh<> p2 = Mesh<>::coffee_mug(3.0, 4.0, 9.0, 0.5, 1, 2);

  EXPECT_EQ(p1.hex.shape[0], p2.hex.shape[0]);
  EXPECT_GT(p2.X.data.shape[0], p1.X.data.shape[0]);
  EXPECT_EQ(p2.X.degree, 2u);

  for (uint32_t i = 0; i < p2.X.data.shape[0]; i++) {
    for (uint32_t d = 0; d < 3; d++) {
      EXPECT_TRUE(std::isfinite(p2.X.data(i, d)));
    }
  }

}
