#include <gtest/gtest.h>

#include <iomanip>
#include <iostream>
#include <functional>

#include "femto/mesh.hpp"
#include "femto/domain.hpp"

#include <gtest/gtest.h>

#include "forall.hpp"
#include "fm/types/vec.hpp"
#include "fm/types/matrix.hpp"
#include "containers/ndarray_conversions.hpp"
#include "integrate_residual_cuda_common.cuh"

using namespace femto;
using namespace residual_cuda_tests;

// ----------------------------------------------------------------------------

template < typename vecd, typename F, typename G, typename Answer >
void integrate_source_test(std::string filename,
                           F f,
                           G g,
                           Answer answer,
                           double tolerance) {

  if (!cuda_device_available()) {
    GTEST_SKIP() << "No CUDA-capable device is available";
  }

  auto mesh = Mesh<memory::space::gpu>::load(FEMTO_MESH_DIR + filename);

  std::cout << std::setprecision(15);

  // evaluate g at each quadrature point
  auto g_xi = CovariantWeightedVector<vecd, G>{g};

  for (int p = 1; p < 4; p++) {

    auto f_p = HcurlDofAtDegree<vecd, F>{f, p};

    Field<Family::Hcurl, memory::space::gpu> u = create_field<Family::Hcurl>(mesh, p, 1);
    auto nodes = nodes_for(u, mesh); 
    auto directions = directions_for(u, mesh); 
    u = forall(f_p, nodes, directions);

    BasisFunction phi(u);

    for (int q = p + 1; q < 5; q++) {

      Domain<memory::space::gpu> domain(mesh, MeshQuadratureRule(q));

      nd::gpu_array<double, 3> x_q = evaluate(mesh.X, domain);
      nd::gpu_array<double, 3> dx_dxi_q = evaluate(grad(mesh.X), isoparametric(domain));

      auto g_q = forall(g, x_q);
      auto g_xi_q = forall(g_xi, x_q, dx_dxi_q);

      Residual<Family::Hcurl, memory::space::gpu> r1 = integrate(g_q * phi, domain);
      Residual<Family::Hcurl, memory::space::gpu> r2 = integrate(g_xi_q * phi, isoparametric(domain));

      SCOPED_TRACE("p = " + std::to_string(p) + ", q = " + std::to_string(q));
      EXPECT_NEAR(host_dot(r1, u), answer(p), tolerance);
      EXPECT_NEAR(host_dot(r2, u), answer(p), tolerance);

      EXPECT_NEAR(host_relative_error(r1, r2), 0.0, 1.0e-14);

    }
  }

}

// ----------------------------------------------------------------------------

/* 
  In[] := 
    f[{x_, y_}] := {1, 3};
    g[{x_, y_}] := {1 + y, 2 - x};
    Integrate[f[{x, y}] . g[{x, y}], {x, y} \[Element] Rectangle[{0, 0}, {1, 1}]]
  
  --------------------
  
  Out[] := 
    6.0
*/
struct F1 {
  __host__ __device__ vec2 operator()(vec2, int) const { return vec2{1, 3}; }
};

struct G1 {
  __host__ __device__ vec2 operator()(vec2 x) const { return vec2{1 + x[1], 2 - x[0]}; }
};

struct Answer1 {
  double operator()(int) const { return 6.0; }
};

TEST(IntegrateTest, SourceHcurlTris) { integrate_source_test<vec2>("patch_test_tris.json", F1{}, G1{}, Answer1{}, 1.5e-14); }
TEST(IntegrateTest, SourceHcurlQuads) { integrate_source_test<vec2>("patch_test_quads.json", F1{}, G1{}, Answer1{}, 1.0e-14); }
TEST(IntegrateTest, SourceHcurlBoth) { integrate_source_test<vec2>("patch_test_tris_and_quads.json", F1{}, G1{}, Answer1{}, 1.0e-14); }

/* 
  In[] := 
    f[{x_, y_}] := {1 + 2 y, 3 - 2 x};
    g[{x_, y_}] := {2, 3};
    Integrate[f[{x, y}] . g[{x, y}], {x, y} \[Element] Triangle[{{0, 0}, {1, 0}, {0, 1}}]]
    Integrate[f[{x, y}] . g[{x, y}], {x, y} \[Element] Triangle[{{0, 0}, {1, 0}, {1, 1}}]]
  
  --------------------
  
  Out[] := 
    31.0 / 6.0
    25.0 / 6.0
*/
struct F2 {
  __host__ __device__ vec2 operator()(vec2 x, int) const { return vec2{1 + 2 * x[1], 3 - 2 * x[0]}; }
};

struct G2 {
  __host__ __device__ vec2 operator()(vec2) const { return vec2{2.0, 3.0}; }
};

struct Answer2 {
  double operator()(int) const { return 31.0 / 6.0; }
};

struct Answer3 {
  double operator()(int) const { return 25.0 / 6.0; }
};

TEST(IntegrateTest, SourceHcurlOneTri) { integrate_source_test<vec2>("one_tri.json", F2{}, G2{}, Answer2{}, 3.0e-14); }
TEST(IntegrateTest, SourceHcurlOneTriRotated) { integrate_source_test<vec2>("one_tri_rotated.json", F2{}, G2{}, Answer2{}, 3.0e-14); }
TEST(IntegrateTest, SourceHcurlOneTriSheared) { integrate_source_test<vec2>("one_tri_sheared.json", F2{}, G2{}, Answer3{}, 3.0e-14); }

// ----------------------------------------------------------------------------

/*
  In[] := 
    f[{x_, y_, z_}] := {1, 2, 3};
    g[{x_, y_, z_}] := {2, 2, 4};
    Integrate[f[{x, y, z}] . g[{x, y, z}], {x, y, z} \[Element] Tetrahedron[{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}]]
    Integrate[f[{x, y, z}] . g[{x, y, z}], {x, y, z} \[Element] Cuboid[{0, 0, 0}, {1, 1, 1}]]
    Integrate[f[{x, y, z}] . g[{x, y, z}], {x, y, z} \[Element] Cuboid[{0, 0, 0}, {2, 1, 1}]]
 
  --------------------
 
  Out[] := 
    3
    18
    36
*/
struct F3 {
  __host__ __device__ vec3 operator()(vec3, int) const { return vec3{1, 2, 3}; }
};

struct G3 {
  __host__ __device__ vec3 operator()(vec3) const { return vec3{2, 2, 4}; }
};

struct Answer4 {
  double operator()(int) const { return 3.0; }
};

struct Answer5 {
  double operator()(int) const { return 18.0; }
};

struct Answer6 {
  double operator()(int) const { return 36.0; }
};

TEST(IntegrateTest, SourceHcurlOneTet) { integrate_source_test<vec3>("one_tet.json", F3{}, G3{}, Answer4{}, 1.0e-13); }
TEST(IntegrateTest, SourceHcurlOneHex) { integrate_source_test<vec3>("one_hex.json", F3{}, G3{}, Answer5{}, 1.0e-13); }

TEST(IntegrateTest, SourceHcurlTets) { integrate_source_test<vec3>("patch_test_tets.json", F3{}, G3{}, Answer5{}, 1.0e-13); }
TEST(IntegrateTest, SourceHcurlHexes) { integrate_source_test<vec3>("patch_test_hexes.json", F3{}, G3{}, Answer5{}, 1.0e-13); }
TEST(IntegrateTest, SourceHcurlTetsAndHexes) { integrate_source_test<vec3>("patch_test_tets_and_hexes.json", F3{}, G3{}, Answer6{}, 1.0e-13); }
