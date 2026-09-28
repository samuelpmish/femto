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
  auto g_detJ = DetJWeightedScalar<vecd, G>{g};

  for (int p = 1; p < 4; p++) {

    auto f_p = ScalarAtDegree<vecd, F>{f, p};

    Field<Family::H1, memory::space::gpu> u = create_field<Family::H1>(mesh, p, 1);
    auto nodes = nodes_for(u, mesh); 
    u = forall(f_p, nodes);

    BasisFunction phi(u);

    for (int q = p + 1; q < 5; q++) {

      Domain<memory::space::gpu> domain(mesh, MeshQuadratureRule(q));

      nd::gpu_array<double, 3> x_q = evaluate(mesh.X, domain);
      nd::gpu_array<double, 3> dx_dxi_q = evaluate(grad(mesh.X), isoparametric(domain));

      auto g_q = forall(g, x_q);
      auto g_detJ_q = forall(g_detJ, x_q, dx_dxi_q);

      Residual<Family::H1, memory::space::gpu> r1 = integrate(g_q * phi, domain);
      Residual<Family::H1, memory::space::gpu> r2 = integrate(g_detJ_q * phi, isoparametric(domain));

      SCOPED_TRACE("p = " + std::to_string(p) + ", q = " + std::to_string(q));
      EXPECT_NEAR(host_dot(r1, u), answer(p), tolerance);
      EXPECT_NEAR(host_dot(r2, u), answer(p), tolerance);

    }
  }

}

// ----------------------------------------------------------------------------

/* 
  In[] := 
    f[x_] := x^p + 3;
    g[x_] := x;
    Integrate[f[x]  g[x], {x, 0, 1}]
  
  --------------------
  
  Out[] := 
    (3.0/2.0) + 1.0 / (2.0 + p)
*/
struct F0 {
  __host__ __device__ double operator()(double x, int p) const { return pow(x, p) + 3.0; }
};

struct G0 {
  __host__ __device__ double operator()(double x) const { return x; }
};

struct Answer0 {
  double operator()(int p) const { return 1.5 + 1.0 / (2 + p); }
};

TEST(IntegrateTest, SourceH1Edge) { integrate_source_test<double>("patch_test_edges.json", F0{}, G0{}, Answer0{}, 3.0e-2); }

// ----------------------------------------------------------------------------

/* 
  In[] := 
    f[{x_, y_}] := x^p - y + 3;
    g[{x_, y_}] := y;
    Integrate[f[{x, y}] g[{x, y}], {x, y} \[Element] Rectangle[{0, 0}, {1, 1}]]
  
  --------------------
  
  Out[] := 
    (7.0/6.0) + 1.0 /(2.0 + 2.0 * p)
*/
struct F1 {
  __host__ __device__ double operator()(vec2 x, int p) const { return pow(x[0], p) - x[1] + 3.0; }
};

struct G1 {
  __host__ __device__ double operator()(vec2 x) const { return x[1]; }
};

struct Answer1 {
  double operator()(int p) const { return (7.0 / 6.0) + 1.0 / (2.0 + 2.0 * p); }
};

TEST(IntegrateTest, SourceH1Tris) { integrate_source_test<vec2>("patch_test_tris.json", F1{}, G1{}, Answer1{}, 3.0e-15); }
TEST(IntegrateTest, SourceH1Quads) { integrate_source_test<vec2>("patch_test_quads.json", F1{}, G1{}, Answer1{}, 1.0e-15); }
TEST(IntegrateTest, SourceH1Both) { integrate_source_test<vec2>("patch_test_tris_and_quads.json", F1{}, G1{}, Answer1{}, 1.0e-15); }

// ----------------------------------------------------------------------------

/*
  In[] := 
    f[{x_, y_, z_}] := x^p - y - z + 3;
    g[{x_, y_, z_}] := 1;
    Integrate[f[{x, y, z}] g[{x, y, z}], {x, y, z} \[Element] Cuboid[{0, 0, 0}, {1, 1, 1}]]
 
  --------------------
 
  Out[] := 
    2.0 + 1.0 / (1.0 + p)
*/
struct F2 {
  __host__ __device__ double operator()(vec3 x, int p) const { return pow(x[0], p) - x[1] - x[2] + 3.0; }
};

struct G2 {
  __host__ __device__ double operator()(vec3) const { return 1.0; }
};

struct Answer2 {
  double operator()(int p) const { return 2.0 + 1.0 / (1.0 + p); }
};

TEST(IntegrateTest, SourceH1Tets) { integrate_source_test<vec3>("patch_test_tets.json", F2{}, G2{}, Answer2{}, 7.0e-15); }
TEST(IntegrateTest, SourceH1Hexes) { integrate_source_test<vec3>("patch_test_hexes.json", F2{}, G2{}, Answer2{}, 1.0e-15); }
TEST(IntegrateTest, SourceH1TetsFine) { integrate_source_test<vec3>("unit_cube_of_tets_fine.json", F2{}, G2{}, Answer2{}, 5.0e-14); }
TEST(IntegrateTest, SourceH1HexesFine) { integrate_source_test<vec3>("unit_cube_of_hexes_fine.json", F2{}, G2{}, Answer2{}, 2.0e-14); }
