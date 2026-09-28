#include <gtest/gtest.h>

#include <iomanip>
#include <iostream>
#include <functional>

#include "femto/mesh.hpp"
#include "femto/domain.hpp"

#include "forall.hpp"
#include "fm/types/vec.hpp"
#include "fm/types/matrix.hpp"
#include "containers/ndarray_conversions.hpp"
#include "integrate_residual_cuda_common.cuh"

using namespace fm;
using namespace femto;
using namespace residual_cuda_tests;

// ----------------------------------------------------------------------------

template < typename vecd, typename F, typename G, typename Answer >
void flux_test(std::string filename,
               F f,
               G g,
               Answer answer,
               double tolerance) {

  if (!cuda_device_available()) {
    GTEST_SKIP() << "No CUDA-capable device is available";
  }

  auto mesh = Mesh<memory::space::gpu>::load(FEMTO_MESH_DIR + filename);

  // evaluate g at each quadrature point
  auto g_xi = CovariantWeightedVector<vecd, G>{g};

  for (int p = 1; p < 4; p++) {

    auto f_p = ScalarAtDegree<vecd, F>{f, p};

    Field<Family::DG, memory::space::gpu> u = create_field<Family::DG>(mesh, p, 1);
    auto nodes = nodes_for(u, mesh); 
    u = forall(f_p, nodes);

    BasisFunction phi(u);

    for (int q = p + 1; q < 5; q++) {

      Domain<memory::space::gpu> domain(mesh, MeshQuadratureRule(q));

      nd::gpu_array<double, 3> x_q = evaluate(mesh.X, domain);
      nd::gpu_array<double, 3> dx_dxi_q = evaluate(grad(mesh.X), isoparametric(domain));

      auto g_q = forall(g, x_q);
      auto g_xi_q = forall(g_xi, x_q, dx_dxi_q);

      Residual<Family::DG, memory::space::gpu> r1 = integrate(dot(g_q, grad(phi)), domain);
      Residual<Family::DG, memory::space::gpu> r2 = integrate(dot(g_xi_q, grad(phi)), isoparametric(domain));

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
    gradf[x_] := p  x^(p-1);
    g[x_] := x;
    Integrate[gradf[x] g[x], {x, 0, 1}]
  
   --------------------
  
  Out[] := 
    1/2
*/

struct F3 {
  __host__ __device__ double operator()(double x, int p) const { return pow(x, p); }
};

struct G3 {
  __host__ __device__ double operator()(double x) const { return x; }
};

struct Answer3 {
  double operator()(int p) const { return double(p) / double(1 + p); }
};

TEST(IntegrateTest, FluxH1Edges) { flux_test<double>("patch_test_edges.json", F3{}, G3{}, Answer3{}, 3.0e-15); }

// ----------------------------------------------------------------------------

/* 
  In[] := 
    f[{x_, y_}] := x^p - y + 3;
    gradf[{x_, y_}] := {p x^(-1 + p), -1};
    g[{x_, y_}] := {y, -x};
    Integrate[gradf[{x, y}] . g[{x, y}], {x, y} \[Element] Rectangle[{0, 0}, {1, 1}]]
  
   --------------------
  
  Out[] := 
    1
*/

struct F4 {
  __host__ __device__ double operator()(vec2 x, int p) const { return pow(x[0], p) - x[1] + 3.0; }
};

struct G4 {
  __host__ __device__ vec2 operator()(vec2 x) const { return vec2{x[1], -x[0]}; }
};

struct Answer4 {
  double operator()(int) const { return 1.0; }
};

TEST(IntegrateTest, FluxH1Tris) { flux_test<vec2>("patch_test_tris.json", F4{}, G4{}, Answer4{}, 5.0e-15); }
TEST(IntegrateTest, FluxH1Quads) { flux_test<vec2>("patch_test_quads.json", F4{}, G4{}, Answer4{}, 5.0e-15); }
TEST(IntegrateTest, FluxH1TrisAndQuads) { flux_test<vec2>("patch_test_tris_and_quads.json", F4{}, G4{}, Answer4{}, 5.0e-15); }

TEST(IntegrateTest, FluxH1TrisFine) { flux_test<vec2>("unit_square_of_tris_fine.json", F4{}, G4{}, Answer4{}, 8.0e-14); }
TEST(IntegrateTest, FluxH1QuadsFine) { flux_test<vec2>("unit_square_of_quads_fine.json", F4{}, G4{}, Answer4{}, 8.0e-14); }

// ----------------------------------------------------------------------------

/* 
  In[] := 
    f[{x_, y_, z_}] := x^p - y - 2 z + 3;
    gradf[{x_, y_, z_}] := {p x^(-1 + p), -1, -2};
    g[{x_, y_, z_}] := {z, x, -y};
    Integrate[gradf[{x, y, z}] . g[{x, y, z}], {x, y, z} \[Element] Cuboid[{0, 0, 0}, {1, 1, 1}]]
  
  --------------------
  
  Out[] := 
    1
*/
struct F5 {
  __host__ __device__ double operator()(vec3 x, int p) const { return pow(x[0], p) - x[1] - 2 * x[2] + 3.0; }
};

struct G5 {
  __host__ __device__ vec3 operator()(vec3 x) const { return vec3{x[2], x[0], -x[1]}; }
};

struct Answer5 {
  double operator()(int) const { return 1.0; }
};

TEST(IntegrateTest, FluxH1Tets) { flux_test<vec3>("patch_test_tets.json", F5{}, G5{}, Answer5{}, 5.0e-15); }
TEST(IntegrateTest, FluxH1Hexes) { flux_test<vec3>("patch_test_hexes.json", F5{}, G5{}, Answer5{}, 5.0e-15); }
TEST(IntegrateTest, FluxH1TetsFine) { flux_test<vec3>("unit_cube_of_tets_fine.json", F5{}, G5{}, Answer5{}, 5.0e-14); }
TEST(IntegrateTest, FluxH1HexesFine) { flux_test<vec3>("unit_cube_of_hexes_fine.json", F5{}, G5{}, Answer5{}, 5.0e-15); }
