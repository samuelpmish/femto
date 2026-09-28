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

template < typename vec_t, typename F, typename G, typename Answer >
void integrate_flux_test(std::string filename,
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
  auto g_xi = HcurlFluxReference<vec_t, G>{g};

  for (int p = 1; p < 4; p++) {

    auto f_p = HcurlDofAtDegree<vec_t, F>{f, p};

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

      Residual<Family::Hcurl, memory::space::gpu> r1 = integrate(dot(g_q, curl(phi)), domain);
      Residual<Family::Hcurl, memory::space::gpu> r2 = integrate(dot(g_xi_q, curl(phi)), isoparametric(domain));

      SCOPED_TRACE("p = " + std::to_string(p) + ", q = " + std::to_string(q));
      EXPECT_NEAR(host_dot(r1, u), answer(p), tolerance);
      EXPECT_NEAR(host_dot(r2, u), answer(p), tolerance);

    }
  }

}

// ----------------------------------------------------------------------------

/* 
  In[] := 
    f[{x_, y_}] := {3 + y, 2 - x};
    g[{x_, y_}] := x + y;
    Integrate[Curl[f[{x, y}], {x, y}] g[{x, y}], {x, y} \[Element] Rectangle[{0, 0}, {1, 1}]]
  
  --------------------
  
  Out[] := 
    -2.0
*/
struct F1 {
  __host__ __device__ vec2 operator()(vec2 x, int) const { return vec2{3 + x[1], 2 - x[0]}; }
};

struct G1 {
  __host__ __device__ double operator()(vec2 x) const { return x[0] + x[1]; }
};

struct Answer1 {
  double operator()(int) const { return -2.0; }
};

// note: non affinely-transformed low-order quadrilateral elements can't exactly reproduce the 
//       functions in this test, so for p=1 those tests have some small finite error
//       which decreases with mesh refinement
TEST(IntegrateTest, FluxHcurlTris) { integrate_flux_test<vec2>("patch_test_tris.json", F1{}, G1{}, Answer1{}, 5.0e-14); }
TEST(IntegrateTest, FluxHcurlQuads) { integrate_flux_test<vec2>("patch_test_quads.json", F1{}, G1{}, Answer1{}, 1.0e-2); }
TEST(IntegrateTest, FluxHcurlQuadsFine) { integrate_flux_test<vec2>("unit_square_of_quads_fine.json", F1{}, G1{}, Answer1{}, 1.0e-5); }
TEST(IntegrateTest, FluxHcurlBoth) { integrate_flux_test<vec2>("patch_test_tris_and_quads.json", F1{}, G1{}, Answer1{}, 1.0e-3); }
TEST(IntegrateTest, FluxHcurlTrisFine) { integrate_flux_test<vec2>("unit_square_of_tris_fine.json", F1{}, G1{}, Answer1{}, 1.5e-13); }

// ----------------------------------------------------------------------------

/*
  In[] := 
  f[{x_, y_, z_}] := {y + 3, 2 - x - z, 1 + y};
  g[{x_, y_, z_}] := {y, z, 2 x};
  Integrate[Curl[f[{x, y, z}], {x, y, z}]  g[{x, y, z}], {x, y, z} \[Element] Tetrahedron[{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}]]
  Integrate[Curl[f[{x, y, z}], {x, y, z}]  g[{x, y, z}], {x, y, z} \[Element] Cuboid[{0, 0, 0}, {1, 1, 1}]]
  Integrate[Curl[f[{x, y, z}], {x, y, z}]  g[{x, y, z}], {x, y, z} \[Element] Cuboid[{0, 0, 0}, {2, 1, 1}]]
 
  --------------------
 
  Out[] := 
    -1.0 / 12.0
    -1.0
    -6.0
*/
struct F2 {
  __host__ __device__ vec3 operator()(vec3 x, int) const {
    return vec3{3 + x[1], 2 - x[0] - x[2], 1 + x[1]};
  }
};

struct G2 {
  __host__ __device__ vec3 operator()(vec3 x) const { return vec3{x[1], x[2], 2 * x[0]}; }
};

struct Answer2 {
  double operator()(int) const { return -1.0 / 12.0; }
};

struct Answer3 {
  double operator()(int) const { return -1.0; }
};

struct Answer4 {
  double operator()(int) const { return -6.0; }
};

TEST(IntegrateTest, FluxHcurlOneTet) { integrate_flux_test<vec3>("one_tet.json", F2{}, G2{}, Answer2{}, 1.0e-13); }
TEST(IntegrateTest, FluxHcurlOneHex) { integrate_flux_test<vec3>("one_hex.json", F2{}, G2{}, Answer3{}, 1.0e-13); }

TEST(IntegrateTest, FluxHcurlTets) { integrate_flux_test<vec3>("patch_test_tets.json", F2{}, G2{}, Answer3{}, 1.0e-13); }
TEST(IntegrateTest, FluxHcurlHexes) { integrate_flux_test<vec3>("patch_test_hexes.json", F2{}, G2{}, Answer3{}, 1.0e-1); }
TEST(IntegrateTest, FluxHcurlHexesFine) { integrate_flux_test<vec3>("unit_cube_of_hexes_fine.json", F2{}, G2{}, Answer3{}, 5.0e-4); }
TEST(IntegrateTest, FluxHcurlTetsAndHexes) { integrate_flux_test<vec3>("patch_test_tets_and_hexes.json", F2{}, G2{}, Answer4{}, 1.0e-1); }
