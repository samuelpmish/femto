#include <gtest/gtest.h>

#include "femto/mesh.hpp"
#include "femto/domain.hpp"

#include "forall.hpp"
#include "fm/types/vec.hpp"
#include "fm/types/matrix.hpp"
#include "containers/ndarray_conversions.hpp"
#include "integrate_residual_cuda_common.cuh"

using namespace femto;
using namespace residual_cuda_tests;

// ----------------------------------------------------------------------------

// the gpu counterpart of integrate_residual_H1_bdr_source_test.cpp
template < typename vecd, typename F, typename G, typename Answer >
void integrate_bdr_source_test(std::string filename,
                               F f,
                               G g,
                               Answer answer,
                               double tolerance) {

  if (!cuda_device_available()) {
    GTEST_SKIP() << "No CUDA-capable device is available";
  }

  auto mesh = Mesh<memory::space::gpu>::load(FEMTO_MESH_DIR + filename);

  auto g_dS = FacetWeightedScalar<vecd, G>{g};

  for (int p = 1; p < 4; p++) {

    auto f_p = ScalarAtDegree<vecd, F>{f, p};

    Field<Family::H1, memory::space::gpu> u = create_field<Family::H1>(mesh, p, 1);
    auto nodes = nodes_for(u, mesh);
    u = forall(f_p, nodes);

    BasisFunction phi(u);

    for (int q = p + 1; q < 5; q++) {

      Domain<memory::space::gpu> bdr(boundary_of(mesh), MeshQuadratureRule(q));

      nd::gpu_array<double, 3> x_q = evaluate(mesh.X, bdr);
      nd::gpu_array<double, 3> dx_dxi_q = evaluate(grad(mesh.X), isoparametric(bdr));

      auto g_q = forall(g, x_q);
      auto g_dS_q = forall(g_dS, x_q, dx_dxi_q);

      Residual<Family::H1, memory::space::gpu> r1 = integrate(g_q * phi, bdr);
      Residual<Family::H1, memory::space::gpu> r2 = integrate(g_dS_q * phi, isoparametric(bdr));

      SCOPED_TRACE("p = " + std::to_string(p) + ", q = " + std::to_string(q));
      EXPECT_NEAR(host_dot(r1, u), answer(p), tolerance);
      EXPECT_NEAR(host_dot(r2, u), answer(p), tolerance);

    }
  }

}

// ----------------------------------------------------------------------------

// boundary of the unit square: 29/6 + 1/(1 + p), see the .cpp test
struct F1 {
  __host__ __device__ double operator()(vec2 x, int p) const { return pow(x[0], p) - x[1] + 3.0; }
};

struct G1 {
  __host__ __device__ double operator()(vec2 x) const { return x[1]; }
};

struct Answer1 {
  double operator()(int p) const { return 29.0 / 6.0 + 1.0 / (1.0 + p); }
};

TEST(IntegrateTest, BdrSourceH1Tris) { integrate_bdr_source_test<vec2>("patch_test_tris.json", F1{}, G1{}, Answer1{}, 5.0e-14); }
TEST(IntegrateTest, BdrSourceH1Quads) { integrate_bdr_source_test<vec2>("patch_test_quads.json", F1{}, G1{}, Answer1{}, 5.0e-14); }
TEST(IntegrateTest, BdrSourceH1Both) { integrate_bdr_source_test<vec2>("patch_test_tris_and_quads.json", F1{}, G1{}, Answer1{}, 5.0e-14); }
TEST(IntegrateTest, BdrSourceH1TrisFine) { integrate_bdr_source_test<vec2>("unit_square_of_tris_fine.json", F1{}, G1{}, Answer1{}, 5.0e-13); }
TEST(IntegrateTest, BdrSourceH1QuadsFine) { integrate_bdr_source_test<vec2>("unit_square_of_quads_fine.json", F1{}, G1{}, Answer1{}, 5.0e-13); }

// ----------------------------------------------------------------------------

// boundary of the unit cube: 13 + 4/(1 + p), see the .cpp test
struct F2 {
  __host__ __device__ double operator()(vec3 x, int p) const { return pow(x[0], p) - x[1] - x[2] + 3.0; }
};

struct G2 {
  __host__ __device__ double operator()(vec3) const { return 1.0; }
};

struct Answer2 {
  double operator()(int p) const { return 13.0 + 4.0 / (1.0 + p); }
};

TEST(IntegrateTest, BdrSourceH1Tets) { integrate_bdr_source_test<vec3>("patch_test_tets.json", F2{}, G2{}, Answer2{}, 1.0e-13); }
TEST(IntegrateTest, BdrSourceH1Hexes) { integrate_bdr_source_test<vec3>("patch_test_hexes.json", F2{}, G2{}, Answer2{}, 1.0e-13); }
TEST(IntegrateTest, BdrSourceH1TetsFine) { integrate_bdr_source_test<vec3>("unit_cube_of_tets_fine.json", F2{}, G2{}, Answer2{}, 1.0e-12); }
TEST(IntegrateTest, BdrSourceH1HexesFine) { integrate_bdr_source_test<vec3>("unit_cube_of_hexes_fine.json", F2{}, G2{}, Answer2{}, 1.0e-12); }
