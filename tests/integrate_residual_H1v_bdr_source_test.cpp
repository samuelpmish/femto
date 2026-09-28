#include <gtest/gtest.h>

#include <functional>

#include "femto/mesh.hpp"
#include "femto/domain.hpp"

#include "forall.hpp"
#include "fm/types/vec.hpp"
#include "fm/types/matrix.hpp"
#include "containers/ndarray_conversions.hpp"

using namespace femto;

// the measure of a boundary facet with (sdim x sdim - 1) jacobian J
template < uint32_t sdim, uint32_t gdim >
double facet_measure(const mat<sdim, gdim> & J) { return sqrt(det(dot(transpose(J), J))); }

// ----------------------------------------------------------------------------

// a vector-valued source on the boundary: component i of u interpolates
// (i + 1) f, every component of the source is g, so dot(r, u) is
// (1 + 2 [+ 3]) times the boundary integral of f * g
template < typename vecd >
void integrate_bdr_source_test(std::string filename,
                               std::function< double(vecd, int) > f,
                               std::function< double(vecd) > g,
                               std::function< double(int) > answer,
                               double tolerance) {

  constexpr uint32_t dim = dimension(vecd{});
  using jacd = mat<dim, dim - 1>;

  double scale_factor = (dim == 2) ? 3.0 : 6.0;

  auto mesh = Mesh<>::load(FEMTO_MESH_DIR + filename);

  std::function< vecd(vecd) > g_vec = [g](vecd x) {
    vecd output{};
    for (int i = 0; i < dim; i++) { output[i] = g(x); }
    return output;
  };

  std::function< vecd(vecd, jacd) > g_dS = [g](vecd x, jacd J) {
    vecd output{};
    for (int i = 0; i < dim; i++) { output[i] = g(x) * facet_measure(J); }
    return output;
  };

  for (int p = 1; p < 4; p++) {

    std::function< vecd(vecd) > f_p = [f, p](vecd x) {
      vecd output;
      for (int i = 0; i < dim; i++) { output[i] = f(x, p) * (i + 1); }
      return output;
    };

    Field u = create_field<Family::H1>(mesh, p, dim);
    auto nodes = nodes_for(u, mesh);
    u = forall(f_p, nodes);

    BasisFunction phi(u);

    for (int q = p + 1; q < 5; q++) {

      Domain bdr(boundary_of(mesh), MeshQuadratureRule(q));

      nd::cpu_array<double, 3> x_q = evaluate(mesh.X, bdr);
      nd::cpu_array<double, 3> dx_dxi_q = evaluate(grad(mesh.X), isoparametric(bdr));

      auto g_q = forall(g_vec, x_q);
      auto g_dS_q = forall(g_dS, x_q, dx_dxi_q);

      Residual<Family::H1> r1 = integrate(dot(g_q, phi), bdr);
      Residual<Family::H1> r2 = integrate(dot(g_dS_q, phi), isoparametric(bdr));

      SCOPED_TRACE("p = " + std::to_string(p) + ", q = " + std::to_string(q));
      EXPECT_NEAR(dot(r1, u), scale_factor * answer(p), tolerance);
      EXPECT_NEAR(dot(r2, u), scale_factor * answer(p), tolerance);

    }
  }

}

// ----------------------------------------------------------------------------

// boundary of the unit square: 29/6 + 1/(1 + p), see integrate_residual_H1_bdr_source_test.cpp
std::function<double(vec2,int)> f1([](vec2 x, int p){ return pow(x[0], p) - x[1] + 3.0; });
std::function<double(vec2)> g1([](vec2 x){ return x[1]; });
std::function<double(int)> answer1([](int p){ return 29.0 / 6.0 + 1.0 / (1.0 + p); });

TEST(IntegrateTest, BdrSourceH1vTris) { integrate_bdr_source_test("patch_test_tris.json", f1, g1, answer1, 2.0e-13); }
TEST(IntegrateTest, BdrSourceH1vQuads) { integrate_bdr_source_test("patch_test_quads.json", f1, g1, answer1, 2.0e-13); }
TEST(IntegrateTest, BdrSourceH1vBoth) { integrate_bdr_source_test("patch_test_tris_and_quads.json", f1, g1, answer1, 2.0e-13); }
TEST(IntegrateTest, BdrSourceH1vTrisFine) { integrate_bdr_source_test("unit_square_of_tris_fine.json", f1, g1, answer1, 2.0e-12); }
TEST(IntegrateTest, BdrSourceH1vQuadsFine) { integrate_bdr_source_test("unit_square_of_quads_fine.json", f1, g1, answer1, 2.0e-12); }

// ----------------------------------------------------------------------------

// boundary of the unit cube: 13 + 4/(1 + p)
std::function<double(vec3,int)> f2([](vec3 x, int p){ return pow(x[0], p) - x[1] - x[2] + 3.0; });
std::function<double(vec3)> g2([](vec3 x){ return 1.0; });
std::function<double(int)> answer2([](int p){ return 13.0 + 4.0 / (1.0 + p); });

TEST(IntegrateTest, BdrSourceH1vTets) { integrate_bdr_source_test("patch_test_tets.json", f2, g2, answer2, 1.0e-12); }
TEST(IntegrateTest, BdrSourceH1vHexes) { integrate_bdr_source_test("patch_test_hexes.json", f2, g2, answer2, 1.0e-12); }
TEST(IntegrateTest, BdrSourceH1vTetsFine) { integrate_bdr_source_test("unit_cube_of_tets_fine.json", f2, g2, answer2, 1.0e-11); }
TEST(IntegrateTest, BdrSourceH1vHexesFine) { integrate_bdr_source_test("unit_cube_of_hexes_fine.json", f2, g2, answer2, 1.0e-11); }
