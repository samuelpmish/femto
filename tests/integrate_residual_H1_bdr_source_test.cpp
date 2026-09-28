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

// like integrate_residual_H1_source_tests, but the source is applied on the
// boundary of the mesh: dot(r, u) must be the boundary integral of f * g
template < typename vecd >
void integrate_bdr_source_test(std::string filename,
                               std::function< double(vecd, int) > f,
                               std::function< double(vecd) > g,
                               std::function< double(int) > answer,
                               double tolerance) {

  constexpr uint32_t sdim = dimension(vecd{});
  using jacd = mat<sdim, sdim - 1>;

  auto mesh = Mesh<>::load(FEMTO_MESH_DIR + filename);

  // in parent coordinates the integrand carries the facet measure explicitly
  std::function< double(vecd, jacd) > g_dS = [g](vecd x, jacd J) {
    return g(x) * facet_measure(J);
  };

  for (int p = 1; p < 4; p++) {

    std::function< double(vecd) > f_p = [f, p](vecd x) { return f(x, p); };

    Field u = create_field<Family::H1>(mesh, p, 1);
    auto nodes = nodes_for(u, mesh);
    u = forall(f_p, nodes);

    BasisFunction phi(u);

    for (int q = p + 1; q < 5; q++) {

      Domain bdr(boundary_of(mesh), MeshQuadratureRule(q));

      nd::cpu_array<double, 3> x_q = evaluate(mesh.X, bdr);
      nd::cpu_array<double, 3> dx_dxi_q = evaluate(grad(mesh.X), isoparametric(bdr));

      auto g_q = forall(g, x_q);
      auto g_dS_q = forall(g_dS, x_q, dx_dxi_q);

      Residual<Family::H1> r1 = integrate(g_q * phi, bdr);
      Residual<Family::H1> r2 = integrate(g_dS_q * phi, isoparametric(bdr));

      SCOPED_TRACE("p = " + std::to_string(p) + ", q = " + std::to_string(q));
      EXPECT_NEAR(dot(r1, u), answer(p), tolerance);
      EXPECT_NEAR(dot(r2, u), answer(p), tolerance);

    }
  }

}

// ----------------------------------------------------------------------------

/*
  In[] :=
    f[{x_, y_}] := x^p - y + 3;
    g[{x_, y_}] := y;
    Integrate[f[{x, y}] g[{x, y}], {x, y} \[Element] RegionBoundary[Rectangle[{0, 0}, {1, 1}]]]

  --------------------

  Out[] :=
    29/6 + 1/(1 + p)
*/
std::function<double(vec2,int)> f1([](vec2 x, int p){ return pow(x[0], p) - x[1] + 3.0; });
std::function<double(vec2)> g1([](vec2 x){ return x[1]; });
std::function<double(int)> answer1([](int p){ return 29.0 / 6.0 + 1.0 / (1.0 + p); });

TEST(IntegrateTest, BdrSourceH1Tris) { integrate_bdr_source_test("patch_test_tris.json", f1, g1, answer1, 5.0e-14); }
TEST(IntegrateTest, BdrSourceH1Quads) { integrate_bdr_source_test("patch_test_quads.json", f1, g1, answer1, 5.0e-14); }
TEST(IntegrateTest, BdrSourceH1Both) { integrate_bdr_source_test("patch_test_tris_and_quads.json", f1, g1, answer1, 5.0e-14); }
TEST(IntegrateTest, BdrSourceH1TrisFine) { integrate_bdr_source_test("unit_square_of_tris_fine.json", f1, g1, answer1, 5.0e-13); }
TEST(IntegrateTest, BdrSourceH1QuadsFine) { integrate_bdr_source_test("unit_square_of_quads_fine.json", f1, g1, answer1, 5.0e-13); }

// ----------------------------------------------------------------------------

/*
  In[] :=
    f[{x_, y_, z_}] := x^p - y - z + 3;
    g[{x_, y_, z_}] := 1;
    Integrate[f[{x, y, z}] g[{x, y, z}], {x, y, z} \[Element] RegionBoundary[Cuboid[{0, 0, 0}, {1, 1, 1}]]]

  --------------------

  Out[] :=
    13 + 4/(1 + p)
*/
std::function<double(vec3,int)> f2([](vec3 x, int p){ return pow(x[0], p) - x[1] - x[2] + 3.0; });
std::function<double(vec3)> g2([](vec3 x){ return 1.0; });
std::function<double(int)> answer2([](int p){ return 13.0 + 4.0 / (1.0 + p); });

TEST(IntegrateTest, BdrSourceH1Tets) { integrate_bdr_source_test("patch_test_tets.json", f2, g2, answer2, 1.0e-13); }
TEST(IntegrateTest, BdrSourceH1Hexes) { integrate_bdr_source_test("patch_test_hexes.json", f2, g2, answer2, 1.0e-13); }
TEST(IntegrateTest, BdrSourceH1TetsFine) { integrate_bdr_source_test("unit_cube_of_tets_fine.json", f2, g2, answer2, 1.0e-12); }
TEST(IntegrateTest, BdrSourceH1HexesFine) { integrate_bdr_source_test("unit_cube_of_hexes_fine.json", f2, g2, answer2, 1.0e-12); }
