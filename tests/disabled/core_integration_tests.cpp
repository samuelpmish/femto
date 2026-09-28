#include <gtest/gtest.h>

#include <iomanip>
#include <iostream>
#include <functional>

#include "femto/domain.hpp"

#include <gtest/gtest.h>

#include "femto/mesh.hpp"

#include "forall.hpp"

#include "fm/types/vec.hpp"
#include "fm/types/matrix.hpp"
#include "containers/ndarray_conversions.hpp"

#if 0
TEST(IntegrationTest, DirectIntegration) {

  Mesh<> m = Mesh<>::load(FEMTO_MESH_DIR"ball.json");

  Domain domain(m, MeshQuadratureRule{3});

  double volume = integrate([](vec3 x) { return 1.0; }, domain);
  std::cout << volume << std::endl;

  mat3 moment_of_inertia = integrate([](vec3 x) { return outer(x, x); }, domain);
  std::cout << moment_of_inertia << std::endl;

  auto body_force = [](vec3 x) { return x; };

  TestFunction phi(m.nodes);

  Residual r = integrate(body_force * phi, domain);

}
#endif

// ----------------------------------------------------------------------------

template < typename vecd >
void source_test(std::string filename,
                 std::function< double(vecd, int) > f,
                 std::function< double(vecd) > g,
                 std::function< double(int) > answer,
                 double tolerance) {

  using matd = decltype(outer(vecd{}, vecd{}));

  auto mesh = Mesh<>::load(FEMTO_MESH_DIR + filename);

  std::cout << std::setprecision(15);

  // evaluate g at each quadrature point
  std::function< double(vecd, matd) > g_detJ = [g](vecd x, matd J) { 
    return g(x) * det(J); 
  };

  for (int p = 1; p < 4; p++) {

    std::function< double(vecd) > f_p = [f, p](vecd x) { return f(x,p); };

    Field u = create_field(mesh, Family::H1, p, 1);
    nd::cpu_array<double, 2> nodes = nodes_for(u, mesh); 
    u = forall(f_p, nodes);

    BasisFunction phi(u);

    for (int q = p + 1; q < 5; q++) {

      Domain domain(mesh, MeshQuadratureRule(q));

      auto x_q = evaluate(mesh.X, domain);
      auto dx_dxi_q = evaluate(grad_wrt_xi(mesh.X), domain);

      auto g_q = forall(g_detJ, x_q, dx_dxi_q);

      Residual r = integrate(g_q * phi, domain);

      EXPECT_NEAR(dot(r, u), answer(p), tolerance);

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
std::function<double(double,int)> f0([](double x, int p){ return pow(x, p) + 3.0; });
std::function<double(double)> g0([](double x){ return x; });
std::function<double(int)> answer0([](int p){ return 1.5 + 1.0 / (2 + p); });

TEST(SourceTest, PatchTest1D) { source_test("patch_test_edges.json", f0, g0, answer0, 3.0e-2); }

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
std::function<double(vec2,int)> f1([](vec2 x, int p){ return pow(x[0], p) - x[1] + 3.0; });
std::function<double(vec2)> g1([](vec2 x){ return x[1]; });
std::function<double(int)> answer1([](int p){ return (7.0/6.0) + 1.0 /(2.0 + 2.0 * p); });

TEST(SourceTest, PatchTest2DTris) { source_test("patch_test_tris.json", f1, g1, answer1, 3.0e-15); }
TEST(SourceTest, PatchTest2DQuads) { source_test("patch_test_quads.json", f1, g1, answer1, 1.0e-15); }
TEST(SourceTest, PatchTest2DBoth) { source_test("patch_test_tris_and_quads.json", f1, g1, answer1, 1.0e-15); }

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
std::function<double(vec3,int)> f2([](vec3 x, int p){ return pow(x[0], p) - x[1] - x[2] + 3.0; });
std::function<double(vec3)> g2([](vec3 x){ return 1.0; });
std::function<double(int)> answer2([](int p){ return 2.0 + 1.0 / (1.0 + p); });

TEST(SourceTest, PatchTest3DTets) { source_test("patch_test_tets.json", f2, g2, answer2, 7.0e-15); }
TEST(SourceTest, PatchTest3DHexes) { source_test("patch_test_hexes.json", f2, g2, answer2, 1.0e-15); }

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

template < typename vecd >
void flux_test(std::string filename,
               std::function< double(vecd, int) > f,
               std::function< vecd(vecd) > g,
               std::function< double(int) > answer,
               double tolerance) {

  using matd = decltype(outer(vecd{}, vecd{}));

  auto mesh = Mesh<>::load(FEMTO_MESH_DIR + filename);

  // evaluate g at each quadrature point
  std::function< vecd(vecd, matd) > qf = [g](vecd x, matd J) { 
    return dot(inv(J), g(x)) * det(J); 
  };

  for (int p = 1; p < 4; p++) {

    std::function< double(vecd) > f_p = [f, p](vecd x) { return f(x, p); };

    Field u = create_field(mesh, Family::H1, p, 1);
    nd::cpu_array<double, 2> nodes = nodes_for(u, mesh); 
    u = forall(f_p, nodes);

    BasisFunction phi(u);

    for (int q = p + 1; q < 5; q++) {

      Domain domain(mesh, MeshQuadratureRule(q));

      auto x_q = evaluate(mesh.X, domain);
      auto dx_dxi_q = evaluate(grad_wrt_xi(mesh.X), domain);

      auto g_q = forall(qf, x_q, dx_dxi_q);

      Residual r = integrate(dot(g_q, grad_wrt_xi(phi)), domain);

      EXPECT_NEAR(dot(r, u), answer(p), tolerance);

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

std::function<double(double,int)> f3([](double x, int p){ return pow(x, p); });
std::function<double(double)> g3([](double x){ return x; });
std::function<double(int)> answer3([](int p){ return double(p) / double(1 + p); });

TEST(FluxTest, PatchTest1DEdges) { flux_test("patch_test_edges.json", f3, g3, answer3, 3.0e-15); }

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
std::function<double(vec2,int)> f4([](vec2 x, int p){ return pow(x[0], p) - x[1] + 3.0; });
std::function<vec2(vec2)> g4([](vec2 x){ return vec2{x[1], -x[0]}; });
std::function<double(int)> answer4([](int){ return 1.0; });

TEST(FluxTest, PatchTest2DTris) { flux_test("patch_test_tris.json", f4, g4, answer4, 3.0e-15); }
TEST(FluxTest, PatchTest2DQuads) { flux_test("patch_test_quads.json", f4, g4, answer4, 1.5e-15); }
TEST(FluxTest, PatchTest2DBoth) { flux_test("patch_test_tris_and_quads.json", f4, g4, answer4, 1.5e-15); }

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
std::function<double(vec3,int)> f5([](vec3 x, int p){ return pow(x[0], p) - x[1] - 2 * x[2] + 3.0; });
std::function<vec3(vec3)> g5([](vec3 x){ return vec3{x[2], x[0], -x[1]}; });
std::function<double(int)> answer5([](int){ return 1.0; });

TEST(FluxTest, PatchTest3DTets) { flux_test("patch_test_tets.json", f5, g5, answer5, 4.0e-15); }
TEST(FluxTest, PatchTest3DHexes) { flux_test("patch_test_hexes.json", f5, g5, answer5, 1.0e-15); }

// ----------------------------------------------------------------------------
