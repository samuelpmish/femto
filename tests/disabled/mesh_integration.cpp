#include <gtest/gtest.h>

#include "femto/mesh.hpp"

#include "forall.hpp"

#include "fm/types/vec.hpp"
#include "fm/types/matrix.hpp"
#include "containers/ndarray_conversions.hpp"

#include <iostream>
#include <functional>

#define MESHDIR "../../data/meshes/"

template < int dim >
void source_test(std::string filename,
                 std::function< double(vec<dim>, int) > f,
                 std::function< double(vec<dim>) > g,
                 std::function< double(int) > answer,
                 double tolerance) {

  using vecd = vec<dim>;
  using matd = mat<dim,dim>;

  auto mesh = Mesh<>::load(MESHDIR + filename);

  for (int p = 1; p < 4; p++) {
    for (int q = p + 1; q < 5; q++) {

      std::function< double(vecd) > f_p = [f, p](vecd x) { return f(x,p); };
      QuadratureRule rule = gauss_legendre_rule(q);

      Field u = create_field(mesh, Family::H1, p, 1);
      nd::cpu_array<double, 2> nodes = nodes_for(u, mesh); 
      u = forall(f_p, nodes);

      // evaluate g at each quadrature point
      std::function< double(vecd, matd) > g_detJ = [g](vecd x, matd J) { return g(x) * det(J); };
      auto x_q = interpolate(mesh.X, mesh, rule);
      auto dx_dxi_q = gradient_wrt_xi(mesh.X, mesh, rule);
      auto g_q = forall(g_detJ, x_q, dx_dxi_q);

      nd::cpu_array<double, 2> r = integrate_source(g_q, mesh, rule, u.family, u.degree);

      EXPECT_NEAR(dot(view(r), view(u.data)), answer(p), tolerance);
    }
  }

}

template < int dim >
void flux_test(std::string filename,
               std::function< double(vec<dim>, int) > f,
               std::function< vec<dim>(vec<dim>) > g,
               std::function< double(int) > answer,
               double tolerance) {

  using vecd = vec<dim>;
  using matd = mat<dim,dim>;

  auto mesh = Mesh<>::load(MESHDIR + filename);

  for (int p = 1; p < 4; p++) {
    for (int q = p + 1; q < 5; q++) {

      std::function< double(vecd) > f_p = [f, p](vecd x) { return f(x, p); };
      QuadratureRule rule = gauss_legendre_rule(q);

      Field u = create_field(mesh, Family::H1, p, 1);
      nd::cpu_array<double, 2> nodes = nodes_for(u, mesh); 
      u = forall(f_p, nodes);

      // evaluate g at each quadrature point
      std::function< mat<1,dim>(vecd, matd) > g_inv_JT_detJ = [g](vecd x, matd J) { 
        return mat<1,dim>{dot(g(x), inv(transpose(J)))} * det(J); 
      };
      auto x_q = interpolate(mesh.X, mesh, rule);
      auto dx_dxi_q = gradient_wrt_xi(mesh.X, mesh, rule);

      auto g_q = forall(g_inv_JT_detJ, x_q, dx_dxi_q);

      nd::cpu_array<double, 2> r = integrate_flux(g_q, mesh, rule, u.family, u.degree);

      EXPECT_NEAR(dot(view(r), view(u.data)), answer(p), tolerance);
    }
  }

}

// ------------------------------------------------------

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

// ------------------------------------------------------

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


// ------------------------------------------------------

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
std::function<double(vec2,int)> f3([](vec2 x, int p){ return pow(x[0], p) - x[1] + 3.0; });
std::function<vec2(vec2)> g3([](vec2 x){ return vec2{x[1], -x[0]}; });
std::function<double(int)> answer3([](int){ return 1.0; });

TEST(FluxTest, PatchTest2DTris) { flux_test("patch_test_tris.json", f3, g3, answer3, 3.0e-15); }
TEST(FluxTest, PatchTest2DQuads) { flux_test("patch_test_quads.json", f3, g3, answer3, 1.0e-15); }
TEST(FluxTest, PatchTest2DBoth) { flux_test("patch_test_tris_and_quads.json", f3, g3, answer3, 1.0e-15); }

// ------------------------------------------------------

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
std::function<double(vec3,int)> f4([](vec3 x, int p){ return pow(x[0], p) - x[1] - 2 * x[2] + 3.0; });
std::function<vec3(vec3)> g4([](vec3 x){ return vec3{x[2], x[0], -x[1]}; });
std::function<double(int)> answer4([](int){ return 1.0; });

TEST(FluxTest, PatchTest3DTets) { flux_test("patch_test_tets.json", f4, g4, answer4, 4.0e-15); }
TEST(FluxTest, PatchTest3DHexes) { flux_test("patch_test_hexes.json", f4, g4, answer4, 8.0e-16); }

// ------------------------------------------------------
