#include <gtest/gtest.h>

#include "femto/mesh.hpp"

#include "forall.hpp"

#include "fm/types/vec.hpp"
#include "fm/types/matrix.hpp"
#include "containers/ndarray_conversions.hpp"

#include <iostream>
#include <functional>

static constexpr double k = 3.0;

template < int dim >
void h1_patch_test(std::function< double(vec<dim>) > f, 
std::function<vec<dim>(vec<dim>) > grad_f, std::string mesh_file, int p, int q, double tolerance) {

  using vecd = vec<dim>;
  using matd = mat<dim,dim>;

  Mesh<> mesh = Mesh<>::load(FEMTO_MESH_DIR + mesh_file);
  SubMesh boundary = boundary_of(mesh);

  QuadratureRule rule = gauss_legendre_rule(q);

  Field u = create_field(mesh, Family::H1, p);
  u = forall(f, nodes_for(u, mesh));

  auto du_dxi_q = gradient_wrt_xi(u, mesh, rule);
  auto dx_dxi_q = gradient_wrt_xi(mesh.X, mesh, rule);

  auto heat_flux_q = forall(std::function< mat<1,dim>(vecd, matd) >([](vecd du_dxi, matd dx_dxi) {
    matd dxi_dx = inv(dx_dxi);
    vecd du_dx = dot(du_dxi, dxi_dx);
    vecd heat_flux = -k * du_dx;
    return mat<1,dim>{dot(heat_flux, transpose(dxi_dx)) * det(dx_dxi)};
  }), du_dxi_q, dx_dxi_q);

  auto bdr_x_q = interpolate(mesh.X, boundary, rule);
  auto bdr_dx_dxi_q = gradient_wrt_xi(mesh.X, boundary, rule);

  auto bdr_source_q = forall(std::function< double(vecd, mat<dim,dim-1>) >([=](vecd x, mat<dim,dim-1> dx_dxi) {
    vecd normal = cross(dx_dxi);
    vecd heat_flux = -k * grad_f(x);
    return dot(heat_flux, normal);
  }), bdr_x_q, bdr_dx_dxi_q);

  auto r1 = integrate_flux(heat_flux_q, mesh, rule, u.family, u.degree);
  auto r2 = integrate_source(bdr_source_q, boundary, rule, u.family, u.degree);
  auto error = r2 - r1;

  double relative_error = norm(error) / norm(r1);

  EXPECT_NEAR(relative_error, 0, tolerance);

}

std::function< double(vec2) > f1 = [](vec2 x){ return x[0] * x[1]; };
std::function< vec2(vec2) > grad_f1 = [](vec2 x){ return vec2{x[1], x[0]}; };

TEST(PatchTest2D, OneQuadXY) { h1_patch_test(f1, grad_f1, "one_quad.json", 1, 2, 1.0e-16); }
TEST(PatchTest2D, TrisXY) { h1_patch_test(f1, grad_f1, "patch_test_tris.json", 2, 3, 4.0e-15); }
TEST(PatchTest2D, QuadsXY) { h1_patch_test(f1, grad_f1, "patch_test_quads.json", 2, 3, 2.0e-15); }

std::function< double(vec2) > f2 = [](vec2 x){ return x[0] + 2.0 * x[1]; };
std::function< vec2(vec2) > grad_f2 = [](vec2 x){ return vec2{1.0, 2.0}; };

TEST(PatchTest2D, TrisP1) { h1_patch_test(f2, grad_f2, "patch_test_tris.json", 1, 2, 3.0e-15); }
TEST(PatchTest2D, QuadsP1) { h1_patch_test(f2, grad_f2, "patch_test_quads.json", 1, 2, 6.0e-16); }

TEST(PatchTest2D, TrisP2) { h1_patch_test(f2, grad_f2, "patch_test_tris.json", 2, 3, 3.5e-15); }
TEST(PatchTest2D, QuadsP2) { h1_patch_test(f2, grad_f2, "patch_test_quads.json", 2, 3, 2.5e-15); }

TEST(PatchTest2D, TrisP3) { h1_patch_test(f2, grad_f2, "patch_test_tris.json", 3, 4, 9.0e-15); }
TEST(PatchTest2D, QuadsP3) { h1_patch_test(f2, grad_f2, "patch_test_quads.json", 3, 4, 1.0e-14); }

std::function< double(vec3) > f3 = [](vec3 x){ return x[0] + 2.0 * x[1] + 3.0 * x[2]; };
std::function< vec3(vec3) > grad_f3 = [](vec3 x){ return vec3{1.0, 2.0, 3.0}; };

TEST(PatchTest3D, TetsP1) { h1_patch_test(f3, grad_f3, "patch_test_tets.json", 1, 2, 3.0e-15); }
TEST(PatchTest3D, HexesP1) { h1_patch_test(f3, grad_f3, "patch_test_hexes.json", 1, 2, 8.0e-16); }

TEST(PatchTest3D, TetsP2) { h1_patch_test(f3, grad_f3, "patch_test_tets.json", 2, 3, 4.0e-15); }
TEST(PatchTest3D, HexesP2) { h1_patch_test(f3, grad_f3, "patch_test_hexes.json", 2, 3, 3.0e-15); }

TEST(PatchTest3D, TetsP3) { h1_patch_test(f3, grad_f3, "patch_test_tets.json", 3, 4, 1.0e-14); }
TEST(PatchTest3D, HexesP3) { h1_patch_test(f3, grad_f3, "patch_test_hexes.json", 3, 4, 1.0e-14); }
