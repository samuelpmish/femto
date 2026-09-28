#include <gtest/gtest.h>

#include "femto/mesh.hpp"

#include "forall.hpp"

TEST(Boundary, PatchTest2D) {
  Mesh<> mesh = Mesh<>::load(FEMTO_MESH_DIR"patch_test_tris_and_quads.json");
  SubMesh bdr = boundary_of(mesh);
  std::cout << mesh.edge.shape[0] << " " << bdr.edge.shape[0] << std::endl;

  auto rule = gauss_legendre_rule(4);
  auto x_bdr = interpolate(mesh.X, bdr, rule);
  auto dx_bdr = gradient_wrt_xi(mesh.X, bdr, rule);

  auto s_q = forall(std::function<double(vec2)>([](vec2 tangent){ return norm(tangent); }), dx_bdr);

  auto r = integrate_source(s_q, bdr, rule, Family::H1, 1);

  print(r);
}

TEST(Boundary, Ball3D) {
  Mesh<> mesh = Mesh<>::load(FEMTO_MESH_DIR"ball.json");
  SubMesh bdr = boundary_of(mesh);
  std::cout << mesh.edge.shape[0] << " " << bdr.edge.shape[0] << std::endl;
}