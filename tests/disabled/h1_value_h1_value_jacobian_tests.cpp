#include <gtest/gtest.h>

#include <iostream>
#include <functional>

#include "common.hpp"

#include "femto/domain.hpp"

#include <gtest/gtest.h>

#include "femto/mesh.hpp"

#include "forall.hpp"

#include "fm/types/vec.hpp"
#include "fm/types/matrix.hpp"
#include "containers/ndarray_conversions.hpp"

//////////
// mass //
//////////
template < int dim >
double f(double u, mat<dim, dim> dX_dxi) {
  return 3.0 * u * det(dX_dxi);
}

template < int dim >
double df(mat<dim,dim> dX_dxi) {
  return 3.0 * det(dX_dxi);
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template < typename return_t, typename dreturn_t, typename arg1_t, typename arg2_t >
void check_qfunc_derivatives(return_t (*f)(arg1_t, arg2_t), dreturn_t (*df)(arg2_t)) {
  double eps = 1.0e-6;
  
  arg1_t arg1 = femto::random(arg1_t{});
  arg1_t darg1 = femto::random(arg1_t{});

  arg2_t arg2 = femto::random(arg2_t{});

  return_t df1 = (f(arg1 + eps * darg1, arg2) - 
                  f(arg1 - eps * darg1, arg2)) / (2 * eps);

  return_t df2 = dot(df(arg2), darg1);

  //std::cout << df1 - df2 << std::endl;
  EXPECT_NEAR(norm(df1 - df2), 0.0, 1.0e-8);
}

TEST(verify_derivative_qfunctions, 2D) { check_qfunc_derivatives(f<2>, df<2>); }
TEST(verify_derivative_qfunctions, 3D) { check_qfunc_derivatives(f<3>, df<3>); }

template < int dim >
void verify_h1_value_h1_value(std::string mesh_filename, uint32_t p1, uint32_t p2) {

  Mesh<> mesh = Mesh<>::load(FEMTO_MESH_DIR + mesh_filename);

  Field u = create_field(mesh, Family::H1, p1);

  auto X = nodes_for(u, mesh);
  u = forall(std::function< double(vec<dim>)> ([](vec<dim> X){ return X[0]; }), X);

  Domain domain(mesh, MeshQuadratureRule(p1 + 1));

  nd::cpu_array<double, 3> dX_dxi_q = evaluate(grad(mesh.X), domain);
  nd::cpu_array<double, 2> u_q = evaluate(u, domain);

  auto s_q = forall(f<dim>, u_q, dX_dxi_q);

  TestFunction psi(FunctionSpace{Family::H1, p1});
  TestFunction phi(FunctionSpace{Family::H1, p2});
  Residual r = integrate(s_q * phi, domain);

  nd::cpu_array<double, 2> rho_q = forall(df<dim>, dX_dxi_q);
  femto::sparse_matrix M = integrate(dot(psi, rho_q, phi), domain);

  //femto::export_matrix_market(M, "M.mtx");

  femto::vector du(u.data.size());
  int count = 0;
  for (int i = 0; i < u.data.shape[0]; i++) {
    for (int j = 0; j < u.data.shape[1]; j++) {
      du[count++] = u.data(i,j);
    }
  }

  femto::vector dr = dot(M, du);

  double error = 0.0;
  count = 0;
  for (int i = 0; i < r.data.shape[0]; i++) {
    for (int j = 0; j < r.data.shape[1]; j++) {
      double diff = (dr[count] - r.data(i,j));
      error += diff * diff;
      count++;
    }
  }
  EXPECT_NEAR(sqrt(error), 0.0, 1.0e-12);
  //std::cout << sqrt(error) << std::endl;

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

#if 0
  for (int i = 0; i < u.data.size(); i++) {
    for (int j = 0; j < u.data.size(); j++) {
      u.data(j, 0) = (i == j);
    }

    nd::cpu_array<double, 2> u_q2 = evaluate(u, domain);
    auto s_q2 = forall(f<dim>, u_q2, dX_dxi_q);
    Residual r2 = integrate(s_q2 * phi, domain);

    for (int j = 0; j < u.data.size(); j++) {
      std::cout << r2.data(j, 0) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  for (int i = 0; i < u.data.size(); i++) {
    for (int j = 0; j < u.data.size(); j++) {
      du[j] = (i == j);
    }

    femto::vector r3 = dot(M, du);

    for (int j = 0; j < u.data.size(); j++) {
      std::cout << r3[j] << " ";
    }
    std::cout << std::endl;
  }
#endif

}

TEST(h1_value_h1_value, tri_1)  { verify_h1_value_h1_value<2>("patch_test_tris.json", 1, 1); }
TEST(h1_value_h1_value, tri_2)  { verify_h1_value_h1_value<2>("patch_test_tris.json", 2, 2); }
TEST(h1_value_h1_value, tri_3)  { verify_h1_value_h1_value<2>("patch_test_tris.json", 3, 3); }
TEST(h1_value_h1_value, tri_mixed_1_3)  { verify_h1_value_h1_value<2>("patch_test_tris.json", 1, 3); }
TEST(h1_value_h1_value, tri_mixed_3_1)  { verify_h1_value_h1_value<2>("patch_test_tris.json", 3, 1); }

//TEST(h1_value_h1_value, quad_1) { verify_h1_value_h1_value<2>("patch_test_quads.json", 1, 1); }
//TEST(h1_value_h1_value, quad_2) { verify_h1_value_h1_value<2>("patch_test_quads.json", 2, 2); }
//TEST(h1_value_h1_value, quad_3) { verify_h1_value_h1_value<2>("patch_test_quads.json", 3, 3); }

TEST(h1_value_h1_value, tet_1)  { verify_h1_value_h1_value<3>("patch_test_tets.json", 1, 1); }
TEST(h1_value_h1_value, tet_2)  { verify_h1_value_h1_value<3>("patch_test_tets.json", 2, 2); }
TEST(h1_value_h1_value, tet_3)  { verify_h1_value_h1_value<3>("patch_test_tets.json", 3, 3); }
TEST(h1_value_h1_value, tet_mixed_1_3)  { verify_h1_value_h1_value<3>("patch_test_tets.json", 1, 3); }
TEST(h1_value_h1_value, tet_mixed_3_1)  { verify_h1_value_h1_value<3>("patch_test_tets.json", 3, 1); }

TEST(h1_value_h1_value, tet_ball_3)  { verify_h1_value_h1_value<3>("ball.json", 3, 3); }

//TEST(h1_value_h1_value, hex_1)  { verify_h1_value_h1_value<3>("patch_test_hexes.json", 1, 1); }
//TEST(h1_value_h1_value, hex_2)  { verify_h1_value_h1_value<3>("patch_test_hexes.json", 2, 2); }
//TEST(h1_value_h1_value, hex_3)  { verify_h1_value_h1_value<3>("patch_test_hexes.json", 3, 3); }

