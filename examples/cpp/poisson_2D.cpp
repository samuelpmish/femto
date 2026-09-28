#include "femto/mesh.hpp"
#include "femto/field.hpp"
#include "femto/domain.hpp"

#include "forall.hpp"

#include "misc/enzyme_wrapper.hpp"

#include "femto/assert.hpp"
#include <functional>

using namespace femto;

vec2 qfunction(const vec2 & du_dX) {
  vec2 heat_flux = 3.0 * du_dX;
  return heat_flux;
}

mat2 qfunction_derivative(const mat2 & du_dX) {
  return 3.0 * Identity<2>();
}

vec2 qfunction_xi(const vec2 & du_dxi, const mat2 & dX_dxi) {
  mat2 dxi_dX = inv(dX_dxi);
  vec2 du_dX = dot(du_dxi, dxi_dX);
  vec2 heat_flux = 3.0 * du_dX;
  return dot(heat_flux, transpose(dxi_dX)) * det(dX_dxi);
}

mat2 qfunction_xi_derivative(const vec2 & du_dxi, const mat2 & dX_dxi) {
  mat2 dxi_dX = inv(dX_dxi);
  return dot(dxi_dX, transpose(dxi_dX)) * (3.0 * det(dX_dxi));
}

void check_qfunc_derivative() {
  double eps = 1.0e-6;
  vec2 e1 = {1.0, 0.0};
  vec2 e2 = {0.0, 1.0};

  vec2 du_dxi = {1.0, 2.0};
  mat2 dX_dxi = {{{1.0, 2.0}, {3.0, 4.0}}};

  mat2 df1 = transpose(mat2{
    qfunction_xi(du_dxi + eps * e1, dX_dxi) - qfunction_xi(du_dxi - eps * e1, dX_dxi),
    qfunction_xi(du_dxi + eps * e2, dX_dxi) - qfunction_xi(du_dxi - eps * e2, dX_dxi)
  } / (2 * eps));

  mat2 df2 = qfunction_xi_derivative(du_dxi, dX_dxi);

  std::cout << df1 - df2 << std::endl;
}

int main() {

  check_qfunc_derivative();

  constexpr int dim = 2;

  //std::string filename = "wrench.json";
  //std::string filename = "one_tri.json";
  //std::string filename = "one_quad.json";
  //std::string filename = "patch_test_tris.json";
  std::string filename = "patch_test_quads.json";

  Mesh<> mesh = Mesh<>::load(FEMTO_MESH_DIR + filename);

  int polynomial_order = 1;
  Field u = create_field<Family::H1>(mesh, polynomial_order);
  BasisFunction phi(u);

  auto X = nodes_for(u, mesh);
  u = forall(std::function< double(vec2)> ([](vec2 X){ return X[0]; }), X);

  Domain domain(mesh, MeshQuadratureRule(polynomial_order + 1));

  #if 1
  nd::cpu_array<double, 3> du_dX_q = evaluate(grad(u), domain);
  nd::cpu_array<double, 2> f_q = forall(qfunction, du_dX_q);
  Residual<Family::H1> r = integrate(dot(f_q, grad(phi)), domain);

  nd::cpu_array<double, 3> k_q = forall(qfunction_derivative, du_dX_q);
  sparse_matrix K = integrate(dot(grad(phi), k_q, grad(phi)), domain);
  #else
  nd::cpu_array<double, 3> dX_dxi_q = evaluate(grad(mesh.X), isoparametric(domain));
  nd::cpu_array<double, 3> du_dxi_q = evaluate(grad(u), isoparametric(domain));

  nd::cpu_array<double, 2> f_q = forall(qfunction_xi, du_dxi_q, dX_dxi_q);
  Residual<Family::H1> r = integrate(dot(f_q, grad(phi)), domain);

  nd::cpu_array<double, 3> k_q = forall(qfunction_xi_derivative, du_dxi_q, dX_dxi_q);
  sparse_matrix K = integrate(dot(grad(phi), k_q, grad(phi)), isoparametric(domain));
  #endif

}