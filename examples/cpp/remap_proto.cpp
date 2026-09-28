#include "femto/mesh.hpp"
#include "femto/field.hpp"
#include "femto/domain.hpp"
#include "linear_algebra/sparse_direct.hpp"

#include "forall.hpp"

using namespace femto;

int main() {

  constexpr int dim = 2;

  int p_mesh = 2;
  double h = 0.01;
  double r = 1.0;
  vec2 center = {0.0, 0.0};

  Mesh<> source_mesh = Mesh<>::disk(center, r, h, p_mesh, Geometry::Triangle);
  Mesh<> target_mesh = source_mesh;
  
  double phi = 1.7;
  mat2 R = RotationMatrix(phi);
  transform([&](vec2 X) { return dot(R, X); }, target_mesh);

  int p_solution = 2;
  Field source_solution = create_field<Family::H1>(source_mesh, p_solution);
  BasisFunction psi(source_solution);

  Field target_solution = create_field<Family::H1>(target_mesh, p_solution);
  BasisFunction phi(target_solution);

  Domain overlap_domain = overlap(source_mesh, target_mesh, MeshQuadratureRule{3});

  sparse_matrix A = integrate(dot(phi, phi), overlap_domain);

  nd::cpu_array<double, 3> source_q = evaluate(source_solution, overlap_domain);
  vector b = integrate(dot(source_q, phi), overlap_domain);

  auto invA = inv(A);

  target_solution.v() = dot(invA, b);

#if 0
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
#endif

}