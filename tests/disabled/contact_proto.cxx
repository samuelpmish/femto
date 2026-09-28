#include <gtest/gtest.h>

#include "femto/mesh.hpp"

#include "forall.hpp"

#include "fm/types/vec.hpp"
#include "fm/types/matrix.hpp"
#include "containers/ndarray_conversions.hpp"

#include <iostream>
#include <functional>

int main() {

  using vecd = vec<dim>;
  using matd = mat<dim,dim>;

  std::string mesh_file = "";
  Mesh<> mesh = Mesh<>::load("../data/meshes/" + mesh_file);

  Field<> & X = mesh.X;
  Field u = create_field(mesh, Family::H1, p);
  u = forall(f, nodes_for(u, mesh));

  QuadratureRule rule = gauss_legendre_rule(q);
  auto du_dxi_q = gradient_wrt_xi(u, mesh, rule);
  auto dX_dxi_q = gradient_wrt_xi(X, mesh, rule);

  std::function< mat3(mat3, mat3) > calculate_stress([](mat3 du_dxi, mat3 dX_dxi) {
    mat3 dxi_dX = inv(dX_dxi);
    mat3 du_dX = dot(du_dxi, dxi_dX);
    mat3 stress = -k * du_dX;
    return dot(stress, transpose(dxi_dX)) * det(dX_dxi);
  });

  auto stress_q = forall(calculate_stress, du_dxi_q, dx_dxi_q);

  Mesh<> contact_interface = find_contact(mesh, u, ContactMethod::MORTAR);

  QuadratureRule rule = gauss_legendre_rule(q);
  auto x_q = interpolate(X + u, contact_interface, rule);
  auto n_q = calculate_normal(X + u, contact_interface, rule);

  double penalty = 1e5;
  std::function< vec3(vec3, vec3) > calculate_contact_forces([](vec3 x1, vec3 x2, vec3 n) {
    return (penalty * dot(x2 - x1, n)) * n;
  });

  auto f_q = forall(calculate_contact_forces, x_q, n_q);

  auto r = integrate(f_q * phi, contact_interface, rule) +
           integrate(dot(stress_q, dphi_dx), mesh, rule);

}