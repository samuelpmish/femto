#include "femto/mesh.hpp"
#include "femto/field.hpp"
#include "femto/domain.hpp"

#include "forall.hpp"

#include "misc/enzyme_wrapper.hpp"

#include "materials/linear_elasticity.hpp"

#include <functional>

using namespace femto;

LinearElasticModel material{1.0, 1.0};

tuple<vec2, mat2> qfunction(vec2 X, mat2 dX_dxi, vec2 u, mat2 du_dxi) {
  double r = X[0];
  mat2 dxi_dX = inv(dX_dxi);
  mat2 du_dX = dot(du_dxi, dxi_dX);

  mat3 sigma = material(mat3{
    vec3{du_dX[0][0], du_dX[0][1], 0.0},
    vec3{du_dX[1][0], du_dX[1][1], 0.0},
    vec3{        0.0,         0.0, u[0] / r}
  });

  vec2 body_force{sigma[2][2] / r, 0.0};

  mat2 stress{
    vec2{sigma[0][0], sigma[0][1]},
    vec2{sigma[1][0], sigma[1][1]}
  };

  double J = 6.28 * r * det(dX_dxi);

  return tuple{body_force * J, dot(stress, dxi_dX) * J};
}

int main() {

  constexpr int dim = 2;

  std::string filename = "wrench.json";

  Mesh<> mesh = Mesh<>::load(FEMTO_MESH_DIR + filename);

  int polynomial_order = 2;
  int num_components = dim;
  Field u = create_field<Family::H1>(mesh, polynomial_order, num_components);

  Domain domain(mesh, MeshQuadratureRule(polynomial_order + 1));

  nd::cpu_array<double, 3> X_q = evaluate(mesh.nodes, domain);
  nd::cpu_array<double, 3> dX_dxi_q = evaluate(grad_wrt_xi(mesh.nodes), domain);
  nd::cpu_array<double, 3> u_q = evaluate(u, domain);
  nd::cpu_array<double, 3> du_dxi_q = evaluate(grad_wrt_xi(u), domain);

  auto s_q = forall(qfunction, X_q, dX_dxi_q, u_q, du_dxi_q);

  //auto grad_s_q = forall(+femto::jacobian<qfunction>(), du_dxi_q, dx_dxi_q);

  BasisFunction phi(u);
  Residual<Family::H1> r = integrate(dot(s_q, grad(phi)), domain);

  //nd::cpu_array<double, 5> tmp{};
  //femto::sparse_matrix K = integrate(dot(grad(phi), tmp, grad(phi)), domain);

  //grads_q(0, 0) * femto::tuple{dudxi, dxdxi};

}