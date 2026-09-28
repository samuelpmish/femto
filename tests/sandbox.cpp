#include <iomanip>
#include <iostream>
#include <functional>

#include "femto/mesh.hpp"
#include "femto/domain.hpp"
#include "femto/piola_transformations.hpp"
#include "materials/neohookean.hpp"

#include <gtest/gtest.h>

#include "forall.hpp"
#include "fm/types/vec.hpp"
#include "fm/types/matrix.hpp"
#include "containers/ndarray_conversions.hpp"

using namespace femto;

constexpr double lambda = 3.0;
constexpr double mu = 1.0;

template < uint32_t dim >
mat< dim,dim > df_dtheta(const vec<dim> & dtheta_dX) {
  return 3.0 * Identity<dim>();
}

template < uint32_t dim >
mat< dim,dim, mat<dim,dim> > qfunction_jac(const mat<dim,dim> & du_dX) {

  /*

    eps[i][j] = 0.5 * (F[m][i] * F[m][j] - delta[i][j])

    deps[i][j]_dF[k][l] = 0.5 * (dF[m][i]_dF[k][l] * F[m][j] + F[m][i] * dF[m][j]_dF[k][l])
      = 0.5 * ((m==k) * (i == l) * F[m][j] + F[m][i] * (m == k) * (j == l))
      = 0.5 * ((i == l) * F[k][j] + F[k][i] * (j == l))

    deps[m][m]_dF[k][l] = 0.5 * ((m == l) * F[k][m] + F[k][m] * (m == l)) = F[k][l]

    --------------------------------------------------------------------------------

    sigma[i][j] = lambda * eps[m][m] * delta[i][j] + mu * eps[i][j];

    dsigma[i][j]_dF[k][l] = d_dF[k][l](lambda * eps[m][m] * delta[i][j] + mu * eps[i][j])

      = lambda * deps[m][m]_dF[k][l] * delta[i][j] + mu * deps[i][j]_dF[k][l]

      = lambda * F[k][l] * delta[i][j] + 0.5 * mu * ((i == l) * F[k][j] + F[k][i] * (j == l))

  */

  mat< dim,dim, mat<dim,dim> > output{};

  mat<dim,dim> F = Identity<dim>() + du_dX;

  for (uint32_t i = 0; i < dim; i++) {
    for (uint32_t j = 0; j < dim; j++) {
      for (uint32_t k = 0; k < dim; k++) {
        for (uint32_t l = 0; l < dim; l++) {
          output[i][j][k][l] = lambda * F[k][l] * (i == j) + 0.5 * mu * ((i == l) * F[k][j] + F[k][i] * (j == l));
        }
      }
    }
  }

  return output;
}

int main() {

  double tolerance = 1.0e-13;
  constexpr int dim = 3;
  constexpr int p = 1;

#if 0
  std::string filename = "octane_fine.msh";
  Mesh mesh = Mesh<>::load(FEMTO_MESH_DIR + filename);
#else
  //Mesh mesh = Mesh<>::load("/home/sam/code/femto/linear_tets_only.msh");
  //Mesh mesh = Mesh<>::load("/home/sam/code/femto/linear_tets_only.msh");
  Mesh mesh = Mesh< memory::space::cpu >::load(FEMTO_DATA_DIR "/meshes/ball1000000.msh");
#endif

  Field u = create_field<Family::H1>(mesh, p, dim);
  nd::cpu_array<double, 2> nodes = nodes_for(u, mesh);

  int q = 2;
  Domain domain(mesh, MeshQuadratureRule(q));

  NeoHookeanModel material{1.0, 1.0};

  // evaluate f, df_dx directly at each quadrature point
  //auto P_q = forall(std::function([&](mat3 du_dX){ return material(du_dX); }), du_dX_q);

  BasisFunction phi(u);
  nd::cpu_array<double, 3> du_dX_q = evaluate(grad(u), domain);
  auto dsigma_ddudX_q = forall(qfunction_jac<dim>, du_dX_q);
  femto::sparse_matrix K = integrate(dot(grad(phi), dsigma_ddudX_q, grad(phi)), domain);

  std::cout << "writing K_elasticity.mtx" << std::endl;
  export_matrix_market(K, "K_elasticity.mtx");

  Field theta = create_field<Family::H1>(mesh, p);
  BasisFunction psi(theta);

  nd::cpu_array<double, 3> dtheta_dX_q = evaluate(grad(theta), domain);
  auto df_dtheta_q = forall(df_dtheta<dim>, dtheta_dX_q);
  femto::sparse_matrix K2 = integrate(dot(grad(psi), df_dtheta_q, grad(psi)), domain);

  std::cout << "writing K_poisson.mtx" << std::endl;
  export_matrix_market(K2, "K_poisson.mtx");

}
