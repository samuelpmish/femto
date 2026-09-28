#include "femto/mesh.hpp"
#include "femto/field.hpp"
#include "femto/domain.hpp"

#include "linear_algebra/krylov.hpp"

#include "forall.hpp"

#include "misc/binary_io.hpp"
#include "misc/enzyme_wrapper.hpp"

#include <functional>

using namespace femto;

static constexpr int dim = 3;

struct NeoHookeanModel {
  __host__ __device__ mat3 operator()(mat3 du_dX) const {
    mat3 F = Identity<3>() + du_dX;
    double J = det(F);
    mat3 invFT = transpose(inv(F));
    return (lambda * log(J) - mu) * invFT + mu * F;
  }

  double lambda, mu;
};

struct NeoHookeanModelIsoparametric {
  __host__ __device__ mat3 operator()(mat3 du_dxi, mat3 dX_dxi) const {
    mat3 dxi_dX = inv(dX_dxi);
    mat3 du_dX = dot(du_dxi, dxi_dX);
    mat3 F = Identity<3>() + du_dX;
    double J = det(F);
    mat3 invFT = transpose(inv(F));
    mat3 P = (lambda * log(J) - mu) * invFT + mu * F;
    return dot(P, transpose(dxi_dX)) * det(dX_dxi);
  }

  double lambda, mu;
};

int main() {

  //auto mesh = Mesh< memory::space::cpu >::load("/home/sam/code/femto/linear_tets_only.msh");
  auto mesh = Mesh< memory::space::cpu >::load(FEMTO_DATA_DIR "/meshes/ball1000000.msh");

  uint32_t p = 2; // quadratic
  std::cout << "  p = " << 2 << std::endl;

  double lambda = 10.0;
  double mu = 10.0;
  NeoHookeanModel material{lambda, mu};
  NeoHookeanModelIsoparametric isoparametric_material{lambda, mu};

  Field<Family::H1, memory::space::cpu> u = create_field<Family::H1>(mesh, p, dim);

  BasisFunction phi(u);

  Domain< memory::space::cpu > domain(mesh, MeshQuadratureRule(p));

#if 1
  std::cout << "mesh has:" << std::endl;
  std::cout << "  " << mesh.vert.shape[0] << " vertices" << std::endl;
  std::cout << "  " << mesh.edge.shape[0] << " edges" << std::endl;
  std::cout << "  " << mesh.tri.shape[0] << " triangles" << std::endl;
  std::cout << "  " << mesh.quad.shape[0] << " quadrilaterals" << std::endl;
  std::cout << "  " << mesh.tet.shape[0] << " tetrahedra" << std::endl;
  std::cout << "  " << mesh.hex.shape[0] << " hexahedra" << std::endl;
  std::cout << "  " << total(domain.num_qpts) << " quadrature points" << std::endl;

  //sparse_matrix A = blank_sparse_matrix(phi, phi, domain);
  //export_matrix_market(A, "K_elasticity.mtx");
  //std::cout << "(vector) Stiffness matrix has " << A.nrows << " rows and " << A.nnz << " nonzeros" << std::endl;

  //BasisFunction<Family::H1> psi(p);
  //A = blank_sparse_matrix(psi, psi, domain);
  //export_matrix_market(A, "K_poisson.mtx");
  //std::cout << "(scalar) Stiffness matrix has " << A.nrows << " rows and " << A.nnz << " nonzeros" << std::endl;
#endif

  {
    for (int k = 0; k < 3; k++) {
      nd::array<double, 3, memory::space::cpu > du_dX_q = evaluate(grad(u), domain);
      nd::array<double, 3, memory::space::cpu > P_q = forall(material, du_dX_q);
      Residual<Family::H1, memory::space::cpu> r = integrate(dot(P_q, grad(phi)), domain);
    }
  }

  {
    for (int k = 0; k < 3; k++) {
      nd::array<double, 3, memory::space::cpu> dX_dxi_q = evaluate(grad(mesh.X), isoparametric(domain));
      nd::array<double, 3, memory::space::cpu> du_dxi_q = evaluate(grad(u), isoparametric(domain));
      nd::array<double, 3, memory::space::cpu> P_q = forall(isoparametric_material, du_dxi_q, dX_dxi_q);
      Residual<Family::H1, memory::space::cpu> r = integrate(dot(P_q, grad(phi)), isoparametric(domain));
    }
  }

}
