#include "femto/mesh.hpp"
#include "femto/field.hpp"
#include "femto/domain.hpp"
#include "materials/neohookean.hpp"
#include "materials/plane_strain.hpp"

#include "linear_algebra/krylov.hpp"

#include "forall.hpp"

#include "misc/binary_io.hpp"
#include "misc/enzyme_wrapper.hpp"

#include <functional>

using namespace femto;

template < uint32_t dim >
struct LinearDiffusionModel {
  __host__ __device__ vec<dim> operator()(vec<dim> du_dxi, mat<dim,dim> dX_dxi) const {
    mat<dim,dim> dxi_dX = inv(dX_dxi);
    vec<dim> du_dX = dot(du_dxi, dxi_dX);
    vec<dim> flux = -k * du_dX;
    return dot(flux, transpose(dxi_dX)) * det(dX_dxi);
  }

  double k;
};

template < int dim >
void diffusion_perf_test(const Mesh<> & mesh, int p) {

  std::cout << "  p = " << p << std::endl;

  LinearDiffusionModel<dim> material{1.0};

  Field u = create_field<Family::H1>(mesh, p);

  BasisFunction phi(u);

  Domain domain(mesh, MeshQuadratureRule(p + 1));

  for (int k = 0; k < 3; k++) {
    nd::cpu_array<double, 3> dX_dxi_q = evaluate(grad(mesh.X), isoparametric(domain));
    nd::cpu_array<double, 3> du_dxi_q = evaluate(grad(u), isoparametric(domain));
    //nd::cpu_array<double, 2> f_q = forall(material, du_dxi_q, dX_dxi_q);
    nd::cpu_array<double, 2> f_q = forall(material, du_dxi_q, dX_dxi_q);
    Residual<Family::H1> r = integrate(dot(f_q, grad(phi)), isoparametric(domain));
  }
  
}

int main() {
  int linear = 1;
  int quadratic = 2;
  int cubic = 3;

  {
    std::cout << "triangle tests" << std::endl;
    auto mesh =  Mesh<>::load("/home/sam/Dropbox/meshes/disk1000000.msh");
    diffusion_perf_test<2>(mesh, linear);
    diffusion_perf_test<2>(mesh, quadratic);
    diffusion_perf_test<2>(mesh, cubic);
  }

  {
    std::cout << "quadrilateral tests" << std::endl;
    auto mesh = Mesh<>::cuboid({1000, 1000}, vec2{1.0, 1.0});
    diffusion_perf_test<2>(mesh, linear);
    diffusion_perf_test<2>(mesh, quadratic);
    diffusion_perf_test<2>(mesh, cubic);
  }

  {
    std::cout << "tetrahedron tests" << std::endl;
    auto mesh = Mesh<>::load("/home/sam/Dropbox/meshes/ball1000000.msh");
    diffusion_perf_test<3>(mesh, linear);
    diffusion_perf_test<3>(mesh, quadratic);
    diffusion_perf_test<3>(mesh, cubic);
  }

  {
    std::cout << "hexahedron tests" << std::endl;
    auto mesh = Mesh<>::cuboid({100, 100, 100}, vec3{1.0, 1.0, 1.0});
    diffusion_perf_test<3>(mesh, linear);
    diffusion_perf_test<3>(mesh, quadratic);
    diffusion_perf_test<3>(mesh, cubic);
  }

}