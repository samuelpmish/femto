// sweep the elements-per-block launch parameter of the GPU `evaluate`
// kernels, timing an isoparametric gradient evaluation for scalar and
// vector H1 fields on 2D and 3D meshes
//
// run directly (or under nsys/ncu):  ./tests/cuda_evaluation_tuning

#include "femto/mesh.hpp"
#include "femto/field.hpp"
#include "femto/domain.hpp"

#include <chrono>
#include <cstdio>

using namespace femto;

static double best_eval_ms(const Field<Family::H1, memory::space::gpu> & u,
                           const Domain<memory::space::gpu> & domain) {
  double best = 1e30;
  for (int i = 0; i < 5; i++) {
    auto t0 = std::chrono::steady_clock::now();
    nd::array<double, 3, memory::space::gpu> du_dxi_q = evaluate(grad(u), isoparametric(domain));
    cudaDeviceSynchronize();
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    best = std::min(best, ms);
  }
  return best;
}

template < int dim >
void elem_per_block_test(const Mesh<> & mesh, int p) {

  Mesh<memory::space::gpu> d_mesh = mesh;

  Field<Family::H1, memory::space::gpu> u = create_field<Family::H1>(d_mesh, p);
  Field<Family::H1, memory::space::gpu> uvec = create_field<Family::H1>(d_mesh, p, dim);

  Domain<memory::space::gpu> domain(d_mesh, MeshQuadratureRule(p + 1));

  int kmax = (dim == 2) ? 32 : 16;

  printf("  p = %d\n", p);
  printf("    %5s %12s %12s\n", "k", "scalar (ms)", "vector (ms)");
  for (int k = 1; k <= kmax; k++) {
    set_elems_per_block(k);
    printf("    %5d %12.3f %12.3f\n", k, best_eval_ms(u, domain), best_eval_ms(uvec, domain));
  }
  set_elems_per_block(0);  // restore the default heuristic

}

int main() {
  int linear = 1;
  int quadratic = 2;
  int cubic = 3;

  {
    printf("triangle mesh (disk):\n");
    auto mesh = Mesh<>::disk(vec2{0.0, 0.0}, 1.0, 0.01, 1, Geometry::Triangle);
    elem_per_block_test<2>(mesh, linear);
    elem_per_block_test<2>(mesh, quadratic);
    elem_per_block_test<2>(mesh, cubic);
  }

  {
    printf("quadrilateral mesh (cuboid):\n");
    auto mesh = Mesh<>::cuboid({300, 300}, vec2{1.0, 1.0});
    elem_per_block_test<2>(mesh, linear);
    elem_per_block_test<2>(mesh, quadratic);
    elem_per_block_test<2>(mesh, cubic);
  }

  {
    printf("tetrahedron mesh (ball):\n");
    auto mesh = Mesh<>::ball(vec3{0.0, 0.0, 0.0}, 1.0, 0.05, 1, Geometry::Tetrahedron);
    elem_per_block_test<3>(mesh, linear);
    elem_per_block_test<3>(mesh, quadratic);
    elem_per_block_test<3>(mesh, cubic);
  }

  {
    printf("hexahedron mesh (cuboid):\n");
    auto mesh = Mesh<>::cuboid({40, 40, 40}, vec3{1.0, 1.0, 1.0});
    elem_per_block_test<3>(mesh, linear);
    elem_per_block_test<3>(mesh, quadratic);
    elem_per_block_test<3>(mesh, cubic);
  }

}
