#include "femto/mesh.hpp"
#include "femto/field.hpp"
#include "femto/domain.hpp"

#include "forall.hpp"

#include "misc/timer.hpp"

#include <functional>

#include <CLI/CLI.hpp>

namespace compiler {
static void please_do_not_optimize_away([[maybe_unused]] void* p) { asm volatile("" : : "g"(p) : "memory"); }
}

using namespace femto;

timer stopwatch;

template < typename matd >
mat<dim> qfunction(matd du_dxi, matd dX_dxi) {
  constexpr double lambda = 1.0;
  constexpr double mu = 1.0;
  constexpr mat<dim,dim> I = Identity<dim>();
  matd dxi_dX = inv(dX_dxi);
  matd du_dX = dot(du_dxi, dxi_dX);
  matd strain = sym(du_dX);
  matd stress = lambda * tr(strain) * I + mu * strain;
  return dot(stress, transpose(dxi_dX)) * det(dX_dxi);
}

template < uint32_t dim >
mat<dim,dim> qfunction_derivative(mat<dim,dim> dX_dxi) {
  mat<dim,dim> dxi_dX = inv(dX_dxi);
  return dot(dxi_dX, transpose(dxi_dX)) * (3.0 * det(dX_dxi));
}

template < int dim >
void run_test(const Mesh<> & mesh, int p, int q) {

  femto::timer stopwatch;

  Field u = create_field(mesh, Family::H1, p);

  TestFunction phi(u);

  auto X = nodes_for(u, mesh);
  u = forall(std::function< double(vec<dim>)> ([](vec<dim> X){ return X[0]; }), X);

  Domain domain(mesh, MeshQuadratureRule(p + 1));

  nd::cpu_array<double, 3> dX_dxi_q = evaluate(grad(mesh.X), domain);

  for (int i = 0; i < 5; i++) {
    stopwatch.start();
    nd::cpu_array<double, 3> du_dxi_q = evaluate(grad(u), domain);
    stopwatch.stop();
    std::cout << stopwatch.elapsed() * 1000.0 << " ";
    std::cout << std::flush;

    stopwatch.start();
    nd::cpu_array<double, 2> f_q = forall(qfunction<dim>, du_dxi_q, dX_dxi_q);
    stopwatch.stop();
    std::cout << stopwatch.elapsed() * 1000.0 << " ";
    std::cout << std::flush;

    stopwatch.start();
    Residual r = integrate(dot(f_q, grad(phi)), domain);
    stopwatch.stop();
    std::cout << stopwatch.elapsed() * 1000.0 << " ";
    std::cout << std::flush;

    nd::cpu_array<double, 3> k_q = forall(qfunction_derivative<dim>, dX_dxi_q);

    stopwatch.start();
    sparse_matrix K = integrate(dot(grad(phi), k_q, grad(phi)), domain);
    stopwatch.stop();
    std::cout << stopwatch.elapsed() * 1000.0 << std::endl;
  }

}

int main(int argc, char **argv) {

  CLI::App app("command line tool for measuring performance of poisson-like FEM kernels");

  std::string meshfile;
  app.add_option("-m", meshfile, "mesh file name")->required();

  int order = 1;
  app.add_option("-o", order, "polynomial order of solution field");

  int q = 2;
  app.add_option("-q", q, "quadrature points per dimension");

  CLI11_PARSE(app, argc, argv);

  std::cout << "loading mesh ... ";
  Mesh<> mesh = Mesh<>::load(meshfile);
  std::cout << "finished" << std::endl;

  std::cout << "mesh has:" << std::endl;
  std::cout << "  " << mesh.vert.shape[0] << " vertices" << std::endl;
  std::cout << "  " << mesh.edge.shape[0] << " edges" << std::endl;
  std::cout << "  " << mesh.tri.shape[0] << " triangles" << std::endl;
  std::cout << "  " << mesh.quad.shape[0] << " quadrilaterals" << std::endl;
  std::cout << "  " << mesh.tet.shape[0] << " tetrahedra" << std::endl;
  std::cout << "  " << mesh.hex.shape[0] << " hexahedra" << std::endl;

  if (mesh.spatial_dimension == 2) { run_test<2>(mesh, order, q); }
  if (mesh.spatial_dimension == 3) { run_test<3>(mesh, order, q); }

}