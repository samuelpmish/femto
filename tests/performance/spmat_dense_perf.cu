// CSR scatter vs element-matrix store: the library's CUDA sparse matrix
// kernel against the library's element_integrate() (integrate_spmat_deterministic.cuh),
// which writes K(e, ci*NC+cj, I*NN+J) instead of searching the CSR row and
// accumulating with atomics.  Same problems and meshes as ../fem_shootout's
// bench_femto:
//
//   ./tests/cuda_spmat_dense_perf <mesh.msh> <poisson|elasticity|magnetostatics> <order> [reps]
//
// Reports both times, verifies K_e by assembling it on the host into the CSR
// pattern, and prints the element-matrix array size.
#include "femto/mesh.hpp"
#include "femto/field.hpp"
#include "femto/domain.hpp"
#include "femto/piola_transformations.hpp"
#include "linear_algebra/sparse_matrix.hpp"
#include "forall.hpp"
#include "fm/types/vec.hpp"
#include "fm/types/matrix.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace femto;
static constexpr uint32_t dim = 3;

struct PoissonJacV {
  __host__ __device__ void operator()(vec<dim>, mat<dim,dim> dX_dxi, mat<dim,dim> & out) const {
    mat<dim,dim> dxi_dX = inv(dX_dxi);
    out = dot(dxi_dX, transpose(dxi_dX)) * det(dX_dxi);
  }
};
struct ElasticityJacV {
  __host__ __device__ void operator()(mat<dim,dim>, mat<dim,dim> dX_dxi, mat<dim,dim,mat<dim,dim>> & out) const {
    const double lambda = 1.0, mu = 1.0;
    mat<dim,dim> dxi_dX = inv(dX_dxi);
    const double wt = det(dX_dxi);
    for (uint32_t i = 0; i < dim; i++)
      for (uint32_t j = 0; j < dim; j++)
        for (uint32_t k = 0; k < dim; k++)
          for (uint32_t l = 0; l < dim; l++) {
            double sum = 0.0;
            for (uint32_t m = 0; m < dim; m++)
              for (uint32_t n = 0; n < dim; n++) {
                double ds = lambda * (k == n) * (i == m) + mu * ((i == k) * (m == n) + (m == k) * (i == n));
                sum += ds * dxi_dX[j][m] * dxi_dX[l][n];
              }
            out[i][j][k][l] = sum * wt;
          }
  }
};
struct CurlCurlJacV {
  __host__ __device__ void operator()(vec<dim>, mat<dim,dim> dX_dxi, mat<dim,dim> & out) const {
    auto Q = covariant_piola(dX_dxi);
    out = dot(transpose(Q), Q) * det(dX_dxi);
  }
};
struct MassJacV {
  __host__ __device__ void operator()(vec<dim>, mat<dim,dim> dX_dxi, mat<dim,dim> & out) const {
    mat<dim,dim> dxi_dX = inv(dX_dxi);
    out = dot(dxi_dX, transpose(dxi_dX)) * det(dX_dxi);
  }
};

template < typename callable >
static double bench(callable && f, int reps) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start); cudaEventCreate(&stop);
  f(); cudaDeviceSynchronize();
  double best = 1e30;
  for (int i = 0; i < reps; i++) {
    cudaEventRecord(start); f(); cudaEventRecord(stop); cudaEventSynchronize(stop);
    float ms; cudaEventElapsedTime(&ms, start, stop); best = std::min(best, double(ms));
  }
  return best;
}

// run one weighted integrand both ways
template < Family family, uint32_t n, DerivedQuantity test_dq, DerivedQuantity trial_dq >
static void run_pair(const char * label, const WeightedIntegrand<BasisFunctionOp<test_dq, family>, nd::array<double, n, memory::space::gpu>, BasisFunctionOp<trial_dq, family>> & integrand,
                     const DomainWithType<memory::space::gpu> & dwt, uint32_t nelem, uint32_t nodes, uint32_t comps, int reps) {
  // CSR (library)
  sparse_matrix<memory::space::gpu> K;
  K = integrate(integrand, dwt);
  double t_csr = bench([&]() { K = integrate(integrand, dwt); }, reps);

  // dense element matrices
  uint32_t gdim = dwt.domain.mesh.geometry_dimension;
  auto phi = integrand.test.function.space;
  auto psi = integrand.trial.function.space;
  stack::array<uint32_t, 5> shape5D = {integrand.qdata.shape[0], phi.components, qshape(family, test_dq, gdim),
                                       psi.components, qshape(family, trial_dq, gdim)};
  nd::view<const double, 5, memory::space::gpu> q5D{integrand.qdata.data(), shape5D};
  ElementMatrices<memory::space::gpu> em = element_integrate(integrand, dwt);
  auto dense = [&]() {
    femto::impl::element_matrices_into<family, test_dq, family, trial_dq>(phi, q5D, psi, dwt.domain, dwt.type, em.K, em.test_ids, em.trial_ids);
  };
  double t_dense = bench(dense, reps);

  // verify: assemble K_e on the host into K's pattern
  sparse_matrix<> Kh = K;
  nd::array<double, 3, memory::space::cpu> Ke_h = em.K;
  nd::array<uint32_t, 2, memory::space::cpu> tid_h = em.test_ids, uid_h = em.trial_ids;
  std::vector<double> A(Kh.nnz, 0.0);
  for (uint32_t e = 0; e < nelem; e++)
    for (uint32_t I = 0; I < nodes; I++)
      for (uint32_t i = 0; i < comps; i++) {
        int row = int(tid_h(e, I) * comps + i);
        for (uint32_t J = 0; J < nodes; J++)
          for (uint32_t j = 0; j < comps; j++) {
            int col = int(uid_h(e, J) * comps + j);
            int lo = Kh.row_ptr[row], hi = Kh.row_ptr[row + 1];
            while (lo < hi) { int mid = (lo + hi) / 2; if (Kh.col_ind[mid] < col) lo = mid + 1; else hi = mid; }
            A[lo] += Ke_h(e, i * comps + j, I * nodes + J);
          }
      }
  double num = 0.0, den = 0.0;
  for (size_t k = 0; k < Kh.nnz; k++) { double d = A[k] - Kh.values[k]; num += d * d; den += Kh.values[k] * Kh.values[k]; }
  // compulsory traffic: K_e + id stores, qdata + connectivity reads
  double write_bytes = double(em.K.size()) * 8.0 + 2.0 * double(nelem) * nodes * 4.0;
  const bool tets = dwt.domain.mesh[Geometry::Tetrahedron].shape[0] > 0;
  uint32_t conn_row = tets ? dwt.domain.mesh[Geometry::Tetrahedron].shape[1]
                           : dwt.domain.mesh[Geometry::Hexahedron].shape[1];
  double read_bytes = double(integrand.qdata.size()) * 8.0
                    + double(nelem) * conn_row * sizeof(Connection);
  double total_bytes = write_bytes + read_bytes;
  printf("RESULT,%s,csr_ms,%.4f,dense_ms,%.4f,speedup,%.3f,K_e_GB,%.3f,write_GBps,%.1f,total_GBps,%.1f,rel_err,%.2e,nnz,%zu\n",
         label, t_csr, t_dense, t_csr / t_dense, double(em.K.size()) * 8.0e-9,
         write_bytes / (t_dense * 1e-3) * 1e-9, total_bytes / (t_dense * 1e-3) * 1e-9,
         std::sqrt(num / den), size_t(Kh.nnz));
}

int main(int argc, char * argv[]) {
  if (argc < 4) { printf("usage: %s mesh problem order [reps]\n", argv[0]); return 1; }
  std::string mesh_file = argv[1], prob = argv[2];
  int order = std::atoi(argv[3]);
  int reps = (argc > 4) ? std::atoi(argv[4]) : 5;
  constexpr memory::space gpu = memory::space::gpu;

  Mesh<> mesh_cpu = Mesh<>::load(mesh_file);
  auto mesh = copy_to<gpu>(mesh_cpu);
  const bool simplex = (mesh_cpu[Geometry::Tetrahedron].shape[0] > 0);
  const uint32_t nelem = simplex ? mesh_cpu[Geometry::Tetrahedron].shape[0] : mesh_cpu[Geometry::Hexahedron].shape[0];
  const Family family = (prob == "magnetostatics") ? Family::Hcurl : Family::H1;
  const uint32_t ncomp = (prob == "elasticity") ? dim : 1;
  const uint32_t q = simplex ? uint32_t(2 * order) : uint32_t(order + 1);

  Domain<gpu> domain(mesh, MeshQuadratureRule(q));
  auto dwt = isoparametric(domain);
  uint32_t nodes = simplex ? (family == Family::H1 ? FiniteElement<Geometry::Tetrahedron, Family::H1>{uint32_t(order)}.num_nodes()
                                                   : FiniteElement<Geometry::Tetrahedron, Family::Hcurl>{uint32_t(order)}.num_nodes())
                           : (family == Family::H1 ? FiniteElement<Geometry::Hexahedron, Family::H1>{uint32_t(order)}.num_nodes()
                                                   : FiniteElement<Geometry::Hexahedron, Family::Hcurl>{uint32_t(order)}.num_nodes());

  nd::array<double, 3, gpu> dX_dxi_q = evaluate(grad(mesh.X), dwt);
  std::string label = prob + "_p" + std::to_string(order) + "_" + (simplex ? "tet" : "hex");

  if (prob == "poisson") {
    Field u = create_field<Family::H1>(mesh, uint32_t(order), ncomp);
    BasisFunction psi(u), phi(u);
    nd::array<double, 3, gpu> g = evaluate(grad(u), dwt);
    nd::array<double, 3, gpu> C; forall(PoissonJacV{}, g, dX_dxi_q, C);
    run_pair<Family::H1>(label.c_str(), dot(grad(psi), C, grad(phi)), dwt, nelem, nodes, 1, reps);
  } else if (prob == "elasticity") {
    Field u = create_field<Family::H1>(mesh, uint32_t(order), ncomp);
    BasisFunction psi(u), phi(u);
    nd::array<double, 3, gpu> g = evaluate(grad(u), dwt);
    nd::array<double, 5, gpu> C; forall(ElasticityJacV{}, g, dX_dxi_q, C);
    run_pair<Family::H1>(label.c_str(), dot(grad(psi), C, grad(phi)), dwt, nelem, nodes, 3, reps);
  } else {
    Field u = create_field<Family::Hcurl>(mesh, uint32_t(order), ncomp);
    BasisFunction psi(u), phi(u);
    nd::array<double, 3, gpu> B = evaluate(curl(u), dwt);
    nd::array<double, 3, gpu> E = evaluate(u, dwt);
    nd::array<double, 3, gpu> Cc; forall(CurlCurlJacV{}, B, dX_dxi_q, Cc);
    nd::array<double, 3, gpu> Cm; forall(MassJacV{}, E, dX_dxi_q, Cm);
    run_pair<Family::Hcurl>((label + "_curlcurl").c_str(), dot(curl(psi), Cc, curl(phi)), dwt, nelem, nodes, 1, reps);
    run_pair<Family::Hcurl>((label + "_mass").c_str(), dot(psi, Cm, phi), dwt, nelem, nodes, 1, reps);
  }
  return 0;
}
