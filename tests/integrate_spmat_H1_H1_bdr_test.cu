#include "common.hpp"

#include <gtest/gtest.h>

#include "containers/ndarray_conversions.hpp"
#include "femto/domain.hpp"
#include "femto/mesh.hpp"
#include "forall.hpp"
#include "fm/types/matrix.hpp"
#include "fm/types/vec.hpp"
#include "integrate_residual_cuda_common.cuh"

using namespace femto;
using namespace residual_cuda_tests;

////////////////////////////////////////////////////////////////////////////////

// a nonlinear boundary source s(u) = rho u^2 on a spatial boundary domain:
// the kernels supply the facet measure
constexpr double rho = 3.0;

struct QFunction {
  __host__ __device__ double operator()(double u) const { return rho * u * u; }
};

struct QFunctionJac {
  __host__ __device__ double operator()(double u) const { return 2 * rho * u; }
};

struct QFunctionJvp {
  __host__ __device__ double operator()(double u, double du) const { return 2 * rho * u * du; }
};

template < uint32_t dim >
struct LinearField {
  __host__ __device__ double operator()(vec<dim> X) const {
    double sum = 0.0;
    for (int i = 0; i < dim; i++) { sum += (i + 1) * X[i]; }
    return sum;
  }
};

void expect_same_sparsity(const sparse_matrix<> & expected, const sparse_matrix<> & actual) {
  ASSERT_EQ(expected.nrows, actual.nrows);
  ASSERT_EQ(expected.ncols, actual.ncols);
  ASSERT_EQ(expected.nnz, actual.nnz);
  for (uint32_t i = 0; i < expected.row_ptr.size(); i++) {
    EXPECT_EQ(expected.row_ptr[i], actual.row_ptr[i]) << "row_ptr[" << i << "]";
  }
  for (uint32_t i = 0; i < expected.col_ind.size(); i++) {
    EXPECT_EQ(expected.col_ind[i], actual.col_ind[i]) << "col_ind[" << i << "]";
  }
}

////////////////////////////////////////////////////////////////////////////////

// the finite difference of the gpu boundary residual must agree with the gpu
// boundary matrix times du, and that matrix with the cpu one
template < uint32_t dim >
void H1_H1_bdr_gpu_test(std::string filename, double tolerance) {
  if (!cuda_device_available()) {
    GTEST_SKIP() << "No CUDA-capable device is available";
  }

  double epsilon = 1.0e-6;

  auto mesh_cpu = Mesh<>::load(FEMTO_MESH_DIR + filename);
  Mesh<memory::space::gpu> mesh_gpu = copy_to<memory::space::gpu>(mesh_cpu);

  for (int p1 = 1; p1 < 4; p1++) {
    Field u_cpu = create_field<Family::H1>(mesh_cpu, p1, 1);
    Field du_cpu = create_field<Family::H1>(mesh_cpu, p1, 1);
    Field<Family::H1, memory::space::gpu> u_gpu = create_field<Family::H1>(mesh_gpu, p1, 1);
    Field<Family::H1, memory::space::gpu> du_gpu = create_field<Family::H1>(mesh_gpu, p1, 1);

    auto nodes_cpu = nodes_for(u_cpu, mesh_cpu);
    auto u0 = forall(LinearField<dim>{}, nodes_cpu);

    du_cpu.data = femto::random(u_cpu.data.shape);
    du_gpu = du_cpu;

    BasisFunction<Family::H1> psi(p1, 1);

    for (int p2 = 1; p2 < 4; p2++) {
      BasisFunction<Family::H1> phi(p2, 1);

      for (int q = 1; q < 5; q++) {
        Domain<> bdr_cpu(boundary_of(mesh_cpu), MeshQuadratureRule(q));
        Domain<memory::space::gpu> bdr_gpu(boundary_of(mesh_gpu), MeshQuadratureRule(q));

        // the gpu boundary residual, brought back to the host
        auto r = [&](const Field<Family::H1, memory::space::gpu> & u_) -> Residual<Family::H1> {
          nd::gpu_array<double, 3> u_q = evaluate(u_, bdr_gpu);
          auto s_q = forall(QFunction{}, u_q);
          Residual<Family::H1, memory::space::gpu> r_gpu = integrate(dot(s_q, phi), bdr_gpu);
          return r_gpu;
        };

        // finite difference approximation of jvp
        u_cpu = u0 + epsilon * du_cpu.data;
        u_gpu = u_cpu;
        Residual<Family::H1> rp = r(u_gpu);

        u_cpu = u0 - epsilon * du_cpu.data;
        u_gpu = u_cpu;
        Residual<Family::H1> rm = r(u_gpu);

        auto dr1 = (rp.data - rm.data) / (2 * epsilon);

        // "matrix-free" jvp
        u_cpu = u0;
        u_gpu = u_cpu;
        nd::gpu_array<double, 3> u_q = evaluate(u_gpu, bdr_gpu);
        nd::gpu_array<double, 3> du_q = evaluate(du_gpu, bdr_gpu);
        auto ds_q = forall(QFunctionJvp{}, u_q, du_q);
        Residual<Family::H1, memory::space::gpu> dr2_gpu = integrate(dot(ds_q, phi), bdr_gpu);
        Residual<Family::H1> dr2 = dr2_gpu;

        // "sparse matrix" jvp
        auto ds_du_q = forall(QFunctionJac{}, u_q);
        sparse_matrix<memory::space::gpu> K_gpu = integrate(dot(psi, ds_du_q, phi), bdr_gpu);
        sparse_matrix<> K_gpu_cpu = K_gpu;

        Residual<Family::H1> dr3(FunctionSpace(Family::H1, p2, 1), mesh_cpu);
        dr3.v() = K_gpu_cpu(du_cpu.v());

        // the cpu matrix, for reference
        nd::cpu_array<double, 3> u_q_cpu = evaluate(u_cpu, bdr_cpu);
        auto ds_du_q_cpu = forall(QFunctionJac{}, u_q_cpu);
        sparse_matrix<> K_cpu = integrate(dot(psi, ds_du_q_cpu, phi), bdr_cpu);

        SCOPED_TRACE("p1 = " + std::to_string(p1) + ", p2 = " + std::to_string(p2) + ", q = " + std::to_string(q));

        EXPECT_NEAR(relative_error(dr1, dr2.data), 0.0, tolerance);
        EXPECT_NEAR(relative_error(dr1, dr3.data), 0.0, tolerance);
        EXPECT_NEAR(relative_error(dr2.data, dr3.data), 0.0, 1.0e-12);

        expect_same_sparsity(K_cpu, K_gpu_cpu);
        EXPECT_NEAR(relative_error(K_cpu.values, K_gpu_cpu.values), 0.0, 1.0e-12);
      }
    }
  }
}

// ----------------------------------------------------------------------------

TEST(IntegrateTest, H1_H1_bdr_tris) { H1_H1_bdr_gpu_test<2>("patch_test_tris.json", 1.0e-8); }
TEST(IntegrateTest, H1_H1_bdr_quads) { H1_H1_bdr_gpu_test<2>("patch_test_quads.json", 1.0e-8); }
TEST(IntegrateTest, H1_H1_bdr_tris_and_quads) { H1_H1_bdr_gpu_test<2>("patch_test_tris_and_quads.json", 1.0e-8); }

TEST(IntegrateTest, H1_H1_bdr_tets) { H1_H1_bdr_gpu_test<3>("patch_test_tets.json", 1.0e-8); }
TEST(IntegrateTest, H1_H1_bdr_hexes) { H1_H1_bdr_gpu_test<3>("patch_test_hexes.json", 1.0e-8); }
TEST(IntegrateTest, H1_H1_bdr_tets_and_hexes) { H1_H1_bdr_gpu_test<3>("patch_test_tets_and_hexes.json", 1.0e-8); }
