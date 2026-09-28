#include "common.hpp"

#include <gtest/gtest.h>

#include <functional>
#include <iomanip>
#include <iostream>

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

constexpr double k = 3.0;

template < uint32_t dim >
__host__ __device__ vec<dim> material(vec<dim> du_dX) {
  return (k + du_dX[0]) * du_dX;
}

template < uint32_t dim >
__host__ __device__ mat<dim, dim> material_jac(vec<dim> du_dX) {
  return outer(du_dX, Identity<dim>()[0]) + (k + du_dX[0]) * Identity<dim>();
}

template < uint32_t dim >
__host__ __device__ vec<dim> qfunction(vec<dim> du_dxi, mat<dim, dim> dX_dxi) {
  mat<dim, dim> dxi_dX = inv(dX_dxi);
  vec<dim> du_dX = dot(du_dxi, dxi_dX);
  vec<dim> flux = material<dim>(du_dX);
  return dot(flux, transpose(dxi_dX)) * det(dX_dxi);
}

template < uint32_t dim >
__host__ __device__ mat<dim, dim> qfunction_jac(vec<dim> du_dxi, mat<dim, dim> dX_dxi) {
  mat<dim, dim> dxi_dX = inv(dX_dxi);
  vec<dim> du_dX = dot(du_dxi, dxi_dX);
  return dot(dxi_dX, dot(material_jac<dim>(du_dX), transpose(dxi_dX))) * det(dX_dxi);
}

template < uint32_t dim >
__host__ __device__ vec<dim> qfunction_jvp(vec<dim> du_dxi, vec<dim> ddu_dxi, mat<dim, dim> dX_dxi) {
  return dot(qfunction_jac<dim>(du_dxi, dX_dxi), ddu_dxi);
}

template < uint32_t dim >
struct LinearField {
  __host__ __device__ double operator()(vec<dim> X) const {
    double sum = 0.0;
    for (int i = 0; i < dim; i++) {
      sum += (i + 1) * X[i];
    }
    return sum;
  }
};

template < uint32_t dim >
struct QFunctionJac {
  __host__ __device__ mat<dim, dim> operator()(vec<dim> du_dxi, mat<dim, dim> dX_dxi) const {
    return qfunction_jac<dim>(du_dxi, dX_dxi);
  }
};

template < uint32_t dim >
struct QFunctionJvp {
  __host__ __device__ vec<dim> operator()(vec<dim> du_dxi, vec<dim> ddu_dxi, mat<dim, dim> dX_dxi) const {
    return qfunction_jvp<dim>(du_dxi, ddu_dxi, dX_dxi);
  }
};

////////////////////////////////////////////////////////////////////////////////

template < uint32_t dim >
void verify_dH1_dH1_derivatives() {
  double eps = 1.0e-5;
  vec<dim> du_dxi = femto::random_vec<dim>();
  vec<dim> ddu_dxi = femto::random_vec<dim>();
  mat<dim, dim> dX_dxi = femto::random_mat<dim, dim>() + Identity<dim>();

  vec<dim> dflux0 = (qfunction<dim>(du_dxi + eps * ddu_dxi, dX_dxi) -
                     qfunction<dim>(du_dxi - eps * ddu_dxi, dX_dxi)) / (2 * eps);
  vec<dim> dflux1 = dot(qfunction_jac<dim>(du_dxi, dX_dxi), ddu_dxi);

  for (uint32_t i = 0; i < dim; i++) {
    EXPECT_NEAR(dflux0[i], dflux1[i], 1.0e-10);
  }
}

void expect_same_sparsity(const sparse_matrix<> & expected, const sparse_matrix<> & actual) {
  ASSERT_EQ(expected.nrows, actual.nrows);
  ASSERT_EQ(expected.ncols, actual.ncols);
  ASSERT_EQ(expected.nnz, actual.nnz);
  ASSERT_EQ(expected.row_ptr.size(), actual.row_ptr.size());
  ASSERT_EQ(expected.col_ind.size(), actual.col_ind.size());

  for (uint32_t i = 0; i < expected.row_ptr.size(); i++) {
    EXPECT_EQ(expected.row_ptr[i], actual.row_ptr[i]) << "row_ptr[" << i << "]";
  }

  for (uint32_t i = 0; i < expected.col_ind.size(); i++) {
    EXPECT_EQ(expected.col_ind[i], actual.col_ind[i]) << "col_ind[" << i << "]";
  }
}

template < uint32_t dim >
void dH1_dH1_gpu_test(std::string filename, double tolerance) {
  if (!cuda_device_available()) {
    GTEST_SKIP() << "No CUDA-capable device is available";
  }

  auto mesh_cpu = Mesh<>::load(FEMTO_MESH_DIR + filename);
  Mesh<memory::space::gpu> mesh_gpu = copy_to<memory::space::gpu>(mesh_cpu);

  std::cout << std::setprecision(15);

  for (int p1 = 1; p1 < 4; p1++) {
    Field u_cpu = create_field<Family::H1>(mesh_cpu, p1, 1);
    Field du_cpu = create_field<Family::H1>(mesh_cpu, p1, 1);

    Field<Family::H1, memory::space::gpu> u_gpu = create_field<Family::H1>(mesh_gpu, p1, 1);
    Field<Family::H1, memory::space::gpu> du_gpu = create_field<Family::H1>(mesh_gpu, p1, 1);

    auto nodes_cpu = nodes_for(u_cpu, mesh_cpu);
    auto nodes_gpu = nodes_for(u_gpu, mesh_gpu);

    auto u0_cpu = forall(LinearField<dim>{}, nodes_cpu);
    auto u0_gpu = forall(LinearField<dim>{}, nodes_gpu);

    u_cpu = u0_cpu;
    u_gpu = u0_gpu;

    du_cpu.data = femto::random(u_cpu.data.shape);
    du_gpu.data = nd::copy_to<memory::space::gpu>(du_cpu.data);

    BasisFunction<Family::H1> psi(p1, 1);

    for (int p2 = 1; p2 < 4; p2++) {
      BasisFunction<Family::H1> phi(p2, 1);

      for (int q = 1; q < 5; q++) {
        Domain<> domain_cpu(mesh_cpu, MeshQuadratureRule(q));
        Domain<memory::space::gpu> domain_gpu(mesh_gpu, MeshQuadratureRule(q));

        nd::cpu_array<double, 3> dX_dxi_q_cpu = evaluate(grad(mesh_cpu.X), isoparametric(domain_cpu));
        nd::cpu_array<double, 3> du_dxi_q_cpu = evaluate(grad(u_cpu), isoparametric(domain_cpu));
        nd::cpu_array<double, 3> ddu_dxi_q_cpu = evaluate(grad(du_cpu), isoparametric(domain_cpu));

        auto df_q_cpu = forall(QFunctionJvp<dim>{}, du_dxi_q_cpu, ddu_dxi_q_cpu, dX_dxi_q_cpu);
        Residual<Family::H1> dr_cpu_r = integrate(dot(df_q_cpu, grad(phi)), isoparametric(domain_cpu));
        auto dr_cpu = dr_cpu_r.data;

        auto df_ddudxi_q_cpu = forall(QFunctionJac<dim>{}, du_dxi_q_cpu, dX_dxi_q_cpu);
        sparse_matrix<> K_cpu = integrate(dot(grad(psi), df_ddudxi_q_cpu, grad(phi)), isoparametric(domain_cpu));

        nd::gpu_array<double, 3> dX_dxi_q_gpu = evaluate(grad(mesh_gpu.X), isoparametric(domain_gpu));
        nd::gpu_array<double, 3> du_dxi_q_gpu = evaluate(grad(u_gpu), isoparametric(domain_gpu));
        nd::gpu_array<double, 3> ddu_dxi_q_gpu = evaluate(grad(du_gpu), isoparametric(domain_gpu));

        auto df_q_gpu = forall(QFunctionJvp<dim>{}, du_dxi_q_gpu, ddu_dxi_q_gpu, dX_dxi_q_gpu);
        Residual<Family::H1, memory::space::gpu> dr_gpu = integrate(dot(df_q_gpu, grad(phi)), isoparametric(domain_gpu));

        auto df_ddudxi_q_gpu = forall(QFunctionJac<dim>{}, du_dxi_q_gpu, dX_dxi_q_gpu);
        sparse_matrix<memory::space::gpu> K_gpu =
          integrate(dot(grad(psi), df_ddudxi_q_gpu, grad(phi)), isoparametric(domain_gpu));
        sparse_matrix<> K_gpu_cpu = K_gpu;

        Residual<Family::H1> dr_gpu_cpu = dr_gpu;
        Residual<Family::H1> K_cpu_du(FunctionSpace(Family::H1, p2, 1), mesh_cpu);
        Residual<Family::H1> K_gpu_du(FunctionSpace(Family::H1, p2, 1), mesh_cpu);
        K_cpu_du.v() = K_cpu(du_cpu.v());
        K_gpu_du.v() = K_gpu_cpu(du_cpu.v());

        SCOPED_TRACE("p1 = " + std::to_string(p1) +
                     ", p2 = " + std::to_string(p2) +
                     ", q = " + std::to_string(q));

        EXPECT_NEAR(relative_error(dr_cpu, dr_gpu_cpu.data), 0.0, tolerance);

        expect_same_sparsity(K_cpu, K_gpu_cpu);
        EXPECT_NEAR(relative_error(K_cpu.values, K_gpu_cpu.values), 0.0, tolerance);

        EXPECT_NEAR(relative_error(dr_cpu, K_gpu_du.data), 0.0, tolerance);
        EXPECT_NEAR(relative_error(K_cpu_du.data, K_gpu_du.data), 0.0, tolerance);
      }
    }
  }
}

// ----------------------------------------------------------------------------

TEST(IntegrateTest, verify_dH1_dH1_derivatives2D) { verify_dH1_dH1_derivatives<2>(); }
TEST(IntegrateTest, verify_dH1_dH1_derivatives3D) { verify_dH1_dH1_derivatives<3>(); }

// ----------------------------------------------------------------------------

TEST(IntegrateTest, dH1_dH1_edges) { dH1_dH1_gpu_test<1>("patch_test_edges.json", 1.0e-8); }

TEST(IntegrateTest, dH1_dH1_tris) { dH1_dH1_gpu_test<2>("patch_test_tris.json", 1.0e-8); }
TEST(IntegrateTest, dH1_dH1_quads) { dH1_dH1_gpu_test<2>("patch_test_quads.json", 1.0e-8); }
TEST(IntegrateTest, dH1_dH1_tris_and_quads) { dH1_dH1_gpu_test<2>("patch_test_tris_and_quads.json", 1.0e-8); }

TEST(IntegrateTest, dH1_dH1_tets) { dH1_dH1_gpu_test<3>("patch_test_tets.json", 1.0e-8); }
TEST(IntegrateTest, dH1_dH1_hexes) { dH1_dH1_gpu_test<3>("patch_test_hexes.json", 1.0e-8); }
TEST(IntegrateTest, dH1_dH1_tets_and_hexes) { dH1_dH1_gpu_test<3>("patch_test_tets_and_hexes.json", 1.0e-8); }
