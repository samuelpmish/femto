// GPU counterpart of integrate_spmat_dHcurl_dHcurl_tests.cpp: checks that the
// CUDA sparse matrix assembly kernel agrees with the (independently verified)
// CPU kernel for curl-curl Hcurl operators.
//
// this also covers two things that are easy to get silently wrong:
//   - the piola transformation helpers in femto/piola_transformations.hpp must
//     be callable from device code; if they aren't, nvcc emits only a warning
//     and the qfunctions below quietly evaluate to zero on the GPU
//   - high-order 3D Hcurl shape function tables exceed the 48 KB of dynamic
//     shared memory a kernel can request by default, so the assembly kernel
//     has to fall back to reading them from global memory

#include "common.hpp"

#include <gtest/gtest.h>

#include <functional>
#include <iomanip>
#include <iostream>

#include "containers/ndarray_conversions.hpp"
#include "femto/domain.hpp"
#include "femto/mesh.hpp"
#include "femto/piola_transformations.hpp"
#include "forall.hpp"
#include "fm/types/matrix.hpp"
#include "fm/types/vec.hpp"
#include "integrate_residual_cuda_common.cuh"

using namespace femto;
using namespace residual_cuda_tests;

////////////////////////////////////////////////////////////////////////////////
// same nonlinear magnetostatics-like material as the CPU test

constexpr double inv_mu = 3.0;

// number of components in curl(A) -- 1 in 2D, 3 in 3D
template < uint32_t dim >
static constexpr uint32_t curl_components = (dim == 2) ? 1 : dim;

template < uint32_t c >
__host__ __device__ vec<c> material(vec<c> B) {
  return (inv_mu + B[0]) * B;
}

template < uint32_t c >
__host__ __device__ mat<c, c> material_jac(vec<c> B) {
  return outer(B, Identity<c>()[0]) + (inv_mu + B[0]) * Identity<c>();
}

template < uint32_t dim >
struct QFunction {
  static constexpr uint32_t c = curl_components<dim>;
  __host__ __device__ vec<c> operator()(vec<c> B_xi, mat<dim, dim> dX_dxi) const {
    auto Q = covariant_piola(dX_dxi);
    vec<c> B = dot(Q, B_xi);
    vec<c> H = material<c>(B);
    return dot(transpose(Q), H) * det(dX_dxi);
  }
};

template < uint32_t dim >
struct QFunctionJac {
  static constexpr uint32_t c = curl_components<dim>;
  __host__ __device__ mat<c, c> operator()(vec<c> B_xi, mat<dim, dim> dX_dxi) const {
    auto Q = covariant_piola(dX_dxi);
    vec<c> B = dot(Q, B_xi);
    return dot(transpose(Q), dot(material_jac<c>(B), Q)) * det(dX_dxi);
  }
};

template < uint32_t dim >
struct QFunctionJvp {
  static constexpr uint32_t c = curl_components<dim>;
  __host__ __device__ vec<c> operator()(vec<c> B_xi, vec<c> dB_xi, mat<dim, dim> dX_dxi) const {
    return dot(QFunctionJac<dim>{}(B_xi, dX_dxi), dB_xi);
  }
};

// nodal dof values for an Hcurl field: the tangential component of a linear field
template < uint32_t dim >
struct TangentialDof {
  __host__ __device__ double operator()(vec<dim> X, vec<dim> n) const { return dot(X, n); }
};

////////////////////////////////////////////////////////////////////////////////

static void expect_same_sparsity(const sparse_matrix<> & expected, const sparse_matrix<> & actual) {
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
void dHcurl_dHcurl_gpu_test(std::string filename, double tolerance) {
  if (!cuda_device_available()) {
    GTEST_SKIP() << "No CUDA-capable device is available";
  }

  auto mesh_cpu = Mesh<>::load(FEMTO_MESH_DIR + filename);
  Mesh<memory::space::gpu> mesh_gpu = copy_to<memory::space::gpu>(mesh_cpu);

  std::cout << std::setprecision(15);

  for (int p1 = 1; p1 < 4; p1++) {
    Field A_cpu = create_field<Family::Hcurl>(mesh_cpu, p1, 1);
    Field dA_cpu = create_field<Family::Hcurl>(mesh_cpu, p1, 1);

    Field<Family::Hcurl, memory::space::gpu> A_gpu = create_field<Family::Hcurl>(mesh_gpu, p1, 1);
    Field<Family::Hcurl, memory::space::gpu> dA_gpu = create_field<Family::Hcurl>(mesh_gpu, p1, 1);

    auto nodes_cpu = nodes_for(A_cpu, mesh_cpu);
    auto directions_cpu = directions_for(A_cpu, mesh_cpu);
    auto nodes_gpu = nodes_for(A_gpu, mesh_gpu);
    auto directions_gpu = directions_for(A_gpu, mesh_gpu);

    A_cpu = forall(TangentialDof<dim>{}, nodes_cpu, directions_cpu);
    A_gpu = forall(TangentialDof<dim>{}, nodes_gpu, directions_gpu);

    dA_cpu.data = femto::random(A_cpu.data.shape);
    dA_gpu.data = nd::copy_to<memory::space::gpu>(dA_cpu.data);

    BasisFunction<Family::Hcurl> psi(p1, 1);

    for (int p2 = 1; p2 < 4; p2++) {
      BasisFunction<Family::Hcurl> phi(p2, 1);

      for (int q = 1; q < 5; q++) {
        Domain<> domain_cpu(mesh_cpu, MeshQuadratureRule(q));
        Domain<memory::space::gpu> domain_gpu(mesh_gpu, MeshQuadratureRule(q));

        nd::cpu_array<double, 3> dX_dxi_q_cpu = evaluate(grad(mesh_cpu.X), isoparametric(domain_cpu));
        nd::cpu_array<double, 3> B_xi_q_cpu = evaluate(curl(A_cpu), isoparametric(domain_cpu));
        nd::cpu_array<double, 3> dB_xi_q_cpu = evaluate(curl(dA_cpu), isoparametric(domain_cpu));

        auto dH_xi_q_cpu = forall(QFunctionJvp<dim>{}, B_xi_q_cpu, dB_xi_q_cpu, dX_dxi_q_cpu);
        Residual<Family::Hcurl> dr_cpu_r = integrate(dot(dH_xi_q_cpu, curl(phi)), isoparametric(domain_cpu));
        auto dr_cpu = dr_cpu_r.data;

        auto dHxi_dBxi_q_cpu = forall(QFunctionJac<dim>{}, B_xi_q_cpu, dX_dxi_q_cpu);
        sparse_matrix<> K_cpu = integrate(dot(curl(psi), dHxi_dBxi_q_cpu, curl(phi)), isoparametric(domain_cpu));

        nd::gpu_array<double, 3> dX_dxi_q_gpu = evaluate(grad(mesh_gpu.X), isoparametric(domain_gpu));
        nd::gpu_array<double, 3> B_xi_q_gpu = evaluate(curl(A_gpu), isoparametric(domain_gpu));
        nd::gpu_array<double, 3> dB_xi_q_gpu = evaluate(curl(dA_gpu), isoparametric(domain_gpu));

        auto dH_xi_q_gpu = forall(QFunctionJvp<dim>{}, B_xi_q_gpu, dB_xi_q_gpu, dX_dxi_q_gpu);
        Residual<Family::Hcurl, memory::space::gpu> dr_gpu = integrate(dot(dH_xi_q_gpu, curl(phi)), isoparametric(domain_gpu));

        auto dHxi_dBxi_q_gpu = forall(QFunctionJac<dim>{}, B_xi_q_gpu, dX_dxi_q_gpu);
        sparse_matrix<memory::space::gpu> K_gpu =
          integrate(dot(curl(psi), dHxi_dBxi_q_gpu, curl(phi)), isoparametric(domain_gpu));

        sparse_matrix<> K_gpu_cpu = K_gpu;
        Residual<Family::Hcurl> dr_gpu_cpu = dr_gpu;

        Residual<Family::Hcurl> K_cpu_dA(FunctionSpace(Family::Hcurl, p2, 1), mesh_cpu);
        Residual<Family::Hcurl> K_gpu_dA(FunctionSpace(Family::Hcurl, p2, 1), mesh_cpu);
        K_cpu_dA.v() = K_cpu(dA_cpu.v());
        K_gpu_dA.v() = K_gpu_cpu(dA_cpu.v());

        SCOPED_TRACE("p1 = " + std::to_string(p1) +
                     ", p2 = " + std::to_string(p2) +
                     ", q = " + std::to_string(q));

        // the qfunction evaluated on the device must match the host: a
        // host-only piola helper would leave the device values identically zero
        nd::array<double, 3, memory::space::cpu> dHxi_dBxi_q_gpu_cpu = dHxi_dBxi_q_gpu;
        EXPECT_NEAR(relative_error(dHxi_dBxi_q_cpu, dHxi_dBxi_q_gpu_cpu), 0.0, tolerance);

        EXPECT_NEAR(relative_error(dr_cpu, dr_gpu_cpu.data), 0.0, tolerance);

        // the assembled matrices must agree entry for entry
        expect_same_sparsity(K_cpu, K_gpu_cpu);
        EXPECT_NEAR(relative_error(K_cpu.values, K_gpu_cpu.values), 0.0, tolerance);

        // ... and act identically on a vector, consistent with the matrix-free jvp
        EXPECT_NEAR(relative_error(K_cpu_dA.data, K_gpu_dA.data), 0.0, tolerance);
        EXPECT_NEAR(relative_error(dr_cpu, K_gpu_dA.data), 0.0, tolerance);
      }
    }
  }
}

// ----------------------------------------------------------------------------

TEST(IntegrateTest, dHcurl_dHcurl_tris) { dHcurl_dHcurl_gpu_test<2>("patch_test_tris.json", 1.0e-8); }
TEST(IntegrateTest, dHcurl_dHcurl_quads) { dHcurl_dHcurl_gpu_test<2>("patch_test_quads.json", 1.0e-8); }
TEST(IntegrateTest, dHcurl_dHcurl_tris_and_quads) { dHcurl_dHcurl_gpu_test<2>("patch_test_tris_and_quads.json", 1.0e-8); }

TEST(IntegrateTest, dHcurl_dHcurl_one_tet) { dHcurl_dHcurl_gpu_test<3>("one_tet.json", 1.0e-8); }
TEST(IntegrateTest, dHcurl_dHcurl_one_hex) { dHcurl_dHcurl_gpu_test<3>("one_hex.json", 1.0e-8); }
TEST(IntegrateTest, dHcurl_dHcurl_tets) { dHcurl_dHcurl_gpu_test<3>("patch_test_tets.json", 1.0e-8); }
TEST(IntegrateTest, dHcurl_dHcurl_hexes) { dHcurl_dHcurl_gpu_test<3>("patch_test_hexes.json", 1.0e-8); }
TEST(IntegrateTest, dHcurl_dHcurl_tets_and_hexes) { dHcurl_dHcurl_gpu_test<3>("patch_test_tets_and_hexes.json", 1.0e-8); }
