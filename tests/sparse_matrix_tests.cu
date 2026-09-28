#include <gtest/gtest.h>

#include "femto/mesh.hpp"
#include "linear_algebra/sparse_matrix.hpp"
#include "integrate_residual_cuda_common.cuh"

#include "misc/timer.hpp"

namespace {

constexpr double tolerance = 1.0e-14;

} // namespace

#if 0
TEST(SparseMatrix, Construction) {
    femto::sparse_matrix<femto::memory::space::gpu> spmat(10, 10);
    EXPECT_EQ(spmat.num_rows(), 10);
    EXPECT_EQ(spmat.num_cols(), 10);
    EXPECT_EQ(spmat.nnz(), 0);
}
#endif

TEST(SparseMatrix, CudaMatvec) {
  if (!residual_cuda_tests::cuda_device_available()) {
    GTEST_SKIP() << "No CUDA-capable device is available";
  }

  std::vector<femto::triplet> entries{
    {0, 0, 1.0},
    {0, 1, 2.0},
    {1, 1, 3.0},
    {2, 0, 4.0},
    {2, 2, 5.0}
  };

  femto::sparse_matrix<> A_cpu = femto::sparse_matrix<>::from_triplets(entries, 3, 3);
  femto::sparse_matrix<memory::space::gpu> A_gpu(A_cpu);

  femto::vector x_cpu({1.0, 2.0, 3.0});
  femto::gpu_vector x_gpu(x_cpu);
  femto::gpu_vector y_gpu = femto::dot(A_gpu, x_gpu);
  femto::vector y_cpu(y_gpu);

  EXPECT_NEAR(y_cpu[0], 5.0, tolerance);
  EXPECT_NEAR(y_cpu[1], 6.0, tolerance);
  EXPECT_NEAR(y_cpu[2], 19.0, tolerance);
}

TEST(SparseMatrix, CudaRelativeError) {
  if (!residual_cuda_tests::cuda_device_available()) {
    GTEST_SKIP() << "No CUDA-capable device is available";
  }

  nd::array<double, 1, memory::space::cpu> a_cpu({4});
  nd::array<double, 1, memory::space::cpu> b_cpu({4});
  for (uint32_t i = 0; i < a_cpu.size(); i++) {
    a_cpu(i) = double(i + 1);
    b_cpu(i) = double(i + 1) + 0.1;
  }

  nd::array<double, 1, memory::space::gpu> a_gpu(a_cpu);
  nd::array<double, 1, memory::space::gpu> b_gpu(b_cpu);

  EXPECT_NEAR(relative_error(a_gpu, b_gpu), relative_error(a_cpu, b_cpu), tolerance);
}
