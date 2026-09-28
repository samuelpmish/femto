#include <cmath>

#include <gtest/gtest.h>

#include "linear_algebra/sparse_direct.hpp"

using namespace femto;

static constexpr memory::space GPU = memory::space::gpu;

// 5-point Laplacian on a g x g grid: symmetric positive definite,
// with an optional convection term that makes it unsymmetric
static sparse_matrix<> model_matrix(int g, bool unsymmetric) {
  auto id = [g](int i, int j) { return i * g + j; };
  std::vector<triplet> triplets;
  for (int i = 0; i < g; i++) {
    for (int j = 0; j < g; j++) {
      triplets.push_back({id(i, j), id(i, j), 4.0});
      if (i > 0) { triplets.push_back({id(i, j), id(i - 1, j), -1.0 - (unsymmetric ? 0.3 : 0.0)}); }
      if (i < g - 1) { triplets.push_back({id(i, j), id(i + 1, j), -1.0 + (unsymmetric ? 0.3 : 0.0)}); }
      if (j > 0) { triplets.push_back({id(i, j), id(i, j - 1), -1.0}); }
      if (j < g - 1) { triplets.push_back({id(i, j), id(i, j + 1), -1.0}); }
    }
  }
  auto A = sparse_matrix<>::from_triplets(triplets, g * g, g * g);
  if (!unsymmetric) {
    A.symmetry = Symmetry::Symmetric;
    A.definiteness = Definiteness::PositiveDefinite;
  }
  return A;
}

static double relative_residual(const sparse_matrix<> & A, const vector & x, const vector & b) {
  vector r = A(x);
  double num = 0.0;
  double den = 0.0;
  for (uint32_t i = 0; i < b.size(); i++) {
    num += (r[i] - b[i]) * (r[i] - b[i]);
    den += b[i] * b[i];
  }
  return sqrt(num / den);
}

// device matrix in, device vectors through the solve
static void check_device(SparseFactorizationBackend backend, bool symmetric) {
  sparse_matrix<> A = model_matrix(20, !symmetric);
  sparse_matrix<GPU> A_gpu = A;
  vector b = ones(int(A.nrows));
  gpu_vector b_gpu = b;

  auto invA = inv(A_gpu, backend);
  EXPECT_EQ(invA.backend(), backend);
  vector x = dot(invA, b_gpu);
  EXPECT_LT(relative_residual(A, x, b), 1.0e-10);

  // refactorize with scaled values (same pattern): solution should track it
  for (uint32_t i = 0; i < A.nnz; i++) { A.values(i) *= 2.0; }
  A_gpu = A;
  invA.update(A_gpu);
  vector y = dot(invA, b_gpu);
  EXPECT_LT(relative_residual(A, y, b), 1.0e-10);
}

// host matrix in: copied to the device, solved there, solution copied back
static void check_host(SparseFactorizationBackend backend, bool symmetric) {
  sparse_matrix<> A = model_matrix(20, !symmetric);
  vector b = ones(int(A.nrows));
  auto invA = inv(A, backend);
  vector x = dot(invA, b);
  EXPECT_LT(relative_residual(A, x, b), 1.0e-10);
  for (uint32_t i = 0; i < A.nnz; i++) { A.values(i) *= 2.0; }
  invA.update(A);
  vector y = dot(invA, b);
  EXPECT_LT(relative_residual(A, y, b), 1.0e-10);
}

TEST(SparseDirectCuDSS, DeviceLU) { check_device(SparseFactorizationBackend::CUDSS_LU, false); }
TEST(SparseDirectCuDSS, DeviceLLT) { check_device(SparseFactorizationBackend::CUDSS_LLT, true); }
TEST(SparseDirectCuDSS, DeviceLDLT) { check_device(SparseFactorizationBackend::CUDSS_LDLT, true); }
TEST(SparseDirectCuDSS, HostLU) { check_host(SparseFactorizationBackend::CUDSS_LU, false); }
TEST(SparseDirectCuDSS, HostLLT) { check_host(SparseFactorizationBackend::CUDSS_LLT, true); }

// the default for a device matrix follows its declared properties
TEST(SparseDirectCuDSS, DefaultSelection) {
  sparse_matrix<GPU> spd = model_matrix(10, false);
  sparse_matrix<GPU> general = model_matrix(10, true);
  EXPECT_EQ(inv(spd).backend(), SparseFactorizationBackend::CUDSS_LLT);
  EXPECT_EQ(inv(general).backend(), SparseFactorizationBackend::CUDSS_LU);
}
