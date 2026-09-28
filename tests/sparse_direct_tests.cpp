#include <cmath>

#include <gtest/gtest.h>

#include "linear_algebra/sparse_direct.hpp"

using namespace femto;

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

static void check_backend(SparseFactorizationBackend backend, bool symmetric) {
  sparse_matrix<> A = model_matrix(20, !symmetric);
  vector b = ones(int(A.nrows));

  auto invA = inv(A, backend);
  EXPECT_EQ(invA.backend(), backend);
  vector x = dot(invA, b);
  EXPECT_LT(relative_residual(A, x, b), 1.0e-10);

  // refactorize with scaled values (same pattern): solution should track it
  for (uint32_t i = 0; i < A.nnz; i++) { A.values(i) *= 2.0; }
  invA.update(A);
  vector y = dot(invA, b);
  EXPECT_LT(relative_residual(A, y, b), 1.0e-10);
}

TEST(SparseDirect, EigenSparseLU) { check_backend(SparseFactorizationBackend::EIGEN_SPARSE_LU, false); }
TEST(SparseDirect, EigenSimplicialLLT) { check_backend(SparseFactorizationBackend::EIGEN_SIMPLICIAL_LLT, true); }
TEST(SparseDirect, EigenSimplicialLDLT) { check_backend(SparseFactorizationBackend::EIGEN_SIMPLICIAL_LDLT, true); }

#ifdef FEMTO_ENABLE_MKL
TEST(SparseDirect, PardisoLU) { check_backend(SparseFactorizationBackend::MKL_PARDISO_LU, false); }
TEST(SparseDirect, PardisoLLT) { check_backend(SparseFactorizationBackend::MKL_PARDISO_LLT, true); }
TEST(SparseDirect, PardisoLDLT) { check_backend(SparseFactorizationBackend::MKL_PARDISO_LDLT, true); }
#endif

#ifdef FEMTO_ENABLE_PASTIX
TEST(SparseDirect, PastixLU) { check_backend(SparseFactorizationBackend::PASTIX_LU, false); }
TEST(SparseDirect, PastixLLT) { check_backend(SparseFactorizationBackend::PASTIX_LLT, true); }
TEST(SparseDirect, PastixLDLT) { check_backend(SparseFactorizationBackend::PASTIX_LDLT, true); }
#endif

// inv(A) without an explicit backend prefers MKL, then PaStiX, then Eigen
TEST(SparseDirect, DefaultBackendSelection) {
  sparse_matrix<> A = model_matrix(20, false);
  vector b = ones(int(A.nrows));

  using enum SparseFactorizationBackend;
  auto invA = inv(A);
#if defined(FEMTO_ENABLE_MKL)
  EXPECT_EQ(invA.backend(), MKL_PARDISO_LLT);
#elif defined(FEMTO_ENABLE_PASTIX)
  EXPECT_EQ(invA.backend(), PASTIX_LLT);
#else
  EXPECT_EQ(invA.backend(), EIGEN_SIMPLICIAL_LLT);
#endif
  EXPECT_LT(relative_residual(A, dot(invA, b), b), 1.0e-10);
}
