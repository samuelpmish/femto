#pragma once

#include <memory>

#include "linear_algebra/sparse_matrix.hpp"
#include "linear_algebra/vector.hpp"

namespace femto {

// a factorization of a sparse matrix, usable as a direct solver:
//
//   A.symmetry = Symmetry::Symmetric;
//   A.definiteness = Definiteness::PositiveDefinite;
//   auto invA = inv(A); // picks Cholesky, based on the properties above
//   vector x = dot(invA, b);
//   ...
//   invA.update(A_new); // refactorize (sparsity pattern must be unchanged)
//
// the backend can also be specified explicitly:
//
//   auto invA = inv(A, SparseFactorizationBackend::EIGEN_SPARSE_LU);
//
// and device-memory matrices (CUDA builds) factorize on the GPU with cuDSS:
//
//   sparse_matrix<memory::space::gpu> A_gpu = A;
//   auto invA = inv(A_gpu);            // CUDSS_LLT / LDLT / LU from A's properties
//   gpu_vector x = dot(invA, b_gpu);   // solve on the device
//
// the actual solver types are kept out of this header
// (see src/linear_algebra/sparse_factorization_{eigen,mkl,pastix}.cpp and
// sparse_factorization_cudss.cu)
struct sparse_factorization {

  sparse_factorization();
  sparse_factorization(sparse_factorization &&) noexcept;
  sparse_factorization & operator=(sparse_factorization &&) noexcept;
  ~sparse_factorization();

  // recompute the numeric factorization for a matrix with
  // the same sparsity pattern as the one originally factorized
  void update(const sparse_matrix<> & A);
#ifdef NDARRAY_ENABLE_CUDA
  void update(const sparse_matrix<memory::space::gpu> & A);
#endif

  SparseFactorizationBackend backend() const;

  struct impl;
  std::unique_ptr<impl> pimpl;

};

#ifdef FEMTO_ENABLE_PASTIX
// solver benchmark (used by the web demos): factor an SPD matrix by Cholesky
// with Eigen's simplicial solver or PaStiX under a given fill-reducing
// ordering (and, for PaStiX, thread count), and time the factorization and
// the average of `reps` triangular solves
enum class Ordering { NATURAL, AMD, NESTED_DISSECTION };
struct SolverTiming { double factor_ms, solve_ms; long nnz_factor; };
SolverTiming benchmark_eigen_llt(const sparse_matrix<> & A, Ordering ordering, int reps);
SolverTiming benchmark_pastix_llt(const sparse_matrix<> & A, Ordering ordering, int threads, int reps);
#endif

}  // namespace femto
