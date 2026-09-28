#pragma once

// internal interface shared by the sparse direct solver backends
// (sparse_factorization{_eigen,_mkl,_pastix}.cpp) -- not part of the
// public headers in inc/

#include <cstdlib>
#include <iostream>
#include <memory>

#include <Eigen/SparseCore>

#include "linear_algebra/sparse_direct.hpp"

namespace femto {

// Eigen's sparse direct solvers require column-major storage, and the
// other backends consume the same compressed-column arrays
using eigen_sparse_matrix = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;

// femto's CSR arrays, viewed (without copying) as an Eigen row-major matrix
inline Eigen::Map<const Eigen::SparseMatrix<double, Eigen::RowMajor, int>> to_eigen(const sparse_matrix<> & A) {
  return Eigen::Map<const Eigen::SparseMatrix<double, Eigen::RowMajor, int>>(
      A.nrows, A.ncols, A.nnz, A.row_ptr.data(), A.col_ind.data(), A.values.data());
}

[[noreturn]] inline void backend_error(const char * message) {
  std::cout << "error: " << message << std::endl;
  exit(1);
}

struct sparse_factorization::impl {
  virtual ~impl() = default;

  // host matrices: the Eigen-interfaced backends take the column-major copy,
  // other backends may override the CSR entry point directly
  virtual void factorize(const eigen_sparse_matrix & A, bool analyze) = 0;
  virtual void factorize(const sparse_matrix<> & A, bool analyze) { factorize(to_eigen(A), analyze); }
  virtual void solve(Eigen::Map<const Eigen::VectorXd> b, Eigen::Map<Eigen::VectorXd> x) const = 0;

  // device matrices / vectors (pointers into device memory): cuDSS only
  virtual void factorize_gpu(const sparse_matrix<memory::space::gpu> &, bool) { backend_error("this sparse direct solver backend does not accept device-memory matrices"); }
  virtual void solve_gpu(const double *, double *, uint32_t) const { backend_error("this sparse direct solver backend does not solve with device-memory vectors"); }

  virtual SparseFactorizationBackend backend() const = 0;
};

// adapter for any solver with Eigen's decomposition interface
// (Eigen's own solvers and the MKL PARDISO wrappers)
template < SparseFactorizationBackend which, typename eigen_solver >
struct eigen_impl final : public sparse_factorization::impl {
  eigen_solver solver;

  void factorize(const eigen_sparse_matrix & A, bool analyze) override {
    if (analyze) {
      solver.analyzePattern(A);
    }
    solver.factorize(A);
    if (solver.info() != Eigen::Success) {
      std::cout << "error: sparse factorization failed" << std::endl;
      exit(1);
    }
  }

  void solve(Eigen::Map<const Eigen::VectorXd> b, Eigen::Map<Eigen::VectorXd> x) const override {
    x = solver.solve(b);
  }

  SparseFactorizationBackend backend() const override { return which; }
};

std::unique_ptr<sparse_factorization::impl> make_eigen_impl(SparseFactorizationBackend which);

#ifdef FEMTO_ENABLE_MKL
std::unique_ptr<sparse_factorization::impl> make_mkl_impl(SparseFactorizationBackend which);
#endif

#ifdef FEMTO_ENABLE_PASTIX
std::unique_ptr<sparse_factorization::impl> make_pastix_impl(SparseFactorizationBackend which);
std::unique_ptr<sparse_factorization::impl> make_eigen_nested_dissection_impl();
#endif

#ifdef FEMTO_ENABLE_CUDA
std::unique_ptr<sparse_factorization::impl> make_cudss_impl(SparseFactorizationBackend which);
#endif

}  // namespace femto
