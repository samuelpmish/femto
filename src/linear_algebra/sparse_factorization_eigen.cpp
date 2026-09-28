// Eigen's built-in sparse direct solvers: always available, but slower
// than the MKL PARDISO / PaStiX backends

#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>

#include "sparse_factorization_impl.hpp"

namespace femto {

std::unique_ptr<sparse_factorization::impl> make_eigen_impl(SparseFactorizationBackend which) {
  using enum SparseFactorizationBackend;
  switch (which) {
    case EIGEN_SPARSE_LU:
      return std::make_unique<eigen_impl<EIGEN_SPARSE_LU, Eigen::SparseLU<eigen_sparse_matrix>>>();
    case EIGEN_SIMPLICIAL_LLT:
      return std::make_unique<eigen_impl<EIGEN_SIMPLICIAL_LLT, Eigen::SimplicialLLT<eigen_sparse_matrix>>>();
    case EIGEN_SIMPLICIAL_LDLT:
      return std::make_unique<eigen_impl<EIGEN_SIMPLICIAL_LDLT, Eigen::SimplicialLDLT<eigen_sparse_matrix>>>();
    case EIGEN_SIMPLICIAL_LLT_NATURAL:
      return std::make_unique<eigen_impl<EIGEN_SIMPLICIAL_LLT_NATURAL,
        Eigen::SimplicialLLT<eigen_sparse_matrix, Eigen::Lower, Eigen::NaturalOrdering<int>>>>();
    default:
      return nullptr;  // unreachable, make_impl only forwards Eigen backends
  }
}

}  // namespace femto
