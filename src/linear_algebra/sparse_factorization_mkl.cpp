#ifdef FEMTO_ENABLE_MKL

// MKL PARDISO backends, through Eigen's wrapper types. The LLT/LDLT
// wrappers extract the triangular part themselves, so all three accept
// the same fully-populated matrix as the other backends.

#include <Eigen/PardisoSupport>

#include "sparse_factorization_impl.hpp"

namespace femto {

std::unique_ptr<sparse_factorization::impl> make_mkl_impl(SparseFactorizationBackend which) {
  using enum SparseFactorizationBackend;
  switch (which) {
    case MKL_PARDISO_LU:
      return std::make_unique<eigen_impl<MKL_PARDISO_LU, Eigen::PardisoLU<eigen_sparse_matrix>>>();
    case MKL_PARDISO_LLT:
      return std::make_unique<eigen_impl<MKL_PARDISO_LLT, Eigen::PardisoLLT<eigen_sparse_matrix>>>();
    case MKL_PARDISO_LDLT:
      return std::make_unique<eigen_impl<MKL_PARDISO_LDLT, Eigen::PardisoLDLT<eigen_sparse_matrix>>>();
    default:
      return nullptr;  // unreachable, make_impl only forwards MKL backends
  }
}

}  // namespace femto

#endif
