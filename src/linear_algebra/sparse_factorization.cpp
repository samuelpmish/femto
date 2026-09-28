// backend-independent parts of the sparse direct solver interface: backend
// selection and the public inv() / dot() / update() entry points for host
// matrices (the device-memory overloads live in sparse_factorization_cudss.cu)

#include "sparse_factorization_impl.hpp"

namespace femto {

namespace {


std::unique_ptr<sparse_factorization::impl> make_impl(SparseFactorizationBackend backend) {
  using enum SparseFactorizationBackend;
  switch (backend) {
    case EIGEN_SPARSE_LU:
    case EIGEN_SIMPLICIAL_LLT:
    case EIGEN_SIMPLICIAL_LDLT:
    case EIGEN_SIMPLICIAL_LLT_NATURAL:
      return make_eigen_impl(backend);
    case EIGEN_SIMPLICIAL_LDLT_NESTED_DISSECTION:
#ifdef FEMTO_ENABLE_PASTIX
      return make_eigen_nested_dissection_impl();
#else
      break;
#endif
    case MKL_PARDISO_LU:
    case MKL_PARDISO_LLT:
    case MKL_PARDISO_LDLT:
#ifdef FEMTO_ENABLE_MKL
      return make_mkl_impl(backend);
#else
      break;
#endif
    case PASTIX_LU:
    case PASTIX_LLT:
    case PASTIX_LDLT:
#ifdef FEMTO_ENABLE_PASTIX
      return make_pastix_impl(backend);
#else
      break;
#endif
    case CUDSS_LU:
    case CUDSS_LLT:
    case CUDSS_LDLT:
#ifdef FEMTO_ENABLE_CUDA
      return make_cudss_impl(backend);
#else
      break;
#endif
  }
  std::cout << "error: sparse factorization backend not compiled in" << std::endl;
  exit(1);
}

}  // namespace

// the backend family is decided at compile time (MKL PARDISO when enabled,
// then PaStiX, then Eigen's built-in solvers), and the variant within the
// family follows the matrix properties. Entries are real-valued, so
// Hermitian and Symmetric are equivalent.
SparseFactorizationBackend select_backend(const sparse_matrix<> & A) {
  using enum SparseFactorizationBackend;
  bool symmetric = (A.symmetry != Symmetry::Unsymmetric);
  bool spd = symmetric && (A.definiteness == Definiteness::PositiveDefinite);
#if defined(FEMTO_ENABLE_MKL)
  // PARDISO's LDLT pivots, so it covers the symmetric indefinite case too
  if (spd) { return MKL_PARDISO_LLT; }
  if (symmetric) { return MKL_PARDISO_LDLT; }
  return MKL_PARDISO_LU;
#elif defined(FEMTO_ENABLE_PASTIX)
  if (spd) { return PASTIX_LLT; }
  if (symmetric) { return PASTIX_LDLT; }
  return PASTIX_LU;
#else
  if (spd) { return EIGEN_SIMPLICIAL_LLT; }
  if (symmetric && A.definiteness == Definiteness::PositiveSemidefinite) {
    return EIGEN_SIMPLICIAL_LDLT;
  }
  // LU handles the unsymmetric and symmetric-indefinite cases
  // (Eigen's LDLT lacks the 2x2 pivoting needed for the latter)
  return EIGEN_SPARSE_LU;
#endif
}

sparse_factorization::sparse_factorization() = default;
sparse_factorization::sparse_factorization(sparse_factorization &&) noexcept = default;
sparse_factorization & sparse_factorization::operator=(sparse_factorization &&) noexcept = default;
sparse_factorization::~sparse_factorization() = default;

void sparse_factorization::update(const sparse_matrix<> & A) {
  pimpl->factorize(A, false);
}

SparseFactorizationBackend sparse_factorization::backend() const {
  return pimpl->backend();
}

sparse_factorization inv(const sparse_matrix<> & A, SparseFactorizationBackend backend) {
  if (A.nrows != A.ncols) {
    std::cout << "error: requesting factorization of rectangular matrix" << std::endl;
    exit(1);
  }

  sparse_factorization invA;
  invA.pimpl = make_impl(backend);
  invA.pimpl->factorize(A, true);
  return invA;
}

sparse_factorization inv(const sparse_matrix<> & A) {
  return inv(A, select_backend(A));
}

vector dot(const sparse_factorization & invA, const vector & b) {
  vector x(b.size());
  Eigen::Map<const Eigen::VectorXd> b_map(&b[0], b.size());
  Eigen::Map<Eigen::VectorXd> x_map(&x[0], x.size());
  invA.pimpl->solve(b_map, x_map);
  return x;
}

}  // namespace femto
