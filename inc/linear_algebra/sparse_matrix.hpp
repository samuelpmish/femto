#pragma once

#include <algorithm>
#include <cstddef>
#include <string>
#include <functional>
#include <tuple>
#include <vector>

#include "vector.hpp"

#include "containers/ndarray.hpp"

#include "femto/assert.hpp"
#include "misc/timer.hpp"

namespace femto {

template < memory::space mem_space = memory::space::cpu >
struct sparsity_pattern {
  size_t nrows = 0, ncols = 0, nnz = 0;
  nd::array<int, 1, mem_space> row_ptr;
  nd::array<int, 1, mem_space> col_ind;

  sparsity_pattern() = default;
  sparsity_pattern(const sparsity_pattern & other) = default;
  sparsity_pattern& operator=(const sparsity_pattern & other) = default;

  template < memory::space other_space >
  sparsity_pattern(const sparsity_pattern<other_space> & other) :
    nrows(other.nrows),
    ncols(other.ncols),
    nnz(other.nnz),
    row_ptr(other.row_ptr),
    col_ind(other.col_ind) {}

  template < memory::space other_space >
  sparsity_pattern& operator=(const sparsity_pattern<other_space> & other) {
    nrows = other.nrows;
    ncols = other.ncols;
    nnz = other.nnz;
    row_ptr = other.row_ptr;
    col_ind = other.col_ind;
    return *this;
  }
};

#ifdef NDARRAY_ENABLE_CUDA
template <>
template <>
sparsity_pattern<memory::space::cpu>::sparsity_pattern(const sparsity_pattern<memory::space::gpu> & other);

template <>
template <>
sparsity_pattern<memory::space::cpu>&
sparsity_pattern<memory::space::cpu>::operator=(const sparsity_pattern<memory::space::gpu> & other);
#endif

using triplet = std::tuple<int, int, double>;

// structural properties of a matrix, declared by whoever assembles it
// (they are not verified), and used to select appropriate algorithms
// (e.g. which factorization inv() performs)
enum class Symmetry { Unsymmetric, Symmetric, Hermitian };
enum class Definiteness { Indefinite, PositiveDefinite, PositiveSemidefinite };

template < memory::space mem_space = memory::space::cpu >
struct sparse_matrix;

template < memory::space mem_space = memory::space::cpu >
struct sparse_matrix_slice {
  sparse_matrix<mem_space> * spmat;
  std::vector<int> rows;
  std::vector<int> cols; // sorted

  void operator=(std::function< double(int, int) > func);
};

template < memory::space mem_space >
struct sparse_matrix : public sparsity_pattern<mem_space> {

  static constexpr int ROW = 0;
  static constexpr int COL = 1;
  static constexpr int VALUE = 2;

  using pattern_type = sparsity_pattern<mem_space>;
  using slice_type = sparse_matrix_slice<mem_space>;

  using pattern_type::nrows;
  using pattern_type::ncols;
  using pattern_type::nnz;
  using pattern_type::row_ptr;
  using pattern_type::col_ind;

  static sparse_matrix with_sparsity(pattern_type pattern);
  static sparse_matrix with_dimensions(const size_t rows, const size_t cols, const size_t nonzeros);
  static sparse_matrix from_matrix_market(std::string filename);
  static sparse_matrix from_triplets(std::vector<triplet>& triplets, const size_t rows = 0, const size_t cols = 0);

  sparse_matrix() = default;
  sparse_matrix(const sparse_matrix & other) = default;
  sparse_matrix& operator=(const sparse_matrix & other) = default;

  template < memory::space other_space >
  sparse_matrix(const sparse_matrix<other_space> & other) :
    pattern_type(other),
    values(other.values),
    symmetry(other.symmetry),
    definiteness(other.definiteness) {}

  template < memory::space other_space >
  sparse_matrix& operator=(const sparse_matrix<other_space> & other) {
    pattern_type::operator=(other);
    values = other.values;
    symmetry = other.symmetry;
    definiteness = other.definiteness;
    return *this;
  }

  sparse_matrix(const slice_type & slice);

  void operator=(const slice_type & slice);

  sparse_matrix(const std::function< void(sparse_matrix &) > func) {
    func(*this);
  }

  sparse_matrix& operator=(const std::function< void(sparse_matrix &) > func) {
    func(*this);
    return *this;
  }

  slice_type operator()(const std::vector<int> & rows, const std::vector<int> & cols) {
    std::vector<int> cols_sorted = cols;
    std::sort(cols_sorted.begin(), cols_sorted.end());
    return slice_type{this, rows, cols_sorted};
  }

  vector operator()(const vector & x) const {
    CHECK_FOR_SIZE_MISMATCH(x.sz, ncols);
    vector Ax(nrows);
    for (int i = 0; i < nrows; i++) {
      Ax[i] = 0.0;
      for (int p = row_ptr[i]; p < row_ptr[i+1]; p++) {
        Ax[i] += values[p] * x[col_ind[p]];
      }
    }
    return Ax;
  }

  vector_expr operator()(const vector_view & x) const {
    CHECK_FOR_SIZE_MISMATCH(x.sz, ncols);
    return vector_expr{
      [&](double * Ax) {
        femto::timer stopwatch;
        stopwatch.start();
        for (int i = 0; i < nrows; i++) {
          Ax[i] = 0.0;
          for (int p = row_ptr[i]; p < row_ptr[i+1]; p++) {
            Ax[i] += values[p] * x[col_ind[p]];
          }
        }
        stopwatch.stop();
        if (print_timings) {
          std::cout << "sparse matvec time: " << stopwatch.elapsed() * 1000.0 << "ms" << std::endl;
        }
      },
      uint32_t(nrows)
    };
  }

  // assumes that col exists in the given row
  int search_row_for_given_column(int row, int col) const {
    for (int i = row_ptr[row]; i < row_ptr[row+1]; i++) {
      if (col_ind[i] == col) {
        return i;
      }
    }
    return -1;
  }

  double & at(int i, int j) {
    int id = search_row_for_given_column(i, j);
    return values[id];
  }

  const double & at(int i, int j) const {
    int id = search_row_for_given_column(i, j);
    return values[id];
  }

  nd::array<double, 1, mem_space> values;

  Symmetry symmetry = Symmetry::Unsymmetric;
  Definiteness definiteness = Definiteness::Indefinite;

};

#ifdef NDARRAY_ENABLE_CUDA
template <>
template <>
sparse_matrix<memory::space::cpu>::sparse_matrix(const sparse_matrix<memory::space::gpu> & other);

template <>
template <>
sparse_matrix<memory::space::cpu>&
sparse_matrix<memory::space::cpu>::operator=(const sparse_matrix<memory::space::gpu> & other);
#endif

void export_matlab(const sparse_matrix<> & A, std::string filename);

template < memory::space mem_space = memory::space::cpu >
sparse_matrix<mem_space> import_matrix_market(std::string filename);

template < memory::space mem_space >
void export_matrix_market(const sparse_matrix<mem_space> & A, std::string filename);

inline vector dot(const sparse_matrix<> & A, const vector & x) {
  return A(x);
}

#ifdef NDARRAY_ENABLE_CUDA
gpu_vector dot(
  const sparse_matrix<memory::space::gpu> & A,
  gpu_const_vector_view x);

inline gpu_vector dot(
  const sparse_matrix<memory::space::gpu> & A,
  const gpu_vector & x) {
  return dot(A, gpu_const_vector_view(x));
}
#endif

sparse_matrix<> triu(const sparse_matrix<> & A);

sparse_matrix<> tril(const sparse_matrix<> & A);

// sparse direct solver backends. The Eigen solvers are always available;
// MKL PARDISO is compiled in when found, PaStiX whenever MKL isn't (or when
// FEMTO_ENABLE_PASTIX is set), and cuDSS in every CUDA build. Requesting a
// backend that wasn't compiled in is a runtime error.
enum class SparseFactorizationBackend {
  EIGEN_SPARSE_LU,        // general square matrices
  EIGEN_SIMPLICIAL_LLT,   // symmetric positive definite matrices
  EIGEN_SIMPLICIAL_LDLT,  // symmetric quasi-definite matrices
  EIGEN_SIMPLICIAL_LLT_NATURAL,  // LLT without reordering: for SPD matrices whose rows are
                                 // already banded (e.g. level-ordered extruded meshes), where
                                 // AMD only adds fill and the solve is bandwidth-bound
  EIGEN_SIMPLICIAL_LDLT_NESTED_DISSECTION,  // LDLT on Scotch's nested-dissection ordering (borrowed from
                                            // PaStiX's analysis, so it needs the PaStiX build): the least
                                            // fill and the fastest solves of the CPU backends on 2D meshes
  MKL_PARDISO_LU,         // multithreaded MKL PARDISO variants of the above
  MKL_PARDISO_LLT,        // (LDLT pivots, so it also covers symmetric
  MKL_PARDISO_LDLT,       //  indefinite matrices)
  PASTIX_LU,              // multithreaded PaStiX variants, built from source
  PASTIX_LLT,             // (the backend wherever MKL is unavailable,
  PASTIX_LDLT,            //  including webassembly builds)
  CUDSS_LU,               // NVIDIA cuDSS, factorizing and solving on the GPU:
  CUDSS_LLT,              // the native backend for sparse_matrix<gpu>, and
  CUDSS_LDLT              // usable from host matrices (copied to the device)
};

struct sparse_factorization;

// factorize A with an explicitly-chosen backend
sparse_factorization inv(const sparse_matrix<> & A, SparseFactorizationBackend backend);

// the backend inv(A) would pick for A (MKL PARDISO when enabled, then PaStiX,
// then Eigen; the variant follows A's symmetry / definiteness)
SparseFactorizationBackend select_backend(const sparse_matrix<> & A);

#ifdef NDARRAY_ENABLE_CUDA
// device-memory matrices are factorized on the GPU with cuDSS, and the
// solves stay on the device: x = dot(invA, b) with gpu vectors
sparse_factorization inv(const sparse_matrix<memory::space::gpu> & A, SparseFactorizationBackend backend);
sparse_factorization inv(const sparse_matrix<memory::space::gpu> & A);
SparseFactorizationBackend select_backend(const sparse_matrix<memory::space::gpu> & A);
gpu_vector dot(const sparse_factorization & invA, const gpu_vector & b);
#endif

// factorize A, selecting a variant from A's symmetry / definiteness
// (LLT for symmetric positive definite, LDLT for other symmetric matrices,
// LU otherwise) from the preferred backend family: MKL PARDISO when enabled,
// then PaStiX, then Eigen's built-in solvers
sparse_factorization inv(const sparse_matrix<> & A);

vector dot(const sparse_factorization & invA, const vector & b);

// TODO: decide on consistent containers (instead of vector/buffer/nd::array)
nd::array<double, 1, memory::space::cpu> diagonal(const sparse_matrix<> & A);

}  // namespace femto
