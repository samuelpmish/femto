#include "linear_algebra/sparse_matrix.hpp"

#include "misc/macros.hpp"

namespace femto {

namespace {

__global__ void csr_matvec_kernel(
  size_t nrows,
  const int * row_ptr,
  const int * col_ind,
  const double * values,
  const double * x,
  double * Ax) {
  size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= nrows) {
    return;
  }

  double sum = 0.0;
  for (int p = row_ptr[row]; p < row_ptr[row + 1]; p++) {
    sum += values[p] * x[col_ind[p]];
  }
  Ax[row] = sum;
}

template < typename T >
void copy_device_to_host(
  nd::array<T, 1, memory::space::cpu> & host,
  const nd::array<T, 1, memory::space::gpu> & device) {
  host.resize(device.shape);
  if (device.sz > 0) {
    CUDA_CHECK(cudaMemcpy(host.data(), device.data(), device.sz * sizeof(T), cudaMemcpyDeviceToHost));
  }
}

} // namespace

template <>
template <>
sparsity_pattern<memory::space::cpu>::sparsity_pattern(
  const sparsity_pattern<memory::space::gpu> & other) {
  *this = other;
}

template <>
template <>
sparsity_pattern<memory::space::cpu>&
sparsity_pattern<memory::space::cpu>::operator=(
  const sparsity_pattern<memory::space::gpu> & other) {
  nrows = other.nrows;
  ncols = other.ncols;
  nnz = other.nnz;
  copy_device_to_host(row_ptr, other.row_ptr);
  copy_device_to_host(col_ind, other.col_ind);
  return *this;
}

template <>
template <>
sparse_matrix<memory::space::cpu>::sparse_matrix(
  const sparse_matrix<memory::space::gpu> & other) {
  *this = other;
}

template <>
template <>
sparse_matrix<memory::space::cpu>&
sparse_matrix<memory::space::cpu>::operator=(
  const sparse_matrix<memory::space::gpu> & other) {
  sparsity_pattern<memory::space::cpu>::operator=(other);
  copy_device_to_host(values, other.values);
  symmetry = other.symmetry;
  definiteness = other.definiteness;
  return *this;
}

template <>
sparse_matrix<memory::space::gpu>
sparse_matrix<memory::space::gpu>::with_sparsity(sparsity_pattern<memory::space::gpu> pattern) {
  sparse_matrix<memory::space::gpu> A;
  static_cast<sparsity_pattern<memory::space::gpu> &>(A) = pattern;
  A.values.resize(A.nnz);
  return A;
}

template <>
sparse_matrix<memory::space::gpu>
sparse_matrix<memory::space::gpu>::with_dimensions(const size_t rows, const size_t cols, const size_t nonzeros) {
  sparse_matrix<memory::space::gpu> A;
  A.nrows = rows;
  A.ncols = cols;
  A.nnz = nonzeros;
  A.row_ptr.resize(rows + 1);
  A.col_ind.resize(nonzeros);
  A.values.resize(nonzeros);
  return A;
}

template <>
sparse_matrix<memory::space::gpu>
sparse_matrix<memory::space::gpu>::from_triplets(std::vector<triplet>& triplets, const size_t rows, const size_t cols) {
  return sparse_matrix<memory::space::gpu>(sparse_matrix<>::from_triplets(triplets, rows, cols));
}

template <>
sparse_matrix<memory::space::gpu>
sparse_matrix<memory::space::gpu>::from_matrix_market(std::string filename) {
  return sparse_matrix<memory::space::gpu>(sparse_matrix<>::from_matrix_market(filename));
}

template <>
sparse_matrix<memory::space::gpu>
import_matrix_market<memory::space::gpu>(std::string filename) {
  return sparse_matrix<memory::space::gpu>::from_matrix_market(filename);
}

template <>
void export_matrix_market<memory::space::gpu>(
  const sparse_matrix<memory::space::gpu> & A,
  std::string filename) {
  sparse_matrix<> A_host(A);
  export_matrix_market<memory::space::cpu>(A_host, filename);
}

gpu_vector dot(
  const sparse_matrix<memory::space::gpu> & A,
  gpu_const_vector_view x) {
  CHECK_FOR_SIZE_MISMATCH(x.sz, A.ncols);

  gpu_vector Ax(uint32_t(A.nrows));
  constexpr uint32_t block_size = 256;
  uint32_t grid_size = (uint32_t(A.nrows) + block_size - 1) / block_size;

  if (A.nrows > 0) {
    csr_matvec_kernel<<<grid_size, block_size>>>(
      A.nrows,
      A.row_ptr.data(),
      A.col_ind.data(),
      A.values.data(),
      x.ptr,
      Ax.ptr);
    CUDA_CHECK(cudaGetLastError());
  }

  return Ax;
}

} // namespace femto
