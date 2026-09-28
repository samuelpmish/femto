#include "containers/ndarray.hpp"

#include "misc/macros.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace {

template < uint32_t rank >
__global__ void relative_error_kernel(
  nd::view<const double, rank, memory::space::gpu> x,
  nd::view<const double, rank, memory::space::gpu> y,
  double * block_error,
  double * block_norm) {
  extern __shared__ double shmem[];
  double * error_sums = shmem;
  double * norm_sums = shmem + blockDim.x;

  uint32_t tid = threadIdx.x;
  uint32_t stride = blockDim.x * gridDim.x;
  uint32_t sz = x.size();

  double error = 0.0;
  double norm = 0.0;
  for (uint32_t i = blockIdx.x * blockDim.x + tid; i < sz; i += stride) {
    double avg = 0.5 * (x.values[i] + y.values[i]);
    double diff = x.values[i] - y.values[i];
    error += diff * diff;
    norm += avg * avg;
  }

  error_sums[tid] = error;
  norm_sums[tid] = norm;
  __syncthreads();

  for (uint32_t offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (tid < offset) {
      error_sums[tid] += error_sums[tid + offset];
      norm_sums[tid] += norm_sums[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    block_error[blockIdx.x] = error_sums[0];
    block_norm[blockIdx.x] = norm_sums[0];
  }
}

template < uint32_t rank >
double relative_error_impl(
  nd::view<const double, rank, memory::space::gpu> x,
  nd::view<const double, rank, memory::space::gpu> y) {
  if (x.shape != y.shape) {
    std::cout << "shape mismatch" << std::endl;
  }

  constexpr uint32_t block_size = 256;
  uint32_t grid_size = std::max(1u, std::min(1024u, (x.size() + block_size - 1) / block_size));

  nd::array<double, 1, memory::space::gpu> block_error({grid_size});
  nd::array<double, 1, memory::space::gpu> block_norm({grid_size});

  relative_error_kernel<rank><<<grid_size, block_size, 2 * block_size * sizeof(double)>>>(
    x,
    y,
    block_error.data(),
    block_norm.data());
  CUDA_CHECK(cudaGetLastError());

  nd::array<double, 1, memory::space::cpu> host_error = block_error;
  nd::array<double, 1, memory::space::cpu> host_norm = block_norm;

  double error = 0.0;
  double norm = 0.0;
  for (uint32_t i = 0; i < grid_size; i++) {
    error += host_error(i);
    norm += host_norm(i);
  }

  return sqrt(error / norm);
}

} // namespace

double relative_error(
  nd::view<const double, 1, memory::space::gpu> a,
  nd::view<const double, 1, memory::space::gpu> b) {
  return relative_error_impl(a, b);
}

double relative_error(
  nd::view<const double, 2, memory::space::gpu> a,
  nd::view<const double, 2, memory::space::gpu> b) {
  return relative_error_impl(a, b);
}

double relative_error(
  nd::view<const double, 3, memory::space::gpu> a,
  nd::view<const double, 3, memory::space::gpu> b) {
  return relative_error_impl(a, b);
}

double relative_error(
  nd::view<const double, 4, memory::space::gpu> a,
  nd::view<const double, 4, memory::space::gpu> b) {
  return relative_error_impl(a, b);
}

double relative_error(
  nd::view<const double, 5, memory::space::gpu> a,
  nd::view<const double, 5, memory::space::gpu> b) {
  return relative_error_impl(a, b);
}
