#include "containers/memory.hpp"

#include <cstdlib>

#include "misc/macros.hpp"

namespace memory {

template <>
void * allocate<space::gpu>(uint64_t n) {
  void * ptr;
  CUDA_CHECK(cudaMalloc(&ptr, n));
  return ptr;
}

template <>
void * allocate<space::unified>(uint64_t n) {
  void * ptr;
  CUDA_CHECK(cudaMallocManaged(&ptr, n));
  return ptr;
}

template <>
void deallocate<space::gpu>(void * ptr) {
  CUDA_CHECK(cudaFree(ptr));
}

template <>
void deallocate<space::unified>(void * ptr) {
  CUDA_CHECK(cudaFree(ptr));
}

template <>
void memcpy<space::gpu, space::cpu>(void * dest, const void * src, uint64_t n) {
  CUDA_CHECK(cudaMemcpy(dest, src, n, cudaMemcpyDefault));
}

template <>
void memcpy<space::gpu, space::gpu>(void * dest, const void * src, uint64_t n) {
  CUDA_CHECK(cudaMemcpy(dest, src, n, cudaMemcpyDefault));
}

template <>
void memcpy<space::gpu, space::unified>(void * dest, const void * src, uint64_t n) {
  CUDA_CHECK(cudaMemcpy(dest, src, n, cudaMemcpyDefault));
}

template <>
void memcpy<space::cpu, space::gpu>(void * dest, const void * src, uint64_t n) {
  CUDA_CHECK(cudaMemcpy(dest, src, n, cudaMemcpyDefault));
}

template <>
void memcpy<space::cpu, space::unified>(void * dest, const void * src, uint64_t n) {
  CUDA_CHECK(cudaMemcpy(dest, src, n, cudaMemcpyDefault));
}

template <>
void memcpy<space::unified, space::cpu>(void * dest, const void * src, uint64_t n) {
  CUDA_CHECK(cudaMemcpy(dest, src, n, cudaMemcpyDefault));
}

template <>
void memcpy<space::unified, space::gpu>(void * dest, const void * src, uint64_t n) {
  CUDA_CHECK(cudaMemcpy(dest, src, n, cudaMemcpyDefault));
}

template <>
void memcpy<space::unified, space::unified>(void * dest, const void * src, uint64_t n) {
  CUDA_CHECK(cudaMemcpy(dest, src, n, cudaMemcpyDefault));
}

template <>
void zero<space::gpu>(void * ptr, uint64_t n) {
  CUDA_CHECK(cudaMemset(ptr, 0, n));
}

template <>
void zero<space::unified>(void * ptr, uint64_t n) {
  CUDA_CHECK(cudaMemset(ptr, 0, n));
}

}
