#pragma once

#include <cinttypes>

namespace memory {
  enum class space {cpu, gpu, unified};

  inline constexpr space CPU = space::cpu;
  inline constexpr space GPU = space::gpu;
  inline constexpr space UNIFIED = space::unified;

////////////////////////////////////////////////////////////////////////////////

  template < space mem_space >
  void * allocate(uint64_t n);

  template < space mem_space >
  void deallocate(void * ptr);

  template < space dest_space, space src_space >
  void memcpy(void * dest, const void * src, uint64_t n);

  template < space mem_space >
  void zero(void * ptr, uint64_t n);

////////////////////////////////////////////////////////////////////////////////

  template < typename T, space mem_space >
  T * allocate(uint64_t n) {
    return static_cast<T*>(allocate<mem_space>(n * sizeof(T)));
  }

  template < typename T, space mem_space >
  void deallocate(T * ptr) {
    deallocate<mem_space>(static_cast<void*>(ptr));
  }

  template < typename T, space dest_space, space src_space >
  void memcpy(T * dest, const T * src, uint64_t n) {
    memcpy<dest_space, src_space>(static_cast<void*>(dest), static_cast<const void*>(src), n * sizeof(T));
  }

  template < typename T, space mem_space >
  void zero(T * ptr, uint64_t n) {
    zero<mem_space>(static_cast<void*>(ptr), n * sizeof(T));
  }

}
