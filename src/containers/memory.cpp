#include "containers/memory.hpp"

#include <cstdlib>
#include <cstring>

namespace memory {

template <>
void * allocate<space::cpu>(uint64_t n) {
  return std::malloc(n);
}

template <>
void deallocate<space::cpu>(void * ptr) {
  std::free(ptr);
}

template <>
void memcpy<space::cpu, space::cpu>(void * dest, const void * src, uint64_t n) {
  std::memcpy(dest, src, n);
}

template <>
void zero<space::cpu>(void * ptr, uint64_t n) {
  std::memset(ptr, 0, n);
}

}
