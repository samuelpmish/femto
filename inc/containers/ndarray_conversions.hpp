#pragma once

#include "containers/ndarray.hpp"

#include "fm/types/vec.hpp"
#include "fm/types/matrix.hpp"

template < int n, typename T, memory::space mem_space >
auto view(nd::array<T, 2, mem_space> & arr) {
  nd::view< fm::vec<n,T>, 1, mem_space > output;
  output.values = reinterpret_cast<fm::vec<n,T> *>(arr.begin());
  output.stride[0] = 1;
  output.shape[0] = arr.shape[0];
  output.sz = arr.shape[0];
  return output;
}

template < int m, int n, typename T, memory::space mem_space >
auto view(nd::array<T, 3, mem_space> & arr) {
  nd::view< fm::mat<m,n,T>, 1, mem_space > output;
  output.values = reinterpret_cast<fm::mat<m,n,T> *>(arr.begin());
  output.stride[0] = 1;
  output.shape[0] = arr.shape[0];
  output.sz = arr.shape[0];
  return output;
}
