#pragma once

#include "containers/ndarray.hpp"

#include "femto/assert.hpp"

template < typename T, uint32_t n, memory::space mem_space >
auto operator+(const nd::array<T, n, mem_space> & x, const nd::array<T, n, mem_space> & y) {
  FEMTO_ASSERT(x.shape == y.shape, "adding arrays with incompatible shapes");
  FEMTO_ASSERT(x.stride == y.stride, "adding arrays with incompatible strides");

  nd::array< T, n, mem_space > output(x.shape);
  for (int i = 0; i < output.size(); i++) {
    output[i] = x[i] + y[i];
  }
  return output;
}

template < typename T, uint32_t n, memory::space mem_space >
auto operator-(const nd::array<T, n, mem_space> & x, const nd::array<T, n, mem_space> & y) {
  FEMTO_ASSERT(x.shape == y.shape, "subtracting arrays with incompatible shapes");
  FEMTO_ASSERT(x.stride == y.stride, "subtracting arrays with incompatible strides");

  nd::array< T, n, mem_space > output(x.shape);
  for (int i = 0; i < output.size(); i++) {
    output[i] = x[i] - y[i];
  }
  return output;
}

template < typename T, uint32_t n, memory::space mem_space >
auto operator*(const nd::array<T, n, mem_space> & x, double y) {
  nd::array< decltype(T{} * double{}), n, mem_space > output(x.shape);
  for (int i = 0; i < output.size(); i++) {
    output[i] = x[i] * y;
  }
  return output;
}

template < typename T, uint32_t n, memory::space mem_space >
auto operator*(double x, const nd::array<T, n, mem_space> & y) {
  nd::array< decltype(T{} * double{}), n, mem_space > output(y.shape);
  for (int i = 0; i < output.size(); i++) {
    output[i] = x * y[i];
  }
  return output;
}

template < typename T, uint32_t n, memory::space mem_space >
auto operator/(const nd::array<T, n, mem_space> & x, double y) {
  nd::array< decltype(T{} / double{}), n, mem_space > output(x.shape);
  for (int i = 0; i < output.size(); i++) {
    output[i] = x[i] / y;
  }
  return output;
}

template < typename T, uint32_t n, memory::space mem_space >
auto operator/(double x, const nd::array<T, n, mem_space> & y) {
  nd::array< decltype(T{} / double{}), n, mem_space > output(y.shape);
  for (int i = 0; i < output.size(); i++) {
    output[i] = x / y[i];
  }
  return output;
}
