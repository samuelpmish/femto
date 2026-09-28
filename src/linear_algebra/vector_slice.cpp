#include "linear_algebra/vector.hpp"

#include "femto/assert.hpp"

#include <iostream>

namespace femto {

  double norm(const_vector_slice v) {
    double total{};
    for (int i = 0; i < v.sz; i++) {
      double value = v.ptr[v.ids[i]];
      total += value * value;
    }
    return sqrt(total);
  }

  double total(const_vector_slice v) {
    double sum = 0.0;
    for (int i = 0; i < v.sz; i++) {
      sum += v.ptr[v.ids[i]];
    }
    return sum;
  }

  std::ostream & operator<<(std::ostream & out, const_vector_slice v) {
    out << "{";
    for (int i = 0; i < v.sz; i++) {
      out << v.ptr[v.ids[i]];
      if (i+1 < v.sz) { out << ", "; }
    }
    out << "}";
    return out;
  }

}