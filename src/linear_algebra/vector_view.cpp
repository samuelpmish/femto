#include "linear_algebra/vector.hpp"

#include "femto/assert.hpp"

#include <fstream>
#include <iostream>

namespace femto {

  using u32 = uint32_t;

  void operator+=(vector_view u, const const_vector_view v) {
    if (u.size() != v.size()) {
      std::cout << "error: cannot add vectors of different sizes" << std::endl;
      exit(1);
    }

    for (u32 i = 0; i < v.sz; i++) {
      u[i] += v[i];
    }
  }

  void operator-=(vector_view u, const const_vector_view v) {
    if (u.size() != v.size()) {
      std::cout << "error: cannot subtract vectors of different sizes" << std::endl;
      exit(1);
    }

    for (u32 i = 0; i < v.sz; i++) {
      u[i] -= v[i];
    }
  }

  void operator*=(vector_view u, double scale) {
    for (u32 i = 0; i < u.sz; i++) {
      u[i] *= scale;
    }
  }

  void operator/=(vector_view u, double scale) {
    double inv_scale = 1.0 / scale;
    for (u32 i = 0; i < u.sz; i++) {
      u[i] *= inv_scale;
    }
  }

  double dot(const const_vector_view & u, const const_vector_view & v) {
    if (u.size() != v.size()) {
      std::cout << "error: cannot take the dot product of vectors with different sizes" << std::endl;
      exit(1);
    }

    double uTv = 0.0;
    for (int i = 0; i < u.size(); i++) {
      uTv += u[i] * v[i];
    }

    return uTv;
  }

  double norm(const_vector_view v) {
    return sqrt(dot(v, v));
  }

  double total(const_vector_view v) {
    double sum = 0.0;
    for (int i = 0; i < v.size(); i++) {
      sum += v[i];
    }
    return sum;
  }

  std::ostream & operator<<(std::ostream & out, const_vector_view v) {
    out << "{";
    for (int i = 0; i < v.size(); i++) {
      out << v[i];
      if (i+1 < v.size()) {
        out << ", ";
      }
    }
    out << "}";
    return out;
  }

  void export_vector(const_vector_view v, std::string filename) {
    std::ofstream outfile(filename, std::ios::binary);

    if (outfile) {
      outfile.write((char*)&v[0], sizeof(double) * v.size());
    } else {
      std::cout << "file not found: " << filename << std::endl;
    }

    outfile.close();
  }

}