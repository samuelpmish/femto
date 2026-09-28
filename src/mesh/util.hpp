#pragma once

#include "femto/mesh/io.hpp"

#include <cmath>
#include <array>
#include <tuple>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>

inline void exit_with_error(std::string msg) {
  std::cout << msg << std::endl;
  exit(1);
}

template< size_t n >
static std::array<int, n> operator-(const std::array<int, n> & a, int offset) {
  std::array<int,n> output = a;
  for (auto & entry : output) entry -= offset; 
  return output;
}

template< size_t n >
static std::array<int, n> operator+(const std::array<int, n> & a, int offset) {
  std::array<int,n> output = a;
  for (auto & entry : output) entry += offset; 
  return output;
}

template < typename T, size_t n >
std::ostream & operator<<(std::ostream & out, const std::array<T,n> & arr) {
  for (int i = 0; i < n; i++) {
    out << arr[i];
    if (i != (n-1)) out << " ";
  }  
  return out;
}

template < typename T, size_t n >
std::array<T,n> binary_read_array(std::istream & infile) {
  std::array<T,n> out; 
  infile.read((char*)&out, sizeof(T) * n);
  return out;
}

template < typename T, size_t n >
void binary_write_array(std::ofstream & outfile, const std::array<T,n> & arr) {
  outfile.write((char*)&arr, sizeof(T) * n);
}


template < typename T >
T ascii_read(std::ifstream & infile) {
  T out; 
  infile >> out;
  return out;
}

template < typename ... T >
std::tuple<T...> ascii_read_tuple(std::ifstream & infile) {
  return std::tuple{ascii_read<T>(infile) ...};
}

template < typename T, size_t n >
fm::vec<n, T> ascii_read_array(std::istream & infile) {
  fm::vec<n, T> out; 
  for (int i = 0; i < n; i++) infile >> out[i];
  return out;
}  

using vec3f = std::array<float, 3>;

template < typename T, size_t n > 
inline std::array<T, n> normalize(const std::array<T,n> & v) {
  T norm = 0;
  for (size_t i = 0; i < n; i++) {
    norm += v[i] * v[i];
  }
  norm = std::sqrt(norm);
  T scale = 1.0 / norm;

  std::array<T,n> output;
  for (size_t i = 0; i < n; i++) {
    output[i] = v[i] * scale;
  }
  return output;
}

template < typename T >
auto cross(const std::array<T,3> & u, const std::array<T,3> & v) { 
  return std::array<T, 3>{
    u[1]*v[2]-u[2]*v[1],
    u[2]*v[0]-u[0]*v[2],
    u[0]*v[1]-u[1]*v[0]
  };
}

inline vec3f operator-(vec3f x, vec3f y) {
  return {x[0] - y[0], x[1] - y[1], x[2] - y[2]};
}

inline vec3f operator+(vec3f x, vec3f y) {
  return {x[0] + y[0], x[1] + y[1], x[2] + y[2]};
}

inline vec3f operator*(vec3f x, float scale) {
  return {x[0] * scale, x[1] * scale, x[2] * scale};
}

inline std::vector<int> identity_permutation(int n) {
  std::vector<int> P(n);
  for (int i = 0; i < n; i++) { P[i] = i; }
  return P;
}

static const bool is_big_endian = [](){
  int64_t x16 = 1;
  int32_t *x8 = reinterpret_cast<int32_t*>(&x16);
  return !*x8;
}();

template <typename T>
T to_big_endian(T value) {
  if (!is_big_endian) {
    auto it = reinterpret_cast<uint8_t*>(&value);
    std::reverse(it, it + sizeof(T)); // value is now in swapped from little to big endianness
  }
  return value;
}

template <typename T>
T from_big_endian(T value) { to_big_endian(value); }