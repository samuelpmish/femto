#pragma once

// This file is not meant to be included directly

template < typename V, typename G, int n, typename T >
auto operator*(const dual<V,G> a, const vec<n,T> & x) {
  vec<n, decltype(dual<V,G>{} * T{})> z{};
  for (int i = 0; i < n; i++) z[i] = a * x[i];
  return z;
}

template < typename V, typename G, int n, typename T >
auto operator*(const vec<n,T> & x, const dual<V,G> a) {
  vec<n, decltype(T{} * dual<V,G>{})> z{};
  for (int i = 0; i < n; i++) z[i] = x[i] * a;
  return z;
}

template < int n >
auto gradient_wrt(vec<n, double> x) {
  vec<n, dual<double, vec<n, double> > > x_dual;
  for (int i = 0; i < n; i++) {
    x_dual[i].value = x[i];
    for (int j = 0; j < n; j++) {
      x_dual[i].gradient[j] = (i == j);
    }
  }
  return x_dual;
}

template < int n >
auto get_gradient(vec<n, dual< double, double > > x) {
  vec<n, double> output;
  for (int i = 0; i < n; i++) output[i] = x[i].gradient;
  return output;
}

template < int n >
auto hessian_wrt(vec<n, double> x) {
  vec< n, dual< dual< double, vec<n> >, vec<n, dual< double, vec <n> > > > > x_dual{};
  for (int i = 0; i < n; i++) {
    x_dual[i].value.value = x[i];
    for (int j = 0; j < n; j++) {
      x_dual[i].gradient[j].value = (i == j);
      x_dual[i].value.gradient[j] = (i == j);
    }
  }
  return x_dual;
}

#if defined FEMTO_MAT
  #include "dual_vec_mat.hpp"
#endif