#pragma once

// This file is not meant to be included directly

template < int n, int m >
auto get_gradient(vec<n, dual< double, vec<m> > > x) {
  mat<n,m, double> output;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      output(i,j) = x[i].gradient[j];
    }
  }
  return output;
}
