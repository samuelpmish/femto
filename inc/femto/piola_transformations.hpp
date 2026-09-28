#pragma once

#include "fm/types/vec.hpp"
#include "fm/types/matrix.hpp"

namespace femto {

// these are the piola transformations that user-written qfunctions call, so
// they need to be callable from device code as well as from the host

template < typename grad_t, typename jac_t >
__host__ __device__ auto contravariant_piola(const grad_t & du_dxi, const jac_t & dx_dxi) {
  return dot(du_dxi, inv(dx_dxi));
};

template < typename curl_t, typename jac_t >
__host__ __device__ curl_t covariant_piola(const curl_t & curl_xi, const jac_t & dx_dxi) {
  if constexpr (std::is_same< jac_t, mat2 >::value) {
    return curl_xi / det(dx_dxi);
  } else {
    return dot(dx_dxi, curl_xi) / det(dx_dxi);
  }
}

template < typename jac_t >
__host__ __device__ auto contravariant_piola(const jac_t & dx_dxi) {
  return inv(dx_dxi);
};

template < typename jac_t >
__host__ __device__ auto covariant_piola(const jac_t & dx_dxi) {
  if constexpr (std::is_same< jac_t, mat2 >::value) {
    return mat<1,1,double>{1.0 / det(dx_dxi)};
  } else {
    return dx_dxi / det(dx_dxi);
  }
}

}