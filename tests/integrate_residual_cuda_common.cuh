#pragma once

#include <gtest/gtest.h>

#include "femto/domain.hpp"
#include "femto/mesh.hpp"
#include "forall.hpp"

#include <cuda_runtime.h>

#include <type_traits>

namespace residual_cuda_tests {

using namespace femto;
using namespace fm;

inline bool cuda_device_available() {
  int device_count = 0;
  cudaError_t error = cudaGetDeviceCount(&device_count);
  if (error != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return device_count > 0;
}

template < Family family, memory::space mem_space >
double host_dot(const Residual<family, mem_space> & r, const Field<family, mem_space> & u) {
  Residual<family> r_cpu = r;
  Field<family> u_cpu = u;
  return dot(r_cpu, u_cpu);
}

template < Family family, memory::space lhs_space, memory::space rhs_space >
double host_relative_error(const Residual<family, lhs_space> & lhs, const Residual<family, rhs_space> & rhs) {
  Residual<family> lhs_cpu = lhs;
  Residual<family> rhs_cpu = rhs;
  return relative_error(lhs_cpu.data, rhs_cpu.data);
}

template < typename Vec, typename F >
struct ScalarAtDegree {
  __host__ __device__ double operator()(Vec x) const { return f(x, p); }

  F f;
  int p;
};

template < typename Vec, typename F >
struct VectorAtDegree {
  __host__ __device__ Vec operator()(Vec x) const {
    constexpr int dim = dimension(Vec{});
    Vec output{};
    for (int i = 0; i < dim; i++) {
      output[i] = f(x, p) * (i + 1);
    }
    return output;
  }

  F f;
  int p;
};

template < typename Vec, typename F >
struct HcurlDofAtDegree {
  __host__ __device__ double operator()(Vec x, Vec n) const { return dot(f(x, p), n); }

  F f;
  int p;
};

template < typename Vec, typename G >
struct DetJWeightedScalar {
  using mat_t = decltype(outer(Vec{}, Vec{}));

  __host__ __device__ double operator()(Vec x, mat_t J) const { return g(x) * det(J); }

  G g;
};

template < typename Vec, typename G >
struct CovariantWeightedVector {
  using mat_t = decltype(outer(Vec{}, Vec{}));

  __host__ __device__ Vec operator()(Vec x, mat_t J) const {
    if constexpr (std::is_same<Vec, double>::value) {
      return g(x);
    } else {
      return dot(inv(J), g(x)) * det(J);
    }
  }

  G g;
};

template < typename Vec, typename G >
struct HcurlFluxReference {
  using mat_t = decltype(outer(Vec{}, Vec{}));

  __host__ __device__ auto operator()(Vec x, mat_t J) const {
    if constexpr (std::is_same<mat_t, mat2>::value) {
      return g(x);
    } else {
      return dot(transpose(J), g(x));
    }
  }

  G g;
};

template < typename Vec, typename G >
struct H1VectorFluxReference {
  using mat_t = decltype(outer(Vec{}, Vec{}));

  __host__ __device__ mat_t operator()(Vec x, mat_t J) const {
    constexpr int dim = dimension(Vec{});
    mat_t output{};
    Vec transformed = dot(inv(J), g(x)) * det(J);
    for (int i = 0; i < dim; i++) {
      output[i] = transformed * (i + 1);
    }
    return output;
  }

  G g;
};

// boundary-domain counterparts of the above: a facet's (dim x dim - 1)
// jacobian has no determinant, the isoparametric weight is its measure
template < uint32_t sdim, uint32_t gdim >
__host__ __device__ double facet_measure(const mat<sdim, gdim> & J) {
  return sqrt(det(dot(transpose(J), J)));
}

template < typename Vec, typename G >
struct FacetWeightedScalar {
  using jac_t = mat<dimension(Vec{}), dimension(Vec{}) - 1>;

  __host__ __device__ double operator()(Vec x, jac_t J) const { return g(x) * facet_measure(J); }

  G g;
};

// the same scalar in every component, for vector-valued test spaces
template < typename Vec, typename G >
struct VectorFromScalar {
  __host__ __device__ Vec operator()(Vec x) const {
    constexpr int dim = dimension(Vec{});
    Vec output{};
    for (int i = 0; i < dim; i++) { output[i] = g(x); }
    return output;
  }

  G g;
};

template < typename Vec, typename G >
struct FacetWeightedVector {
  using jac_t = mat<dimension(Vec{}), dimension(Vec{}) - 1>;

  __host__ __device__ Vec operator()(Vec x, jac_t J) const {
    constexpr int dim = dimension(Vec{});
    Vec output{};
    for (int i = 0; i < dim; i++) { output[i] = g(x) * facet_measure(J); }
    return output;
  }

  G g;
};

} // namespace residual_cuda_tests
