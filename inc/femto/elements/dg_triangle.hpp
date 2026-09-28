#pragma once

#include "femto/connection.hpp" 

namespace femto {

// clang-format off
template <>
struct FiniteElement<Geometry::Triangle, Family::DG> : private FiniteElement<Geometry::Triangle, Family::H1> {

  static constexpr int dim = 2;

  using source_type = vec1;
  using flux_type = vec2;

  using value_type = vec1;
  using grad_type = vec2;

  using FiniteElement<Geometry::Triangle, Family::H1>::nodes;

  using FiniteElement<Geometry::Triangle, Family::H1>::shape_function;
  using FiniteElement<Geometry::Triangle, Family::H1>::shape_function_gradient;
  using FiniteElement<Geometry::Triangle, Family::H1>::shape_function_derivative;

  using FiniteElement<Geometry::Triangle, Family::H1>::evaluate_shape_functions;
  using FiniteElement<Geometry::Triangle, Family::H1>::evaluate_shape_function_gradients;
  using FiniteElement<Geometry::Triangle, Family::H1>::evaluate_weighted_shape_functions;
  using FiniteElement<Geometry::Triangle, Family::H1>::evaluate_weighted_shape_function_gradients;

  FiniteElement(uint32_t p_) : FiniteElement<Geometry::Triangle, Family::H1>{p_}, two_sided{} {}

  __host__ __device__ uint32_t num_nodes() const { 
    uint32_t multiplier = two_sided ? 2 : 1;
    return multiplier * FiniteElement<Geometry::Triangle, Family::H1>::num_nodes();
  }

  __host__ __device__ uint32_t num_interior_nodes() const { 
    return FiniteElement<Geometry::Triangle, Family::H1>::num_nodes(); // all DG nodes are interior
  }

  void interior_nodes(nd::view<double, 2> xi) const {
    return FiniteElement<Geometry::Triangle, Family::H1>::nodes(xi); // all DG nodes are interior
  }

  __host__ __device__ void indices(const GeometryInfo & offsets, const Connection * tri, uint32_t * indices) const {
    if (two_sided) {
      // TODO
    } else {
      uint32_t nodes_per_tri = num_nodes();
      uint32_t tri_id = tri[Triangle::cell_offset].index;
      uint32_t base_offset = offsets.tri + tri_id * nodes_per_tri;
      for (uint32_t i = 0; i < nodes_per_tri; i++) {
        indices[i] = base_offset + i;
      } 
    }
  }

  __host__ __device__ uint32_t batch_interpolation_scratch_space(nd::view<const double,2> xi) const {
    return 0;
  }

  void interpolate(nd::view<value_type> values_q, nd::view<const double, 1> values_e, nd::view<const double, 2> shape_fns, double * /*buffer*/) const {
    int nnodes = num_nodes();
    int nqpts = values_q.shape[0];

    for (int q = 0; q < nqpts; q++) {
      double sum = 0.0;
      for (int i = 0; i < nnodes; i++) {
        sum += shape_fns(q, i) * values_e(i);
      }
      values_q(q) = sum;
    }
  }

  void gradient(nd::view<grad_type> gradients_q, nd::view<const double, 1> values_e, nd::view<const double, 3> shape_fn_grads, double * /*buffer*/) const {
    int nnodes = num_nodes();
    int nqpts = gradients_q.shape[0];
    for (int j = 0; j < nqpts; j++) {
      grad_type sum{};
      for (int i = 0; i < nnodes; i++) {
        sum[0] += values_e(i) * shape_fn_grads(j, i, 0);
        sum[1] += values_e(i) * shape_fn_grads(j, i, 1);
      }
      gradients_q(j) = sum;
    }
  }

  __host__ __device__ void integrate_source(nd::view<double> residual_e, nd::view<const source_type> source_q, nd::view<const double, 2> shape_fn, double * /*buffer*/) const {
    int nnodes = num_nodes();
    int nqpts = source_q.shape[0];

    for (int i = 0; i < nnodes; i++) {
      double sum = 0.0;
      for (int q = 0; q < nqpts; q++) {
        sum += shape_fn(q, i) * source_q(q);
      }
      residual_e(i) = sum;
    }
  }

  void integrate_flux(nd::view<double> residual_e, nd::view<const flux_type> flux_q, nd::view<const double, 3> shape_fn_grads, double * /*buffer*/) const {
    int nnodes = num_nodes();
    int nqpts = flux_q.shape[0];

    for (int i = 0; i < nnodes; i++) {
      double sum = 0.0;
      for (int q = 0; q < nqpts; q++) {
        for (int d = 0; d < dim; d++) {
          sum += shape_fn_grads(q, i, d) * flux_q(q)[d];
        }
      }
      residual_e(i) = sum;
    }
  }

  #ifdef __CUDACC__
  __device__ void cuda_interpolate(nd::view<value_type> values_q, nd::view<const double, 1> values_e, nd::view<const double, 2> shape_fns, double * /*buffer*/) const {
    int nnodes = num_nodes();
    int nqpts = values_q.shape[0];

    for (int q = threadIdx.x; q < nqpts; q += blockDim.x) {
      double sum = 0.0;
      for (int i = 0; i < nnodes; i++) {
        sum += shape_fns(q, i) * values_e(i);
      }
      values_q(q) = sum;
    }
  }

  __device__ void cuda_gradient(nd::view<grad_type> gradients_q, nd::view<const double, 1> values_e, nd::view<const double, 3> shape_fn_grads, double * /*buffer*/) const {
    int nnodes = num_nodes();
    int nqpts = gradients_q.shape[0];
    for (int j = threadIdx.x; j < nqpts; j += blockDim.x) {
      grad_type sum{};
      for (int i = 0; i < nnodes; i++) {
        sum[0] += values_e(i) * shape_fn_grads(j, i, 0);
        sum[1] += values_e(i) * shape_fn_grads(j, i, 1);
      }
      gradients_q(j) = sum;
    }
  }

  __device__ void cuda_integrate_source(nd::view<double> residual_e, nd::view<const source_type> source_q, nd::view<const double, 2> shape_fn, double * /*buffer*/) const {
    int nnodes = num_nodes();
    int nqpts = source_q.shape[0];

    for (int i = threadIdx.x; i < nnodes; i += blockDim.x) {
      double sum = 0.0;
      for (int q = 0; q < nqpts; q++) {
        sum += shape_fn(q, i) * source_q(q)[0];
      }
      residual_e(i) = sum;
    }
  }

  __device__ void cuda_integrate_flux(nd::view<double> residual_e, nd::view<const flux_type> flux_q, nd::view<const double, 3> shape_fn_grads, double * /*buffer*/) const {
    int nnodes = num_nodes();
    int nqpts = flux_q.shape[0];

    for (int i = threadIdx.x; i < nnodes; i += blockDim.x) {
      double sum = 0.0;
      for (int q = 0; q < nqpts; q++) {
        for (int d = 0; d < dim; d++) {
          sum += shape_fn_grads(q, i, d) * flux_q(q)[d];
        }
      }
      residual_e(i) = sum;
    }
  }
  #endif

  bool two_sided;

};
// clang-format on

} // namespace femto
