#pragma once

#include "fm/macros.hpp"
#include "femto/connection.hpp" 
#include "femto/interpolation.hpp"

namespace femto {

using namespace fm;

template <>
struct FiniteElement<Geometry::Edge, Family::DG> : private FiniteElement< Geometry::Edge, Family::H1 >{

  using source_type = vec1;
  using flux_type = vec1;

  using value_type = vec1;
  using grad_type = vec1;

  using FiniteElement<Geometry::Edge, Family::H1>::nodes;
  using FiniteElement<Geometry::Edge, Family::H1>::num_nodes;

  using FiniteElement<Geometry::Edge, Family::H1>::shape_function;
  using FiniteElement<Geometry::Edge, Family::H1>::shape_function_gradient;
  using FiniteElement<Geometry::Edge, Family::H1>::shape_function_derivative;

  using FiniteElement<Geometry::Edge, Family::H1>::evaluate_shape_functions;
  using FiniteElement<Geometry::Edge, Family::H1>::evaluate_shape_function_gradients;
  using FiniteElement<Geometry::Edge, Family::H1>::evaluate_weighted_shape_functions;
  using FiniteElement<Geometry::Edge, Family::H1>::evaluate_weighted_shape_function_gradients;

  FiniteElement(uint32_t p_) : FiniteElement<Geometry::Edge, Family::H1>{p_} {}

  __host__ __device__ uint32_t num_interior_nodes() const { return num_nodes(); }

  void interior_nodes(nd::view<double, 2> xi) const { nodes(xi); }

  __host__ __device__ void indices(const GeometryInfo & offsets, 
               const Connection * edge, 
               uint32_t * ids) const {
    uint32_t nodes_per_edge = num_nodes();
    uint32_t edge_id = edge[Edge::cell_offset].index;
    uint32_t base_offset = offsets.edge + edge_id * nodes_per_edge;
    for (uint32_t i = 0; i < nodes_per_edge; i++) {
      ids[i] = base_offset + i;
    } 
  }

  __host__ __device__ uint32_t batch_interpolation_scratch_space(nd::view<const double,2> xi) const {
    return 0;
  }

  __host__ __device__ void interpolate(nd::view<value_type> values_q, nd::view<const double> values_e, nd::view<const double, 2> shape_fns, double * /*buffer*/) const {
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

  __host__ __device__ void gradient(nd::view<grad_type> gradients_q, nd::view<const double, 1> values_e, nd::view<const double, 2> shape_fn_grads, double * /*buffer*/) const {
    int nnodes = num_nodes();
    int nqpts = gradients_q.shape[0];
    for (int j = 0; j < nqpts; j++) {
      double sum = 0.0;
      for (int i = 0; i < nnodes; i++) {
        sum += values_e(i) * shape_fn_grads(j, i);
      }
      gradients_q(j) = sum;
    }
  }

  nd::array< double, 2, memory::space::cpu > evaluate_weighted_shape_functions(nd::view<const double,2> xi,
                                                                        nd::view<const double,1> weights) const {
    uint32_t nnodes = num_nodes();
    uint32_t q = xi.shape[0];
    nd::array<double, 2, memory::space::cpu> shape_fns({q, nnodes});
    for (uint32_t i = 0; i < q; i++) {
      GaussLobattoInterpolation(xi(i, 0), p+1, &shape_fns(i, 0));
      for (uint32_t j = 0; j < nnodes; j++) {
        shape_fns(i, j) = shape_fns(i, j) * weights(i);
      }
    }
    return shape_fns;
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

  nd::array< double, 2, memory::space::cpu > evaluate_weighted_shape_function_gradients(nd::view<const double, 2> xi,
                                                                                 nd::view<const double, 1> weights) const {
    uint32_t nnodes = num_nodes();
    uint32_t q = xi.shape[0];
    nd::array<double, 2, memory::space::cpu> shape_fn_grads({q, num_nodes()});
    for (uint32_t i = 0; i < q; i++) {
      GaussLobattoInterpolationDerivative(xi(i, 0), p + 1, &shape_fn_grads(i, 0));
      for (uint32_t j = 0; j < nnodes; j++) {
        shape_fn_grads(i, j) = shape_fn_grads(i, j) * weights(i);
      }
    }
    return shape_fn_grads;
  }

  __host__ __device__ void integrate_flux(nd::view<double> output_e, nd::view<const flux_type> flux_q, nd::view<const double, 2> shape_fn_grads, double * /*buffer*/) const {
    int nnodes = num_nodes();
    int nqpts = flux_q.shape[0];
    for (int i = 0; i < nnodes; i++) {
      double sum = 0.0;
      for (int q = 0; q < nqpts; q++) {
        sum += shape_fn_grads(q, i) * flux_q(q);
      }
      output_e(i) = sum;
    }
  }

  #ifdef __CUDACC__
  __device__ void cuda_interpolate(nd::view<value_type> values_q, nd::view<const double> values_e, nd::view<const double, 2> shape_fns, double * /*buffer*/) const {
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

  __device__ void cuda_gradient(nd::view<grad_type, 1> gradients_q, nd::view<const double, 1> values_e, nd::view<const double, 2> shape_fn_grads, double * /*buffer*/) const {
    int nnodes = num_nodes();
    int nqpts = gradients_q.shape[0];
    for (int j = threadIdx.x; j < nqpts; j += blockDim.x) {
      double sum = 0.0;
      for (int i = 0; i < nnodes; i++) {
        sum += values_e(i) * shape_fn_grads(j, i);
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

  __device__ void cuda_integrate_flux(nd::view<double> residual_e, nd::view<const flux_type> flux_q, nd::view<const double, 2> shape_fn_grads, double * /*buffer*/) const {
    int nnodes = num_nodes();
    int nqpts = flux_q.shape[0];

    for (int i = threadIdx.x; i < nnodes; i += blockDim.x) {
      double sum = 0.0;
      for (int q = 0; q < nqpts; q++) {
        sum += shape_fn_grads(q, i) * flux_q(q);
      }
      residual_e(i) = sum;
    }
  }
  #endif

};

} // namespace femto
