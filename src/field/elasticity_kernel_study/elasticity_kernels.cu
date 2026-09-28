#include "femto/elasticity_kernels.hpp"

#include "../common.hpp"

#include "misc/macros.hpp"

#include <algorithm>
#include <iostream>

#ifdef FEMTO_ENABLE_CUDA

namespace femto {

namespace impl {

namespace stiffness_cuda {

namespace {

using tet_h1_element = FiniteElement<Geometry::Tetrahedron, Family::H1>;
using grad_qtype = vec3;
using dX_dxi_type = mat3;

static __device__ __forceinline__ int find_column_in_sparse_row(
  nd::view<const int, 1, memory::space::gpu> col_ind,
  int row_start,
  int row_end,
  int col) {
  int lo = row_start;
  int hi = row_end;
  while (lo < hi) {
    int mid = lo + (hi - lo) / 2;
    if (col_ind[mid] < col) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return (lo < row_end && col_ind[lo] == col) ? lo : -1;
}

template < uint32_t shape_rank >
__device__ grad_qtype load_tet_h1_grad(
  nd::view<double, shape_rank> shape_fn_grads,
  uint32_t q,
  uint32_t i) {
  static_assert(shape_rank == 3);
  return {
    shape_fn_grads(q, i, 0),
    shape_fn_grads(q, i, 1),
    shape_fn_grads(q, i, 2)
  };
}

template < uint32_t grad_shape_rank, uint32_t X_shape_rank, bool need_to_compute_dX_dxi >
__global__ void h1_tet_grad_grad_spmat_kernel(
  nd::view<double, 1, memory::space::gpu> values,
  nd::view<const int, 1, memory::space::gpu> row_ptr,
  nd::view<const int, 1, memory::space::gpu> col_ind,
  tet_h1_element element,
  nd::view<const double, 5, memory::space::gpu> qdata,
  FunctionSpace space,
  GeometryInfo offsets,
  nd::view<const Connection, 2, memory::space::gpu> connectivity,
  nd::view<const int, 1, memory::space::gpu> elements,
  nd::view<const double, grad_shape_rank, memory::space::gpu> shape_fn_grads,
  nd::view<const double, grad_shape_rank, memory::space::gpu> weighted_shape_fn_grads,
  tet_h1_element X_el,
  GeometryInfo X_offsets,
  nd::view<const double, 2, memory::space::gpu> X,
  nd::view<const double, X_shape_rank, memory::space::gpu> X_shape_functions,
  uint32_t qpts_per_element,
  uint32_t element_scratch_size,
  uint32_t X_scratch_size) {

  uint32_t shmem_offset = 0;
  extern __shared__ char shmem[];

  uint32_t nodes_per_element = element.num_nodes();
  uint32_t X_nodes_per_element = X_el.num_nodes();
  uint32_t components = space.components;
  uint32_t i = threadIdx.y;
  uint32_t j = threadIdx.z;
  uint32_t component_pair_id = i * components + j;
  uint32_t component_pair_count = components * components;
  uint32_t local_tid = threadIdx.x;
  uint32_t local_stride = blockDim.x;
  uint32_t block_tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  uint32_t block_stride = blockDim.x * blockDim.y * blockDim.z;

  nd::view<Connection, 1> shr_connectivity((Connection *)(shmem + shmem_offset), {connectivity.shape[1]});
  shmem_offset += round_up_to_multiple_of_128(shr_connectivity.size() * sizeof(Connection));

  nd::view<uint32_t, 1> shr_node_ids((uint32_t *)(shmem + shmem_offset), {nodes_per_element});
  shmem_offset += round_up_to_multiple_of_128(shr_node_ids.size() * sizeof(uint32_t));

  nd::view<grad_qtype, 3> shr_flux((grad_qtype *)(shmem + shmem_offset), {components, components, qpts_per_element});
  shmem_offset += round_up_to_multiple_of_128(shr_flux.size() * sizeof(grad_qtype));

  nd::view<double, 3> shr_r_e((double *)(shmem + shmem_offset), {components, components, nodes_per_element});
  shmem_offset += round_up_to_multiple_of_128(shr_r_e.size() * sizeof(double));

  nd::view<double, grad_shape_rank> shr_shape_fn_grads((double *)(shmem + shmem_offset), shape_fn_grads.shape);
  shmem_offset += round_up_to_multiple_of_128(shr_shape_fn_grads.size() * sizeof(double));

  nd::view<double, grad_shape_rank> shr_weighted_shape_fn_grads((double *)(shmem + shmem_offset), weighted_shape_fn_grads.shape);
  shmem_offset += round_up_to_multiple_of_128(shr_weighted_shape_fn_grads.size() * sizeof(double));

  nd::view<double, 2> shr_element_scratch((double *)(shmem + shmem_offset), {component_pair_count, element_scratch_size});
  shmem_offset += round_up_to_multiple_of_128(shr_element_scratch.size() * sizeof(double));

  nd::view<uint32_t, 1> shr_X_node_ids;
  nd::view<double, 1> shr_X_e;
  nd::view<double, X_shape_rank> shr_X_shape_fn;
  nd::view<grad_qtype, 1> shr_X_grad_q;
  nd::view<double, 1> shr_X_scratch;

  if constexpr (need_to_compute_dX_dxi) {
    shr_X_node_ids = nd::view<uint32_t, 1>((uint32_t *)(shmem + shmem_offset), {X_nodes_per_element});
    shmem_offset += round_up_to_multiple_of_128(shr_X_node_ids.size() * sizeof(uint32_t));

    shr_X_e = nd::view<double, 1>((double *)(shmem + shmem_offset), {X_nodes_per_element});
    shmem_offset += round_up_to_multiple_of_128(shr_X_e.size() * sizeof(double));

    shr_X_shape_fn = nd::view<double, X_shape_rank>((double *)(shmem + shmem_offset), X_shape_functions.shape);
    shmem_offset += round_up_to_multiple_of_128(shr_X_shape_fn.size() * sizeof(double));

    shr_X_grad_q = nd::view<grad_qtype, 1>((grad_qtype *)(shmem + shmem_offset), {qpts_per_element});
    shmem_offset += round_up_to_multiple_of_128(shr_X_grad_q.size() * sizeof(grad_qtype));

    shr_X_scratch = nd::view<double, 1>((double *)(shmem + shmem_offset), {X_scratch_size});
  }

  for (uint32_t idx = block_tid; idx < shr_shape_fn_grads.size(); idx += block_stride) {
    shr_shape_fn_grads[idx] = shape_fn_grads[idx];
  }
  for (uint32_t idx = block_tid; idx < shr_weighted_shape_fn_grads.size(); idx += block_stride) {
    shr_weighted_shape_fn_grads[idx] = weighted_shape_fn_grads[idx];
  }
  if constexpr (need_to_compute_dX_dxi) {
    for (uint32_t idx = block_tid; idx < shr_X_shape_fn.size(); idx += block_stride) {
      shr_X_shape_fn[idx] = X_shape_functions[idx];
    }
  }
  __syncthreads();

  uint32_t e = blockIdx.x;
  bool active = e < elements.shape[0];
  uint32_t source_e = active ? e : 0;
  uint32_t elem_id = elements(source_e);
  uint32_t q = local_tid;
  uint32_t qid = source_e * qpts_per_element + q;
  mat3 C{};
  if (q < qpts_per_element) {
    for (uint32_t k = 0; k < 3; k++) {
      for (uint32_t m = 0; m < 3; m++) {
        C(k, m) = qdata(qid, i, k, j, m);
      }
    }
  }

  for (uint32_t idx = block_tid; idx < shr_connectivity.shape[0]; idx += block_stride) {
    shr_connectivity(idx) = connectivity(elem_id, idx);
  }
  __syncthreads();

  if (block_tid == 0) {
    element.indices(offsets, shr_connectivity.data(), shr_node_ids.data());
  }
  __syncthreads();

  if constexpr (need_to_compute_dX_dxi) {
    dX_dxi_type dX_dxi{};

    if (block_tid == 0) {
      X_el.indices(X_offsets, shr_connectivity.data(), shr_X_node_ids.data());
    }
    __syncthreads();

    for (uint32_t d = 0; d < 3; d++) {
      for (uint32_t idx = block_tid; idx < X_nodes_per_element; idx += block_stride) {
        shr_X_e(idx) = X(shr_X_node_ids(idx), d);
      }
      __syncthreads();

      if (i == 0 && j == 0) {
        X_el.cuda_gradient(
          shr_X_grad_q,
          shr_X_e,
          shr_X_shape_fn,
          shr_X_scratch.data()
        );
      }
      __syncthreads();

      if (q < qpts_per_element) {
        dX_dxi[d] = shr_X_grad_q(q);
      }
      __syncthreads();
    }

    if (q < qpts_per_element) {
      C = fm::dot(fm::dot(inv(dX_dxi), C), transpose(adj(dX_dxi)));
    }
  }

  if (!active) {
    return;
  }

  for (uint32_t I = 0; I < nodes_per_element; I++) {
    uint32_t row_id = shr_node_ids(I) * components + i;

    if (q < qpts_per_element) {
      grad_qtype dphi_I = load_tet_h1_grad(shr_shape_fn_grads, q, I);

      shr_flux(i, j, q) = fm::dot(dphi_I, C);
    }
    __syncthreads();

    element.cuda_integrate_flux(
      shr_r_e(i, j),
      shr_flux(i, j),
      shr_weighted_shape_fn_grads,
      shr_element_scratch(component_pair_id).data()
    );
    __syncthreads();

    int row_start = row_ptr(row_id);
    int row_end = row_ptr(row_id + 1);
    for (uint32_t J = local_tid; J < nodes_per_element; J += local_stride) {
      uint32_t col_id = shr_node_ids(J) * components + j;
      int position = find_column_in_sparse_row(col_ind, row_start, row_end, col_id);
      if (position >= 0) {
        atomicAdd(&values(position), shr_r_e(i, j, J));
      }
    }
    __syncthreads();
  }
}

template < uint32_t grad_shape_rank, uint32_t X_shape_rank, bool need_to_compute_dX_dxi >
__global__ void h1_tet_grad_grad_emat_kernel(
  nd::view<double, 5, memory::space::gpu> element_matrices,
  tet_h1_element element,
  nd::view<const double, 5, memory::space::gpu> qdata,
  FunctionSpace space,
  nd::view<const Connection, 2, memory::space::gpu> connectivity,
  nd::view<const int, 1, memory::space::gpu> elements,
  nd::view<const double, grad_shape_rank, memory::space::gpu> shape_fn_grads,
  nd::view<const double, grad_shape_rank, memory::space::gpu> weighted_shape_fn_grads,
  tet_h1_element X_el,
  GeometryInfo X_offsets,
  nd::view<const double, 2, memory::space::gpu> X,
  nd::view<const double, X_shape_rank, memory::space::gpu> X_shape_functions,
  uint32_t qpts_per_element,
  uint32_t element_scratch_size,
  uint32_t X_scratch_size) {

  uint32_t shmem_offset = 0;
  extern __shared__ char shmem[];

  uint32_t nodes_per_element = element.num_nodes();
  uint32_t X_nodes_per_element = X_el.num_nodes();
  uint32_t components = space.components;
  uint32_t i = threadIdx.y;
  uint32_t j = threadIdx.z;
  uint32_t component_pair_id = i * components + j;
  uint32_t component_pair_count = components * components;
  uint32_t local_tid = threadIdx.x;
  uint32_t local_stride = blockDim.x;
  uint32_t block_tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  uint32_t block_stride = blockDim.x * blockDim.y * blockDim.z;

  nd::view<Connection, 1> shr_connectivity((Connection *)(shmem + shmem_offset), {connectivity.shape[1]});
  shmem_offset += round_up_to_multiple_of_128(shr_connectivity.size() * sizeof(Connection));

  nd::view<grad_qtype, 3> shr_flux((grad_qtype *)(shmem + shmem_offset), {components, components, qpts_per_element});
  shmem_offset += round_up_to_multiple_of_128(shr_flux.size() * sizeof(grad_qtype));

  nd::view<double, 3> shr_r_e((double *)(shmem + shmem_offset), {components, components, nodes_per_element});
  shmem_offset += round_up_to_multiple_of_128(shr_r_e.size() * sizeof(double));

  nd::view<double, grad_shape_rank> shr_shape_fn_grads((double *)(shmem + shmem_offset), shape_fn_grads.shape);
  shmem_offset += round_up_to_multiple_of_128(shr_shape_fn_grads.size() * sizeof(double));

  nd::view<double, grad_shape_rank> shr_weighted_shape_fn_grads((double *)(shmem + shmem_offset), weighted_shape_fn_grads.shape);
  shmem_offset += round_up_to_multiple_of_128(shr_weighted_shape_fn_grads.size() * sizeof(double));

  nd::view<double, 2> shr_element_scratch((double *)(shmem + shmem_offset), {component_pair_count, element_scratch_size});
  shmem_offset += round_up_to_multiple_of_128(shr_element_scratch.size() * sizeof(double));

  nd::view<uint32_t, 1> shr_X_node_ids;
  nd::view<double, 1> shr_X_e;
  nd::view<double, X_shape_rank> shr_X_shape_fn;
  nd::view<grad_qtype, 1> shr_X_grad_q;
  nd::view<double, 1> shr_X_scratch;

  if constexpr (need_to_compute_dX_dxi) {
    shr_X_node_ids = nd::view<uint32_t, 1>((uint32_t *)(shmem + shmem_offset), {X_nodes_per_element});
    shmem_offset += round_up_to_multiple_of_128(shr_X_node_ids.size() * sizeof(uint32_t));

    shr_X_e = nd::view<double, 1>((double *)(shmem + shmem_offset), {X_nodes_per_element});
    shmem_offset += round_up_to_multiple_of_128(shr_X_e.size() * sizeof(double));

    shr_X_shape_fn = nd::view<double, X_shape_rank>((double *)(shmem + shmem_offset), X_shape_functions.shape);
    shmem_offset += round_up_to_multiple_of_128(shr_X_shape_fn.size() * sizeof(double));

    shr_X_grad_q = nd::view<grad_qtype, 1>((grad_qtype *)(shmem + shmem_offset), {qpts_per_element});
    shmem_offset += round_up_to_multiple_of_128(shr_X_grad_q.size() * sizeof(grad_qtype));

    shr_X_scratch = nd::view<double, 1>((double *)(shmem + shmem_offset), {X_scratch_size});
  }

  for (uint32_t idx = block_tid; idx < shr_shape_fn_grads.size(); idx += block_stride) {
    shr_shape_fn_grads[idx] = shape_fn_grads[idx];
  }
  for (uint32_t idx = block_tid; idx < shr_weighted_shape_fn_grads.size(); idx += block_stride) {
    shr_weighted_shape_fn_grads[idx] = weighted_shape_fn_grads[idx];
  }
  if constexpr (need_to_compute_dX_dxi) {
    for (uint32_t idx = block_tid; idx < shr_X_shape_fn.size(); idx += block_stride) {
      shr_X_shape_fn[idx] = X_shape_functions[idx];
    }
  }
  __syncthreads();

  uint32_t e = blockIdx.x;
  bool active = e < elements.shape[0];
  uint32_t source_e = active ? e : 0;
  uint32_t elem_id = elements(source_e);
  uint32_t q = local_tid;
  uint32_t qid = source_e * qpts_per_element + q;
  mat3 C{};
  if (q < qpts_per_element) {
    for (uint32_t k = 0; k < 3; k++) {
      for (uint32_t m = 0; m < 3; m++) {
        C(k, m) = qdata(qid, i, k, j, m);
      }
    }
  }

  for (uint32_t idx = block_tid; idx < shr_connectivity.shape[0]; idx += block_stride) {
    shr_connectivity(idx) = connectivity(elem_id, idx);
  }
  __syncthreads();

  if constexpr (need_to_compute_dX_dxi) {
    dX_dxi_type dX_dxi{};

    if (block_tid == 0) {
      X_el.indices(X_offsets, shr_connectivity.data(), shr_X_node_ids.data());
    }
    __syncthreads();

    for (uint32_t d = 0; d < 3; d++) {
      for (uint32_t idx = block_tid; idx < X_nodes_per_element; idx += block_stride) {
        shr_X_e(idx) = X(shr_X_node_ids(idx), d);
      }
      __syncthreads();

      if (i == 0 && j == 0) {
        X_el.cuda_gradient(
          shr_X_grad_q,
          shr_X_e,
          shr_X_shape_fn,
          shr_X_scratch.data()
        );
      }
      __syncthreads();

      if (q < qpts_per_element) {
        dX_dxi[d] = shr_X_grad_q(q);
      }
      __syncthreads();
    }

    if (q < qpts_per_element) {
      C = fm::dot(fm::dot(inv(dX_dxi), C), transpose(adj(dX_dxi)));
    }
  }

  if (!active) {
    return;
  }

  for (uint32_t I = 0; I < nodes_per_element; I++) {
    if (q < qpts_per_element) {
      grad_qtype dphi_I = load_tet_h1_grad(shr_shape_fn_grads, q, I);

      shr_flux(i, j, q) = fm::dot(dphi_I, C);
    }
    __syncthreads();

    element.cuda_integrate_flux(
      shr_r_e(i, j),
      shr_flux(i, j),
      shr_weighted_shape_fn_grads,
      shr_element_scratch(component_pair_id).data()
    );
    __syncthreads();

    for (uint32_t J = local_tid; J < nodes_per_element; J += local_stride) {
      element_matrices(e, j, i, I, J) = shr_r_e(i, j, J);
    }
    __syncthreads();
  }
}

void integrate_h1_tet_grad_grad_spmat_impl(
  sparse_matrix<memory::space::gpu> & A,
  nd::view<const double, 5, memory::space::gpu> qdata,
  FunctionSpace space,
  GeometryInfo offsets,
  const Field<Family::H1, memory::space::gpu> & X,
  DomainType type,
  nd::view<const Connection, 2, memory::space::gpu> connectivity,
  nd::view<const int, 1, memory::space::gpu> elements,
  nd::view<const double, 2> xi,
  nd::view<const double, 1> weights) {
  uint32_t num_elements = elements.size();
  if (num_elements == 0) return;

  tet_h1_element element{space.degree};
  tet_h1_element X_el{X.degree};

  auto shape_fn_grads = element.evaluate_shape_function_gradients(xi);
  auto weighted_shape_fn_grads = element.evaluate_weighted_shape_function_gradients(xi, weights);
  auto X_shape_fns = X_el.evaluate_shape_function_gradients(xi);

  constexpr uint32_t grad_shape_rank = array_rank<decltype(shape_fn_grads)>::value;
  constexpr uint32_t X_shape_rank = array_rank<decltype(X_shape_fns)>::value;

  nd::array<double, grad_shape_rank, memory::space::gpu> shape_fn_grads_device = shape_fn_grads;
  nd::array<double, grad_shape_rank, memory::space::gpu> weighted_shape_fn_grads_device = weighted_shape_fn_grads;
  nd::array<double, X_shape_rank, memory::space::gpu> X_shape_fns_device = X_shape_fns;

  uint32_t qpts_per_element = qpe<Geometry::Tetrahedron>(xi.shape[0]);
  uint32_t nodes_per_element = element.num_nodes();
  uint32_t X_nodes_per_element = X_el.num_nodes();
  uint32_t element_scratch_size = element.batch_interpolation_scratch_space(xi);
  uint32_t X_scratch_size = X_el.batch_interpolation_scratch_space(xi);

  FEMTO_ASSERT(nodes_per_element >= qpts_per_element, "H1 tet grad-grad sparse kernel assumes blockDim.x >= qpts_per_element");

  uint32_t component_pairs = space.components * space.components;

  auto base_shmem_size = [&]() {
    uint32_t shmem_size = 0;
    shmem_size += round_up_to_multiple_of_128(connectivity.shape[1] * sizeof(Connection));
    shmem_size += round_up_to_multiple_of_128(nodes_per_element * sizeof(uint32_t));
    shmem_size += round_up_to_multiple_of_128(component_pairs * qpts_per_element * sizeof(grad_qtype));
    shmem_size += round_up_to_multiple_of_128(component_pairs * nodes_per_element * sizeof(double));
    shmem_size += round_up_to_multiple_of_128(shape_fn_grads.size() * sizeof(double));
    shmem_size += round_up_to_multiple_of_128(weighted_shape_fn_grads.size() * sizeof(double));
    shmem_size += round_up_to_multiple_of_128(component_pairs * element_scratch_size * sizeof(double));
    return shmem_size;
  };

  auto spatial_shmem_size = [&]() {
    uint32_t shmem_size = base_shmem_size();
    shmem_size += round_up_to_multiple_of_128(X_nodes_per_element * sizeof(uint32_t));
    shmem_size += round_up_to_multiple_of_128(X_nodes_per_element * sizeof(double));
    shmem_size += round_up_to_multiple_of_128(X_shape_fns.size() * sizeof(double));
    shmem_size += round_up_to_multiple_of_128(qpts_per_element * sizeof(grad_qtype));
    shmem_size += round_up_to_multiple_of_128(X_scratch_size * sizeof(double));
    return shmem_size;
  };

  dim3 block_size = {nodes_per_element, space.components, space.components};
  uint32_t grid_size = num_elements;

  cudaEvent_t kernel_start{};
  cudaEvent_t kernel_stop{};
  CUDA_CHECK(cudaEventCreate(&kernel_start));
  CUDA_CHECK(cudaEventCreate(&kernel_stop));
  CUDA_CHECK(cudaEventRecord(kernel_start));

  if (type == DomainType::SPATIAL) {
    h1_tet_grad_grad_spmat_kernel<grad_shape_rank, X_shape_rank, true>
      <<<grid_size, block_size, spatial_shmem_size()>>>(
        A.values,
        A.row_ptr,
        A.col_ind,
        element,
        qdata,
        space,
        offsets,
        connectivity,
        elements,
        shape_fn_grads_device,
        weighted_shape_fn_grads_device,
        X_el,
        X.offsets,
        X.data,
        X_shape_fns_device,
        qpts_per_element,
        element_scratch_size,
        X_scratch_size);
  } else {
    h1_tet_grad_grad_spmat_kernel<grad_shape_rank, X_shape_rank, false>
      <<<grid_size, block_size, base_shmem_size()>>>(
        A.values,
        A.row_ptr,
        A.col_ind,
        element,
        qdata,
        space,
        offsets,
        connectivity,
        elements,
        shape_fn_grads_device,
        weighted_shape_fn_grads_device,
        X_el,
        X.offsets,
        X.data,
        X_shape_fns_device,
        qpts_per_element,
        element_scratch_size,
        X_scratch_size);
  }

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(kernel_stop));
  CUDA_CHECK(cudaEventSynchronize(kernel_stop));

  float kernel_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop));
  CUDA_CHECK(cudaEventDestroy(kernel_start));
  CUDA_CHECK(cudaEventDestroy(kernel_stop));

  std::cout << "H1 tet sparse matrix assembly kernel time: " << kernel_ms << "ms" << std::endl;
}

void integrate_h1_tet_grad_grad_emat_impl(
  nd::array<double, 5, memory::space::gpu> & element_matrices,
  nd::view<const double, 5, memory::space::gpu> qdata,
  FunctionSpace space,
  const Field<Family::H1, memory::space::gpu> & X,
  DomainType type,
  nd::view<const Connection, 2, memory::space::gpu> connectivity,
  nd::view<const int, 1, memory::space::gpu> elements,
  nd::view<const double, 2> xi,
  nd::view<const double, 1> weights) {

  tet_h1_element element{space.degree};
  tet_h1_element X_el{X.degree};

  uint32_t num_elements = elements.size();
  uint32_t nodes_per_element = element.num_nodes();
  uint32_t X_nodes_per_element = X_el.num_nodes();
  element_matrices.resize({num_elements, space.components, space.components, nodes_per_element, nodes_per_element});
  if (num_elements == 0) return;

  auto shape_fn_grads = element.evaluate_shape_function_gradients(xi);
  auto weighted_shape_fn_grads = element.evaluate_weighted_shape_function_gradients(xi, weights);
  auto X_shape_fns = X_el.evaluate_shape_function_gradients(xi);

  constexpr uint32_t grad_shape_rank = array_rank<decltype(shape_fn_grads)>::value;
  constexpr uint32_t X_shape_rank = array_rank<decltype(X_shape_fns)>::value;

  nd::array<double, grad_shape_rank, memory::space::gpu> shape_fn_grads_device = shape_fn_grads;
  nd::array<double, grad_shape_rank, memory::space::gpu> weighted_shape_fn_grads_device = weighted_shape_fn_grads;
  nd::array<double, X_shape_rank, memory::space::gpu> X_shape_fns_device = X_shape_fns;

  uint32_t qpts_per_element = qpe<Geometry::Tetrahedron>(xi.shape[0]);
  uint32_t element_scratch_size = element.batch_interpolation_scratch_space(xi);
  uint32_t X_scratch_size = X_el.batch_interpolation_scratch_space(xi);

  FEMTO_ASSERT(nodes_per_element >= qpts_per_element, "H1 tet grad-grad element matrix kernel assumes blockDim.x >= qpts_per_element");

  uint32_t component_pairs = space.components * space.components;

  auto base_shmem_size = [&]() {
    uint32_t shmem_size = 0;
    shmem_size += round_up_to_multiple_of_128(connectivity.shape[1] * sizeof(Connection));
    shmem_size += round_up_to_multiple_of_128(component_pairs * qpts_per_element * sizeof(grad_qtype));
    shmem_size += round_up_to_multiple_of_128(component_pairs * nodes_per_element * sizeof(double));
    shmem_size += round_up_to_multiple_of_128(shape_fn_grads.size() * sizeof(double));
    shmem_size += round_up_to_multiple_of_128(weighted_shape_fn_grads.size() * sizeof(double));
    shmem_size += round_up_to_multiple_of_128(component_pairs * element_scratch_size * sizeof(double));
    return shmem_size;
  };

  auto spatial_shmem_size = [&]() {
    uint32_t shmem_size = base_shmem_size();
    shmem_size += round_up_to_multiple_of_128(X_nodes_per_element * sizeof(uint32_t));
    shmem_size += round_up_to_multiple_of_128(X_nodes_per_element * sizeof(double));
    shmem_size += round_up_to_multiple_of_128(X_shape_fns.size() * sizeof(double));
    shmem_size += round_up_to_multiple_of_128(qpts_per_element * sizeof(grad_qtype));
    shmem_size += round_up_to_multiple_of_128(X_scratch_size * sizeof(double));
    return shmem_size;
  };

  dim3 block_size = {nodes_per_element, space.components, space.components};
  uint32_t grid_size = num_elements;

  cudaEvent_t kernel_start{};
  cudaEvent_t kernel_stop{};
  CUDA_CHECK(cudaEventCreate(&kernel_start));
  CUDA_CHECK(cudaEventCreate(&kernel_stop));
  CUDA_CHECK(cudaEventRecord(kernel_start));

  if (type == DomainType::SPATIAL) {
    h1_tet_grad_grad_emat_kernel<grad_shape_rank, X_shape_rank, true>
      <<<grid_size, block_size, spatial_shmem_size()>>>(
        element_matrices,
        element,
        qdata,
        space,
        connectivity,
        elements,
        shape_fn_grads_device,
        weighted_shape_fn_grads_device,
        X_el,
        X.offsets,
        X.data,
        X_shape_fns_device,
        qpts_per_element,
        element_scratch_size,
        X_scratch_size);
  } else {
    h1_tet_grad_grad_emat_kernel<grad_shape_rank, X_shape_rank, false>
      <<<grid_size, block_size, base_shmem_size()>>>(
        element_matrices,
        element,
        qdata,
        space,
        connectivity,
        elements,
        shape_fn_grads_device,
        weighted_shape_fn_grads_device,
        X_el,
        X.offsets,
        X.data,
        X_shape_fns_device,
        qpts_per_element,
        element_scratch_size,
        X_scratch_size);
  }

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(kernel_stop));
  CUDA_CHECK(cudaEventSynchronize(kernel_stop));

  float kernel_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop));
  CUDA_CHECK(cudaEventDestroy(kernel_start));
  CUDA_CHECK(cudaEventDestroy(kernel_stop));

  std::cout << "H1 tet element matrix assembly kernel time: " << kernel_ms << "ms" << std::endl;
}

} // namespace

bool can_integrate_h1_tet_spmat(
  BasisFunctionOp<DerivedQuantity::GRAD, Family::H1> test,
  BasisFunctionOp<DerivedQuantity::GRAD, Family::H1> trial,
  const Domain<memory::space::gpu> & domain) {
  const FunctionSpace space = test.function.space;

  return test.function == trial.function &&
         test.mod == trial.mod &&
         domain.mesh.geometry_dimension == 3 &&
         domain.active_elements.vert.size() == 0 &&
         domain.active_elements.edge.size() == 0 &&
         domain.active_elements.tri.size() == 0 &&
         domain.active_elements.quad.size() == 0 &&
         domain.active_elements.tet.size() > 0 &&
         domain.active_elements.hex.size() == 0;
}

void integrate_h1_tet_spmat(
  sparse_matrix<memory::space::gpu> & A,
  BasisFunctionOp<DerivedQuantity::GRAD, Family::H1> test,
  nd::view<const double, 5, memory::space::gpu> qdata,
  BasisFunctionOp<DerivedQuantity::GRAD, Family::H1> trial,
  const Domain<memory::space::gpu> & domain,
  DomainType type) {
  FEMTO_ASSERT(can_integrate_h1_tet_spmat(test, trial, domain), "unsupported H1 tetrahedron grad-grad sparse integration path");

  const FunctionSpace space = test.function.space;
  constexpr uint32_t gdim = 3;

  stack::array<uint32_t, 5> shape5D = {
    qdata.shape[0],
    space.components, gdim,
    space.components, gdim
  };
  FEMTO_ASSERT(compatible_shapes(qdata.shape, shape5D), "incompatible array shapes");

  if (A.nnz == 0) {
    A = blank_sparse_matrix(test, trial, domain);
  } else {
    zero(A.values);
  }

  GeometryInfo counts = domain.mesh.geometry_counts();
  GeometryInfo offsets = scan(interior_nodes_per_geom(space, gdim) * counts);

  constexpr Geometry geom = Geometry::Tetrahedron;
  nd::view<const Connection, 2, memory::space::gpu> connectivity = domain.mesh[geom];
  nd::view<const int, 1, memory::space::gpu> elements = domain.active_elements[geom];
  nd::view<const double, 2> xi = domain.rule[geom].points;
  nd::view<const double, 1> weights = domain.rule[geom].weights;

  stack::array<uint32_t, 5> tet_qdata_shape{
    domain.num_qpts[geom],
    shape5D[1],
    shape5D[2],
    shape5D[3],
    shape5D[4]
  };
  nd::view<const double, 5, memory::space::gpu> tet_qdata{&qdata(0, 0, 0, 0, 0), tet_qdata_shape};

  integrate_h1_tet_grad_grad_spmat_impl(
    A,
    tet_qdata,
    space,
    offsets,
    domain.mesh.X,
    type,
    connectivity,
    elements,
    xi,
    weights);
}

void integrate_h1_tet_emat(
  nd::array<double, 5, memory::space::gpu> & element_matrices,
  BasisFunctionOp<DerivedQuantity::GRAD, Family::H1> test,
  nd::view<const double, 5, memory::space::gpu> qdata,
  BasisFunctionOp<DerivedQuantity::GRAD, Family::H1> trial,
  const Domain<memory::space::gpu> & domain,
  DomainType type) {
  FEMTO_ASSERT(can_integrate_h1_tet_spmat(test, trial, domain), "unsupported H1 tetrahedron grad-grad element matrix integration path");

  const FunctionSpace space = test.function.space;
  constexpr uint32_t gdim = 3;

  stack::array<uint32_t, 5> shape5D = {
    qdata.shape[0],
    space.components, gdim,
    space.components, gdim
  };
  FEMTO_ASSERT(compatible_shapes(qdata.shape, shape5D), "incompatible array shapes");

  constexpr Geometry geom = Geometry::Tetrahedron;
  nd::view<const Connection, 2, memory::space::gpu> connectivity = domain.mesh[geom];
  nd::view<const int, 1, memory::space::gpu> elements = domain.active_elements[geom];
  nd::view<const double, 2> xi = domain.rule[geom].points;
  nd::view<const double, 1> weights = domain.rule[geom].weights;

  stack::array<uint32_t, 5> tet_qdata_shape{
    domain.num_qpts[geom],
    shape5D[1],
    shape5D[2],
    shape5D[3],
    shape5D[4]
  };
  nd::view<const double, 5, memory::space::gpu> tet_qdata{&qdata(0, 0, 0, 0, 0), tet_qdata_shape};

  integrate_h1_tet_grad_grad_emat_impl(
    element_matrices,
    tet_qdata,
    space,
    domain.mesh.X,
    type,
    connectivity,
    elements,
    xi,
    weights);
}

} // namespace stiffness_cuda

} // namespace impl

} // namespace femto

#endif
