#pragma once

// Reference only -- not included anywhere.  The "rows in flight" CUDA sparse
// matrix kernel from the Aug 2026 tuning experiments (sparse_matrix_experiments.md
// section 8d): block shape {trial nodes, nr, elements}, nr rows of the element
// matrix computed concurrently with the element's cuda_integrate_* routines on
// pulled-back, weighted qdata staged per block.  Drop-in replacement for
// integrate_spmat.cuh (same batched_integrate_spmat_cuda / integrate_sparse_matrix
// interface).

#include "common.hpp"
#include "test_shape_functions.hpp"

#include "femto/assert.hpp"
#include "femto/domain.hpp"

#include "misc/macros.hpp"

#include <algorithm>
#include <functional>
#include <type_traits>

#ifdef FEMTO_ENABLE_CUDA

namespace femto {

namespace impl {

namespace spmat_cuda {

// A row of the element matrix is the residual of a linear material driven by
// one test function, so the kernel assembles the matrix rows at a time with
// the same element routines integrate() uses for a residual.
//
// Block shape {trial nodes, nr, elements}: the x threads are what the
// element's cuda_integrate_* routines index, the y index picks one of nr rows
// being worked on concurrently, and z picks the element.  Per block, the qdata
// of its elements is staged once, pulled back to the parent element and
// multiplied by the quadrature weight (C -> testA . C . trialA^T . w).  A row
// group then loops over its test nodes I: each thread keeps phi_I at its
// quadrature points in registers while the (test component, trial component)
// sub-rows are formed as hat_f(q) = C_q^T phi_I(q), integrated against the
// trial basis (sum-factorized on tensor-product elements), reoriented and
// scattered.
//
// See sparse_matrix_experiments.md for the measurements behind this layout.

static __device__ __forceinline__ int find_column_in_sparse_row(nd::view<const int, 1, memory::space::gpu> col_ind, int row_start, int row_end, int col) {
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

// C -> testA . C . trialA^T, the pull-back of both basis operators to the
// parent element (the same transformations integrate_residual applies to the
// test and trial sides separately)
template < Family test_family, DerivedQuantity test_op, Family trial_family, DerivedQuantity trial_op, uint32_t tq, uint32_t uq, uint32_t dim >
__device__ __forceinline__ mat<tq, uq> pull_back(const mat<tq, uq> & C, const mat<dim, dim> & dX_dxi) {
  auto testA = piola_transformation<test_family, test_op>(dX_dxi);
  auto trialA = weighted_piola_transformation<trial_family, trial_op>(dX_dxi);
  mat<tq, uq> output{};
  for (uint32_t k = 0; k < tq; k++) {
    vec<tq> phi{};
    phi[k] = 1.0;
    if constexpr (std::is_arithmetic_v<decltype(testA)>) { phi = phi * testA; } else { phi = fm::dot(phi, testA); }
    vec<uq> hat_f = fm::dot(phi, C);
    if constexpr (std::is_arithmetic_v<decltype(trialA)>) { hat_f = hat_f * trialA; } else { hat_f = fm::dot(trialA, hat_f); }
    for (uint32_t m = 0; m < uq; m++) { output(k, m) = hat_f[m]; }
  }
  return output;
}

template < uint32_t test_shape_rank,
           uint32_t trial_shape_rank,
           uint32_t X_shape_rank,
           Geometry geom,
           Family test_family,
           DerivedQuantity test_op,
           Family trial_family,
           DerivedQuantity trial_op,
           bool need_to_compute_dX_dxi >
__global__ void batched_integrate_spmat_kernel(
  nd::view<double, 1, memory::space::gpu> values,
  nd::view<const int, 1, memory::space::gpu> row_ptr,
  nd::view<const int, 1, memory::space::gpu> col_ind,
  FiniteElement<geom, test_family> test_el,
  FiniteElement<geom, trial_family> trial_el,
  nd::view<const double, 5, memory::space::gpu> qdata,
  FunctionSpace trial_space,
  FunctionSpace test_space,
  GeometryInfo trial_offsets,
  GeometryInfo test_offsets,
  nd::view<const Connection, 2, memory::space::gpu> connectivity,
  nd::view<const int, 1, memory::space::gpu> elements,
  nd::view<const double, test_shape_rank, memory::space::gpu> test_shape_functions,
  nd::view<const double, trial_shape_rank, memory::space::gpu> trial_shape_functions,
  nd::view<const double, 1, memory::space::gpu> weights,
  FiniteElement<geom, Family::H1> X_el,
  GeometryInfo X_offsets,
  nd::view<const double, 2, memory::space::gpu> X,
  nd::view<const double, X_shape_rank, memory::space::gpu> X_shape_functions,
  uint32_t qpts_per_element,
  uint32_t trial_scratch_size,
  bool stage_trial_shape_fns) {

  constexpr uint32_t gdim = dimension(geom);
  constexpr uint32_t tq = qshape(test_family, test_op, gdim);
  constexpr uint32_t uq = qshape(trial_family, trial_op, gdim);
  constexpr uint32_t max_qpts_per_thread = 4;   // phi_I cache; beyond it, recomputed

  using mat_t = mat<tq, uq>;
  using trial_qtype = vec<uq>;
  using dX_dxi_type = mat<gdim, gdim>;

  const uint32_t test_components = test_space.components;
  const uint32_t trial_components = trial_space.components;
  const uint32_t block_size = test_components * trial_components * tq * uq;

  uint32_t shmem_offset = 0;
  extern __shared__ char shmem[];

  const uint32_t nr = blockDim.y;
  const uint32_t elem_per_block = blockDim.z;
  const uint32_t test_nodes_per_element = test_el.num_nodes();
  const uint32_t trial_nodes_per_element = trial_el.num_nodes();
  const uint32_t X_nodes_per_element = X_el.num_nodes();
  const uint32_t slot = threadIdx.y;
  const uint32_t le = threadIdx.z;
  const uint32_t elem_tid = threadIdx.x + blockDim.x * threadIdx.y;      // within the element
  const uint32_t elem_stride = blockDim.x * blockDim.y;
  const uint32_t block_tid = elem_tid + elem_stride * threadIdx.z;
  const uint32_t block_stride = elem_stride * blockDim.z;

  nd::view<Connection, 2> shr_connectivity((Connection *)(shmem + shmem_offset), {elem_per_block, connectivity.shape[1]});
  shmem_offset += round_up_to_multiple_of_128(shr_connectivity.size() * sizeof(Connection));

  nd::view<uint32_t, 2> shr_test_node_ids((uint32_t *)(shmem + shmem_offset), {elem_per_block, test_nodes_per_element});
  shmem_offset += round_up_to_multiple_of_128(shr_test_node_ids.size() * sizeof(uint32_t));

  nd::view<uint32_t, 2> shr_trial_node_ids((uint32_t *)(shmem + shmem_offset), {elem_per_block, trial_nodes_per_element});
  shmem_offset += round_up_to_multiple_of_128(shr_trial_node_ids.size() * sizeof(uint32_t));

  nd::view<uint32_t, 2> shr_X_node_ids((uint32_t *)(shmem + shmem_offset), {elem_per_block, X_nodes_per_element});
  shmem_offset += round_up_to_multiple_of_128(shr_X_node_ids.size() * sizeof(uint32_t));

  nd::view<int8_t, 2> shr_test_transform((int8_t *)(shmem + shmem_offset), {elem_per_block, test_nodes_per_element});
  shmem_offset += round_up_to_multiple_of_128(shr_test_transform.size() * sizeof(int8_t));

  // pulled-back, weighted qdata: {elem, qpt, (i, j, k, m)}
  nd::view<double, 3> shr_C((double *)(shmem + shmem_offset), {elem_per_block, qpts_per_element, block_size});
  shmem_offset += round_up_to_multiple_of_128(shr_C.size() * sizeof(double));

  nd::view<trial_qtype, 3> shr_hat_f((trial_qtype *)(shmem + shmem_offset), {elem_per_block, nr, qpts_per_element});
  shmem_offset += round_up_to_multiple_of_128(shr_hat_f.size() * sizeof(trial_qtype));

  nd::view<double, 3> shr_r_e((double *)(shmem + shmem_offset), {elem_per_block, nr, trial_nodes_per_element});
  shmem_offset += round_up_to_multiple_of_128(shr_r_e.size() * sizeof(double));

  nd::view<double, 3> shr_trial_scratch((double *)(shmem + shmem_offset), {elem_per_block, nr, trial_scratch_size});
  shmem_offset += round_up_to_multiple_of_128(shr_trial_scratch.size() * sizeof(double));

  nd::view<double, trial_shape_rank> shr_trial_shape_fn(const_cast<double *>(&trial_shape_functions[0]), trial_shape_functions.shape);
  if (stage_trial_shape_fns) {
    shr_trial_shape_fn = nd::view<double, trial_shape_rank>((double *)(shmem + shmem_offset), trial_shape_functions.shape);
    shmem_offset += round_up_to_multiple_of_128(trial_shape_functions.size() * sizeof(double));
  }

  nd::view<dX_dxi_type, 2> shr_dX_dxi_q;
  if constexpr (need_to_compute_dX_dxi) {
    shr_dX_dxi_q = nd::view<dX_dxi_type, 2>((dX_dxi_type *)(shmem + shmem_offset), {elem_per_block, qpts_per_element});
    shmem_offset += round_up_to_multiple_of_128(shr_dX_dxi_q.size() * sizeof(dX_dxi_type));
  }

  if (stage_trial_shape_fns) {
    for (uint32_t i = block_tid; i < shr_trial_shape_fn.size(); i += block_stride) {
      shr_trial_shape_fn[i] = trial_shape_functions[i];
    }
  }

  const uint32_t num_elements = elements.shape[0];
  const uint32_t e = blockIdx.x * elem_per_block + le;
  const bool active = e < num_elements;

  if (active) {
    uint32_t elem_id = elements[e];
    for (uint32_t i = elem_tid; i < shr_connectivity.shape[1]; i += elem_stride) {
      shr_connectivity(le, i) = connectivity(elem_id, i);
    }
  }
  __syncthreads();

  if (active && elem_tid == 0) {
    test_el.indices(test_offsets, &shr_connectivity(le, 0), &shr_test_node_ids(le, 0));
    trial_el.indices(trial_offsets, &shr_connectivity(le, 0), &shr_trial_node_ids(le, 0));
    X_el.indices(X_offsets, &shr_connectivity(le, 0), &shr_X_node_ids(le, 0));
    if constexpr (is_vector_valued(test_family)) {
      test_el.reorient(TransformationType::TransposePhysicalToParent, &shr_connectivity(le, 0), &shr_test_transform(le, 0));
    } else {
      for (uint32_t i = 0; i < test_nodes_per_element; i++) { shr_test_transform(le, i) = 0; }
    }
  }
  __syncthreads();

  // number of quadrature points per direction, for the tensor-product tables
  uint32_t q1D = 1;
  if constexpr (geom == Geometry::Quadrilateral) {
    while (q1D * q1D < qpts_per_element) { q1D++; }
  } else if constexpr (geom == Geometry::Hexahedron) {
    while (q1D * q1D * q1D < qpts_per_element) { q1D++; }
  }

  // dX_dxi(d, k) = sum_I X(I, d) dN_I/dxi_k, one thread per quadrature point
  if constexpr (need_to_compute_dX_dxi) {
    if (active) {
      for (uint32_t q = elem_tid; q < qpts_per_element; q += elem_stride) {
        dX_dxi_type J{};
        for (uint32_t I = 0; I < X_nodes_per_element; I++) {
          vec<gdim> dN = load_raw_test_shape<geom, Family::H1, DerivedQuantity::GRAD, gdim>(X_shape_functions, X_el.p, q1D, q, I);
          uint32_t node = shr_X_node_ids(le, I);
          for (uint32_t d = 0; d < gdim; d++) {
            double x = X(node, d);
            for (uint32_t k = 0; k < gdim; k++) { J(d, k) += x * dN[k]; }
          }
        }
        shr_dX_dxi_q(le, q) = J;
      }
    }
    __syncthreads();
  }

  // stage the qdata block for these elements, pulled back and weighted
  const uint32_t first_elem = blockIdx.x * elem_per_block;
  const uint32_t components = test_components * trial_components;
  for (uint32_t idx = block_tid; idx < elem_per_block * qpts_per_element * components; idx += block_stride) {
    uint32_t cj = idx % trial_components;
    uint32_t ci = (idx / trial_components) % test_components;
    uint32_t q = (idx / components) % qpts_per_element;
    uint32_t l = idx / (components * qpts_per_element);
    if (first_elem + l >= num_elements) { continue; }
    uint32_t qid = (first_elem + l) * qpts_per_element + q;

    mat_t C;
    for (uint32_t k = 0; k < tq; k++) {
      for (uint32_t m = 0; m < uq; m++) {
        C(k, m) = qdata(qid, ci, k, cj, m);
      }
    }
    if constexpr (need_to_compute_dX_dxi) {
      C = pull_back<test_family, test_op, trial_family, trial_op>(C, shr_dX_dxi_q(l, q));
    }
    double w = integration_weight<geom>(q, weights);
    double * dst = &shr_C(l, q, (ci * trial_components + cj) * tq * uq);
    for (uint32_t k = 0; k < tq; k++) {
      for (uint32_t m = 0; m < uq; m++) {
        dst[k * uq + m] = C(k, m) * w;
      }
    }
  }
  __syncthreads();

  // rows: each group of blockDim.x threads takes test node I = slot, slot + nr, ...
  // every group runs the same number of iterations (and inactive elements stay
  // in) so the block-wide syncs inside the element routines line up
  const uint32_t iterations = (test_nodes_per_element + nr - 1) / nr;
  for (uint32_t it = 0; it < iterations; it++) {
    uint32_t I = it * nr + slot;
    const bool valid = active && (I < test_nodes_per_element);
    I = min(I, test_nodes_per_element - 1);
    int8_t tr = valid ? shr_test_transform(le, I) : 0;

    // this thread's quadrature points are q = threadIdx.x + s * blockDim.x;
    // phi_I at the first few is cached across the component sub-rows
    vec<tq> phi_cache[max_qpts_per_thread];
    for (uint32_t s = 0; s < max_qpts_per_thread; s++) {
      uint32_t q = threadIdx.x + s * blockDim.x;
      if (q < qpts_per_element) {
        phi_cache[s] = load_reoriented_test_shape<geom, test_family, test_op, tq>(test_shape_functions, test_space.degree, q1D, q, I, tr);
      }
    }

    for (uint32_t ci = 0; ci < test_components; ci++) {
      for (uint32_t cj = 0; cj < trial_components; cj++) {
        uint32_t offset = (ci * trial_components + cj) * tq * uq;
        for (uint32_t s = 0, q = threadIdx.x; q < qpts_per_element; s++, q += blockDim.x) {
          vec<tq> phi = (s < max_qpts_per_thread)
            ? phi_cache[s]
            : load_reoriented_test_shape<geom, test_family, test_op, tq>(test_shape_functions, test_space.degree, q1D, q, I, tr);
          const double * Cij = &shr_C(le, q, offset);
          trial_qtype hat_f{};
          for (uint32_t k = 0; k < tq; k++) {
            for (uint32_t m = 0; m < uq; m++) { hat_f[m] += phi[k] * Cij[k * uq + m]; }
          }
          shr_hat_f(le, slot, q) = hat_f;
        }
        __syncthreads();

        if constexpr (trial_op == DerivedQuantity::VALUE) {
          trial_el.cuda_integrate_source(shr_r_e(le, slot), shr_hat_f(le, slot), shr_trial_shape_fn, &shr_trial_scratch(le, slot, 0));
        } else {
          trial_el.cuda_integrate_flux(shr_r_e(le, slot), shr_hat_f(le, slot), shr_trial_shape_fn, &shr_trial_scratch(le, slot, 0));
        }
        __syncthreads();

        if constexpr (is_vector_valued(trial_family)) {
          if (valid && threadIdx.x == 0) {
            trial_el.reorient(TransformationType::TransposePhysicalToParent, &shr_connectivity(le, 0), &shr_r_e(le, slot, 0));
          }
          __syncthreads();
        }

        if (valid) {
          int row_id = int(shr_test_node_ids(le, I) * test_components + ci);
          int row_start = row_ptr[row_id];
          int row_end = row_ptr[row_id + 1];
          for (uint32_t J = threadIdx.x; J < trial_nodes_per_element; J += blockDim.x) {
            int col_id = int(shr_trial_node_ids(le, J) * trial_components + cj);
            int position = find_column_in_sparse_row(col_ind, row_start, row_end, col_id);
            if (position >= 0) {
              atomicAdd(&values[position], shr_r_e(le, slot, J));
            }
          }
        }
        __syncthreads();
      }
    }
  }
}

template < Geometry geom, Family test_family, DerivedQuantity test_op, Family trial_family, DerivedQuantity trial_op >
void batched_integrate_spmat_cuda(sparse_matrix<memory::space::gpu> & A,
                                  nd::view<const double, 5, memory::space::gpu> qdata,
                                  FunctionSpace trial_space,
                                  FunctionSpace test_space,
                                  GeometryInfo trial_offsets,
                                  GeometryInfo test_offsets,
                                  const Field<memory::space::gpu> & X,
                                  const DomainType type,
                                  nd::view<const Connection, 2, memory::space::gpu> connectivity,
                                  const nd::view<const int, 1, memory::space::gpu> elements,
                                  const nd::view<const double, 2> xi,
                                  const nd::view<const double, 1> weights) {
  uint32_t num_elements = elements.size();
  if (num_elements == 0) return;

  constexpr uint32_t gdim = dimension(geom);
  constexpr uint32_t tq = qshape(test_family, test_op, gdim);
  constexpr uint32_t uq = qshape(trial_family, trial_op, gdim);

  FiniteElement<geom, test_family> test_el{test_space.degree};
  FiniteElement<geom, trial_family> trial_el{trial_space.degree};
  FiniteElement<geom, Family::H1> X_el{X.degree};

  // test tables are the minimal ones (1D factors on tensor-product elements),
  // read from global memory; trial tables are in the layout cuda_integrate_*
  // expects, unweighted since the quadrature weights are folded into the
  // staged qdata
  nd::array<double, 1, memory::space::cpu> ones(weights.shape);
  for (uint32_t i = 0; i < ones.size(); i++) { ones(i) = 1.0; }
  auto test_shape_fns = evaluate_test_shape_functions<geom, test_family, test_op>(test_el, xi);
  auto trial_shape_fns = evaluate_weighted_trial_shape_functions<geom, trial_family, trial_op>(trial_el, xi, ones);
  auto X_shape_fns = X_el.evaluate_shape_function_gradients(xi);
  constexpr uint32_t test_shape_rank = array_rank<decltype(test_shape_fns)>::value;
  constexpr uint32_t trial_shape_rank = array_rank<decltype(trial_shape_fns)>::value;
  constexpr uint32_t X_shape_rank = array_rank<decltype(X_shape_fns)>::value;
  nd::array<double, test_shape_rank, memory::space::gpu> test_shape_fns_device = test_shape_fns;
  nd::array<double, trial_shape_rank, memory::space::gpu> trial_shape_fns_device = trial_shape_fns;
  nd::array<double, X_shape_rank, memory::space::gpu> X_shape_fns_device = X_shape_fns;

  nd::array<double, 1, memory::space::cpu> weights_host(weights.shape);
  for (uint32_t i = 0; i < weights.shape[0]; i++) { weights_host(i) = weights(i); }
  nd::array<double, 1, memory::space::gpu> weights_device = weights_host;

  using dX_dxi_type = mat<gdim, gdim>;

  uint32_t qpts_per_element = qpe<geom>(xi.shape[0]);
  uint32_t test_nodes_per_element = test_el.num_nodes();
  uint32_t trial_nodes_per_element = trial_el.num_nodes();
  uint32_t X_nodes_per_element = X_el.num_nodes();
  uint32_t trial_scratch_size = trial_el.batch_interpolation_scratch_space(xi);
  uint32_t block_size = test_space.components * trial_space.components * tq * uq;

  // block shape {trial nodes, rows in flight, elements}: x is what the element
  // routines index; y and z fill the block up to ~512 threads
  const bool spatial = (type == DomainType::SPATIAL);
  bool stage_trial_shape_fns = true;
  uint32_t bx = trial_nodes_per_element;
  uint32_t nr = std::clamp(256u / bx, 1u, test_nodes_per_element);
  uint32_t elem_per_block = std::clamp(256u / (bx * nr), 1u, std::min(32u, num_elements));

  auto shmem_size = [&]() {
    uint32_t size = 0;
    size += round_up_to_multiple_of_128(elem_per_block * connectivity.shape[1] * sizeof(Connection));
    size += round_up_to_multiple_of_128(elem_per_block * test_nodes_per_element * sizeof(uint32_t));
    size += round_up_to_multiple_of_128(elem_per_block * trial_nodes_per_element * sizeof(uint32_t));
    size += round_up_to_multiple_of_128(elem_per_block * X_nodes_per_element * sizeof(uint32_t));
    size += round_up_to_multiple_of_128(elem_per_block * test_nodes_per_element * sizeof(int8_t));
    size += round_up_to_multiple_of_128(elem_per_block * qpts_per_element * block_size * sizeof(double));
    size += round_up_to_multiple_of_128(elem_per_block * nr * qpts_per_element * sizeof(vec<uq>));
    size += round_up_to_multiple_of_128(elem_per_block * nr * trial_nodes_per_element * sizeof(double));
    size += round_up_to_multiple_of_128(elem_per_block * nr * trial_scratch_size * sizeof(double));
    if (stage_trial_shape_fns) { size += round_up_to_multiple_of_128(trial_shape_fns.size() * sizeof(double)); }
    if (spatial) { size += round_up_to_multiple_of_128(elem_per_block * qpts_per_element * sizeof(dX_dxi_type)); }
    return size;
  };

  constexpr uint32_t max_default_dynamic_shmem = 48 * 1024;
  while (elem_per_block > 1 && shmem_size() > max_default_dynamic_shmem) { elem_per_block--; }
  while (nr > 1 && shmem_size() > max_default_dynamic_shmem) { nr--; }
  if (shmem_size() > max_default_dynamic_shmem) { stage_trial_shape_fns = false; }
  FEMTO_ASSERT(shmem_size() <= max_default_dynamic_shmem, "sparse matrix kernel exceeds available shared memory");

  dim3 block_dims = {bx, nr, elem_per_block};
  uint32_t grid_size = (num_elements + elem_per_block - 1) / elem_per_block;

  auto launch = [&](auto spatial_c) {
    constexpr bool spatial_ = decltype(spatial_c)::value;
    batched_integrate_spmat_kernel<test_shape_rank, trial_shape_rank, X_shape_rank, geom, test_family, test_op, trial_family, trial_op, spatial_>
      <<<grid_size, block_dims, shmem_size()>>>(
        A.values, A.row_ptr, A.col_ind, test_el, trial_el, qdata, trial_space, test_space, trial_offsets, test_offsets,
        connectivity, elements, test_shape_fns_device, trial_shape_fns_device, weights_device,
        X_el, X.offsets, X.data, X_shape_fns_device, qpts_per_element, trial_scratch_size, stage_trial_shape_fns);
  };

  if (spatial) { launch(std::true_type{}); } else { launch(std::false_type{}); }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace spmat_cuda

template <Family test_family, Family trial_family>
std::function<void(femto::sparse_matrix<memory::space::gpu>&)> integrate_sparse_matrix(
  BasisFunctionOp test,
  const nd::view<const double, 5, memory::space::gpu> qdata,
  BasisFunctionOp trial,
  const Domain<memory::space::gpu> & domain,
  const DomainType type) {

  return [&, type, test, trial, qdata](femto::sparse_matrix<memory::space::gpu> & A) {
    auto phi = test.function.space;
    auto psi = trial.function.space;

    uint32_t gdim = domain.mesh.geometry_dimension;

    stack::array<uint32_t, 5> shape5D = {
      qdata.shape[0],
      phi.components, qshape(phi.family, test.op, gdim),
      psi.components, qshape(psi.family, trial.op, gdim)
    };

    FEMTO_ASSERT(compatible_shapes(qdata.shape, shape5D), "incompatible array shapes");

    nd::view<const double, 5, memory::space::gpu> q5D{qdata.data(), shape5D};

    GeometryInfo counts = domain.mesh.geometry_counts();
    GeometryInfo test_offsets = scan(interior_nodes_per_geom(phi, gdim) * counts);
    GeometryInfo trial_offsets = scan(interior_nodes_per_geom(psi, gdim) * counts);

    if (A.nnz == 0) {
      A = blank_sparse_matrix(test, trial, domain);
    } else {
      zero(A.values);
    }

    uint32_t qoffset = 0;
    foreach_operation([&](auto test_op) {
      if constexpr (is_supported_combination(test_family, test_op)) {
        foreach_operation([&](auto trial_op) {
          if constexpr (is_supported_combination(trial_family, trial_op)) {
            if (test.op == test_op && trial.op == trial_op) {
              foreach_geometry([&](auto geom) {
                nd::view<const int, 1, memory::space::gpu> elements = domain.active_elements[geom];
                if (gdim == dimension(geom) && elements.size() > 0) {
                  nd::view<const Connection, 2, memory::space::gpu> connectivity = domain.mesh[geom];
                  nd::view<const double, 2> xi = domain.rule[geom].points;
                  nd::view<const double, 1> weights = domain.rule[geom].weights;

                  if constexpr (geom != Geometry::Vertex) {
                    stack::array<uint32_t, 5> qdata_shape{
                      domain.num_qpts[geom],
                      shape5D[1],
                      shape5D[2],
                      shape5D[3],
                      shape5D[4]
                    };
                    nd::view<const double, 5, memory::space::gpu> geom_qdata{&q5D(qoffset, 0, 0, 0, 0), qdata_shape};

                    spmat_cuda::batched_integrate_spmat_cuda<geom, test_family, test_op, trial_family, trial_op>(
                      A,
                      geom_qdata,
                      psi,
                      phi,
                      trial_offsets,
                      test_offsets,
                      domain.mesh.X,
                      type,
                      connectivity,
                      elements,
                      xi,
                      weights
                    );

                    qoffset += domain.num_qpts[geom];
                  }
                }
              });
            }
          }
        });
      }
    });

    return A;
  };
}

} // namespace impl

} // namespace femto

#endif
