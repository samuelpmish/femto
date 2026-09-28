#pragma once

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

// experiment (sparse_matrix_experiments.md, section 17): restructured pair
// loops for Hcurl-Hcurl forms in the simplex kernel, to remove redundant
// floating point work.  Only forms with one component per space participate;
// everything else takes the baseline path.
//   0: baseline pair loop
//   1: stage DB_J(q) = C(q) . psi_J(q) in shared once per element, so each
//      pair contracts a dot product instead of a matrix sandwich
//   2: stage reoriented per-element shape tables in shared, so pairs stop
//      re-applying the orientation weights at every quadrature point
//   3: both
#ifndef FEMTO_SPMAT_HCURL_VARIANT
#define FEMTO_SPMAT_HCURL_VARIANT 3
#endif

// Two kernels, chosen by the element geometry:
//
//   simplex_kernel (edges, triangles, tetrahedra): one thread per (test node,
//   trial node) pair, contracting phi_I . C_q . psi_J over the quadrature
//   points from dense {qpts, nodes, components} parent-space tables staged in
//   shared memory, with the reorientation of vector-valued bases folded into
//   per-pair weights.  Simplex bases have no tensor-product structure to
//   exploit, so the dense contraction is the right one and every pair is
//   independent -- no block syncs in the main loop.
//
//   tensor_product_kernel (quadrilaterals, hexahedra): a row of the element
//   matrix is the residual of a linear material driven by one test function,
//   so rows are formed with the element's own sum-factorized cuda_integrate_*
//   routines.  Block shape {trial nodes, nr, elements}: the x threads are what
//   those routines index, y picks one of nr rows in flight, z the element.
//
// Both stage, per block, the qdata of their elements pulled back to the parent
// element and multiplied by the quadrature weight (C -> testA . C . trialA^T . w).
// See sparse_matrix_experiments.md for the measurements behind the split.

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
// test and trial sides separately), expressed in terms of the precomputed
// inverse jacobian stored on the Domain
template < Family test_family, DerivedQuantity test_op, Family trial_family, DerivedQuantity trial_op, uint32_t tq, uint32_t uq, uint32_t dim >
__device__ __forceinline__ mat<tq, uq> pull_back(const mat<tq, uq> & C, const mat<dim, dim> & dxi_dX, double det_dX_dxi) {
  auto testA = piola_transformation_from_inverse<test_family, test_op>(dxi_dX, det_dX_dxi);
  auto trialA = weighted_piola_transformation_from_inverse<trial_family, trial_op>(dxi_dX, det_dX_dxi);
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

// dense {qpts, nodes, qshape} table of one basis operator, in parent coordinates
template < Geometry geom, Family family, DerivedQuantity op >
nd::array<double, 3, memory::space::cpu> tabulate_shape_functions(FiniteElement<geom, family> el, nd::view<const double, 2> xi) {
  constexpr uint32_t shape = qshape(family, op, dimension(geom));
  uint32_t qpts_per_element = qpe<geom>(xi.shape[0]);
  uint32_t nodes_per_element = el.num_nodes();
  nd::array<double, 3, memory::space::cpu> table({qpts_per_element, nodes_per_element, shape});
  for (uint32_t q = 0; q < qpts_per_element; q++) {
    auto xi_q = quadrature_point<geom>(q, xi);
    for (uint32_t i = 0; i < nodes_per_element; i++) {
      vec<shape> phi = as_qtype<shape>(shape_function<op>(el, xi_q, i, 0));
      for (uint32_t k = 0; k < shape; k++) { table(q, i, k) = phi[k]; }
    }
  }
  return table;
}

// a reoriented shape function is a fixed combination of one or two raw table
// entries: phi_i = w0 * table[i0] + w1 * table[i0 + 1]
__device__ __forceinline__ void reorientation(int8_t transformation, uint32_t i, uint32_t & i0, double & w0, double & w1) {
  i0 = i;
  w0 = 1.0;
  w1 = 0.0;
  if (transformation == -1) {
    w0 = -1.0;
  } else if (transformation != 0) {
    i0 = (i & 0xFFFFFFFEu);
    vec2 weights = face_transformation(transformation)[i % 2];
    w0 = weights[0];
    w1 = weights[1];
  }
}

template < uint32_t shape, typename view_t >
__device__ __forceinline__ vec<shape> load_shape(const view_t & table, uint32_t q, uint32_t i0, double w0, double w1) {
  vec<shape> phi;
  for (uint32_t k = 0; k < shape; k++) { phi[k] = table(q, i0, k) * w0; }
  if (w1 != 0.0) {
    for (uint32_t k = 0; k < shape; k++) { phi[k] += table(q, i0 + 1, k) * w1; }
  }
  return phi;
}

// number of quadrature points per direction, for the tensor-product tables
template < Geometry geom >
__device__ __forceinline__ uint32_t qpts_per_direction(uint32_t qpts_per_element) {
  uint32_t q1D = 1;
  if constexpr (geom == Geometry::Quadrilateral) {
    while (q1D * q1D < qpts_per_element) { q1D++; }
  } else if constexpr (geom == Geometry::Hexahedron) {
    while (q1D * q1D * q1D < qpts_per_element) { q1D++; }
  }
  return q1D;
}

template < Geometry geom,
           Family test_family,
           DerivedQuantity test_op,
           Family trial_family,
           DerivedQuantity trial_op,
           bool spatial >
__global__ void simplex_kernel(
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
  nd::view<const double, 3, memory::space::gpu> test_table,
  nd::view<const double, 3, memory::space::gpu> trial_table,
  nd::view<const double, 1, memory::space::gpu> weights,
  nd::view<const double, 3, memory::space::gpu> dxi_dX_q,
  nd::view<const double, 1, memory::space::gpu> det_dX_dxi_q,
  uint32_t qpts_per_element,
  bool stage_tables) {

  constexpr uint32_t gdim = dimension(geom);
  constexpr uint32_t tq = qshape(test_family, test_op, gdim);
  constexpr uint32_t uq = qshape(trial_family, trial_op, gdim);
  const uint32_t test_components = test_space.components;
  const uint32_t trial_components = trial_space.components;
  const uint32_t block_size = test_components * trial_components * tq * uq;

  constexpr int cvariant = (test_family == Family::Hcurl && trial_family == Family::Hcurl) ? FEMTO_SPMAT_HCURL_VARIANT : 0;
  const bool restructured = (cvariant != 0) && (test_components * trial_components == 1);

  using mat_t = mat<tq, uq>;

  uint32_t shmem_offset = 0;
  extern __shared__ char shmem[];

  const uint32_t elem_per_block = blockDim.z;
  const uint32_t test_nodes_per_element = test_el.num_nodes();
  const uint32_t trial_nodes_per_element = trial_el.num_nodes();
  const uint32_t le = threadIdx.z;
  const uint32_t elem_tid = threadIdx.x;
  const uint32_t elem_stride = blockDim.x;
  const uint32_t block_tid = threadIdx.x + blockDim.x * threadIdx.z;
  const uint32_t block_stride = blockDim.x * blockDim.z;

  nd::view<Connection, 2> shr_connectivity((Connection *)(shmem + shmem_offset), {elem_per_block, connectivity.shape[1]});
  shmem_offset += round_up_to_multiple_of_128(shr_connectivity.size() * sizeof(Connection));

  nd::view<uint32_t, 2> shr_test_node_ids((uint32_t *)(shmem + shmem_offset), {elem_per_block, test_nodes_per_element});
  shmem_offset += round_up_to_multiple_of_128(shr_test_node_ids.size() * sizeof(uint32_t));

  nd::view<uint32_t, 2> shr_trial_node_ids((uint32_t *)(shmem + shmem_offset), {elem_per_block, trial_nodes_per_element});
  shmem_offset += round_up_to_multiple_of_128(shr_trial_node_ids.size() * sizeof(uint32_t));

  nd::view<int8_t, 2> shr_test_transform((int8_t *)(shmem + shmem_offset), {elem_per_block, test_nodes_per_element});
  shmem_offset += round_up_to_multiple_of_128(shr_test_transform.size() * sizeof(int8_t));

  nd::view<int8_t, 2> shr_trial_transform((int8_t *)(shmem + shmem_offset), {elem_per_block, trial_nodes_per_element});
  shmem_offset += round_up_to_multiple_of_128(shr_trial_transform.size() * sizeof(int8_t));

  // pulled-back, weighted qdata: {elem, qpt, (i, j, k, m)}
  nd::view<double, 3> shr_C((double *)(shmem + shmem_offset), {elem_per_block, qpts_per_element, block_size});
  shmem_offset += round_up_to_multiple_of_128(shr_C.size() * sizeof(double));

  const bool same_tables = (test_table.values == trial_table.values);
  nd::view<const double, 3> test_table_view(test_table.values, test_table.shape);
  nd::view<const double, 3> trial_table_view(trial_table.values, trial_table.shape);
  if (stage_tables) {
    double * ptr = (double *)(shmem + shmem_offset);
    test_table_view = nd::view<const double, 3>(ptr, test_table.shape);
    shmem_offset += round_up_to_multiple_of_128(test_table.size() * sizeof(double));
    if (same_tables) {
      trial_table_view = test_table_view;
    } else {
      ptr = (double *)(shmem + shmem_offset);
      trial_table_view = nd::view<const double, 3>(ptr, trial_table.shape);
      shmem_offset += round_up_to_multiple_of_128(trial_table.size() * sizeof(double));
    }
  }

  nd::view<mat<gdim, gdim>, 2> shr_dxi_dX_q;
  nd::view<double, 2> shr_det_q;
  if constexpr (spatial) {
    shr_dxi_dX_q = nd::view<mat<gdim, gdim>, 2>((mat<gdim, gdim> *)(shmem + shmem_offset), {elem_per_block, qpts_per_element});
    shmem_offset += round_up_to_multiple_of_128(shr_dxi_dX_q.size() * sizeof(mat<gdim, gdim>));
    shr_det_q = nd::view<double, 2>((double *)(shmem + shmem_offset), {elem_per_block, qpts_per_element});
    shmem_offset += round_up_to_multiple_of_128(shr_det_q.size() * sizeof(double));
  }

  // restructured-variant buffers (see the FEMTO_SPMAT_HCURL_VARIANT comment);
  // the launcher sizes the allocation with the same conditions
  [[maybe_unused]] nd::view<double, 4> shr_test_phi, shr_trial_psi, shr_DB;
  if constexpr (cvariant == 2 || cvariant == 3) {
    if (restructured) {
      shr_test_phi = nd::view<double, 4>((double *)(shmem + shmem_offset), {elem_per_block, qpts_per_element, test_nodes_per_element, tq});
      shmem_offset += round_up_to_multiple_of_128(shr_test_phi.size() * sizeof(double));
    }
  }
  if constexpr (cvariant == 2) {
    if (restructured) {
      if (same_tables) {
        shr_trial_psi = shr_test_phi;
      } else {
        shr_trial_psi = nd::view<double, 4>((double *)(shmem + shmem_offset), {elem_per_block, qpts_per_element, trial_nodes_per_element, uq});
        shmem_offset += round_up_to_multiple_of_128(shr_trial_psi.size() * sizeof(double));
      }
    }
  }
  if constexpr (cvariant == 1 || cvariant == 3) {
    if (restructured) {
      shr_DB = nd::view<double, 4>((double *)(shmem + shmem_offset), {elem_per_block, qpts_per_element, trial_nodes_per_element, tq});
      shmem_offset += round_up_to_multiple_of_128(shr_DB.size() * sizeof(double));
    }
  }

  if (stage_tables) {
    double * dst = const_cast<double *>(test_table_view.values);
    for (uint32_t i = block_tid; i < test_table.size(); i += block_stride) { dst[i] = test_table[i]; }
    if (!same_tables) {
      dst = const_cast<double *>(trial_table_view.values);
      for (uint32_t i = block_tid; i < trial_table.size(); i += block_stride) { dst[i] = trial_table[i]; }
    }
  }

  const uint32_t num_elements = elements.shape[0];
  const uint32_t first_elem = blockIdx.x * elem_per_block;
  const uint32_t e = first_elem + le;
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
    if constexpr (is_vector_valued(test_family)) {
      test_el.reorient(TransformationType::TransposePhysicalToParent, &shr_connectivity(le, 0), &shr_test_transform(le, 0));
    }
    if constexpr (is_vector_valued(trial_family)) {
      trial_el.reorient(TransformationType::TransposePhysicalToParent, &shr_connectivity(le, 0), &shr_trial_transform(le, 0));
    }
  }
  __syncthreads();

  if constexpr (spatial) {
    if (active) {
      const double * src = &dxi_dX_q(e * qpts_per_element, 0, 0);
      double * dst = (double *)&shr_dxi_dX_q(le, 0);
      for (uint32_t i = elem_tid; i < qpts_per_element * gdim * gdim; i += elem_stride) { dst[i] = src[i]; }
      for (uint32_t q = elem_tid; q < qpts_per_element; q += elem_stride) {
        shr_det_q(le, q) = det_dX_dxi_q(e * qpts_per_element + q);
      }
    }
    __syncthreads();
  }

  // stage per-element reoriented shape values, so the pair loop reads final
  // values instead of combining raw table rows with orientation weights
  if constexpr (cvariant == 2 || cvariant == 3) {
    if (restructured) {
      for (uint32_t idx = block_tid; idx < elem_per_block * qpts_per_element * test_nodes_per_element; idx += block_stride) {
        uint32_t I = idx % test_nodes_per_element;
        uint32_t q = (idx / test_nodes_per_element) % qpts_per_element;
        uint32_t l = idx / (test_nodes_per_element * qpts_per_element);
        if (first_elem + l >= num_elements) { continue; }
        uint32_t I0;
        double w0, w1;
        reorientation(shr_test_transform(l, I), I, I0, w0, w1);
        vec<tq> phi = load_shape<tq>(test_table_view, q, I0, w0, w1);
        for (uint32_t k = 0; k < tq; k++) { shr_test_phi(l, q, I, k) = phi[k]; }
      }
      if constexpr (cvariant == 2) {
        if (!same_tables) {
          for (uint32_t idx = block_tid; idx < elem_per_block * qpts_per_element * trial_nodes_per_element; idx += block_stride) {
            uint32_t J = idx % trial_nodes_per_element;
            uint32_t q = (idx / trial_nodes_per_element) % qpts_per_element;
            uint32_t l = idx / (trial_nodes_per_element * qpts_per_element);
            if (first_elem + l >= num_elements) { continue; }
            uint32_t J0;
            double w0, w1;
            reorientation(shr_trial_transform(l, J), J, J0, w0, w1);
            vec<uq> psi = load_shape<uq>(trial_table_view, q, J0, w0, w1);
            for (uint32_t k = 0; k < uq; k++) { shr_trial_psi(l, q, J, k) = psi[k]; }
          }
        }
      }
    }
  }

  // stage the qdata block for these elements, pulled back and weighted
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
    if constexpr (spatial) {
      C = pull_back<test_family, test_op, trial_family, trial_op>(C, shr_dxi_dX_q(l, q), shr_det_q(l, q));
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

  // stage DB_J(q) = C(q) . psi_J(q): the product only depends on (J, q), so
  // computing it here once removes the matrix sandwich from every pair
  if constexpr (cvariant == 1 || cvariant == 3) {
    if (restructured) {
      for (uint32_t idx = block_tid; idx < elem_per_block * qpts_per_element * trial_nodes_per_element; idx += block_stride) {
        uint32_t J = idx % trial_nodes_per_element;
        uint32_t q = (idx / trial_nodes_per_element) % qpts_per_element;
        uint32_t l = idx / (trial_nodes_per_element * qpts_per_element);
        if (first_elem + l >= num_elements) { continue; }
        uint32_t J0;
        double w0, w1;
        reorientation(shr_trial_transform(l, J), J, J0, w0, w1);
        vec<uq> psi = load_shape<uq>(trial_table_view, q, J0, w0, w1);
        const double * Cq = &shr_C(l, q, 0);
        for (uint32_t k = 0; k < tq; k++) {
          double s = 0.0;
          for (uint32_t m = 0; m < uq; m++) { s += Cq[k * uq + m] * psi[m]; }
          shr_DB(l, q, J, k) = s;
        }
      }
      __syncthreads();
    }
  }

  // one (test node, trial node) pair per thread, one component pair at a time
  const uint32_t pairs_per_element = test_nodes_per_element * trial_nodes_per_element;

  if (restructured) {
    for (uint32_t idx = block_tid; idx < elem_per_block * pairs_per_element; idx += block_stride) {
      uint32_t l = idx / pairs_per_element;
      uint32_t pair = idx % pairs_per_element;
      uint32_t I = pair / trial_nodes_per_element;
      uint32_t J = pair % trial_nodes_per_element;
      if (first_elem + l >= num_elements) { continue; }

      double A = 0.0;
      if constexpr (cvariant == 1) {
        uint32_t I0;
        double w0, w1;
        reorientation(shr_test_transform(l, I), I, I0, w0, w1);
        for (uint32_t q = 0; q < qpts_per_element; q++) {
          vec<tq> phi = load_shape<tq>(test_table_view, q, I0, w0, w1);
          const double * db = &shr_DB(l, q, J, 0);
          for (uint32_t k = 0; k < tq; k++) { A += phi[k] * db[k]; }
        }
      } else if constexpr (cvariant == 2) {
        for (uint32_t q = 0; q < qpts_per_element; q++) {
          const double * phi = &shr_test_phi(l, q, I, 0);
          const double * psi = &shr_trial_psi(l, q, J, 0);
          const double * Cq = &shr_C(l, q, 0);
          for (uint32_t k = 0; k < tq; k++) {
            for (uint32_t m = 0; m < uq; m++) { A += phi[k] * Cq[k * uq + m] * psi[m]; }
          }
        }
      } else if constexpr (cvariant == 3) {
        for (uint32_t q = 0; q < qpts_per_element; q++) {
          const double * phi = &shr_test_phi(l, q, I, 0);
          const double * db = &shr_DB(l, q, J, 0);
          for (uint32_t k = 0; k < tq; k++) { A += phi[k] * db[k]; }
        }
      }

      int row_id = int(shr_test_node_ids(l, I));
      int position = find_column_in_sparse_row(col_ind, row_ptr[row_id], row_ptr[row_id + 1], int(shr_trial_node_ids(l, J)));
      if (position >= 0) {
        atomicAdd(&values[position], A);
      }
    }
    return;
  }

  for (uint32_t idx = block_tid; idx < elem_per_block * pairs_per_element; idx += block_stride) {
    uint32_t l = idx / pairs_per_element;
    uint32_t pair = idx % pairs_per_element;
    uint32_t I = pair / trial_nodes_per_element;
    uint32_t J = pair % trial_nodes_per_element;
    if (first_elem + l >= num_elements) { continue; }

    uint32_t I0 = I, J0 = J;
    double test_w0 = 1.0, test_w1 = 0.0, trial_w0 = 1.0, trial_w1 = 0.0;
    if constexpr (is_vector_valued(test_family)) { reorientation(shr_test_transform(l, I), I, I0, test_w0, test_w1); }
    if constexpr (is_vector_valued(trial_family)) { reorientation(shr_trial_transform(l, J), J, J0, trial_w0, trial_w1); }

    for (uint32_t ci = 0; ci < test_components; ci++) {
      int row_id = int(shr_test_node_ids(l, I) * test_components + ci);
      int row_start = row_ptr[row_id];
      int row_end = row_ptr[row_id + 1];
      for (uint32_t cj = 0; cj < trial_components; cj++) {
        uint32_t offset = (ci * trial_components + cj) * tq * uq;

        double A = 0.0;
        for (uint32_t q = 0; q < qpts_per_element; q++) {
          vec<tq> phi = load_shape<tq>(test_table_view, q, I0, test_w0, test_w1);
          vec<uq> psi = load_shape<uq>(trial_table_view, q, J0, trial_w0, trial_w1);
          const double * Cij = &shr_C(l, q, offset);
          for (uint32_t k = 0; k < tq; k++) {
            for (uint32_t m = 0; m < uq; m++) {
              A += phi[k] * Cij[k * uq + m] * psi[m];
            }
          }
        }

        int col_id = int(shr_trial_node_ids(l, J) * trial_components + cj);
        int position = find_column_in_sparse_row(col_ind, row_start, row_end, col_id);
        if (position >= 0) {
          atomicAdd(&values[position], A);
        }
      }
    }
  }
}

template < uint32_t test_shape_rank,
           uint32_t trial_shape_rank,
           Geometry geom,
           Family test_family,
           DerivedQuantity test_op,
           Family trial_family,
           DerivedQuantity trial_op,
           bool spatial >
__global__ void tensor_product_kernel(
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
  nd::view<const double, 3, memory::space::gpu> dxi_dX_q,
  nd::view<const double, 1, memory::space::gpu> det_dX_dxi_q,
  uint32_t qpts_per_element,
  uint32_t trial_scratch_size,
  bool stage_trial_shape_fns) {

  constexpr uint32_t gdim = dimension(geom);
  constexpr uint32_t tq = qshape(test_family, test_op, gdim);
  constexpr uint32_t uq = qshape(trial_family, trial_op, gdim);
  constexpr uint32_t max_qpts_per_thread = 4;   // phi_I cache; beyond it, recomputed

  using mat_t = mat<tq, uq>;
  using trial_qtype = vec<uq>;

  const uint32_t test_components = test_space.components;
  const uint32_t trial_components = trial_space.components;
  const uint32_t block_size = test_components * trial_components * tq * uq;

  uint32_t shmem_offset = 0;
  extern __shared__ char shmem[];

  const uint32_t nr = blockDim.y;
  const uint32_t elem_per_block = blockDim.z;
  const uint32_t test_nodes_per_element = test_el.num_nodes();
  const uint32_t trial_nodes_per_element = trial_el.num_nodes();
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

  nd::view<mat<gdim, gdim>, 2> shr_dxi_dX_q;
  nd::view<double, 2> shr_det_q;
  if constexpr (spatial) {
    shr_dxi_dX_q = nd::view<mat<gdim, gdim>, 2>((mat<gdim, gdim> *)(shmem + shmem_offset), {elem_per_block, qpts_per_element});
    shmem_offset += round_up_to_multiple_of_128(shr_dxi_dX_q.size() * sizeof(mat<gdim, gdim>));
    shr_det_q = nd::view<double, 2>((double *)(shmem + shmem_offset), {elem_per_block, qpts_per_element});
    shmem_offset += round_up_to_multiple_of_128(shr_det_q.size() * sizeof(double));
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
    if constexpr (is_vector_valued(test_family)) {
      test_el.reorient(TransformationType::TransposePhysicalToParent, &shr_connectivity(le, 0), &shr_test_transform(le, 0));
    } else {
      for (uint32_t i = 0; i < test_nodes_per_element; i++) { shr_test_transform(le, i) = 0; }
    }
  }
  __syncthreads();

  const uint32_t q1D = qpts_per_direction<geom>(qpts_per_element);

  if constexpr (spatial) {
    if (active) {
      const double * src = &dxi_dX_q(e * qpts_per_element, 0, 0);
      double * dst = (double *)&shr_dxi_dX_q(le, 0);
      for (uint32_t i = elem_tid; i < qpts_per_element * gdim * gdim; i += elem_stride) { dst[i] = src[i]; }
      for (uint32_t q = elem_tid; q < qpts_per_element; q += elem_stride) {
        shr_det_q(le, q) = det_dX_dxi_q(e * qpts_per_element + q);
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
    if constexpr (spatial) {
      C = pull_back<test_family, test_op, trial_family, trial_op>(C, shr_dxi_dX_q(l, q), shr_det_q(l, q));
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
                                  nd::view<const double, 3, memory::space::gpu> dxi_dX,
                                  nd::view<const double, 1, memory::space::gpu> det_dX_dxi,
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
  constexpr bool tensor_product = (geom == Geometry::Quadrilateral || geom == Geometry::Hexahedron);

  FiniteElement<geom, test_family> test_el{test_space.degree};
  FiniteElement<geom, trial_family> trial_el{trial_space.degree};

  nd::array<double, 1, memory::space::cpu> weights_host(weights.shape);
  for (uint32_t i = 0; i < weights.shape[0]; i++) { weights_host(i) = weights(i); }
  nd::array<double, 1, memory::space::gpu> weights_device = weights_host;

  uint32_t qpts_per_element = qpe<geom>(xi.shape[0]);
  uint32_t test_nodes_per_element = test_el.num_nodes();
  uint32_t trial_nodes_per_element = trial_el.num_nodes();
  uint32_t block_size = test_space.components * trial_space.components * tq * uq;
  const bool spatial = (type == DomainType::SPATIAL);

  // shared memory common to both kernels
  auto shmem_common = [&](uint32_t elem_per_block) {
    uint32_t size = 0;
    size += round_up_to_multiple_of_128(elem_per_block * connectivity.shape[1] * sizeof(Connection));
    size += round_up_to_multiple_of_128(elem_per_block * test_nodes_per_element * sizeof(uint32_t));
    size += round_up_to_multiple_of_128(elem_per_block * trial_nodes_per_element * sizeof(uint32_t));
    size += round_up_to_multiple_of_128(elem_per_block * test_nodes_per_element * sizeof(int8_t));
    size += round_up_to_multiple_of_128(elem_per_block * qpts_per_element * block_size * sizeof(double));
    if (spatial) {
      size += round_up_to_multiple_of_128(elem_per_block * qpts_per_element * uint32_t(sizeof(mat<gdim, gdim>)));
      size += round_up_to_multiple_of_128(elem_per_block * qpts_per_element * uint32_t(sizeof(double)));
    }
    return size;
  };

  // the block shape is settled once the kernel's register count is known
  // (cudaFuncGetAttributes) and dynamic shared memory above 48 KB is opted in
  auto configure = [&](auto kernel, uint32_t & elem_per_block, auto && shrink, auto && shmem_size) {
    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));
    int device, max_shmem;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&max_shmem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    shrink(uint32_t(attr.maxThreadsPerBlock), uint32_t(max_shmem));
    FEMTO_ASSERT(shmem_size() <= uint32_t(max_shmem), "sparse matrix kernel exceeds available shared memory");
    if (shmem_size() > 48 * 1024) {
      CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(shmem_size())));
    }
    return (num_elements + elem_per_block - 1) / elem_per_block;
  };

  if constexpr (tensor_product) {
    // test tables are the minimal ones (1D factors), read from global memory;
    // trial tables are in the layout cuda_integrate_* expects, unweighted since
    // the quadrature weights are folded into the staged qdata
    nd::array<double, 1, memory::space::cpu> ones(weights.shape);
    for (uint32_t i = 0; i < ones.size(); i++) { ones(i) = 1.0; }
    auto test_shape_fns = evaluate_test_shape_functions<geom, test_family, test_op>(test_el, xi);
    auto trial_shape_fns = evaluate_weighted_trial_shape_functions<geom, trial_family, trial_op>(trial_el, xi, ones);
    constexpr uint32_t test_shape_rank = array_rank<decltype(test_shape_fns)>::value;
    constexpr uint32_t trial_shape_rank = array_rank<decltype(trial_shape_fns)>::value;
    nd::array<double, test_shape_rank, memory::space::gpu> test_shape_fns_device = test_shape_fns;
    nd::array<double, trial_shape_rank, memory::space::gpu> trial_shape_fns_device = trial_shape_fns;
    uint32_t trial_scratch_size = trial_el.batch_interpolation_scratch_space(xi);

    // block shape {trial nodes, rows in flight, elements}: x is what the element
    // routines index.  For scalar problems each row's phi_I is used once, so one
    // row is in flight and elements fill the block; for multi-component problems
    // a row group reuses its phi_I cache over the components^2 sub-rows and all
    // rows in flight at once measured best.  Elements fill the block up to ~256
    // threads (sparse_matrix_experiments.md, section 11).
    bool stage_trial_shape_fns = true;
    uint32_t bx = trial_nodes_per_element;
    uint32_t nr = (test_space.components * trial_space.components > 1) ? test_nodes_per_element : 1;
    uint32_t elem_per_block = std::clamp(256u / (bx * nr), 1u, std::min(32u, num_elements));

    auto shmem_size = [&]() {
      uint32_t size = shmem_common(elem_per_block);
      size += round_up_to_multiple_of_128(elem_per_block * nr * qpts_per_element * sizeof(vec<uq>));
      size += round_up_to_multiple_of_128(elem_per_block * nr * trial_nodes_per_element * sizeof(double));
      size += round_up_to_multiple_of_128(elem_per_block * nr * trial_scratch_size * sizeof(double));
      if (stage_trial_shape_fns) { size += round_up_to_multiple_of_128(trial_shape_fns.size() * sizeof(double)); }
      return size;
    };
    auto shrink = [&](uint32_t max_threads, uint32_t max_shmem) {
      nr = std::min(nr, std::max(1u, max_threads / bx));
      elem_per_block = std::min(elem_per_block, std::max(1u, max_threads / (bx * nr)));
      while (elem_per_block > 1 && shmem_size() > max_shmem) { elem_per_block--; }
      while (nr > 1 && shmem_size() > max_shmem) { nr--; }
      if (shmem_size() > max_shmem) { stage_trial_shape_fns = false; }
    };

    auto launch = [&](auto spatial_c) {
      constexpr bool spatial_ = decltype(spatial_c)::value;
      auto kernel = tensor_product_kernel<test_shape_rank, trial_shape_rank, geom, test_family, test_op, trial_family, trial_op, spatial_>;
      uint32_t grid_size = configure(kernel, elem_per_block, shrink, shmem_size);
      kernel<<<grid_size, dim3{bx, nr, elem_per_block}, shmem_size()>>>(
          A.values, A.row_ptr, A.col_ind, test_el, trial_el, qdata, trial_space, test_space, trial_offsets, test_offsets,
          connectivity, elements, test_shape_fns_device, trial_shape_fns_device, weights_device,
          dxi_dX, det_dX_dxi, qpts_per_element, trial_scratch_size, stage_trial_shape_fns);
    };
    if (spatial) { launch(std::true_type{}); } else { launch(std::false_type{}); }
  } else {
    // dense parent-space tables of both bases, staged in shared memory when they fit
    auto test_table = tabulate_shape_functions<geom, test_family, test_op>(test_el, xi);
    nd::array<double, 3, memory::space::gpu> test_table_device = test_table;
    nd::array<double, 3, memory::space::gpu> trial_table_device;
    const bool same_tables = (test_family == trial_family && test_op == trial_op && test_space.degree == trial_space.degree);
    if (!same_tables) {
      trial_table_device = tabulate_shape_functions<geom, trial_family, trial_op>(trial_el, xi);
    }
    nd::view<const double, 3, memory::space::gpu> trial_table_view = same_tables ? test_table_device : trial_table_device;

    // one pair per thread, flattened over the block; the prologue needs at
    // least max(nodes, qpts) threads per element, and beyond that the block
    // size only sets how many elements share the staged tables
    constexpr int cvariant = (test_family == Family::Hcurl && trial_family == Family::Hcurl) ? FEMTO_SPMAT_HCURL_VARIANT : 0;
    const bool restructured = (cvariant != 0) && (test_space.components * trial_space.components == 1);

    // variants 2 and 3 read per-element reoriented values, so the raw shared
    // copy of the tables would only serve the staging pass -- skip it
    bool stage_tables = !(restructured && (cvariant == 2 || cvariant == 3));
    uint32_t pairs_per_element = test_nodes_per_element * trial_nodes_per_element;
    uint32_t bx = std::max({std::min(pairs_per_element, 128u), qpts_per_element, test_nodes_per_element, trial_nodes_per_element});
    uint32_t elem_per_block = std::clamp(512u / bx, 1u, std::min(32u, num_elements));

    auto shmem_size = [&]() {
      uint32_t size = shmem_common(elem_per_block);
      size += round_up_to_multiple_of_128(elem_per_block * trial_nodes_per_element * sizeof(int8_t));
      if (stage_tables) {
        size += round_up_to_multiple_of_128(test_table.size() * sizeof(double));
        if (!same_tables) { size += round_up_to_multiple_of_128(trial_table_device.size() * sizeof(double)); }
      }
      if (restructured) {
        if (cvariant == 2 || cvariant == 3) {
          size += round_up_to_multiple_of_128(elem_per_block * qpts_per_element * test_nodes_per_element * tq * sizeof(double));
        }
        if (cvariant == 2 && !same_tables) {
          size += round_up_to_multiple_of_128(elem_per_block * qpts_per_element * trial_nodes_per_element * uq * sizeof(double));
        }
        if (cvariant == 1 || cvariant == 3) {
          size += round_up_to_multiple_of_128(elem_per_block * qpts_per_element * trial_nodes_per_element * tq * sizeof(double));
        }
      }
      return size;
    };
    auto shrink = [&](uint32_t max_threads, uint32_t max_shmem) {
      elem_per_block = std::min(elem_per_block, std::max(1u, max_threads / bx));
      // the per-element buffers make shared memory the occupancy limiter, so
      // keep a block's share small enough for at least four resident blocks
      if (restructured) {
        while (elem_per_block > 1 && shmem_size() * 4 > max_shmem) { elem_per_block--; }
      }
      while (elem_per_block > 1 && shmem_size() > max_shmem) { elem_per_block--; }
      if (shmem_size() > max_shmem) { stage_tables = false; }
    };

    auto launch = [&](auto spatial_c) {
      constexpr bool spatial_ = decltype(spatial_c)::value;
      auto kernel = simplex_kernel<geom, test_family, test_op, trial_family, trial_op, spatial_>;
      uint32_t grid_size = configure(kernel, elem_per_block, shrink, shmem_size);
      kernel<<<grid_size, dim3{bx, 1, elem_per_block}, shmem_size()>>>(
          A.values, A.row_ptr, A.col_ind, test_el, trial_el, qdata, trial_space, test_space, trial_offsets, test_offsets,
          connectivity, elements, test_table_device, trial_table_view, weights_device,
          dxi_dX, det_dX_dxi, qpts_per_element, stage_tables);
    };
    if (spatial) { launch(std::true_type{}); } else { launch(std::false_type{}); }
  }

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace spmat_cuda

template < Family test_family, DerivedQuantity test_op, Family trial_family, DerivedQuantity trial_op >
std::function<void(femto::sparse_matrix<memory::space::gpu>&)> integrate_sparse_matrix(
  FunctionSpace phi,
  const nd::view<const double, 5, memory::space::gpu> qdata,
  FunctionSpace psi,
  const Domain<memory::space::gpu> & domain,
  const DomainType type) {

  return [&, type, phi, psi, qdata](femto::sparse_matrix<memory::space::gpu> & A) {

    // dofs are numbered over the whole mesh, quadrature data over the domain
    uint32_t gdim = domain.mesh.geometry_dimension;
    uint32_t domain_gdim = domain.geometry_dimension;

    // on a boundary domain dxi_dX holds a (gdim x sdim) pseudo-inverse the
    // kernel does not stage; only the facet measure det_dX_dxi is used, which
    // is all a scalar VALUE x VALUE product needs
    FEMTO_ASSERT(domain_gdim == gdim || (is_scalar_valued(test_family) && test_op == DerivedQuantity::VALUE &&
                                         is_scalar_valued(trial_family) && trial_op == DerivedQuantity::VALUE),
                 "spatial derivatives and vector-valued fields are not supported on facets embedded in a higher-dimensional space");

    stack::array<uint32_t, 5> shape5D = {
      qdata.shape[0],
      phi.components, qshape(test_family, test_op, domain_gdim),
      psi.components, qshape(trial_family, trial_op, domain_gdim)
    };

    FEMTO_ASSERT(compatible_shapes(qdata.shape, shape5D), "incompatible array shapes");

    nd::view<const double, 5, memory::space::gpu> q5D{qdata.data(), shape5D};

    GeometryInfo counts = domain.mesh.geometry_counts();
    GeometryInfo test_offsets = scan(interior_nodes_per_geom(phi, gdim) * counts);
    GeometryInfo trial_offsets = scan(interior_nodes_per_geom(psi, gdim) * counts);

    if (A.nnz == 0) {
      A = blank_sparse_matrix(phi, psi, domain);
    } else {
      zero(A.values);
    }

    uint32_t qoffset = 0;
    foreach_geometry([&](auto geom) {
    nd::view<const int, 1, memory::space::gpu> elements = domain.active_elements[geom];
    if (domain_gdim == dimension(geom) && elements.size() > 0) {
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

        FEMTO_ASSERT(domain.dxi_dX.shape[0] >= qoffset + domain.num_qpts[geom],
                     "domain is missing its precomputed jacobian inverses");
        nd::view<const double, 3, memory::space::gpu> geom_dxi_dX{
          &domain.dxi_dX(qoffset, 0, 0), {domain.num_qpts[geom], domain_gdim, domain_gdim}};
        nd::view<const double, 1, memory::space::gpu> geom_det{
          &domain.det_dX_dxi(qoffset), {domain.num_qpts[geom]}};

        spmat_cuda::batched_integrate_spmat_cuda<geom, test_family, test_op, trial_family, trial_op>(
          A,
          geom_qdata,
          psi,
          phi,
          trial_offsets,
          test_offsets,
          geom_dxi_dX,
          geom_det,
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

    return A;
  };
}

} // namespace impl

} // namespace femto

#endif
