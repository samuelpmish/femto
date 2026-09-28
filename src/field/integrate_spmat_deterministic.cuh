#pragma once

// Deterministic element-matrix assembly: the restructured simplex pair kernel
// from integrate_spmat.cuh (staged per-element reoriented shape tables, staged
// DB_J(q) = C(q).psi_J(q), one (I, J) pair per thread), but the contributions
// are written to a dense array K(e, ci*NC+cj, I*NN+J) -- component-major,
// pair-minor, so consecutive threads store consecutive doubles -- instead of
// being scattered into a CSR matrix.  No row search, no atomics, and bitwise
// reproducible results.  The per-element test/trial dof node ids are recorded
// so a later gather pass (or the host) can assemble.
//
// Exposed to users as element_integrate() (femto/integrate.hpp).
// Isoparametric domains only, for now; tensor-product elements use the same
// dense pair contraction as the simplices (no sum factorization here).

#include "integrate_spmat.cuh"

#ifdef FEMTO_ENABLE_CUDA

namespace femto {

namespace impl {

namespace spmat_deterministic {

using spmat_cuda::tabulate_shape_functions;
using spmat_cuda::reorientation;
using spmat_cuda::load_shape;

template < Geometry geom,
           Family test_family,
           DerivedQuantity test_op,
           Family trial_family,
           DerivedQuantity trial_op,
           uint32_t NC >
__global__ void element_matrix_kernel(
  nd::view<double, 3, memory::space::gpu> K_e,
  nd::view<uint32_t, 2, memory::space::gpu> test_ids_out,
  nd::view<uint32_t, 2, memory::space::gpu> trial_ids_out,
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
  uint32_t qpts_per_element,
  bool stage_DB) {

  constexpr uint32_t gdim = dimension(geom);
  constexpr uint32_t tq = qshape(test_family, test_op, gdim);
  constexpr uint32_t uq = qshape(trial_family, trial_op, gdim);
  constexpr uint32_t block_size = NC * NC * tq * uq;

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

  // weighted qdata: {elem, qpt, (ci, k, cj, m)}
  nd::view<double, 3> shr_C((double *)(shmem + shmem_offset), {elem_per_block, qpts_per_element, block_size});
  shmem_offset += round_up_to_multiple_of_128(shr_C.size() * sizeof(double));

  // per-element reoriented test shapes {elem, qpt, I, k}, shared with the
  // trial side when the tables coincide
  const bool same_tables = (test_table.values == trial_table.values);
  nd::view<double, 4> shr_test_phi((double *)(shmem + shmem_offset), {elem_per_block, qpts_per_element, test_nodes_per_element, tq});
  shmem_offset += round_up_to_multiple_of_128(shr_test_phi.size() * sizeof(double));

  // DB_J(q) = C(q) . psi_J(q): {elem, qpt, J, (cj*NC + ci)*tq + k}.  When the
  // launcher could not fit this buffer (stage_DB false), the pair loop falls
  // back to contracting shr_C directly, reading psi from shr_trial_psi
  nd::view<double, 4> shr_DB, shr_trial_psi;
  if (stage_DB) {
    shr_DB = nd::view<double, 4>((double *)(shmem + shmem_offset), {elem_per_block, qpts_per_element, trial_nodes_per_element, NC * NC * tq});
    shmem_offset += round_up_to_multiple_of_128(shr_DB.size() * sizeof(double));
  } else if (same_tables) {
    shr_trial_psi = shr_test_phi;
  } else {
    shr_trial_psi = nd::view<double, 4>((double *)(shmem + shmem_offset), {elem_per_block, qpts_per_element, trial_nodes_per_element, uq});
    shmem_offset += round_up_to_multiple_of_128(shr_trial_psi.size() * sizeof(double));
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

  if (active) {
    for (uint32_t i = elem_tid; i < test_nodes_per_element; i += elem_stride) {
      test_ids_out(e, i) = shr_test_node_ids(le, i);
    }
    for (uint32_t i = elem_tid; i < trial_nodes_per_element; i += elem_stride) {
      trial_ids_out(e, i) = shr_trial_node_ids(le, i);
    }
  }

  // stage the reoriented test shapes and the weighted qdata
  for (uint32_t idx = block_tid; idx < elem_per_block * qpts_per_element * test_nodes_per_element; idx += block_stride) {
    uint32_t I = idx % test_nodes_per_element;
    uint32_t q = (idx / test_nodes_per_element) % qpts_per_element;
    uint32_t l = idx / (test_nodes_per_element * qpts_per_element);
    if (first_elem + l >= num_elements) { continue; }
    uint32_t I0 = I;
    double w0 = 1.0, w1 = 0.0;
    if constexpr (is_vector_valued(test_family)) { reorientation(shr_test_transform(l, I), I, I0, w0, w1); }
    vec<tq> phi = load_shape<tq>(test_table, q, I0, w0, w1);
    for (uint32_t k = 0; k < tq; k++) { shr_test_phi(l, q, I, k) = phi[k]; }
  }

  if (!stage_DB && !same_tables) {
    for (uint32_t idx = block_tid; idx < elem_per_block * qpts_per_element * trial_nodes_per_element; idx += block_stride) {
      uint32_t J = idx % trial_nodes_per_element;
      uint32_t q = (idx / trial_nodes_per_element) % qpts_per_element;
      uint32_t l = idx / (trial_nodes_per_element * qpts_per_element);
      if (first_elem + l >= num_elements) { continue; }
      uint32_t J0 = J;
      double w0 = 1.0, w1 = 0.0;
      if constexpr (is_vector_valued(trial_family)) { reorientation(shr_trial_transform(l, J), J, J0, w0, w1); }
      vec<uq> psi = load_shape<uq>(trial_table, q, J0, w0, w1);
      for (uint32_t m = 0; m < uq; m++) { shr_trial_psi(l, q, J, m) = psi[m]; }
    }
  }

  for (uint32_t idx = block_tid; idx < elem_per_block * qpts_per_element * NC * NC; idx += block_stride) {
    uint32_t cj = idx % NC;
    uint32_t ci = (idx / NC) % NC;
    uint32_t q = (idx / (NC * NC)) % qpts_per_element;
    uint32_t l = idx / (NC * NC * qpts_per_element);
    if (first_elem + l >= num_elements) { continue; }
    uint32_t qid = (first_elem + l) * qpts_per_element + q;
    double w = integration_weight<geom>(q, weights);
    double * dst = &shr_C(l, q, (ci * NC + cj) * tq * uq);
    for (uint32_t k = 0; k < tq; k++) {
      for (uint32_t m = 0; m < uq; m++) {
        dst[k * uq + m] = qdata(qid, ci, k, cj, m) * w;
      }
    }
  }
  __syncthreads();

  // DB_J(q) = C(q) . psi_J(q), once per (J, q) instead of once per pair
  if (stage_DB) {
    for (uint32_t idx = block_tid; idx < elem_per_block * qpts_per_element * trial_nodes_per_element; idx += block_stride) {
      uint32_t J = idx % trial_nodes_per_element;
      uint32_t q = (idx / trial_nodes_per_element) % qpts_per_element;
      uint32_t l = idx / (trial_nodes_per_element * qpts_per_element);
      if (first_elem + l >= num_elements) { continue; }
      vec<uq> psi;
      if (same_tables) {
        for (uint32_t m = 0; m < uq; m++) { psi[m] = shr_test_phi(l, q, J, m); }
      } else {
        uint32_t J0 = J;
        double w0 = 1.0, w1 = 0.0;
        if constexpr (is_vector_valued(trial_family)) { reorientation(shr_trial_transform(l, J), J, J0, w0, w1); }
        psi = load_shape<uq>(trial_table, q, J0, w0, w1);
      }
      for (uint32_t ci = 0; ci < NC; ci++) {
        for (uint32_t cj = 0; cj < NC; cj++) {
          const double * Cq = &shr_C(l, q, (ci * NC + cj) * tq * uq);
          double * db = &shr_DB(l, q, J, (cj * NC + ci) * tq);
          for (uint32_t k = 0; k < tq; k++) {
            double s = 0.0;
            for (uint32_t m = 0; m < uq; m++) { s += Cq[k * uq + m] * psi[m]; }
            db[k] = s;
          }
        }
      }
    }
  }
  __syncthreads();

  // one pair per thread: contract and store the NC x NC block, pair-minor
  const uint32_t pairs_per_element = test_nodes_per_element * trial_nodes_per_element;
  for (uint32_t idx = block_tid; idx < elem_per_block * pairs_per_element; idx += block_stride) {
    uint32_t l = idx / pairs_per_element;
    uint32_t pair = idx % pairs_per_element;
    uint32_t I = pair / trial_nodes_per_element;
    uint32_t J = pair % trial_nodes_per_element;
    if (first_elem + l >= num_elements) { continue; }

    double A[NC][NC];
    for (uint32_t ci = 0; ci < NC; ci++) {
      for (uint32_t cj = 0; cj < NC; cj++) { A[ci][cj] = 0.0; }
    }

    if (stage_DB) {
      for (uint32_t q = 0; q < qpts_per_element; q++) {
        const double * phi = &shr_test_phi(l, q, I, 0);
        const double * db = &shr_DB(l, q, J, 0);
        for (uint32_t ci = 0; ci < NC; ci++) {
          for (uint32_t cj = 0; cj < NC; cj++) {
            double s = 0.0;
            for (uint32_t k = 0; k < tq; k++) { s += phi[k] * db[(cj * NC + ci) * tq + k]; }
            A[ci][cj] += s;
          }
        }
      }
    } else {
      for (uint32_t q = 0; q < qpts_per_element; q++) {
        const double * phi = &shr_test_phi(l, q, I, 0);
        const double * psi = &shr_trial_psi(l, q, J, 0);
        for (uint32_t ci = 0; ci < NC; ci++) {
          for (uint32_t cj = 0; cj < NC; cj++) {
            const double * Cq = &shr_C(l, q, (ci * NC + cj) * tq * uq);
            double s = 0.0;
            for (uint32_t k = 0; k < tq; k++) {
              for (uint32_t m = 0; m < uq; m++) { s += phi[k] * Cq[k * uq + m] * psi[m]; }
            }
            A[ci][cj] += s;
          }
        }
      }
    }

    for (uint32_t ci = 0; ci < NC; ci++) {
      for (uint32_t cj = 0; cj < NC; cj++) {
        K_e(first_elem + l, ci * NC + cj, pair) = A[ci][cj];
      }
    }
  }
}

template < Geometry geom, Family test_family, DerivedQuantity test_op, Family trial_family, DerivedQuantity trial_op, uint32_t NC >
void launch_element_matrix_kernel(nd::view<double, 3, memory::space::gpu> K_e,
                                  nd::view<uint32_t, 2, memory::space::gpu> test_ids_out,
                                  nd::view<uint32_t, 2, memory::space::gpu> trial_ids_out,
                                  nd::view<const double, 5, memory::space::gpu> qdata,
                                  FunctionSpace trial_space,
                                  FunctionSpace test_space,
                                  GeometryInfo trial_offsets,
                                  GeometryInfo test_offsets,
                                  nd::view<const Connection, 2, memory::space::gpu> connectivity,
                                  const nd::view<const int, 1, memory::space::gpu> elements,
                                  const nd::view<const double, 2> xi,
                                  const nd::view<const double, 1> weights) {
  constexpr uint32_t gdim = dimension(geom);
  constexpr uint32_t tq = qshape(test_family, test_op, gdim);
  constexpr uint32_t uq = qshape(trial_family, trial_op, gdim);

  uint32_t num_elements = elements.size();
  if (num_elements == 0) { return; }

  FiniteElement<geom, test_family> test_el{test_space.degree};
  FiniteElement<geom, trial_family> trial_el{trial_space.degree};

  auto test_table = tabulate_shape_functions<geom, test_family, test_op>(test_el, xi);
  nd::array<double, 3, memory::space::gpu> test_table_device = test_table;
  nd::array<double, 3, memory::space::gpu> trial_table_device;
  const bool same_tables = (test_family == trial_family && test_op == trial_op && test_space.degree == trial_space.degree);
  if (!same_tables) {
    trial_table_device = tabulate_shape_functions<geom, trial_family, trial_op>(trial_el, xi);
  }
  nd::view<const double, 3, memory::space::gpu> trial_table_view = same_tables ? test_table_device : trial_table_device;

  nd::array<double, 1, memory::space::cpu> weights_host(weights.shape);
  for (uint32_t i = 0; i < weights.shape[0]; i++) { weights_host(i) = weights(i); }
  nd::array<double, 1, memory::space::gpu> weights_device = weights_host;

  uint32_t qpts_per_element = qpe<geom>(xi.shape[0]);
  uint32_t test_nodes_per_element = test_el.num_nodes();
  uint32_t trial_nodes_per_element = trial_el.num_nodes();
  uint32_t pairs_per_element = test_nodes_per_element * trial_nodes_per_element;

  uint32_t bx = std::max({std::min(pairs_per_element, 128u), qpts_per_element, test_nodes_per_element, trial_nodes_per_element});
  uint32_t elem_per_block = std::clamp(512u / bx, 1u, std::min(32u, num_elements));

  // the DB buffer is dropped (and the pair loop falls back to contracting
  // shr_C directly) when it cannot fit in shared memory even at one element
  // per block -- e.g. p2 hex elasticity, where it would be 27q x 27J x 27
  bool stage_DB = true;
  auto shmem_size = [&]() {
    uint32_t size = 0;
    size += round_up_to_multiple_of_128(elem_per_block * connectivity.shape[1] * sizeof(Connection));
    size += round_up_to_multiple_of_128(elem_per_block * test_nodes_per_element * sizeof(uint32_t));
    size += round_up_to_multiple_of_128(elem_per_block * trial_nodes_per_element * sizeof(uint32_t));
    size += round_up_to_multiple_of_128(elem_per_block * test_nodes_per_element * sizeof(int8_t));
    size += round_up_to_multiple_of_128(elem_per_block * trial_nodes_per_element * sizeof(int8_t));
    size += round_up_to_multiple_of_128(elem_per_block * qpts_per_element * NC * NC * tq * uq * sizeof(double));
    size += round_up_to_multiple_of_128(elem_per_block * qpts_per_element * test_nodes_per_element * tq * sizeof(double));
    if (stage_DB) {
      size += round_up_to_multiple_of_128(elem_per_block * qpts_per_element * trial_nodes_per_element * NC * NC * tq * sizeof(double));
    } else if (!same_tables) {
      size += round_up_to_multiple_of_128(elem_per_block * qpts_per_element * trial_nodes_per_element * uq * sizeof(double));
    }
    return size;
  };

  auto kernel = element_matrix_kernel<geom, test_family, test_op, trial_family, trial_op, NC>;
  cudaFuncAttributes attr;
  CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));
  int device, max_shmem;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaDeviceGetAttribute(&max_shmem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  elem_per_block = std::min(elem_per_block, std::max(1u, uint32_t(attr.maxThreadsPerBlock) / bx));
  while (elem_per_block > 1 && shmem_size() * 4 > uint32_t(max_shmem)) { elem_per_block--; }
  if (shmem_size() > uint32_t(max_shmem)) { stage_DB = false; }
  while (elem_per_block > 1 && shmem_size() * 4 > uint32_t(max_shmem)) { elem_per_block--; }
  FEMTO_ASSERT(shmem_size() <= uint32_t(max_shmem), "element matrix kernel exceeds available shared memory");
  if (shmem_size() > 48 * 1024) {
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(shmem_size())));
  }
  uint32_t grid_size = (num_elements + elem_per_block - 1) / elem_per_block;

  kernel<<<grid_size, dim3{bx, 1, elem_per_block}, shmem_size()>>>(
      K_e, test_ids_out, trial_ids_out, test_el, trial_el, qdata, trial_space, test_space,
      trial_offsets, test_offsets, connectivity, elements,
      test_table_device, trial_table_view, weights_device, qpts_per_element, stage_DB);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace spmat_deterministic

template < Family test_family, DerivedQuantity test_op, Family trial_family, DerivedQuantity trial_op >
void element_matrices_into(
  FunctionSpace phi,
  const nd::view<const double, 5, memory::space::gpu> qdata,
  FunctionSpace psi,
  const Domain<memory::space::gpu> & domain,
  const DomainType type,
  nd::view<double, 3, memory::space::gpu> K,
  nd::view<uint32_t, 2, memory::space::gpu> test_ids,
  nd::view<uint32_t, 2, memory::space::gpu> trial_ids) {

  FEMTO_ASSERT(type == DomainType::ISOPARAMETRIC, "element_integrate: isoparametric domains only");
  FEMTO_ASSERT(phi.components == psi.components, "element_integrate: matching component counts only");

  uint32_t gdim = domain.mesh.geometry_dimension;

  stack::array<uint32_t, 5> shape5D = {
    qdata.shape[0],
    phi.components, qshape(test_family, test_op, gdim),
    psi.components, qshape(trial_family, trial_op, gdim)
  };
  FEMTO_ASSERT(compatible_shapes(qdata.shape, shape5D), "incompatible array shapes");
  nd::view<const double, 5, memory::space::gpu> q5D{qdata.data(), shape5D};

  GeometryInfo counts = domain.mesh.geometry_counts();
  GeometryInfo test_offsets = scan(interior_nodes_per_geom(phi, gdim) * counts);
  GeometryInfo trial_offsets = scan(interior_nodes_per_geom(psi, gdim) * counts);

  uint32_t qoffset = 0;
  uint32_t eoffset = 0;
  foreach_geometry([&](auto geom) {
    nd::view<const int, 1, memory::space::gpu> elements = domain.active_elements[geom];
    if (gdim == dimension(geom) && elements.size() > 0) {
      if constexpr (geom == Geometry::Triangle || geom == Geometry::Quadrilateral ||
                    geom == Geometry::Tetrahedron || geom == Geometry::Hexahedron) {
        nd::view<const Connection, 2, memory::space::gpu> connectivity = domain.mesh[geom];
        nd::view<const double, 2> xi = domain.rule[geom].points;
        nd::view<const double, 1> weights = domain.rule[geom].weights;
        stack::array<uint32_t, 5> qdata_shape{domain.num_qpts[geom], shape5D[1], shape5D[2], shape5D[3], shape5D[4]};
        nd::view<const double, 5, memory::space::gpu> geom_qdata{&q5D(qoffset, 0, 0, 0, 0), qdata_shape};
        uint32_t ne = elements.size();
        nd::view<double, 3, memory::space::gpu> K_geom(&K(eoffset, 0, 0), {ne, K.shape[1], K.shape[2]});
        nd::view<uint32_t, 2, memory::space::gpu> tid_geom(&test_ids(eoffset, 0), {ne, test_ids.shape[1]});
        nd::view<uint32_t, 2, memory::space::gpu> uid_geom(&trial_ids(eoffset, 0), {ne, trial_ids.shape[1]});
        auto run = [&](auto nc) {
          spmat_deterministic::launch_element_matrix_kernel<geom, test_family, test_op, trial_family, trial_op, decltype(nc)::value>(
            K_geom, tid_geom, uid_geom, geom_qdata, psi, phi, trial_offsets, test_offsets, connectivity, elements, xi, weights);
        };
        if (phi.components == 1) { run(std::integral_constant<uint32_t, 1>{}); }
        else if (phi.components == 3) { run(std::integral_constant<uint32_t, 3>{}); }
        else { FEMTO_ASSERT(false, "element_integrate: unsupported component count"); }
        qoffset += domain.num_qpts[geom];
        eoffset += ne;
      } else {
        FEMTO_ASSERT(false, "element_integrate: unsupported element geometry");
      }
    }
  });
}

template < Family test_family, DerivedQuantity test_op, Family trial_family, DerivedQuantity trial_op >
ElementMatrices<memory::space::gpu> element_matrices(
  FunctionSpace phi,
  const nd::view<const double, 5, memory::space::gpu> qdata,
  FunctionSpace psi,
  const Domain<memory::space::gpu> & domain,
  const DomainType type) {

  uint32_t gdim = domain.mesh.geometry_dimension;
  GeometryInfo counts = domain.geometry_counts();
  uint32_t num_elements = total(counts);

  // all active elements must share one node count, so K's shape is uniform
  uint32_t test_nodes = 0, trial_nodes = 0;
  foreach_geometry([&](auto geom) {
    if constexpr (geom == Geometry::Triangle || geom == Geometry::Quadrilateral ||
                  geom == Geometry::Tetrahedron || geom == Geometry::Hexahedron) {
      if (gdim == dimension(geom) && counts[geom] > 0) {
        uint32_t tn = FiniteElement<geom, test_family>{phi.degree}.num_nodes();
        uint32_t un = FiniteElement<geom, trial_family>{psi.degree}.num_nodes();
        FEMTO_ASSERT(test_nodes == 0 || (test_nodes == tn && trial_nodes == un),
                     "element_integrate: meshes mixing element types are not supported");
        test_nodes = tn;
        trial_nodes = un;
      }
    }
  });
  FEMTO_ASSERT(test_nodes > 0, "element_integrate: no supported elements in the domain");

  ElementMatrices<memory::space::gpu> out;
  out.K.resize({num_elements, phi.components * psi.components, test_nodes * trial_nodes});
  out.test_ids.resize({num_elements, test_nodes});
  out.trial_ids.resize({num_elements, trial_nodes});
  element_matrices_into<test_family, test_op, trial_family, trial_op>(
    phi, qdata, psi, domain, type, out.K, out.test_ids, out.trial_ids);
  return out;
}

} // namespace impl

} // namespace femto

#endif
