#include "common.hpp"

#include "femto/domain.hpp"
#include "femto/assert.hpp"

#include "fm/types/matrix.hpp"
#include "fm/types/vec.hpp"

#include "misc/for_constexpr.hpp"
#include "misc/macros.hpp"

#include <algorithm>
#include <type_traits>

#ifdef FEMTO_ENABLE_CUDA

#define USE_ATOMICS 0
#define SHARED_GATHER 0

namespace femto {

template <>
const Domain<memory::space::gpu>::AssemblyLUT & Domain<memory::space::gpu>::get(Geometry g, Family f, uint32_t p) const {

  if (!gather_tables.count({g, f, p})) {
    Mesh<> mesh_cpu = copy_to<memory::space::cpu>(mesh);
    Domain<> domain_cpu(mesh_cpu, rule, /* precompute_jacobians = */ false);
    foreach_geometry([&](auto geom) {
      domain_cpu.active_elements[geom] = nd::copy_to<memory::space::cpu>(active_elements[geom]);
    });
    domain_cpu.num_qpts = num_qpts;
    domain_cpu.geometry_dimension = geometry_dimension;
    const Domain<>::AssemblyLUT & cpu_table = domain_cpu.get(g, f, p);

    AssemblyLUT & table = gather_tables[{g, f, p}];
    table.offsets = cpu_table.offsets;
    table.ids = cpu_table.ids;
  }

  return gather_tables.at({g, f, p});
}

template < uint32_t dim >
__global__ void invert_jacobians_kernel(nd::view<double, 3, memory::space::gpu> dxi_dX,
                                        nd::view<double, 1, memory::space::gpu> det_dX_dxi,
                                        nd::view<const double, 3, memory::space::gpu> dX_dxi) {
  uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q < dX_dxi.shape[0]) {
    if constexpr (dim == 1) {
      det_dX_dxi(q) = dX_dxi(q, 0, 0);
      dxi_dX(q, 0, 0) = 1.0 / dX_dxi(q, 0, 0);
    } else {
      mat<dim, dim> A;
      for (uint32_t i = 0; i < dim; i++)
        for (uint32_t j = 0; j < dim; j++) { A(i, j) = dX_dxi(q, i, j); }
      det_dX_dxi(q) = det(A);
      mat<dim, dim> Ainv = inv(A);
      for (uint32_t i = 0; i < dim; i++)
        for (uint32_t j = 0; j < dim; j++) { dxi_dX(q, i, j) = Ainv(i, j); }
    }
  }
}

template < uint32_t sdim, uint32_t gdim >
__global__ void facet_jacobians_kernel(nd::view<double, 3, memory::space::gpu> dxi_dX,
                                       nd::view<double, 1, memory::space::gpu> det_dX_dxi,
                                       nd::view<const double, 3, memory::space::gpu> J) {
  uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q < J.shape[0]) { facet_jacobian<sdim, gdim>(dxi_dX, det_dX_dxi, J, q); }
}

template <>
void Domain<memory::space::gpu>::compute_jacobian_inverses() {
  const uint32_t gdim = geometry_dimension;
  const uint32_t sdim = mesh.X.data.shape[1];
  if (gdim == 0) { return; }
  nd::array<double, 3, memory::space::gpu> J = evaluate(grad(mesh.X), isoparametric(*this));
  dxi_dX.resize({J.shape[0], gdim, sdim});
  det_dX_dxi.resize({J.shape[0]});
  const uint32_t n = J.shape[0];
  if (n == 0) { return; }
  if (sdim != gdim) {
    if (sdim == 2 && gdim == 1) { facet_jacobians_kernel<2, 1><<<(n + 255) / 256, 256>>>(dxi_dX, det_dX_dxi, J); }
    if (sdim == 3 && gdim == 1) { facet_jacobians_kernel<3, 1><<<(n + 255) / 256, 256>>>(dxi_dX, det_dX_dxi, J); }
    if (sdim == 3 && gdim == 2) { facet_jacobians_kernel<3, 2><<<(n + 255) / 256, 256>>>(dxi_dX, det_dX_dxi, J); }
  } else if (gdim == 1) {
    invert_jacobians_kernel<1><<<(n + 255) / 256, 256>>>(dxi_dX, det_dX_dxi, J);
  } else if (gdim == 2) {
    invert_jacobians_kernel<2><<<(n + 255) / 256, 256>>>(dxi_dX, det_dX_dxi, J);
  } else {
    invert_jacobians_kernel<3><<<(n + 255) / 256, 256>>>(dxi_dX, det_dX_dxi, J);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

namespace impl {

template < uint32_t shape_rank,
           uint32_t X_shape_rank,
           Geometry geom,
           Family family,
           DerivedQuantity op,
           typename input_type,
           bool need_to_compute_dX_dxi,
           bool facet >
__global__ void batched_integrate_residual_kernel(
  nd::view<double, 2, memory::space::gpu> r_data,
  GeometryInfo r_offsets,
  FiniteElement< geom, family > r_el,
  nd::view<const input_type, 2, memory::space::gpu> input_q,
  nd::view<const Connection, 2, memory::space::gpu> connectivity,
  nd::view<const int, 1, memory::space::gpu> elements,
  nd::view<const double, shape_rank, memory::space::gpu> r_shape_functions,
  FiniteElement< geom, Family::H1 > X_el,
  GeometryInfo X_offsets,
  nd::view<const double, 2, memory::space::gpu> X,
  nd::view<const double, X_shape_rank, memory::space::gpu> X_shape_functions,
  uint32_t qpts_per_elem,
  uint32_t r_scratch_size,
  uint32_t X_scratch_size
) {

  constexpr uint32_t gdim = dimension(geom);
  using X_grad_type = typename FiniteElement< geom, Family::H1 >::flux_type;
  // a facet of a boundary domain has one more spatial component than reference
  // coordinates, so its jacobian is not square and only its measure is used
  constexpr uint32_t X_rows = facet ? gdim + 1 : gdim;
  using dX_dxi_type = mat<X_rows, gdim>;

  uint32_t shmem_offset = 0;
  extern __shared__ char shmem[];

  uint32_t elem_per_block = blockDim.z;
  uint32_t r_nodes_per_elem = r_el.num_nodes();
  uint32_t local_tid = threadIdx.x + blockDim.x * threadIdx.y;
  uint32_t local_stride = blockDim.x * blockDim.y;

  nd::view< Connection, 2 > shr_connectivity((Connection *)(shmem + shmem_offset), {elem_per_block, connectivity.shape[1]});
  shmem_offset += round_up_to_multiple_of_128(shr_connectivity.size() * sizeof(Connection));

#if USE_ATOMICS
  nd::view< uint32_t, 2 > shr_r_node_ids((uint32_t *)(shmem + shmem_offset), {elem_per_block, r_nodes_per_elem});
  shmem_offset += round_up_to_multiple_of_128(shr_r_node_ids.size() * sizeof(uint32_t));
#endif

  nd::view< input_type, 2 > shr_input_q((input_type *)(shmem + shmem_offset), {elem_per_block, qpts_per_elem});
  shmem_offset += round_up_to_multiple_of_128(shr_input_q.size() * sizeof(input_type));

  nd::view< double, 2 > shr_r_e((double *)(shmem + shmem_offset), {elem_per_block, r_nodes_per_elem});
  shmem_offset += round_up_to_multiple_of_128(shr_r_e.size() * sizeof(double));

  // The shape function tables stay in const global memory rather than being
  // staged in shared memory. They are small and every thread reads the same
  // entries, so the read-only/L1 cache amortizes the reuse for free, while a
  // copy costs a block-wide memcpy, a __syncthreads, and the occupancy that
  // footprint buys.
  //
  // Measured A/B on 1M-element meshes (A100), both variants timed after a warmup
  // sweep: the two are within ~10% almost everywhere. Global wins decisively in
  // the one case where the table is genuinely large -- p=3 Hcurl tetrahedra,
  // where the dense {qpts, 45 nodes, 3} table crushes occupancy when staged
  // (2.26 ms vs 5.07 ms for a spatial evaluate). Shared is 8-14% ahead on
  // hexahedra, where the table is only the 1D factors and the sum-factorized
  // contraction re-reads them often. Global is the better single choice: neutral
  // to slightly behind at worst, 2x ahead at best.
  //
  // Beware when re-measuring: these kernels are fast enough that GPU clock ramp
  // dominates a cold first measurement -- the same code can time 2x slower on
  // the first pass than the third. Always warm up before comparing variants.
  nd::view< double, shape_rank > r_shape_fn(const_cast<double *>(&r_shape_functions[0]), r_shape_functions.shape);

  nd::view< double, 2 > shr_r_scratch((double *)(shmem + shmem_offset), {elem_per_block, r_scratch_size});
  shmem_offset += round_up_to_multiple_of_128(shr_r_scratch.size() * sizeof(double));

  if constexpr (need_to_compute_dX_dxi) {
    uint32_t X_nodes_per_elem = X_el.num_nodes();

    nd::view< uint32_t, 2 > shr_X_node_ids((uint32_t *)(shmem + shmem_offset), {elem_per_block, X_nodes_per_elem});
    shmem_offset += round_up_to_multiple_of_128(shr_X_node_ids.size() * sizeof(uint32_t));

    nd::view< double, 2 > shr_X_e((double *)(shmem + shmem_offset), {elem_per_block, X_nodes_per_elem});
    shmem_offset += round_up_to_multiple_of_128(shr_X_e.size() * sizeof(double));

    nd::view< double, X_shape_rank > X_shape_fn(const_cast<double *>(&X_shape_functions[0]), X_shape_functions.shape);

    nd::view< X_grad_type, 2 > shr_X_grad_q((X_grad_type *)(shmem + shmem_offset), {elem_per_block, qpts_per_elem});
    shmem_offset += round_up_to_multiple_of_128(shr_X_grad_q.size() * sizeof(X_grad_type));

    nd::view< dX_dxi_type, 2 > shr_dX_dxi_q((dX_dxi_type *)(shmem + shmem_offset), {elem_per_block, qpts_per_elem});
    shmem_offset += round_up_to_multiple_of_128(shr_dX_dxi_q.size() * sizeof(dX_dxi_type));

    nd::view< double, 2 > shr_X_scratch((double *)(shmem + shmem_offset), {elem_per_block, X_scratch_size});

    uint32_t e = blockIdx.x * blockDim.z + threadIdx.z;
    uint32_t local_elem_id = threadIdx.z;
    bool active = e < elements.shape[0];
    uint32_t source_e = active ? e : 0;
    uint32_t elem_id = elements(source_e);
    double * r_scratch = shr_r_scratch.data() + local_elem_id * r_scratch_size;
    double * X_scratch = shr_X_scratch.data() + local_elem_id * X_scratch_size;

    for (uint32_t i = local_tid; i < shr_connectivity.shape[1]; i += local_stride) {
      shr_connectivity(local_elem_id, i) = connectivity(elem_id, i);
    }
    __syncthreads();

#if USE_ATOMICS
    if (local_tid == 0) {
      r_el.indices(r_offsets, &shr_connectivity(local_elem_id, 0), &shr_r_node_ids(local_elem_id, 0));
    }
#endif

    if (local_tid == 0) {
      X_el.indices(X_offsets, &shr_connectivity(local_elem_id, 0), &shr_X_node_ids(local_elem_id, 0));
    }
    __syncthreads();

    for (uint32_t d = 0; d < X_rows; d++) {
      for (uint32_t i = local_tid; i < X_nodes_per_elem; i += local_stride) {
        shr_X_e(local_elem_id, i) = X(shr_X_node_ids(local_elem_id, i), d);
      }
      __syncthreads();

      X_el.cuda_gradient(
        shr_X_grad_q(local_elem_id),
        shr_X_e(local_elem_id),
        X_shape_fn,
        X_scratch
      );
      __syncthreads();

      for (uint32_t q = local_tid; q < qpts_per_elem; q += local_stride) {
        shr_dX_dxi_q(local_elem_id, q)[d] = shr_X_grad_q(local_elem_id, q);
      }
      __syncthreads();
    }

    uint32_t num_components = r_data.shape[1];
    for (uint32_t c = 0; c < num_components; c++) {
      for (uint32_t q = local_tid; q < qpts_per_elem; q += local_stride) {
        if constexpr (facet) {
          double A_q = facet_measure(shr_dX_dxi_q(local_elem_id, q));
          shr_input_q(local_elem_id, q) = fm::dot(A_q, input_q(qpts_per_elem * source_e + q, c));
        } else {
          auto A_q = weighted_piola_transformation<family, op>(shr_dX_dxi_q(local_elem_id, q));
          shr_input_q(local_elem_id, q) = fm::dot(A_q, input_q(qpts_per_elem * source_e + q, c));
        }
      }
      __syncthreads();

      if constexpr (op == DerivedQuantity::VALUE) {
        r_el.cuda_integrate_source(
          shr_r_e(local_elem_id),
          shr_input_q(local_elem_id),
          r_shape_fn,
          r_scratch
        );
      }

      if constexpr (op == DerivedQuantity::GRAD ||
                    op == DerivedQuantity::CURL) {
        r_el.cuda_integrate_flux(
          shr_r_e(local_elem_id),
          shr_input_q(local_elem_id),
          r_shape_fn,
          r_scratch
        );
      }
      __syncthreads();

      if constexpr (is_vector_valued(family)) {
        if (local_tid == 0) {
          r_el.reorient(TransformationType::TransposePhysicalToParent, &shr_connectivity(local_elem_id, 0), &shr_r_e(local_elem_id, 0));
        }
        __syncthreads();
      }

#if USE_ATOMICS
      if (active) {
        for (uint32_t i = local_tid; i < r_nodes_per_elem; i += local_stride) {
          atomicAdd(&r_data(shr_r_node_ids(local_elem_id, i), c), shr_r_e(local_elem_id, i));
        }
      }
#else
      if (active) {
        for (uint32_t i = local_tid; i < r_nodes_per_elem; i += local_stride) {
          r_data(e * r_nodes_per_elem + i, c) = shr_r_e(local_elem_id, i);
        }
      }
#endif
      __syncthreads();
    }
  } else {
    uint32_t e = blockIdx.x * blockDim.z + threadIdx.z;
    uint32_t local_elem_id = threadIdx.z;
    bool active = e < elements.shape[0];
    uint32_t source_e = active ? e : 0;
    uint32_t elem_id = elements(source_e);
    double * r_scratch = shr_r_scratch.data() + local_elem_id * r_scratch_size;

    for (uint32_t i = local_tid; i < shr_connectivity.shape[1]; i += local_stride) {
      shr_connectivity(local_elem_id, i) = connectivity(elem_id, i);
    }
    __syncthreads();

#if USE_ATOMICS
    if (local_tid == 0) {
      r_el.indices(r_offsets, &shr_connectivity(local_elem_id, 0), &shr_r_node_ids(local_elem_id, 0));
    }
    __syncthreads();
#endif

    uint32_t num_components = r_data.shape[1];
    for (uint32_t c = 0; c < num_components; c++) {
      for (uint32_t q = local_tid; q < qpts_per_elem; q += local_stride) {
        shr_input_q(local_elem_id, q) = input_q(qpts_per_elem * source_e + q, c);
      }
      __syncthreads();

      if constexpr (op == DerivedQuantity::VALUE) {
        r_el.cuda_integrate_source(
          shr_r_e(local_elem_id),
          shr_input_q(local_elem_id),
          r_shape_fn,
          r_scratch
        );
      }

      if constexpr (op == DerivedQuantity::GRAD ||
                    op == DerivedQuantity::CURL) {
        r_el.cuda_integrate_flux(
          shr_r_e(local_elem_id),
          shr_input_q(local_elem_id),
          r_shape_fn,
          r_scratch
        );
      }
      __syncthreads();

      if constexpr (is_vector_valued(family)) {
        if (local_tid == 0) {
          r_el.reorient(TransformationType::TransposePhysicalToParent, &shr_connectivity(local_elem_id, 0), &shr_r_e(local_elem_id, 0));
        }
        __syncthreads();
      }

#if USE_ATOMICS
      if (active) {
        for (uint32_t i = local_tid; i < r_nodes_per_elem; i += local_stride) {
          atomicAdd(&r_data(shr_r_node_ids(local_elem_id, i), c), shr_r_e(local_elem_id, i));
        }
      }
#else
      if (active) {
        for (uint32_t i = local_tid; i < r_nodes_per_elem; i += local_stride) {
          r_data(e * r_nodes_per_elem + i, c) = shr_r_e(local_elem_id, i);
        }
      }
#endif
      __syncthreads();
    }
  }
}

__global__ void residual_assemble_kernel(nd::view<double, 2, memory::space::gpu> nodal_residuals,
                                         nd::view<const double, 2, memory::space::gpu> element_residuals,
                                         nd::view<const uint32_t, 1, memory::space::gpu> offsets,
                                         nd::view<const uint32_t, 1, memory::space::gpu> ids) {

  uint32_t num_nodes = nodal_residuals.shape[0];
  uint32_t components = nodal_residuals.shape[1];

  uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < num_nodes) {
    uint32_t begin = offsets[i];
    uint32_t end = offsets[i + 1];

    for (uint32_t c = 0; c < components; c++) {
      double sum = 0.0;
      for (uint32_t k = begin; k < end; k++) {
        uint32_t id = ids[k];
        sum += element_residuals(id, c);
      }
      nodal_residuals(i, c) += sum;
    }
  }
}

__global__ void residual_assemble_kernel_shmem(nd::view<double, 2, memory::space::gpu> nodal_residuals,
                                               nd::view<const double, 2, memory::space::gpu> element_residuals,
                                               nd::view<const uint32_t, 1, memory::space::gpu> offsets,
                                               nd::view<const uint32_t, 1, memory::space::gpu> ids) {

  uint32_t num_nodes = nodal_residuals.shape[0];
  uint32_t components = nodal_residuals.shape[1];

  extern __shared__ double shr_buffer[];
  nd::view< double, 2 > shr_nodal_residuals(shr_buffer, {blockDim.x, components});

  uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < num_nodes) {
    uint32_t begin = offsets[i];
    uint32_t end = offsets[i + 1];

    for (uint32_t c = 0; c < components; c++) {
      double sum = 0.0;
      for (uint32_t k = begin; k < end; k++) {
        uint32_t id = ids[k];
        sum += element_residuals(id, c);
      }
      shr_nodal_residuals(threadIdx.x, c) = sum;
    }
    __syncthreads();

    uint32_t active_threads = min(blockDim.x, num_nodes - blockIdx.x * blockDim.x);
    for (uint32_t j = threadIdx.x; j < components * active_threads; j += active_threads) {
      nodal_residuals[components * blockIdx.x * blockDim.x + j] += shr_nodal_residuals[j];
    }
  }
}

template < Geometry geom, Family family, DerivedQuantity op, memory::space residual_space >
void batched_integrate_residual_cuda(Residual<family, residual_space> & r,
                                     const Field<Family::H1, memory::space::gpu> & X,
                                     const DomainType type,
                                     nd::view<const double, 3, memory::space::gpu> f_q,
                                     nd::view<const Connection, 2, memory::space::gpu> connectivity,
                                     const nd::view<const int, 1, memory::space::gpu> elements,
                                     const nd::view<const double, 2> xi,
                                     const nd::view<const double, 1> weights,
                                     const Domain<memory::space::gpu>::AssemblyLUT & table,
                                     nd::array<double, 1, memory::space::gpu> & element_residual_buffer) {

  uint32_t num_elements = elements.size();
  if (num_elements == 0) return;

  FiniteElement< geom, family > r_el{r.space.degree};

  using input_t = std::conditional_t<
    op == DerivedQuantity::VALUE,
    typename FiniteElement< geom, family >::source_type,
    typename FiniteElement< geom, family >::flux_type
  >;

  nd::view<const input_t, 2, memory::space::gpu> input_q((const input_t*)&f_q[0], {f_q.shape[0], f_q.shape[1]});

  constexpr uint32_t gdim = dimension(geom);
  using X_grad_type = typename FiniteElement< geom, Family::H1 >::flux_type;

  // see the kernel: a boundary domain's facets carry a (gdim + 1) x gdim jacobian
  const bool facet = (type == DomainType::SPATIAL) && (X.data.shape[1] != gdim);
  const uint32_t dX_dxi_bytes = facet ? sizeof(mat<gdim + 1, gdim>) : sizeof(mat<gdim, gdim>);

  uint32_t qpts_per_element = impl::qpe<geom>(xi.shape[0]);
  uint32_t num_nodes = r.data.shape[0];
  uint32_t components = r.data.shape[1];
  uint32_t r_nodes_per_element = r_el.num_nodes();

  auto r_shape_fns = [&]() {
    if constexpr (op == DerivedQuantity::VALUE) {
      return r_el.evaluate_weighted_shape_functions(xi, weights);
    }

    if constexpr (op == DerivedQuantity::CURL && family == Family::Hcurl) {
      return r_el.evaluate_weighted_shape_function_curls(xi, weights);
    }

    if constexpr (op == DerivedQuantity::GRAD && is_scalar_valued(family)) {
      return r_el.evaluate_weighted_shape_function_gradients(xi, weights);
    }
  }();

  FiniteElement< geom, Family::H1 > X_el{X.degree};
  uint32_t X_nodes_per_element = X_el.num_nodes();
  auto X_shape_fns = X_el.evaluate_shape_function_gradients(xi);

  constexpr uint32_t r_shape_rank = array_rank< decltype(r_shape_fns) >::value;
  constexpr uint32_t X_shape_rank = array_rank< decltype(X_shape_fns) >::value;

  nd::array<double, r_shape_rank, memory::space::gpu> r_shape_fns_device = r_shape_fns;
  nd::array<double, X_shape_rank, memory::space::gpu> X_shape_fns_device = X_shape_fns;

  uint32_t r_scratch_size = r_el.batch_interpolation_scratch_space(xi);
  uint32_t X_scratch_size = X_el.batch_interpolation_scratch_space(xi);

  uint32_t elem_per_block = elements_per_block(geom, family, r.space.degree);
  elem_per_block = std::max(1u, elem_per_block);

  auto base_shmem_size = [&]() {
    uint32_t shmem_size = 0;
    shmem_size += round_up_to_multiple_of_128(elem_per_block * connectivity.shape[1] * sizeof(Connection));
#if USE_ATOMICS
    shmem_size += round_up_to_multiple_of_128(elem_per_block * r_nodes_per_element * sizeof(uint32_t));
#endif
    shmem_size += round_up_to_multiple_of_128(elem_per_block * qpts_per_element * sizeof(input_t));
    shmem_size += round_up_to_multiple_of_128(elem_per_block * r_nodes_per_element * sizeof(double));
    shmem_size += round_up_to_multiple_of_128(elem_per_block * r_scratch_size * sizeof(double));
    return shmem_size;
  };

  auto spatial_shmem_size = [&]() {
    uint32_t shmem_size = base_shmem_size();
    shmem_size += round_up_to_multiple_of_128(elem_per_block * X_nodes_per_element * sizeof(uint32_t));
    shmem_size += round_up_to_multiple_of_128(elem_per_block * X_nodes_per_element * sizeof(double));
    shmem_size += round_up_to_multiple_of_128(elem_per_block * qpts_per_element * sizeof(X_grad_type));
    shmem_size += round_up_to_multiple_of_128(elem_per_block * qpts_per_element * dX_dxi_bytes);
    shmem_size += round_up_to_multiple_of_128(elem_per_block * X_scratch_size * sizeof(double));
    return shmem_size;
  };

  uint32_t threads_per_element = std::max({r_nodes_per_element, X_nodes_per_element, qpts_per_element});

  constexpr uint32_t max_default_dynamic_shmem = 48 * 1024;
  constexpr uint32_t max_threads_per_block = 1024;
  auto launch_shmem_size = [&]() {
    return (type == DomainType::SPATIAL) ? spatial_shmem_size() : base_shmem_size();
  };

  // a block gets threads_per_element * elem_per_block threads and
  // launch_shmem_size() bytes of dynamic shared memory; back elem_per_block off
  // until both fit. Both limits are reachable in practice: a hexahedron under a
  // 6-point rule has 216 quadrature points, so the requested 8 elements per block
  // would ask for 1728 threads.
  while (elem_per_block > 1 &&
         (launch_shmem_size() > max_default_dynamic_shmem ||
          threads_per_element * elem_per_block > max_threads_per_block)) {
    elem_per_block--;
  }

  FEMTO_ASSERT(threads_per_element <= max_threads_per_block,
               "element needs more threads than a CUDA block can hold");
  FEMTO_ASSERT(launch_shmem_size() <= max_default_dynamic_shmem,
               "element needs more shared memory than a CUDA block can hold");

  dim3 block_size = {threads_per_element, 1, elem_per_block};
  uint32_t grid_size = (num_elements + elem_per_block - 1) / elem_per_block;

#if USE_ATOMICS
  {
    if (type == DomainType::SPATIAL) {
      batched_integrate_residual_kernel<r_shape_rank, X_shape_rank, geom, family, op, input_t, true, false><<<grid_size, block_size, spatial_shmem_size()>>>(
        r.data,
        r.offsets,
        r_el,
        input_q,
        connectivity,
        elements,
        r_shape_fns_device,
        X_el,
        X.offsets,
        X.data,
        X_shape_fns_device,
        qpts_per_element,
        r_scratch_size,
        X_scratch_size
      );
    } else {
      batched_integrate_residual_kernel<r_shape_rank, X_shape_rank, geom, family, op, input_t, false, false><<<grid_size, block_size, base_shmem_size()>>>(
        r.data,
        r.offsets,
        r_el,
        input_q,
        connectivity,
        elements,
        r_shape_fns_device,
        X_el,
        X.offsets,
        X.data,
        X_shape_fns_device,
        qpts_per_element,
        r_scratch_size,
        X_scratch_size
      );
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }
#else
  stack::array<uint32_t, 2> element_residual_shape = {num_elements * r_nodes_per_element, components};

  if (element_residual_buffer.sz < nd::product(element_residual_shape)) {
    element_residual_buffer.resize(nd::product(element_residual_shape));
  }

  nd::view<double, 2, memory::space::gpu> element_residuals(element_residual_buffer.data(), element_residual_shape);

  {
    auto launch = [&](auto spatial, auto facet_) {
      constexpr bool is_spatial = decltype(spatial)::value;
      constexpr bool is_facet = decltype(facet_)::value;
      uint32_t shmem = is_spatial ? spatial_shmem_size() : base_shmem_size();
      batched_integrate_residual_kernel<r_shape_rank, X_shape_rank, geom, family, op, input_t, is_spatial, is_facet><<<grid_size, block_size, shmem>>>(
        element_residuals,
        r.offsets,
        r_el,
        input_q,
        connectivity,
        elements,
        r_shape_fns_device,
        X_el,
        X.offsets,
        X.data,
        X_shape_fns_device,
        qpts_per_element,
        r_scratch_size,
        X_scratch_size
      );
    };

    if (facet) {
      // only the facet measure is defined there (see the cpu overloads of
      // weighted_piola_transformation), so only scalar values are supported
      if constexpr (is_scalar_valued(family) && op == DerivedQuantity::VALUE) {
        launch(std::true_type{}, std::true_type{});
      } else {
        FEMTO_ASSERT(false, "spatial derivatives and vector-valued fields are not supported on facets embedded in a higher-dimensional space");
      }
    } else if (type == DomainType::SPATIAL) {
      launch(std::true_type{}, std::false_type{});
    } else {
      launch(std::false_type{}, std::false_type{});
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

#if SHARED_GATHER
  {
    uint32_t assemble_block_size = 128;
    uint32_t assemble_grid_size = (num_nodes + assemble_block_size - 1) / assemble_block_size;
    uint32_t shmem = assemble_block_size * components * sizeof(double);
    residual_assemble_kernel_shmem<<<assemble_grid_size, assemble_block_size, shmem>>>(
      r.data,
      element_residuals,
      table.offsets,
      table.ids
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }
#else
  {
    uint32_t assemble_block_size = 128;
    uint32_t assemble_grid_size = (num_nodes + assemble_block_size - 1) / assemble_block_size;
    residual_assemble_kernel<<<assemble_grid_size, assemble_block_size>>>(
      r.data,
      element_residuals,
      table.offsets,
      table.ids
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }
#endif

#endif
}

template < Family family, DerivedQuantity op >
void integrate_residual(Residual<family, memory::space::gpu> & r,
                        FunctionSpace space,
                        const nd::view<const double, 3, memory::space::gpu> f_q,
                        const Domain<memory::space::gpu> & domain,
                        const DomainType type) {

  zero(r.data);

  uint32_t gdim = domain.geometry_dimension;

  static nd::array< double, 1, memory::space::gpu > element_residual_buffer;

  uint32_t qoffset = 0;
  foreach_geometry([&](auto geom) {
    nd::view<const int, 1, memory::space::gpu> elements = domain.active_elements[geom];
    if (gdim == dimension(geom) && elements.shape[0] > 0) {
      nd::view<const double, 2> xi = domain.rule[geom].points;
      nd::view<const double, 1> weights = domain.rule[geom].weights;
      nd::view<const Connection, 2, memory::space::gpu> connectivity = domain.mesh[geom];

      nd::view<const double, 3, memory::space::gpu> integrand_geom = {&f_q(qoffset, 0, 0), {domain.num_qpts[geom], f_q.shape[1], f_q.shape[2]}};

      const Domain<memory::space::gpu>::AssemblyLUT & table = domain.get(geom, family, space.degree);

      batched_integrate_residual_cuda<geom, family, op>(
        r,
        domain.mesh.X,
        type,
        integrand_geom,
        connectivity,
        elements,
        xi,
        weights,
        table,
        element_residual_buffer
      );
    }
    qoffset += domain.num_qpts[geom];
  });
}

#define INSTANTIATE_INTEGRATE_RESIDUAL(family, op) \
template void integrate_residual< family, op >( \
  Residual<family, memory::space::gpu> &, \
  FunctionSpace, \
  const nd::view<const double, 3, memory::space::gpu>, \
  const Domain<memory::space::gpu> &, \
  const DomainType);

INSTANTIATE_INTEGRATE_RESIDUAL(Family::H1, DerivedQuantity::VALUE)
INSTANTIATE_INTEGRATE_RESIDUAL(Family::H1, DerivedQuantity::GRAD)
INSTANTIATE_INTEGRATE_RESIDUAL(Family::Hcurl, DerivedQuantity::VALUE)
INSTANTIATE_INTEGRATE_RESIDUAL(Family::Hcurl, DerivedQuantity::CURL)
INSTANTIATE_INTEGRATE_RESIDUAL(Family::DG, DerivedQuantity::VALUE)
INSTANTIATE_INTEGRATE_RESIDUAL(Family::DG, DerivedQuantity::GRAD)

#undef INSTANTIATE_INTEGRATE_RESIDUAL

} // namespace impl

} // namespace femto

#endif
