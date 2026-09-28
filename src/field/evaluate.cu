#include "common.hpp"

#include "femto/domain.hpp"
#include "femto/assert.hpp"

#include "misc/for_constexpr.hpp"
#include "misc/timer.hpp"

#include <algorithm>
#include <type_traits>

#ifdef FEMTO_ENABLE_CUDA

namespace femto {

namespace {

uint32_t e_per_block = 0;

} // namespace

void set_elems_per_block(uint32_t n) {
  e_per_block = n;
}

namespace impl {

template < uint32_t shape_rank,
           uint32_t X_shape_rank,
           Geometry geom,
           Family family,
           DerivedQuantity op,
           typename output_t,
           bool need_to_compute_dX_dxi >
__global__ void batched_interpolate_cuda(
  nd::view<output_t, 2, memory::space::gpu> output_q,
  FiniteElement< geom, family > u_el,
  GeometryInfo u_offsets,
  nd::view<const double, 2, memory::space::gpu> u,
  FiniteElement< geom, Family::H1 > X_el,
  GeometryInfo X_offsets,
  nd::view<const double, 2, memory::space::gpu> X,
  nd::view<const Connection, 2, memory::space::gpu> connectivity,
  nd::view<const int, 1, memory::space::gpu> elements,
  nd::view<const double, shape_rank, memory::space::gpu> u_shape_functions,
  nd::view<const double, X_shape_rank, memory::space::gpu> X_shape_functions,
  uint32_t qpts_per_elem,
  uint32_t u_scratch_size,
  uint32_t X_scratch_size
) {
  constexpr uint32_t gdim = dimension(geom);
  using X_grad_type = typename FiniteElement< geom, Family::H1 >::flux_type;
  using dX_dxi_type = mat<gdim, gdim>;

  uint32_t shmem_offset = 0;
  extern __shared__ char shmem[];

  uint32_t elem_per_block = blockDim.z;
  uint32_t u_nodes_per_elem = u_el.num_nodes();
  uint32_t X_nodes_per_elem = X_el.num_nodes();
  uint32_t local_elem_id = threadIdx.z;
  uint32_t local_tid = threadIdx.x + blockDim.x * threadIdx.y;
  uint32_t local_stride = blockDim.x * blockDim.y;

  nd::view< Connection, 2 > shr_connectivity((Connection *)(shmem + shmem_offset), {elem_per_block, connectivity.shape[1]});
  shmem_offset += round_up_to_multiple_of_128(shr_connectivity.size() * sizeof(Connection));

  nd::view< uint32_t, 2 > shr_u_node_ids((uint32_t *)(shmem + shmem_offset), {elem_per_block, u_nodes_per_elem});
  shmem_offset += round_up_to_multiple_of_128(shr_u_node_ids.size() * sizeof(uint32_t));

  nd::view< double, 2 > shr_u_e((double *)(shmem + shmem_offset), {elem_per_block, u_nodes_per_elem});
  shmem_offset += round_up_to_multiple_of_128(shr_u_e.size() * sizeof(double));

  // the shape function tables stay in const global memory: they are small and
  // every thread reads the same entries, so the read-only/L1 cache amortizes the
  // reuse about as cheaply as a block-wide copy into shared memory would, without
  // spending the shared memory. integrate_residual.cu carries the measurements
  // behind this choice, and a warning about measuring it.
  nd::view< double, shape_rank > u_shape_fn(const_cast<double *>(&u_shape_functions[0]), u_shape_functions.shape);

  nd::view< output_t, 2 > shr_u_xi_q((output_t *)(shmem + shmem_offset), {elem_per_block, qpts_per_elem});
  shmem_offset += round_up_to_multiple_of_128(shr_u_xi_q.size() * sizeof(output_t));

  nd::view< double, 2 > shr_u_scratch((double *)(shmem + shmem_offset), {elem_per_block, u_scratch_size});
  shmem_offset += round_up_to_multiple_of_128(shr_u_scratch.size() * sizeof(double));

  nd::view< uint32_t, 2 > shr_X_node_ids(nullptr, {0, 0});
  nd::view< double, 2 > shr_X_e(nullptr, {0, 0});
  nd::view< double, X_shape_rank > X_shape_fn(const_cast<double *>(&X_shape_functions[0]), X_shape_functions.shape);
  nd::view< X_grad_type, 2 > shr_X_grad_q(nullptr, {0, 0});
  nd::view< dX_dxi_type, 2 > shr_dX_dxi_q(nullptr, {0, 0});
  nd::view< double, 2 > shr_X_scratch(nullptr, {0, 0});

  if constexpr (need_to_compute_dX_dxi) {
    shr_X_node_ids = nd::view< uint32_t, 2 >((uint32_t *)(shmem + shmem_offset), {elem_per_block, X_nodes_per_elem});
    shmem_offset += round_up_to_multiple_of_128(shr_X_node_ids.size() * sizeof(uint32_t));

    shr_X_e = nd::view< double, 2 >((double *)(shmem + shmem_offset), {elem_per_block, X_nodes_per_elem});
    shmem_offset += round_up_to_multiple_of_128(shr_X_e.size() * sizeof(double));

    shr_X_grad_q = nd::view< X_grad_type, 2 >((X_grad_type *)(shmem + shmem_offset), {elem_per_block, qpts_per_elem});
    shmem_offset += round_up_to_multiple_of_128(shr_X_grad_q.size() * sizeof(X_grad_type));

    shr_dX_dxi_q = nd::view< dX_dxi_type, 2 >((dX_dxi_type *)(shmem + shmem_offset), {elem_per_block, qpts_per_elem});
    shmem_offset += round_up_to_multiple_of_128(shr_dX_dxi_q.size() * sizeof(dX_dxi_type));

    shr_X_scratch = nd::view< double, 2 >((double *)(shmem + shmem_offset), {elem_per_block, X_scratch_size});
  }

  uint32_t e = blockIdx.x * blockDim.z + threadIdx.z;
  bool active = e < elements.shape[0];
  uint32_t elem_id = active ? elements(e) : 0;

  if (active) {
    for (uint32_t i = local_tid; i < shr_connectivity.shape[1]; i += local_stride) {
      shr_connectivity(local_elem_id, i) = connectivity(elem_id, i);
    }
  }
  __syncthreads();

  if (active && local_tid == 0) {
    u_el.indices(u_offsets, &shr_connectivity(local_elem_id, 0), &shr_u_node_ids(local_elem_id, 0));
    if constexpr (need_to_compute_dX_dxi) {
      X_el.indices(X_offsets, &shr_connectivity(local_elem_id, 0), &shr_X_node_ids(local_elem_id, 0));
    }
  }
  __syncthreads();

  if constexpr (need_to_compute_dX_dxi) {
    for (uint32_t d = 0; d < gdim; d++) {
      if (active) {
        for (uint32_t i = local_tid; i < X_nodes_per_elem; i += local_stride) {
          shr_X_e(local_elem_id, i) = X(shr_X_node_ids(local_elem_id, i), d);
        }
      }
      __syncthreads();

      X_el.cuda_gradient(
        shr_X_grad_q(local_elem_id),
        shr_X_e(local_elem_id),
        X_shape_fn,
        &shr_X_scratch(local_elem_id, 0)
      );
      __syncthreads();

      if (active) {
        for (uint32_t q = local_tid; q < qpts_per_elem; q += local_stride) {
          shr_dX_dxi_q(local_elem_id, q)[d] = shr_X_grad_q(local_elem_id, q);
        }
      }
      __syncthreads();
    }
  }

  uint32_t num_components = u.shape[1];
  for (uint32_t c = 0; c < num_components; c++) {
    if (active) {
      for (uint32_t i = local_tid; i < u_nodes_per_elem; i += local_stride) {
        shr_u_e(local_elem_id, i) = u(shr_u_node_ids(local_elem_id, i), c);
      }
    }
    __syncthreads();

    if constexpr (is_vector_valued(family)) {
      if (active && local_tid == 0) {
        u_el.reorient(TransformationType::PhysicalToParent, &shr_connectivity(local_elem_id, 0), &shr_u_e(local_elem_id, 0));
      }
      __syncthreads();
    }

    {
      // carry out the appropriate kind of interpolation
      // for the requested family and differential operator
      if constexpr (op == DerivedQuantity::VALUE) {
        u_el.cuda_interpolate(
          shr_u_xi_q(local_elem_id),
          shr_u_e(local_elem_id),
          u_shape_fn,
          &shr_u_scratch(local_elem_id, 0)
        );
      }

      if constexpr (op == DerivedQuantity::GRAD) {
        u_el.cuda_gradient(
          shr_u_xi_q(local_elem_id),
          shr_u_e(local_elem_id),
          u_shape_fn,
          &shr_u_scratch(local_elem_id, 0)
        );
      }

      if constexpr (op == DerivedQuantity::CURL) {
        u_el.cuda_curl(
          shr_u_xi_q(local_elem_id),
          shr_u_e(local_elem_id),
          u_shape_fn,
          &shr_u_scratch(local_elem_id, 0)
        );
      }
    }
    __syncthreads();

    if (active) {
      for (uint32_t q = local_tid; q < qpts_per_elem; q += local_stride) {
        if constexpr (need_to_compute_dX_dxi) {
          auto A_q = piola_transformation<family, op>(shr_dX_dxi_q(local_elem_id, q));
          output_q(qpts_per_elem * e + q, c) = fm::dot(shr_u_xi_q(local_elem_id, q), A_q);
        } else {
          output_q(qpts_per_elem * e + q, c) = shr_u_xi_q(local_elem_id, q);
        }
      }
    }
    __syncthreads();
  }
}

template < Geometry geom, Family family, DerivedQuantity op >
void batched_interpolate(nd::view<double, 3, memory::space::gpu> u_q,
                         const Field<family, memory::space::gpu> & u,
                         const Field<Family::H1, memory::space::gpu> & X,
                         DomainType type,
                         nd::view<const Connection, 2, memory::space::gpu> connectivity,
                         const nd::view<const int, 1, memory::space::gpu> elements,
                         const nd::view<const double, 2> xi) {
  uint32_t num_elements = elements.size();
  if (num_elements == 0) return;

  FiniteElement< geom, family > u_el{u.degree};
  FiniteElement< geom, Family::H1 > X_el{X.degree};

  using output_t = std::conditional_t<
    op == DerivedQuantity::VALUE,
    typename FiniteElement< geom, family >::source_type,
    typename FiniteElement< geom, family >::flux_type
  >;

  constexpr uint32_t gdim = dimension(geom);
  uint32_t qpts_per_element = impl::qpe<geom>(xi.shape[0]);
  uint32_t u_nodes_per_element = u_el.num_nodes();
  uint32_t X_nodes_per_element = X_el.num_nodes();

  auto u_shape_fns = [&]() {
    if constexpr (op == DerivedQuantity::VALUE) {
      return u_el.evaluate_shape_functions(xi);
    }

    if constexpr (op == DerivedQuantity::CURL) {
      return u_el.evaluate_shape_function_curls(xi);
    }

    if constexpr (op == DerivedQuantity::GRAD) {
      return u_el.evaluate_shape_function_gradients(xi);
    }
  }();

  auto X_shape_fns = X_el.evaluate_shape_function_gradients(xi);

  constexpr uint32_t u_shape_rank = array_rank<decltype(u_shape_fns)>::value;
  constexpr uint32_t X_shape_rank = array_rank<decltype(X_shape_fns)>::value;
  using X_grad_type = typename FiniteElement< geom, Family::H1 >::flux_type;

  nd::array<double, u_shape_rank, memory::space::gpu> u_shape_fns_device = u_shape_fns;
  nd::array<double, X_shape_rank, memory::space::gpu> X_shape_fns_device = X_shape_fns;

  auto * output_ptr = reinterpret_cast<output_t *>(&u_q[0]);
  nd::view<output_t, 2, memory::space::gpu> output_q(output_ptr, {u_q.shape[0], u_q.shape[1]});

  constexpr bool spatial_transform_op =
    op == DerivedQuantity::GRAD ||
    op == DerivedQuantity::CURL ||
    op == DerivedQuantity::DIV ||
    is_vector_valued(family);

  // the kernel's dX_dxi is square (see the cpu overloads of piola_transformation)
  FEMTO_ASSERT(!(type == DomainType::SPATIAL && spatial_transform_op) || X.data.shape[1] == gdim,
               "spatial derivatives and vector-valued fields are not supported on facets embedded in a higher-dimensional space");

  uint32_t elem_per_block = elements_per_block(geom, family, u.degree);
  if (e_per_block > 0) elem_per_block = e_per_block;
  elem_per_block = std::max(1u, elem_per_block);
  uint32_t u_scratch_size = u_el.batch_interpolation_scratch_space(xi);
  uint32_t X_scratch_size = X_el.batch_interpolation_scratch_space(xi);

  uint32_t threads_per_element = std::max({u_nodes_per_element, X_nodes_per_element, qpts_per_element});

  auto base_shmem_size = [&]() {
    uint32_t shmem_size = 0;
    shmem_size += round_up_to_multiple_of_128(elem_per_block * connectivity.shape[1] * sizeof(Connection));
    shmem_size += round_up_to_multiple_of_128(elem_per_block * u_nodes_per_element * sizeof(uint32_t));
    shmem_size += round_up_to_multiple_of_128(elem_per_block * u_nodes_per_element * sizeof(double));
    shmem_size += round_up_to_multiple_of_128(elem_per_block * qpts_per_element * sizeof(output_t));
    shmem_size += round_up_to_multiple_of_128(elem_per_block * u_scratch_size * sizeof(double));
    return shmem_size;
  };

  auto spatial_shmem_size = [&]() {
    uint32_t shmem_size = base_shmem_size();
    shmem_size += round_up_to_multiple_of_128(elem_per_block * X_nodes_per_element * sizeof(uint32_t));
    shmem_size += round_up_to_multiple_of_128(elem_per_block * X_nodes_per_element * sizeof(double));
    shmem_size += round_up_to_multiple_of_128(elem_per_block * qpts_per_element * sizeof(X_grad_type));
    shmem_size += round_up_to_multiple_of_128(elem_per_block * qpts_per_element * sizeof(mat<gdim, gdim>));
    shmem_size += round_up_to_multiple_of_128(elem_per_block * X_scratch_size * sizeof(double));
    return shmem_size;
  };

  constexpr uint32_t max_default_dynamic_shmem = 48 * 1024;
  constexpr uint32_t max_threads_per_block = 1024;
  auto launch_shmem_size = [&]() {
    return (type == DomainType::SPATIAL && spatial_transform_op) ? spatial_shmem_size() : base_shmem_size();
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

  auto launch = [&](auto spatial) {
    constexpr bool is_spatial = decltype(spatial)::value;
    uint32_t shmem_size = is_spatial ? spatial_shmem_size() : base_shmem_size();
    batched_interpolate_cuda<u_shape_rank, X_shape_rank, geom, family, op, output_t, is_spatial><<<grid_size, block_size, shmem_size>>>(
      output_q,
      u_el,
      u.offsets,
      u.data,
      X_el,
      X.offsets,
      X.data,
      connectivity,
      elements,
      u_shape_fns_device,
      X_shape_fns_device,
      qpts_per_element,
      u_scratch_size,
      X_scratch_size
    );
  };

  if (type == DomainType::SPATIAL && spatial_transform_op) {
    launch(std::true_type{});
  } else {
    launch(std::false_type{});
  }

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

template < Family family, DerivedQuantity op >
void evaluate_cuda_impl(nd::array<double, 3, memory::space::gpu> & output,
                        const Field<family, memory::space::gpu> & u,
                        const Domain<memory::space::gpu> & domain,
                        const DomainType & type) {

  domain.check_family(family);

  uint32_t gdim = domain.geometry_dimension;
  uint32_t num_components = output.shape[1];
  auto qranges = ranges(domain.num_qpts);

  foreach_geometry([&](auto geom) {
    nd::view<const int, 1, memory::space::gpu> elements = domain.active_elements[geom];
    if (gdim == dimension(geom) && elements.size() > 0) {
      nd::view<const double, 2> xi = domain.rule[geom].points;
      nd::view<const Connection, 2, memory::space::gpu> connectivity = domain.mesh[geom];
      nd::view<double, 3, memory::space::gpu> geom_output = output(qranges[geom]);
      FEMTO_ASSERT(geom_output.shape[1] == num_components, "invalid output component count");
      batched_interpolate<geom, family, op>(geom_output, u, domain.mesh.X, type, connectivity, elements, xi);
    }
  });
}

} // namespace impl

#define INSTANTIATE_EVALUATE(family, op) \
template void impl::evaluate_cuda_impl<family, op>(nd::array<double, 3, memory::space::gpu> &, const Field<family, memory::space::gpu> &, const Domain<memory::space::gpu> &, const DomainType &);

INSTANTIATE_EVALUATE(Family::H1, DerivedQuantity::VALUE)
INSTANTIATE_EVALUATE(Family::H1, DerivedQuantity::GRAD)
INSTANTIATE_EVALUATE(Family::Hcurl, DerivedQuantity::VALUE)
INSTANTIATE_EVALUATE(Family::Hcurl, DerivedQuantity::CURL)
INSTANTIATE_EVALUATE(Family::DG, DerivedQuantity::VALUE)
INSTANTIATE_EVALUATE(Family::DG, DerivedQuantity::GRAD)

#undef INSTANTIATE_EVALUATE

} // namespace femto

#endif
