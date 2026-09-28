#include "femto/domain.hpp"

#include "fm/types/matrix.hpp"
#include "fm/types/vec.hpp"

namespace femto {

namespace impl {

template < Geometry geom, uint32_t sdim, typename T >
auto batched_integrate(nd::view<T> integrand,
                       const Field<Family::H1> & X,  
                       const nd::view<const Connection,2> connectivity,
                       const nd::view<const int> elements,
                       const nd::view<const double, 2> xi,
                       const nd::view<const double, 1> weights) {

  FiniteElement< geom, Family::H1 > el{X.degree};

  constexpr uint32_t gdim = dimension(geom);

  using value_type = vec<1>;
  using grad_type = vec<gdim>;
  using jac_type = mat<sdim, gdim>;

  // allocate storage for an element's nodal values
  uint32_t num_elements = elements.shape[0];
  uint32_t dofs_per_element = el.num_nodes();
  uint32_t qpts_per_element = qpe<geom>(xi.shape[0]);
  std::vector< uint32_t > ids(dofs_per_element);
  nd::array<double, 1, memory::space::cpu> X_e({dofs_per_element});
  nd::array<grad_type, 2, memory::space::cpu> dX_dxi_q({sdim, qpts_per_element});

  // precalculate shape functions for the provided quadrature rule
  auto shape_fn_grads = el.evaluate_shape_function_gradients(xi);

  nd::array< double, 1, memory::space::cpu > scratch({el.batch_interpolation_scratch_space(xi)});

  T output{};

  uint32_t q_id = 0;

  // for each element of this geometry in the domain
  for (uint32_t i = 0; i < num_elements; i++) {

    // figure out which nodal values belong to this element 
    el.indices(X.offsets, &connectivity(elements(i), 0), &ids[0]);

    for (int d = 0; d < sdim; d++) {
      // load the nodal values for this element
      for (int j = 0; j < dofs_per_element; j++) {
        X_e(j) = X.data(ids[j], d);
      }

      el.gradient(dX_dxi_q(d), X_e, shape_fn_grads, scratch.data());
    }

    for (int q = 0; q < qpts_per_element; q++) {
      jac_type J{};
      for (int i = 0; i < sdim; i++) {
        J[i] = dX_dxi_q(i, q);
      }

      double dX;
      if constexpr (sdim == gdim) {
        dX = det(J) * integration_weight<geom>(q, weights);
      } else {
        dX = sqrt(det(dot(transpose(J), J))) * integration_weight<geom>(q, weights);
      }
      output += integrand(q_id++) * dX;
    }

  }

  return output;

}

template < typename T, memory::space mem_space >
T integrate_ndarray(const nd::array< T, 1, mem_space > & integrand, const Domain<> & domain, const DomainType & type) {

  T output{};

  int sdim = domain.mesh.spatial_dimension;
  int gdim = domain.geometry_dimension;

  uint32_t offset = 0;
  foreach_geometry([&](auto geom){
    nd::view<const int> elements = domain.active_elements[geom];
    if (gdim == dimension(geom) && elements.shape[0] > 0) {
      nd::view<const double, 2> xi = domain.rule[geom].points;
      nd::view<const double, 1> weights = domain.rule[geom].weights;
      nd::view<const Connection, 2> connectivity = domain.mesh[geom];
      nd::view<T> geom_integrand = {&integrand(offset), {domain.num_qpts[geom]}};

      if (sdim == 2) {
        output += batched_integrate<geom, 2>(geom_integrand, domain.mesh.X, connectivity, elements, xi, weights); 
      } 
      if (sdim == 3) {
        output += batched_integrate<geom, 3>(geom_integrand, domain.mesh.X, connectivity, elements, xi, weights); 
      }

      offset += domain.num_qpts[geom];
    }
  });

  return output;

}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#define INSTANTIATE_INTEGRATE_NDARRAY(mem_space) \
template double integrate_ndarray<double, mem_space>(const nd::array< double, 1, mem_space > &, const Domain<> &, const DomainType &); \
template vec2 integrate_ndarray<vec2, mem_space>(const nd::array< vec2, 1, mem_space > &, const Domain<> &, const DomainType &); \
template vec3 integrate_ndarray<vec3, mem_space>(const nd::array< vec3, 1, mem_space > &, const Domain<> &, const DomainType &); \
template mat2 integrate_ndarray<mat2, mem_space>(const nd::array< mat2, 1, mem_space > &, const Domain<> &, const DomainType &); \
template mat3 integrate_ndarray<mat3, mem_space>(const nd::array< mat3, 1, mem_space > &, const Domain<> &, const DomainType &);

INSTANTIATE_INTEGRATE_NDARRAY(memory::space::cpu)
#ifdef NDARRAY_ENABLE_CUDA
INSTANTIATE_INTEGRATE_NDARRAY(memory::UNIFIED)
#endif

#undef INSTANTIATE_INTEGRATE_NDARRAY

} // namespace impl

} // namespace femto
