#include "femto/domain.hpp"
#include "femto/assert.hpp"

#include "fm/types/vec.hpp"
#include "fm/types/matrix.hpp"

namespace femto {

namespace impl {

template < Geometry geom, typename return_type, typename arg_type, typename callable >
auto batched_integrate(callable integrand,
                       const Field<Family::H1> & X,
                       const nd::view<const Connection,2> connectivity,
                       const nd::view<const int> elements,
                       const nd::view<const double, 2> xi,
                       const nd::view<const double, 1> weights) {

  FiniteElement< geom, Family::H1 > el{X.degree};

  constexpr uint32_t gdim = dimension(geom);
  constexpr uint32_t sdim = dimension(arg_type{});

  using value_type = vec<1>;
  using grad_type = vec<gdim>;
  using jac_type = mat<sdim, gdim>;

  // allocate storage for an element's nodal values
  uint32_t num_elements = elements.shape[0];
  uint32_t dofs_per_element = el.num_nodes();
  uint32_t qpts_per_element = qpe<geom>(xi.shape[0]);
  std::vector< uint32_t > ids(dofs_per_element);
  nd::array<double, 1, memory::space::cpu> X_e({dofs_per_element});
  nd::array<value_type, 2, memory::space::cpu> X_q({sdim, qpts_per_element});
  nd::array<grad_type, 2, memory::space::cpu> dX_dxi_q({sdim, qpts_per_element});

  // precalculate shape functions for the provided quadrature rule
  auto shape_fns = el.evaluate_shape_functions(xi);
  auto shape_fn_grads = el.evaluate_shape_function_gradients(xi);

  nd::array< double, 1, memory::space::cpu > scratch({el.batch_interpolation_scratch_space(xi)});

  return_type output{};

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

      el.interpolate(X_q(d), X_e, shape_fns, scratch.data());
      el.gradient(dX_dxi_q(d), X_e, shape_fn_grads, scratch.data());
    }

    for (int q = 0; q < qpts_per_element; q++) {
      arg_type X{};
      jac_type J{};
      for (int i = 0; i < sdim; i++) {
        X[i] = X_q(i, q);
        J[i] = dX_dxi_q(i, q);
      }

      double dX;
      if constexpr (sdim == gdim) {
        dX = det(J) * integration_weight<geom>(q, weights);
      } else {
        dX = sqrt(det(dot(transpose(J), J))) * integration_weight<geom>(q, weights);
      }
      output += integrand(X) * dX;
    }

  }

  return output;

}

template < typename return_type, typename arg_type, typename callable >
return_type integrate_callable(callable integrand, const Domain<> & domain, const DomainType & type) {

  FEMTO_ASSERT(type == DomainType::SPATIAL, "integrate(function, domain) not defined over isoparametric domains");

  return_type output{};

  int gdim = domain.geometry_dimension;

  foreach_geometry([&](auto geom){
    nd::view<const int> elements = domain.active_elements[geom];
    if (gdim == dimension(geom) && elements.shape[0] > 0) {
      nd::view<const double, 2> xi = domain.rule[geom].points;
      nd::view<const double, 1> weights = domain.rule[geom].weights;
      nd::view<const Connection, 2> connectivity = domain.mesh[geom];
      output += batched_integrate<geom, return_type, arg_type>(integrand, domain.mesh.X, connectivity, elements, xi, weights); 
    }
  });

  return output;

}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template < typename return_type >
return_type integrate_function_pointer(return_type (*integrand)(vec2), const Domain<> & domain, const DomainType & type) {
  return integrate_callable<return_type, vec2>(integrand, domain, type);
}

template < typename return_type >
return_type integrate_function_pointer(return_type (*integrand)(vec3), const Domain<> & domain, const DomainType & type) {
  return integrate_callable<return_type, vec3>(integrand, domain, type);
}

////////////////////////////////////////////////////////////////////////////////

template < typename return_type >
return_type integrate_stdfunction(std::function< return_type(vec2) > integrand, const Domain<> & domain, const DomainType & type) {
  return integrate_callable<return_type, vec2>(integrand, domain, type);
}

template < typename return_type >
return_type integrate_stdfunction(std::function< return_type(vec3) > integrand, const Domain<> & domain, const DomainType & type) {
  return integrate_callable<return_type, vec3>(integrand, domain, type);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template double integrate_function_pointer(double (*)(vec2), const Domain<> &, const DomainType &);
template double integrate_function_pointer(double (*)(vec3), const Domain<> &, const DomainType &);
template   vec2 integrate_function_pointer(  vec2 (*)(vec2), const Domain<> &, const DomainType &);
template   vec3 integrate_function_pointer(  vec3 (*)(vec3), const Domain<> &, const DomainType &);
template   mat2 integrate_function_pointer(  mat2 (*)(vec2), const Domain<> &, const DomainType &);
template   mat3 integrate_function_pointer(  mat3 (*)(vec3), const Domain<> &, const DomainType &);

template double integrate_stdfunction(std::function< double(vec2) >, const Domain<> &, const DomainType &);
template double integrate_stdfunction(std::function< double(vec3) >, const Domain<> &, const DomainType &);
template   vec2 integrate_stdfunction(std::function<   vec2(vec2) >, const Domain<> &, const DomainType &);
template   vec3 integrate_stdfunction(std::function<   vec3(vec3) >, const Domain<> &, const DomainType &);
template   mat2 integrate_stdfunction(std::function<   mat2(vec2) >, const Domain<> &, const DomainType &);
template   mat3 integrate_stdfunction(std::function<   mat3(vec3) >, const Domain<> &, const DomainType &);

} // namespace impl

} // namespace femto
