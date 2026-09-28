#include "femto/domain.hpp"
#include "femto/threadpool.hpp"

#include "common.hpp"

#include "misc/for_constexpr.hpp"

namespace femto {

namespace impl {

template < Geometry geom, Family family, DerivedQuantity op >
void batched_interpolate(nd::view<double, 3> u_q, 
                         const Field<family> & u,  
                         const Field<Family::H1> & X,  
                         DomainType type,
                         nd::view<const Connection, 2> connectivity,
                         const nd::view<const int> elements,
                         const nd::view<const double, 2> xi) {

  uint32_t num_elements = elements.size();
  if (num_elements == 0) return;

  FiniteElement< geom, family > u_el{u.degree};

  using output_t = std::conditional< 
    op == DerivedQuantity::VALUE, 
    typename FiniteElement< geom, family >::source_type,
    typename FiniteElement< geom, family >::flux_type
  >::type;

  nd::view<output_t, 2> output_q((output_t*)&u_q[0], {u_q.shape[0], u_q.shape[1]});

  constexpr uint32_t gdim = dimension(geom);
  uint32_t qpts_per_element = impl::qpe<geom>(xi.shape[0]);

  using A_type = decltype(piola_transformation<family, op>(mat<gdim,gdim>{}));

  uint32_t u_components = u.data.shape[1];
  uint32_t u_nodes_per_element = u_el.num_nodes();
  auto u_shape_fns = [&](){
    if constexpr(op == DerivedQuantity::VALUE) {
      return u_el.evaluate_shape_functions(xi);
    }  

    if constexpr(op == DerivedQuantity::CURL) {
      return u_el.evaluate_shape_function_curls(xi);
    }  

    if constexpr(op == DerivedQuantity::GRAD) {
      return u_el.evaluate_shape_function_gradients(xi);
    }  
  }();

  FiniteElement< geom, Family::H1 > X_el{X.degree};
  uint32_t X_components = X.data.shape[1];
  uint32_t X_nodes_per_element = X_el.num_nodes();
  auto X_shape_fn_grads = X_el.evaluate_shape_function_gradients(xi);

  //       When do we need to calculate dX/dxi?
  // 
  //     ❌ : don't need to          ✅ : need to 
  // +---------------------+------+-------+------+----+
  // |                     |  H1  | Hcurl | Hdiv | DG |
  // +---------------------+------+-------+------+----+
  // | isoparametric value |  ❌  |  ❌   |  ❌  | ❌ |
  // +---------------------+------+-------+------+----+
  // | isoparametric deriv |  ❌  |  ❌   |  ❌  | ❌ |
  // +---------------------+------+-------+------+----+
  // |       spatial value |  ❌  |  ✅   |  ✅  | ❌ |
  // +---------------------+------+-------+------+----+
  // |       spatial deriv |  ✅  |  ✅   |  ✅  | ✅ |
  // +---------------------+------+-------+------+----+
  const bool need_to_compute_dX_dxi = 
    (type == DomainType::SPATIAL && op == DerivedQuantity::GRAD) || 
    (type == DomainType::SPATIAL && op == DerivedQuantity::CURL) || 
    (type == DomainType::SPATIAL && op == DerivedQuantity::DIV) || 
    (type == DomainType::SPATIAL && is_vector_valued(family));

  // for each element with this geometry
  threadpool::block_parallel_for(num_elements, [&](uint32_t istart, uint32_t iend) {

    nd::array< A_type, 1, memory::space::cpu > A_q;
    nd::array<double, 1, memory::space::cpu> X_e;
    nd::array<uint32_t, 1, memory::space::cpu> X_ids;
    nd::array< double, 1, memory::space::cpu > X_scratch;
    nd::array< vec<gdim>, 2, memory::space::cpu > dX_dxi_q;

    if (need_to_compute_dX_dxi) {
      A_q.resize({qpts_per_element});
      X_e.resize(X_nodes_per_element);
      X_ids.resize(X_nodes_per_element);
      X_scratch.resize({X_el.batch_interpolation_scratch_space(xi)});
      dX_dxi_q.resize({X_components, qpts_per_element});
    }

    nd::array<uint32_t, 1, memory::space::cpu> u_ids({u_nodes_per_element});
    nd::array<double, 1, memory::space::cpu> u_e({u_nodes_per_element});
    nd::array< double, 1, memory::space::cpu > u_scratch({u_el.batch_interpolation_scratch_space(xi)});
    nd::array<output_t, 1, memory::space::cpu> u_xi_q({qpts_per_element});

    for (uint32_t i = istart; i < iend; i++) {

      if (need_to_compute_dX_dxi) {

        // figure out which nodal values belong to this element 
        X_el.indices(X.offsets, connectivity(elements(i)).data(), X_ids.data());

        for (int c = 0; c < X_components; c++) {
          for (int j = 0; j < X_nodes_per_element; j++) {
            X_e(j) = X.data(X_ids(j), c);
          }
          X_el.gradient(dX_dxi_q(c), X_e, X_shape_fn_grads, X_scratch.data());
        }

        for (int q = 0; q < qpts_per_element; q++) {
          A_q[q] = piola_transformation<family, op>(dX_dxi_q, q);
        }

      }

      u_el.indices(u.offsets, connectivity(elements(i)).data(), u_ids.data());

      for (int c = 0; c < u_components; c++) {

        for (int j = 0; j < u_nodes_per_element; j++) {
          u_e(j) = u.data(u_ids(j), c);
        }

        if constexpr (is_vector_valued(family)) {
          u_el.reorient(TransformationType::PhysicalToParent, &connectivity(elements(i), 0), u_e.data()); 
        }

        // carry out the appropriate kind of interpolation
        // for the requested family and differential operator
        if constexpr (op == DerivedQuantity::VALUE) {
          u_el.interpolate(u_xi_q, u_e, u_shape_fns, u_scratch.data());
        } 

        if constexpr (op == DerivedQuantity::GRAD) {
          u_el.gradient(u_xi_q, u_e, u_shape_fns, u_scratch.data());
        }

        if constexpr (op == DerivedQuantity::CURL) {
          u_el.curl(u_xi_q, u_e, u_shape_fns, u_scratch.data());
        } 

        if (need_to_compute_dX_dxi) {
          for (int q = 0; q < qpts_per_element; q++) {
            output_q(i*qpts_per_element+q, c) = fm::dot(u_xi_q[q], A_q[q]);
          }
        } else {
          for (int q = 0; q < qpts_per_element; q++) {
            output_q(i*qpts_per_element+q, c) = u_xi_q[q];
          }
        }

      }

    }

  });

}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template < Family family, DerivedQuantity op >
void evaluate(nd::array<double, 3, memory::space::cpu> & output, const Field<family> & u, const Domain<> & domain, const DomainType & type) {

  domain.check_family(family);

  int gdim = domain.geometry_dimension;
  uint32_t num_components = output.shape[1];

  auto qranges = ranges(domain.num_qpts);

  uint32_t offset = 0;
  foreach_geometry([&](auto geom){
    nd::view<const int> elements = domain.active_elements[geom];
    if (gdim == dimension(geom) && elements.size() > 0) {
      nd::view<const double, 2> xi = domain.rule[geom].points;
      nd::view<const Connection, 2> connectivity = domain.mesh[geom];
      nd::range all{0u, num_components};
      impl::batched_interpolate<geom, family, op>(output(qranges[geom]), u, domain.mesh.X, type, connectivity, elements, xi); 
    }
  });

}

} // namespace impl

#define INSTANTIATE_EVALUATE(family, op) \
template void impl::evaluate<family, op>(nd::array<double, 3, memory::space::cpu> &, const Field<family> &, const Domain<> &, const DomainType &);

INSTANTIATE_EVALUATE(Family::H1, DerivedQuantity::VALUE)
INSTANTIATE_EVALUATE(Family::H1, DerivedQuantity::GRAD)
INSTANTIATE_EVALUATE(Family::Hcurl, DerivedQuantity::VALUE)
INSTANTIATE_EVALUATE(Family::Hcurl, DerivedQuantity::CURL)
INSTANTIATE_EVALUATE(Family::DG, DerivedQuantity::VALUE)
INSTANTIATE_EVALUATE(Family::DG, DerivedQuantity::GRAD)

#undef INSTANTIATE_EVALUATE

} // namespace femto
