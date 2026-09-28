#include "femto/domain.hpp"
#include "femto/assert.hpp"

#include "jacobian_rows_columns.hpp"

#include "misc/timer.hpp"

template < Geometry geom >
void integrate_jacobians_grad_grad(nd::view<double,5> K_e, 
                         nd::view<const double,5> dflux_dgradu,
                         FunctionSpace trial_space,
                         FunctionSpace test_space,
                         const int num_elements,
                         const nd::view<const double, 2> xi,
                         const nd::view<const double, 1> weights) {

  // TODO: support other families
  FiniteElement< geom, Family::H1 > trial_el{trial_space.degree};
  FiniteElement< geom, Family::H1 > test_el{test_space.degree};

  // allocate storage for an element's nodal forces
  constexpr uint32_t gdim = dimension(geom);
  uint32_t qpts_per_element = impl::qpe<geom>(xi.shape[0]);

  uint32_t test_components = test_space.components;
  uint32_t trial_components = trial_space.components;
  uint32_t nodes_per_test_element = test_el.num_nodes();
  uint32_t nodes_per_trial_element = trial_el.num_nodes();

  // precalculate shape functions for the provided quadrature rule
  auto test_shape_fn_grads = test_el.evaluate_shape_function_gradients(xi);
  auto trial_shape_fn_grads = trial_el.evaluate_shape_function_gradients(xi);

  // for each element of this geometry in the domain
  for (uint32_t e = 0; e < num_elements; e++) {

    uint32_t qoffset = e * qpts_per_element;
    for (uint32_t J = 0; J < nodes_per_trial_element; J++) {
      for (uint32_t j = 0; j < trial_space.components; j++) {

        for (uint32_t I = 0; I < nodes_per_test_element; I++) {
          for (uint32_t i = 0; i < test_space.components; i++) {
            double sum = 0.0;
            for (uint32_t q = 0; q < qpts_per_element; q++) {
              vec<gdim> dphi_I;
              vec<gdim> dpsi_J;
              for (int d = 0; d < gdim; d++) {
                dphi_I[d] = test_shape_fn_grads(q, I, d);
                dpsi_J[d] = trial_shape_fn_grads(q, J, d);
              }

              for (int k = 0; k < gdim; k++) {
                for (int m = 0; m < gdim; m++) {
                  sum += dphi_I(k) * dflux_dgradu(q + qoffset, i, k, j, m)  * dpsi_J(m) * weights(q); 
                }
              }
            }
            K_e(e, J, j, I, i) = sum;
          }
        }

      }
    }

  }

}

#if 0
template <>
void integrate_jacobians_grad_grad<Geometry::Quadrilateral>(nd::view<double,5> K_e, 
                         nd::view<const double,5> dflux_dgradu,
                         FunctionSpace trial_space,
                         FunctionSpace test_space,
                         const int num_elements,
                         const nd::view<const double, 2> xi_1D,
                         const nd::view<const double, 1> weights_1D) {
  
  constexpr Geometry geom = Geometry::Quadrilateral;

  // TODO: support other families
  FiniteElement< geom, Family::H1 > trial_el{trial_space.degree};
  FiniteElement< geom, Family::H1 > test_el{test_space.degree};

  // allocate storage for an element's nodal forces
  constexpr uint32_t gdim = dimension(geom);
  uint32_t nqpts_1D = xi_1D.shape[0];
  uint32_t qpts_per_element = impl::qpe<geom>(xi_1D.shape[0]);

  uint32_t test_components = test_space.components;
  uint32_t trial_components = trial_space.components;
  uint32_t nodes_per_test_element = test_el.num_nodes();
  uint32_t nodes_per_trial_element = trial_el.num_nodes();

  // precalculate shape functions for the provided quadrature rule
  auto test_shape_fn_grads = test_el.evaluate_weighted_shape_function_gradients(xi_1D, weights_1D);

  const stack::array< uint32_t, 2 > r_shape = {K_e.shape[3], K_e.shape[4]};
  const stack::array< uint32_t, 2 > r_strides = {K_e.stride[3], K_e.stride[4]};

  // for each element of this geometry in the domain
  for (uint32_t e = 0; e < num_elements; e++) {

    uint32_t qoffset = e * qpts_per_element;

    uint32_t J = 0;
    for (uint32_t Jy = 0; Jy < (trial_space.degree + 1); Jy++) {
    for (uint32_t Jx = 0; Jx < (trial_space.degree + 1); Jx++) {
      for (uint32_t j = 0; j < trial_space.components; j++) {

        nd::cpu_array<double, 3> dflux_Jj({qpts_per_element, trial_components, gdim});

        uint32_t q = 0;
        for (uint32_t qy = 0; qy < nqpts_1D; qy++) {
        for (uint32_t qx = 0; qx < nqpts_1D; qx++) {

          vec2 xi_2D = {xi_1D(qx,0), xi_1D(qy,0)};

          vec2 dphiJ_dxi = trial_el.shape_function_gradient(xi_2D, J);

          for (uint32_t i = 0; i < test_space.components; i++) {
            for (uint32_t k = 0; k < gdim; k++) {
              dflux_Jj(q, i, k) = 0;
              for (uint32_t m = 0; m < gdim; m++) {
                dflux_Jj(q, i, k) += dflux_dgradu(q + qoffset, i, k, j, m) * dphiJ_dxi(m);
              }
            }
          }

          q++;
        }
        }

        nd::view<double, 2> r_e{&K_e(e, J, j, 0, 0), r_shape, r_strides};

        test_el.integrate_flux(r_e, dflux_Jj, test_shape_fn_grads);
      }
      J++;
    }
    }

  }

}
#endif

template <>
void integrate_jacobians_grad_grad<Geometry::Hexahedron>(nd::view<double,5> K_e, 
                         nd::view<const double,5> dflux_dgradu,
                         FunctionSpace trial_space,
                         FunctionSpace test_space,
                         const int num_elements,
                         const nd::view<const double, 2> xi_1D,
                         const nd::view<const double, 1> weights_1D) {
  
  constexpr Geometry geom = Geometry::Hexahedron;

  // TODO: support other families
  FiniteElement< geom, Family::H1 > trial_el{trial_space.degree};
  FiniteElement< geom, Family::H1 > test_el{test_space.degree};

  // allocate storage for an element's nodal forces
  constexpr uint32_t gdim = dimension(geom);
  uint32_t nqpts_1D = xi_1D.shape[0];
  uint32_t q1D = xi_1D.shape[0];
  uint32_t qpts_per_element = impl::qpe<geom>(q1D);

  uint32_t test_components = test_space.components;
  uint32_t trial_components = trial_space.components;
  uint32_t nodes_per_test_element = test_el.num_nodes();
  uint32_t nodes_per_trial_element = trial_el.num_nodes();

  // precalculate shape functions for the provided quadrature rule
  auto test_shape_fn_grads = test_el.evaluate_weighted_shape_function_gradients(xi_1D, weights_1D);

  nd::cpu_array<double, 2> dflux({qpts_per_element, gdim});

  nd::range All{0u, nodes_per_test_element};

  // for each element of this geometry in the domain
  for (uint32_t e = 0; e < num_elements; e++) {

    uint32_t qoffset = e * qpts_per_element;

    for (uint32_t i = 0; i < test_space.components; i++) {
      for (uint32_t j = 0; j < trial_space.components; j++) {
        for (uint32_t J = 0; J < nodes_per_trial_element; J++) {

          for (uint32_t q = 0; q < qpts_per_element; q++) {
            uint32_t qx = i % q1D;
            uint32_t qy = (i % (q1D * q1D)) / q1D;
            uint32_t qz = i / (q1D * q1D);
            vec3 xi_q = {xi_1D(qx,0), xi_1D(qy,0), xi_1D(qz, 0)};

            vec3 dphiJ_dxi = trial_el.shape_function_gradient(xi_q, J);

            for (uint32_t k = 0; k < gdim; k++) {
              double sum = 0.0;
              for (uint32_t m = 0; m < gdim; m++) {
                sum += dflux_dgradu(q + qoffset, i, k, j, m) * dphiJ_dxi(m);
              }
              dflux(q, k) = sum;
            }
          }

          test_el.integrate_flux(K_e(e, J, j, All, i), dflux, test_shape_fn_grads);

        }
      }
    }

  }

}

femto::sparse_matrix<> integrate_impl(BasisFunctionOp test, nd::view<const double,5> qdata, BasisFunctionOp trial, const Domain<> & domain) {

  femto::timer stopwatch;
  double integrate_time = 0.0;

  auto psi = trial.function.space;
  auto f = qdata;
  auto phi = test.function.space;

  auto dofs_per_psi = dofs_per_geom(psi);
  auto dofs_per_phi = dofs_per_geom(phi);

  auto matrix_entries_per_elem = dofs_per_psi * dofs_per_phi;

  auto matrix_entries_per_block = matrix_entries_per_elem * domain.geometry_counts();

  auto total_entries = total(matrix_entries_per_block);
  nd::cpu_array<double, 1> element_matrix_values({total_entries});
  nd::cpu_array<int, 1> element_matrix_rows({total_entries});
  nd::cpu_array<int, 1> element_matrix_cols({total_entries});

  uint32_t gdim = domain.mesh.geometry_dimension;

  bool exclusive = true;
  uint32_t test_components = phi.components;
  uint32_t trial_components = psi.components;
  GeometryInfo counts = domain.mesh.geometry_counts();
  GeometryInfo test_offsets = scan(interior_nodes_per_geom(phi) * counts);
  GeometryInfo trial_offsets = scan(interior_nodes_per_geom(psi) * counts);
  auto nrows = total(interior_dofs_per_geom(phi) * counts);
  auto ncols = total(interior_dofs_per_geom(psi) * counts);

  uint32_t eoffset = 0;
  uint32_t qoffset = 0;
  foreach_geometry([&](auto geom){
    nd::view<const int> elements = domain.active_elements[geom];
    if (gdim == dimension(geom) && elements.size() > 0) {
      nd::view<const Connection, 2> connectivity = domain.mesh[geom];
      nd::view<const double, 2> xi = domain.rule[geom].points;
      nd::view<const double, 1> weights = domain.rule[geom].weights;

      // TODO: enable edge elements
      if constexpr (geom != Geometry::Vertex && geom != Geometry::Edge) {

        FiniteElement< geom, Family::H1 > test_el{phi.degree};
        FiniteElement< geom, Family::H1 > trial_el{psi.degree};

        uint32_t nodes_per_test_element = test_el.num_nodes();
        uint32_t nodes_per_trial_element = trial_el.num_nodes();

        stack::array< uint32_t, 5 > qdata_shape{
          domain.num_qpts[geom],
          test_components, gdim,
          trial_components, gdim
        };
        nd::view<const double, 5> qdata = {&f(qoffset, 0, 0, 0, 0), qdata_shape};

        stack::array< uint32_t, 5 > element_jacobians_shape{
          elements.size(),
          nodes_per_trial_element, trial_components,
          nodes_per_test_element, test_components
        };
        nd::view<double, 5> element_jacobians = {&element_matrix_values(eoffset), element_jacobians_shape};

        stopwatch.start();
        integrate_jacobians_grad_grad<geom>(element_jacobians, qdata, psi, phi, elements.size(), xi, weights);
        stopwatch.stop();
        integrate_time += stopwatch.elapsed();

        nd::view<int, 5> rows = {&element_matrix_rows(eoffset), element_jacobians_shape};
        nd::view<int, 5> cols = {&element_matrix_cols(eoffset), element_jacobians_shape};
        jacobian_rows_and_columns<geom>(rows, cols, psi, phi, trial_offsets, test_offsets, elements, connectivity);

        qoffset += domain.num_qpts[geom];
      }
    }
  });

  if (print_timings) {
    std::cout << "element jacobian calculation time: " << integrate_time * 1000.0 << "ms" << std::endl;
  }

  std::vector< femto::triplet > triplets(total_entries);
  for (int i = 0; i < total_entries; i++) {
    triplets[i] = {
      element_matrix_rows(i),
      element_matrix_cols(i),
      element_matrix_values(i)
    };
  }

  return femto::sparse_matrix<>::from_triplets(triplets, nrows, ncols);
}

femto::sparse_matrix<> integrate(const WeightedIntegrand<BasisFunctionOp, nd::cpu_array<double, 3>, BasisFunctionOp> & integrand, const Domain<> & domain) {
  auto test = integrand.test.function;
  const nd::cpu_array<double, 3> & qdata = integrand.qdata;
  auto trial = integrand.trial.function;

  uint32_t gdim = domain.mesh.geometry_dimension;

  FEMTO_ASSERT(1 == test.space.components, "invalid test space components");
  FEMTO_ASSERT(1 == trial.space.components, "invalid trial space components");
  FEMTO_ASSERT(qdata.shape[0] == total(domain.num_qpts), "invalid array shape[0]");
  FEMTO_ASSERT(qdata.shape[1] == gdim, "invalid array shape[1]");
  FEMTO_ASSERT(qdata.shape[2] == gdim, "invalid array shape[2]");

  nd::view<const double, 5> qdata_view{qdata.data(), {qdata.shape[0],1,gdim,1,gdim}};

  return integrate_impl(integrand.test, qdata_view, integrand.trial, domain);
}

femto::sparse_matrix<> integrate(const WeightedIntegrand<BasisFunctionOp, nd::cpu_array<double, 4>, BasisFunctionOp> & integrand, const Domain<> & domain) {
  auto test = integrand.test.function;
  const nd::cpu_array<double, 4> & qdata = integrand.qdata;
  auto trial = integrand.trial.function;

  uint32_t gdim = domain.mesh.geometry_dimension;

  // TODO
  FEMTO_ASSERT(false, "unimplemented");

  FEMTO_ASSERT(1 == test.space.components, "invalid test space components");
  FEMTO_ASSERT(1 == trial.space.components, "invalid trial space components");
  FEMTO_ASSERT(qdata.shape[0] == total(domain.num_qpts), "invalid array shape[0]");
  FEMTO_ASSERT(qdata.shape[1] == gdim, "incorrect array shape[1]");
  FEMTO_ASSERT(qdata.shape[2] == gdim, "incorrect array shape[2]");

  nd::view<const double, 5> qdata_view{qdata.data(), {qdata.shape[0],1,gdim,1,gdim}};

  return integrate_impl(integrand.test, qdata_view, integrand.trial, domain);
}

femto::sparse_matrix<> integrate(const WeightedIntegrand<BasisFunctionOp, nd::cpu_array<double, 5>, BasisFunctionOp> & integrand, const Domain<> & domain) {
  auto test = integrand.test;
  const nd::cpu_array<double, 5> & qdata = integrand.qdata;
  auto trial = integrand.trial;

  uint32_t gdim = domain.mesh.geometry_dimension;

  FEMTO_ASSERT(qdata.shape[2] == gdim, "invalid array shape[2]");
  FEMTO_ASSERT(qdata.shape[4] == gdim, "invalid array shape[4]");

  return integrate_impl(test, qdata, trial, domain);
}
