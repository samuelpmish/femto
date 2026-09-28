#pragma once

#include "common.hpp"
#include "test_shape_functions.hpp"

#include "misc/timer.hpp"

#include "femto/domain.hpp"
#include "femto/assert.hpp"
#include "femto/threadpool.hpp"

namespace femto {

namespace impl {

template < Geometry geom, 
           Family test_family, 
           DerivedQuantity test_op, 
           Family trial_family, 
           DerivedQuantity trial_op>
void batched_integrate_spmat(nd::view<double> values,
                             nd::view<const int> row_ptr,
                             nd::view<const int> col_ind,
                             nd::view<const double, 5> qdata,
                             FunctionSpace trial_space,
                             FunctionSpace test_space,
                             GeometryInfo trial_offsets,
                             GeometryInfo test_offsets,
                             const Field<Family::H1> & X,
                             const DomainType type,
                             nd::view<const Connection, 2> connectivity,
                             const nd::view<const int> elements,
                             const nd::view<const double, 2> xi,
                             const nd::view<const double, 1> weights) {

  constexpr uint32_t gdim = dimension(geom);
  constexpr uint32_t test_qshape = qshape(test_family, test_op, gdim);
  constexpr uint32_t trial_qshape = qshape(trial_family, trial_op, gdim);

  using test_qtype = vec<test_qshape>;
  using trial_qtype = vec<trial_qshape>;
  using mat_t = mat<test_qshape, trial_qshape>;

  using test_Atype = decltype(piola_transformation<test_family, test_op>(mat<gdim,gdim>{}));
  using trial_Atype = decltype(weighted_piola_transformation<trial_family, trial_op>(mat<gdim,gdim>{}));

  FiniteElement< geom, Family::H1 > X_el{X.degree};
  FiniteElement< geom, test_family > test_el{test_space.degree};
  FiniteElement< geom, trial_family > trial_el{trial_space.degree};

  uint32_t num_elements = elements.size();
  uint32_t test_components = test_space.components;
  uint32_t trial_components = trial_space.components;
  uint32_t test_nodes_per_element = test_el.num_nodes();
  uint32_t trial_nodes_per_element = trial_el.num_nodes();
  uint32_t qpts_per_element = impl::qpe<geom>(xi.shape[0]);

  constexpr int nmutex = 1024;
  std::vector< std::mutex > mutexes(nmutex);

  // precalculate test functions for the provided quadrature rule
  auto psi_wt = [&](){
    if constexpr(trial_op == DerivedQuantity::VALUE) {
      return trial_el.evaluate_weighted_shape_functions(xi, weights);
    }  

    if constexpr(trial_op == DerivedQuantity::CURL && trial_family == Family::Hcurl) {
      return trial_el.evaluate_weighted_shape_function_curls(xi, weights);
    }  

    if constexpr(trial_op == DerivedQuantity::GRAD && is_scalar_valued(trial_family)) {
      return trial_el.evaluate_weighted_shape_function_gradients(xi, weights);
    }  
  }();

  // precalculate the test functions too: for tensor-product elements this
  // tabulates only the 1D factors and each shape function is reconstructed on
  // demand, which also keeps the per-element loop out of the (linear) shape
  // function dispatch
  auto test_shape_fns = evaluate_test_shape_functions<geom, test_family, test_op>(test_el, xi);
  constexpr uint32_t test_shape_rank = array_rank<decltype(test_shape_fns)>::value;
  nd::view<double, test_shape_rank> phi_table(test_shape_fns.data(), test_shape_fns.shape);

  // tabulate the test functions, or evaluate them on the fly from the
  // quadrature points (see TestShapeMode)
  const TestShapeMode test_shape_mode = get_test_shape_mode();
  const bool use_tables = (test_shape_mode == TestShapeMode::MinimalShared ||
                           test_shape_mode == TestShapeMode::MinimalGlobal);

  // number of quadrature points per direction, for the tensor-product paths
  uint32_t q1D = 1;
  if constexpr (geom == Geometry::Quadrilateral) {
    while (q1D * q1D < qpts_per_element) { q1D++; }
  } else if constexpr (geom == Geometry::Hexahedron) {
    while (q1D * q1D * q1D < qpts_per_element) { q1D++; }
  }

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
  // |       spatial value |  ✅  |  ✅   |  ✅  | ✅ |
  // +---------------------+------+-------+------+----+
  // |       spatial deriv |  ✅  |  ✅   |  ✅  | ✅ |
  // +---------------------+------+-------+------+----+
  const bool need_to_compute_dX_dxi = (type == DomainType::SPATIAL);

  // for each element of this geometry in the domain
  threadpool::block_parallel_for(num_elements, [&](uint32_t e_start, uint32_t e_end) {

    //THREAD_SCOPE_TRACE("integrate_spmat", "element block");

    nd::array<test_Atype, 1, memory::space::cpu> testA_q;
    nd::array<trial_Atype, 1, memory::space::cpu> trialA_q;
    nd::array<vec<gdim>, 2, memory::space::cpu> dX_dxi_q;
    nd::array<uint32_t, 1, memory::space::cpu> X_ids;
    nd::array<double, 1, memory::space::cpu> X_e;
    nd::array<double, 1, memory::space::cpu> X_scratch;

    nd::array<double, 1, memory::space::cpu> trial_scratch({trial_el.batch_interpolation_scratch_space(xi)});
    nd::array<trial_qtype, 1, memory::space::cpu> hat_f({qpts_per_element});
    nd::array<uint32_t, 1, memory::space::cpu> test_ids({test_nodes_per_element});
    nd::array<uint32_t, 1, memory::space::cpu> trial_ids({trial_nodes_per_element});
    nd::array<int8_t, 1, memory::space::cpu> transformation({test_nodes_per_element});
    nd::array<double, 1, memory::space::cpu> r_e({trial_el.num_nodes()});

    if (need_to_compute_dX_dxi) {
      testA_q.resize({qpts_per_element});
      trialA_q.resize({qpts_per_element});
      dX_dxi_q.resize({X_components, qpts_per_element});
      X_ids.resize(X_nodes_per_element);
      X_e.resize(X_nodes_per_element);
      X_scratch.resize({X_el.batch_interpolation_scratch_space(xi)});
    }

    for (uint32_t e = e_start; e < e_end; e++) {

      if (need_to_compute_dX_dxi) {
        // figure out which nodal values belong to this element 
        X_el.indices(X.offsets, connectivity(elements(e)).data(), X_ids.data());

        for (int c = 0; c < X_components; c++) {
          for (int j = 0; j < X_nodes_per_element; j++) {
            X_e(j) = X.data(X_ids(j), c);
          }
          X_el.gradient(dX_dxi_q(c), X_e, X_shape_fn_grads, X_scratch.data());
        }

        for (int q = 0; q < qpts_per_element; q++) {
          testA_q[q] = piola_transformation<test_family, test_op>(dX_dxi_q, q);
          trialA_q[q] = weighted_piola_transformation<trial_family, trial_op>(dX_dxi_q, q);
        }
      }

      test_el.indices(test_offsets, connectivity(elements(e)).data(), test_ids.data());
      trial_el.indices(trial_offsets, connectivity(elements(e)).data(), trial_ids.data());

      if constexpr (is_vector_valued(test_family)) {
        test_el.reorient(TransformationType::TransposePhysicalToParent, &connectivity(elements(e), 0), transformation.data()); 
      }

      uint32_t qoffset = e * qpts_per_element;

      for (uint32_t i = 0; i < test_components; i++) {
        for (uint32_t j = 0; j < trial_components; j++) {
          for (uint32_t I = 0; I < test_nodes_per_element; I++) {
            int row_id = test_ids[I] * test_components + i;

            for (uint32_t q = 0; q < qpts_per_element; q++) {
              uint32_t qid = qoffset + q;

              mat_t C{};
              for (uint32_t k = 0; k < test_qshape; k++) {
                for (uint32_t m = 0; m < trial_qshape; m++) {
                  C(k, m) = qdata(qid, i, k, j, m);
                }
              }

              test_qtype phi_I;
              if (use_tables) {
                phi_I = load_reoriented_test_shape<geom, test_family, test_op, test_qshape>(
                  phi_table, test_space.degree, q1D, q, I, transformation[I]);
              } else {
                phi_I = as_qtype<test_qshape>(shape_function<test_op>(
                  test_el, quadrature_point<geom>(q, xi), I, transformation[I]));
              }

              if (need_to_compute_dX_dxi) {
                phi_I = fm::dot(phi_I, testA_q[q]);
              }

              hat_f[q] = fm::dot(phi_I, C);

              if (need_to_compute_dX_dxi) {
                hat_f[q] = fm::dot(trialA_q[q], hat_f[q]);
              }

            }

            if constexpr (trial_op == DerivedQuantity::VALUE) {
              trial_el.integrate_source(r_e, hat_f, psi_wt, trial_scratch.data());
            } 

            if constexpr (trial_op == DerivedQuantity::GRAD || trial_op == DerivedQuantity::CURL) {
              trial_el.integrate_flux(r_e, hat_f, psi_wt, trial_scratch.data());
            }

            if constexpr (is_vector_valued(trial_family)) {
              trial_el.reorient(TransformationType::TransposePhysicalToParent, &connectivity(elements(e), 0), r_e.data()); 
            }

            int row_start = row_ptr[row_id];
            int row_end = row_ptr[row_id+1];
            int which = row_id % nmutex;
            mutexes[which].lock();
            for (uint32_t J = 0; J < trial_nodes_per_element; J++) {
              int col_id = trial_ids[J] * trial_components + j;

              // find the position of the nonzero entry of this row with the right column
              int position = std::lower_bound(&col_ind[row_start], &col_ind[row_end], col_id) - &col_ind[0];

              values[position] += r_e(J);
            }
            mutexes[which].unlock();
          }
        }
      }

  }

  });

}

template < Family test_family, DerivedQuantity test_op, Family trial_family, DerivedQuantity trial_op >
std::function< void(femto::sparse_matrix<>&) > integrate_sparse_matrix(FunctionSpace phi, const nd::view<const double, 5> qdata, FunctionSpace psi, const Domain<> &domain, const DomainType type) {

  return [&, type, phi, psi, qdata](femto::sparse_matrix<> & A) {

    uint32_t test_components = phi.components;
    uint32_t trial_components = psi.components;
    // dofs are numbered over the whole mesh, quadrature data over the domain
    uint32_t gdim = domain.mesh.geometry_dimension;
    uint32_t domain_gdim = domain.geometry_dimension;

    stack::array<uint32_t, 5> shape5D = {
      qdata.shape[0],
      phi.components, qshape(test_family, test_op, domain_gdim),
      psi.components, qshape(trial_family, trial_op, domain_gdim)
    };

    FEMTO_ASSERT(compatible_shapes(qdata.shape, shape5D), "incompatible array shapes");

    nd::view<const double,5> q5D{qdata.data(), shape5D};

    GeometryInfo counts = domain.mesh.geometry_counts();
    GeometryInfo test_offsets = scan(interior_nodes_per_geom(phi, gdim) * counts);
    GeometryInfo trial_offsets = scan(interior_nodes_per_geom(psi, gdim) * counts);
    auto nrows = total(interior_dofs_per_geom(phi, gdim) * counts);
    auto ncols = total(interior_dofs_per_geom(psi, gdim) * counts);

    if (A.nnz == 0) {
      A = blank_sparse_matrix(phi, psi, domain);
    } else {
      zero(A.values);
    }

    uint32_t qoffset = 0;
    foreach_geometry([&](auto geom){

    nd::view<const int> elements = domain.active_elements[geom];
    if (domain_gdim == dimension(geom) && elements.size() > 0) {
      nd::view<const Connection, 2> connectivity = domain.mesh[geom];
      nd::view<const double, 2> xi = domain.rule[geom].points;
      nd::view<const double, 1> weights = domain.rule[geom].weights;

      if constexpr (geom != Geometry::Vertex) {
        stack::array< uint32_t, 5 > qdata_shape{domain.num_qpts[geom], shape5D[1], shape5D[2], shape5D[3], shape5D[4]};
        nd::view<const double, 5> geom_qdata{&q5D(qoffset, 0, 0, 0, 0), qdata_shape};

        batched_integrate_spmat<geom, test_family, test_op, trial_family, trial_op >(
          A.values, A.row_ptr, A.col_ind, 
          geom_qdata, psi, phi, 
          trial_offsets, test_offsets, domain.mesh.X, type,
          connectivity, elements, 
          xi, weights
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
