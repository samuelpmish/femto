#include "femto/assert.hpp"
#include "femto/domain.hpp"

#include "misc/for_constexpr.hpp"

namespace femto {

namespace impl {

/**
 * @brief interpolate quadrature point values 
 * from precomputed shape function values, psi
 * 
 * psi can be 2D or 3D, and outputs are computed as follows: 
 * 
 * values_e := values(elements(e));
 * 2D: output_q(q, c) := values_e(i, c) * psi(q, i)
 * 3D: output_q(q, c, d) := values_e(i, c) * psi(q, i, d)
 */
template < Geometry geom, Family family, uint32_t r >
void element_interpolate(nd::view<double, r> output_q, 
                         const Field<family> & values,
                         nd::view<const Connection, 2> connectivity,
                         const nd::view<const int> elements,
                         const nd::view<const double, r> psi) {

  static_assert(r == 2 || r == 3);

  FiniteElement< geom, family > el{values.degree};

  constexpr int gdim = dimension(geom);

  uint32_t num_elements = elements.size();
  if (num_elements == 0) return;

  uint32_t components = values.data.shape[1];
  uint32_t qpts_per_element = psi.shape[0];
  uint32_t nodes_per_element = psi.shape[1];

  // allocate storage for an element's nodal values
  nd::array<uint32_t, 1, memory::space::cpu> ids({nodes_per_element});
  nd::array<double, 1, memory::space::cpu> values_e({nodes_per_element});

  // for each element of this geometry in the domain
  for (uint32_t e = 0; e < num_elements; e++) {

    // figure out which nodal values belong to this element 
    el.indices(values.offsets, &connectivity(elements(e), 0), ids.data());

    for (int c = 0; c < components; c++) {

      // load the nodal values for this element
      for (int j = 0; j < nodes_per_element; j++) {
        values_e(j) = values.data(ids(j), c);
      }
      
      if constexpr (is_vector_valued(family)) {
        el.reorient_to_parent(&connectivity(elements(e), 0), values_e.data()); 
      }

      // interpolate the quadrature point values for this element
      if constexpr (r == 2) {
        for (int q = 0; q < qpts_per_element; q++) {
          double sum = {};
          for (int i = 0; i < nodes_per_element; i++) {
            sum += values_e(i) * psi(q, i);
          }
          output_q(e * qpts_per_element + q, c) = sum;
        }
      } 

      if constexpr (r == 3) {
        for (int q = 0; q < qpts_per_element; q++) {
          vec<gdim> sum{};
          for (int i = 0; i < nodes_per_element; i++) {
            for (int d = 0; d < gdim; d++) {
              sum[d] += values_e(i) * psi(q, i, d);
            }
          }

          for (int d = 0; d < gdim; d++) {
            output_q(e * qpts_per_element + q, c, d) = sum[d];
          }
        }
      }

    }
  }

}

template < Geometry geom, Family family >
auto calculate_shape_fns(FiniteElement< geom, family > element, 
                         const nd::view<const double, 2> xi) {

  constexpr uint32_t dim = dimension(geom);
  using vecd = vec<dim>;

  uint32_t num_qpts = xi.shape[0];
  uint32_t num_nodes = element.num_nodes();

  nd::array< double, 2, memory::space::cpu > psi({num_qpts, num_nodes});

  for (uint32_t q = 0; q < num_qpts; q++) {
    vecd xi_q{};
    for (int d = 0; d < dim; d++) {
      xi_q[d] = xi(q, d);
    }

    for (uint32_t i = 0; i < num_nodes; i++) {
      psi(q, i) = element.shape_function(xi_q, i);
    }
  }

  return psi;

}

template < Geometry geom, Family family >
auto calculate_vector_shape_fns(FiniteElement< geom, family > element, 
                               const nd::view<const double, 2> xi) {

  constexpr uint32_t dim = dimension(geom);
  using vecd = vec<dim>;

  uint32_t num_qpts = xi.shape[0];
  uint32_t num_nodes = element.num_nodes();

  nd::array< double, 3, memory::space::cpu > psi({num_qpts, num_nodes, dim});

  for (uint32_t q = 0; q < num_qpts; q++) {
    vecd xi_q{};
    for (int d = 0; d < dim; d++) {
      xi_q[d] = xi(q, d);
    }

    for (uint32_t i = 0; i < num_nodes; i++) {
      vecd psi_q = element.shape_function(xi_q, i);
      for (uint32_t d = 0; d < dim; d++) {
        psi(q, i, d) = psi_q[d];
      }
    }
  }

  return psi;

}

template < Geometry geom, Family family >
auto calculate_vector_shape_fns_curl(FiniteElement< geom, family > element, 
                                     const nd::view<const double, 2> xi) {

  constexpr uint32_t dim = dimension(geom);
  using vecd = vec<dim>;

  uint32_t num_qpts = xi.shape[0];
  uint32_t num_nodes = element.num_nodes();

  nd::array< double, 3, memory::space::cpu > psi({num_qpts, num_nodes, dim});

  for (uint32_t q = 0; q < num_qpts; q++) {
    vecd xi_q{};
    for (int d = 0; d < dim; d++) {
      xi_q[d] = xi(q, d);
    }

    for (uint32_t i = 0; i < num_nodes; i++) {
      vecd psi_q = element.shape_function_curl(xi_q, i);
      for (uint32_t d = 0; d < dim; d++) {
        psi(q, i, d) = psi_q[d];
      }
    }
  }

  return psi;

}

template < Family family >
nd::array<double,2,memory::space::cpu> nodes(const Field<family> & u, const Mesh<> & mesh) {

  uint32_t sdim = mesh.spatial_dimension;
  uint32_t gdim = mesh.geometry_dimension;

  nd::view< const double, 2 > x = mesh.X.data;
  uint32_t nverts = x.shape[0];

  GeometryInfo nodes_per_geom = interior_nodes_per_geom(FunctionSpace{u.family, u.degree}, gdim);
  GeometryInfo geom_counts = mesh.geometry_counts();

  auto qranges = ranges(nodes_per_geom * geom_counts);
  uint32_t num_nodes = total(nodes_per_geom * geom_counts);
  nd::array< double, 2, memory::space::cpu > output({num_nodes, mesh.spatial_dimension});

  if (nodes_per_geom.vert != 0) {
    for (uint32_t i = 0; i < nverts; i++) {
      for (uint32_t d = 0; d < sdim; d++) {
        output(i, d) = x(i, d);
      }
    }
  }

  uint32_t offset = 0;
  foreach_geometry([&](auto geom){
    if (qranges[geom].end > qranges[geom].begin) {

      FiniteElement< geom, family > element{u.degree};

      nd::array< double, 2, memory::space::cpu > xi({element.num_interior_nodes(), dimension(geom)});
      element.interior_nodes(xi);

      nd::array<int, 1, memory::space::cpu> elements({geom_counts[geom]});
      for (int i = 0; i < geom_counts[geom]; i++) {
        elements(i) = i;
      }

      auto shape_fns = calculate_shape_fns(FiniteElement< geom, Family::H1 >{mesh.X.degree}, xi);

      element_interpolate<geom, Family::H1, 2>(output(qranges[geom]), mesh.X, mesh[geom], elements, shape_fns);

    }
  });

  return output;
  
}

template < Geometry geom, Family family >
auto calculate_shape_fn_directional_derivatives(FiniteElement< geom, family > element, 
                                                const nd::view<const double, 2> xi,
                                                const nd::view<const double, 2> directions) {


  constexpr uint32_t dim = dimension(geom);
  using vecd = vec<dim>;

  uint32_t num_qpts = xi.shape[0];
  uint32_t num_nodes = element.num_nodes();

  nd::array< double, 2, memory::space::cpu > dpsi({num_qpts, num_nodes});

  for (uint32_t q = 0; q < num_qpts; q++) {
    vecd xi_q{};
    vecd dir_q{};
    for (int d = 0; d < dim; d++) {
      xi_q[d] = xi(q, d);
      dir_q[d] = directions(q, d);
    }

    for (uint32_t i = 0; i < num_nodes; i++) {
      dpsi(q, i) = dot(element.shape_function_gradient(xi_q, i), dir_q);
    }
  }

  return dpsi;

}

template < Family family >
nd::array<double,2,memory::space::cpu> directions(const Field<family> & u, const Mesh<> & mesh) {

  uint32_t sdim = mesh.spatial_dimension;
  uint32_t gdim = mesh.geometry_dimension;
  uint32_t num_components = u.data.shape[1];

  nd::view< const double, 2 > x = mesh.X.data;
  uint32_t nverts = x.shape[0];

  GeometryInfo nodes_per_geom = interior_nodes_per_geom(FunctionSpace{u.family, u.degree}, gdim);
  GeometryInfo geom_counts = mesh.geometry_counts();

  auto qranges = ranges(nodes_per_geom * geom_counts);
  uint32_t num_nodes = total(nodes_per_geom * geom_counts);
  nd::array< double, 2, memory::space::cpu > output({num_nodes, mesh.spatial_dimension});

  uint32_t offset = 0;
  foreach_geometry([&](auto geom){
    if (qranges[geom].end > qranges[geom].begin) {

      FiniteElement< geom, family > element{u.degree};
      nd::array< double, 2, memory::space::cpu > xi({element.num_interior_nodes(), dimension(geom)});
      element.interior_nodes(xi);

      nd::array< double, 2, memory::space::cpu > directions({element.num_interior_nodes(), dimension(geom)});
      element.interior_directions(directions);

      nd::array<int, 1, memory::space::cpu> elements({geom_counts[geom]});
      for (int i = 0; i < geom_counts[geom]; i++) {
        elements(i) = i;
      }

      FiniteElement< geom, Family::H1 > cg_element{mesh.X.degree};

      auto dpsi = calculate_shape_fn_directional_derivatives(cg_element, xi, directions);

      element_interpolate<geom, Family::H1, 2>(output(qranges[geom]), mesh.X, mesh[geom], elements, dpsi);
    }
  });

  return output;
  
}

}

template < Family family >
nd::array<double,2,memory::space::cpu> nodes_for(const Field<family> & u, const Mesh<> & mesh) {
  static_assert(family != Family::Hdiv, "nodes_for(Field, Mesh) not implemented for Family::Hdiv yet");
  return impl::nodes<family>(u, mesh);
}

template nd::array<double,2,memory::space::cpu> nodes_for<Family::H1>(const Field<Family::H1> &, const Mesh<> &);
template nd::array<double,2,memory::space::cpu> nodes_for<Family::Hcurl>(const Field<Family::Hcurl> &, const Mesh<> &);
template nd::array<double,2,memory::space::cpu> nodes_for<Family::DG>(const Field<Family::DG> &, const Mesh<> &);

template < Family family >
nd::array<double,2,memory::space::cpu> directions_for(const Field<family> & u, const Mesh<> & mesh) {
  static_assert(is_vector_valued(family), "directions_for(Field, Mesh) only defined for vector-basis families");
  return impl::directions<family>(u, mesh);
}

template nd::array<double,2,memory::space::cpu> directions_for<Family::Hcurl>(const Field<Family::Hcurl> &, const Mesh<> &);

} // namespace femto
