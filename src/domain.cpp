#include "femto/domain.hpp"

#include "field/common.hpp"

#include "fm/types/AABB.hpp"
#include "fm/types/matrix.hpp"

namespace femto {

template < Geometry geom, Family family >
void compute_multiplicity(nd::view<uint32_t> multiplicity,
                          FunctionSpace space,
                          GeometryInfo offsets,
                          nd::view<const int> elements,
                          nd::view<const Connection, 2> connectivity) {

  FiniteElement< geom, family > el{space.degree};
  uint32_t num_elements = elements.shape[0];
  uint32_t nodes_per_element = el.num_nodes();
  nd::array<uint32_t, 1, memory::space::cpu> ids({nodes_per_element});
  for (uint32_t e = 0; e < num_elements; e++) {
    el.indices(offsets, connectivity(elements(e)).data(), ids.data());
    for (int i = 0; i < nodes_per_element; i++) {
      multiplicity[ids[i]]++;
    }
  };

}

template < Geometry geom, Family family >
void fill_ids(nd::view<uint32_t> multiplicity,
              nd::view<uint32_t> dof_ids,
              nd::view<const uint32_t> offsets,
              FunctionSpace space,
              GeometryInfo goffsets,
              nd::view<const int> elements,
              nd::view<const Connection, 2> connectivity) {

  FiniteElement< geom, family > el{space.degree};
  uint32_t num_elements = elements.shape[0];
  uint32_t nodes_per_element = el.num_nodes();
  nd::array<uint32_t, 1, memory::space::cpu> node_ids({nodes_per_element});
  for (uint32_t e = 0; e < num_elements; e++) {
    el.indices(goffsets, connectivity(elements(e)).data(), node_ids.data());
    for (int i = 0; i < nodes_per_element; i++) {
      uint32_t total_offset = offsets[node_ids[i]] + multiplicity[node_ids[i]];
      dof_ids[total_offset] = e * nodes_per_element + i;
      multiplicity[node_ids[i]]++;
    }
  };
}

template <>
const Domain<>::AssemblyLUT & Domain<>::get(Geometry g, Family f, uint32_t p) const {

  if (gather_tables.count({g, f, p})) {

    return gather_tables.at({g, f, p});

  } else {

    check_family(f);

    FunctionSpace phi{f, p, 1};

    uint32_t gdim = mesh.geometry_dimension;
    GeometryInfo gcounts = mesh.geometry_counts();
    GeometryInfo goffsets = scan(interior_nodes_per_geom(phi, gdim) * gcounts);
    uint32_t num_nodes = total(interior_nodes_per_geom(phi, gdim) * gcounts);

    // first count how many elements each node belongs to
    nd::array<uint32_t, 1, memory::space::cpu> node_multiplicity({num_nodes});
    foreach_geometry([&](auto geom){
      nd::view<const int> elements = active_elements[geom];
      if (g == geom && elements.size() > 0) {
        nd::view<const Connection, 2> connectivity = mesh[geom];
        foreach_constexpr< Family::H1, Family::Hcurl, Family::DG >([&](auto family) {
          if (family == phi.family) {
            compute_multiplicity<geom, family>(node_multiplicity, phi, goffsets, elements, connectivity);
          }
        });
      }
    });

    AssemblyLUT & table = gather_tables[{g, f, p}];

    table.offsets.resize(num_nodes + 1);
    table.offsets[0] = 0;
    for (uint32_t i = 0; i < num_nodes; i++) {
      table.offsets[i+1] = table.offsets[i] + node_multiplicity[i];
    }

    nd::zero(node_multiplicity);
    table.ids.resize(table.offsets[num_nodes]);
    foreach_geometry([&](auto geom){
      nd::view<const int> elements = active_elements[geom];
      if (g == geom && elements.size() > 0) {
        nd::view<const Connection, 2> connectivity = mesh[geom];
        foreach_constexpr< Family::H1, Family::Hcurl, Family::DG >([&](auto family) {
          if (family == phi.family) {
            fill_ids<geom, family>(node_multiplicity, table.ids, table.offsets, phi, goffsets, elements, connectivity);
          }
        });
      }
    });

    return table;

  }

}

template <>
void Domain<>::compute_jacobian_inverses() {
  const uint32_t gdim = geometry_dimension;
  const uint32_t sdim = mesh.X.data.shape[1];
  if (gdim == 0) { return; }
  nd::array<double, 3, memory::space::cpu> J = evaluate(grad(mesh.X), isoparametric(*this));
  dxi_dX.resize({J.shape[0], gdim, sdim});
  det_dX_dxi.resize({J.shape[0]});
  for (uint32_t q = 0; q < J.shape[0]; q++) {
    if (sdim != gdim) {
      if (sdim == 2 && gdim == 1) { facet_jacobian<2, 1, memory::space::cpu>(dxi_dX, det_dX_dxi, J, q); }
      if (sdim == 3 && gdim == 1) { facet_jacobian<3, 1, memory::space::cpu>(dxi_dX, det_dX_dxi, J, q); }
      if (sdim == 3 && gdim == 2) { facet_jacobian<3, 2, memory::space::cpu>(dxi_dX, det_dX_dxi, J, q); }
    } else if (gdim == 1) {
      det_dX_dxi(q) = J(q, 0, 0);
      dxi_dX(q, 0, 0) = 1.0 / J(q, 0, 0);
    } else if (gdim == 2) {
      mat<2, 2> A;
      for (uint32_t i = 0; i < 2; i++)
        for (uint32_t j = 0; j < 2; j++) { A(i, j) = J(q, i, j); }
      det_dX_dxi(q) = det(A);
      mat<2, 2> Ainv = inv(A);
      for (uint32_t i = 0; i < 2; i++)
        for (uint32_t j = 0; j < 2; j++) { dxi_dX(q, i, j) = Ainv(i, j); }
    } else {
      mat<3, 3> A;
      for (uint32_t i = 0; i < 3; i++)
        for (uint32_t j = 0; j < 3; j++) { A(i, j) = J(q, i, j); }
      det_dX_dxi(q) = det(A);
      mat<3, 3> Ainv = inv(A);
      for (uint32_t i = 0; i < 3; i++)
        for (uint32_t j = 0; j < 3; j++) { dxi_dX(q, i, j) = Ainv(i, j); }
    }
  }
}

#if 0
void Domain<>::spatial_sort() {

  foreach_geometry([&](auto geom){
    uint32_t num_elements = active_elements[geom].shape[0];
    if (num_elements > 0) {

      constexpr int dim = dimension(geom);
      uint32_t X_degree = mesh.X.degree;
      auto X_offsets = mesh.X.offsets;
      nd::view< vec<dim> > X(reinterpret_cast<vec<dim>*>(&mesh.X.data[0]), {mesh.X.data.shape[0]});
      const nd::view<int32_t> elements = active_elements[geom];
      const nd::view<Connection, 2> connectivity = mesh[geom];

      fm::AABB<dim> global_AABB;
      for (int i = 0; i < dim; i++) {
        global_AABB.min[i] = +1.0e10;
        global_AABB.max[i] = -1.0e10;
      }

      FiniteElement<geom, Family::H1> el{X_degree};

      uint32_t nodes_per_elem = el.num_nodes();
      std::vector< vec<dim> > X_e(nodes_per_elem);
      std::vector< uint32_t > node_ids(nodes_per_elem);
      std::vector< fm::AABB<dim> > elem_AABBs(num_elements);
      std::vector< std::array<uint32_t,2> > elem_ids;
      for (int i = 0; i < num_elements; i++) {
        el.indices(X_offsets, connectivity(elements[i]).data(), node_ids.data());
        AABB<dim> elem_AABB;
        for (int j = 0; j < nodes_per_elem; j++) {
          vec<dim, float> X_j;
          for (int k = 0; k < dim; k++) {
            X_j[k] = X[node_ids[j]][k];
          };

          if (j == 0) {
            elem_AABB.min = X_j;
            elem_AABB.max = X_j;
          } else {
            elem_AABB.min = fm::min(elem_AABB.min, X_j);
            elem_AABB.max = fm::max(elem_AABB.max, X_j);
          }
        }
        elem_AABBs[i] = elem_AABB;
        global_AABB.min = fm::min(global_AABB.min, elem_AABB.min);
        global_AABB.max = fm::max(global_AABB.max, elem_AABB.max);
      }


    }

  });

}
#endif

}
