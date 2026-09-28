#pragma once

#include "misc/timer.hpp"
#include "misc/for_constexpr.hpp"

#include "fm/macros.hpp"
#include "fm/operations/adjugate.hpp"

#include "femto/domain.hpp"
#include "femto/assert.hpp"
#include "femto/piola_transformations.hpp"

#include "containers/ndarray.hpp"

namespace femto {

uint32_t elements_per_block(Geometry geom, Family family, int p);

template < typename T >
struct array_rank;

template < typename T, uint32_t n, memory::space mem_space >
struct array_rank< nd::view< T, n, mem_space > >{
  static constexpr uint32_t value = n;
};

template < typename T, uint32_t n, memory::space mem_space >
struct array_rank< nd::array< T, n, mem_space > >{
  static constexpr uint32_t value = n;
};

template < typename T >
void foreach_operation(T && function) {
  foreach_constexpr< DerivedQuantity::VALUE,
                     DerivedQuantity::VALUE_AVERAGE,
                     DerivedQuantity::VALUE_JUMP,
                     DerivedQuantity::VALUE_TWO_SIDED,
                     DerivedQuantity::GRAD,
                     DerivedQuantity::GRAD_AVERAGED,
                     DerivedQuantity::GRAD_TWO_SIDED,
                     DerivedQuantity::CURL,
                     DerivedQuantity::DIV >(function);
}

constexpr bool is_supported_combination(Family f, DerivedQuantity op) {
  if (f == Family::H1) {
    return (op == DerivedQuantity::VALUE) || 
           (op == DerivedQuantity::GRAD);
  }

  if (f == Family::Hcurl) {
    return (op == DerivedQuantity::VALUE) || 
           (op == DerivedQuantity::CURL);
  }

  if (f == Family::DG) {
    return (op == DerivedQuantity::VALUE) || 
           //(op == DerivedQuantity::VALUE_JUMP) || 
           //(op == DerivedQuantity::VALUE_AVERAGE) || 
           (op == DerivedQuantity::GRAD);
  }

  return false;
}

template < DerivedQuantity op, Geometry g, Family f, uint32_t n >
__host__ __device__ auto shape_function(const FiniteElement< g, f > & el, vec<n,double> xi, uint32_t I, [[maybe_unused]] int8_t transformation = 0) {

  if constexpr (is_scalar_valued(f) && op == DerivedQuantity::VALUE) {
    return el.shape_function(xi, I);
  }

  if constexpr (is_scalar_valued(f) && op == DerivedQuantity::GRAD) {
    return el.shape_function_gradient(xi, I);
  }

  if constexpr (f == Family::Hcurl && op == DerivedQuantity::VALUE) {
    return el.reoriented_shape_function(xi, I, transformation);
  }

  if constexpr (f == Family::Hcurl && op == DerivedQuantity::CURL) {
    return el.reoriented_shape_function_curl(xi, I, transformation);
  }

}

__host__ __device__ constexpr uint32_t round_up_to_multiple_of_128(uint32_t n) {
  return ((n + 127) / 128) * 128;
};

constexpr uint32_t source_shape(Family f, uint32_t gdim) {
  switch(f) {
    case Family::H1:    return 1;
    case Family::Hcurl: return gdim;
    case Family::Hdiv:  return gdim;
    case Family::DG:    return 1;
  }
  return (1u << 31);
}

constexpr uint32_t flux_shape(Family f, uint32_t gdim) {
  switch(f) {
    case Family::H1:    return gdim;
    case Family::Hcurl: return (gdim == 2) ? 1 : gdim;
    case Family::Hdiv:  return (gdim == 2) ? 1 : gdim;
    case Family::DG:    return gdim;
  }

  return (1u << 31);
}

template < Family f, DerivedQuantity op, uint32_t dim >
__host__ __device__ auto piola_transformation(const mat<dim,dim> & dX_dxi) {
  if constexpr ((f == Family::H1    && op == DerivedQuantity::GRAD) ||
                (f == Family::DG    && op == DerivedQuantity::GRAD) ||
                (f == Family::Hcurl && op == DerivedQuantity::VALUE)) {
    return inv(dX_dxi);
  }

  if constexpr ((f == Family::Hcurl && op == DerivedQuantity::CURL) ||
                (f == Family::Hdiv  && op == DerivedQuantity::VALUE)) {
    if constexpr (dim == 2) {
      return mat<1,1>{1.0 / det(dX_dxi)};
    } else {
      return transpose(dX_dxi) / det(dX_dxi);
    }
  }

  if constexpr ((f == Family::H1    && op == DerivedQuantity::VALUE) ||
                (f == Family::DG    && op == DerivedQuantity::VALUE) || 
                (f == Family::Hdiv  && op == DerivedQuantity::DIV)) {
    // this should never be called, but we implement it here regardless
    // to suppress a compiler warning about incompatible return values
    return 1.0;
  }
}

template < Family f, DerivedQuantity op, uint32_t dim >
__host__ __device__ auto weighted_piola_transformation(const mat<dim,dim> & dX_dxi) {
  if constexpr ((f == Family::H1    && op == DerivedQuantity::GRAD) ||
                (f == Family::DG    && op == DerivedQuantity::GRAD) ||
                (f == Family::Hcurl && op == DerivedQuantity::VALUE)) {
    return adj(dX_dxi);
  }

  if constexpr ((f == Family::Hcurl && op == DerivedQuantity::CURL) ||
                (f == Family::Hdiv  && op == DerivedQuantity::VALUE)) {
    if constexpr (dim == 2) {
      return mat<1,1>{1.0};
    } else {
      return transpose(dX_dxi);
    }
  }

  if constexpr ((f == Family::H1    && op == DerivedQuantity::VALUE) ||
                (f == Family::DG    && op == DerivedQuantity::VALUE) || 
                (f == Family::Hdiv  && op == DerivedQuantity::DIV)) {
    return det(dX_dxi);
  }
}

// the same transformations as above, but expressed in terms of the
// precomputed inverse jacobian A := dxi_dX = inv(dX_dxi) and the precomputed
// determinant d := det(dX_dxi), using
//   inv(dX_dxi) = A,   adj(dX_dxi) = A d,   transpose(dX_dxi) = transpose(adj(A)) d
// (adj(A) is cofactor arithmetic, so no determinant or division is evaluated)
template < Family f, DerivedQuantity op, uint32_t dim >
__host__ __device__ auto piola_transformation_from_inverse(const mat<dim,dim> & dxi_dX, [[maybe_unused]] double det_dX_dxi) {
  if constexpr ((f == Family::H1    && op == DerivedQuantity::GRAD) ||
                (f == Family::DG    && op == DerivedQuantity::GRAD) ||
                (f == Family::Hcurl && op == DerivedQuantity::VALUE)) {
    return dxi_dX;
  }

  if constexpr ((f == Family::Hcurl && op == DerivedQuantity::CURL) ||
                (f == Family::Hdiv  && op == DerivedQuantity::VALUE)) {
    if constexpr (dim == 2) {
      return mat<1,1>{1.0 / det_dX_dxi};
    } else {
      return transpose(adj(dxi_dX));
    }
  }

  if constexpr ((f == Family::H1    && op == DerivedQuantity::VALUE) ||
                (f == Family::DG    && op == DerivedQuantity::VALUE) ||
                (f == Family::Hdiv  && op == DerivedQuantity::DIV)) {
    // this should never be called, but we implement it here regardless
    // to suppress a compiler warning about incompatible return values
    return 1.0;
  }
}

template < Family f, DerivedQuantity op, uint32_t dim >
__host__ __device__ auto weighted_piola_transformation_from_inverse(const mat<dim,dim> & dxi_dX, double det_dX_dxi) {
  if constexpr ((f == Family::H1    && op == DerivedQuantity::GRAD) ||
                (f == Family::DG    && op == DerivedQuantity::GRAD) ||
                (f == Family::Hcurl && op == DerivedQuantity::VALUE)) {
    return dxi_dX * det_dX_dxi;
  }

  if constexpr ((f == Family::Hcurl && op == DerivedQuantity::CURL) ||
                (f == Family::Hdiv  && op == DerivedQuantity::VALUE)) {
    if constexpr (dim == 2) {
      return mat<1,1>{1.0};
    } else {
      return transpose(adj(dxi_dX)) * det_dX_dxi;
    }
  }

  if constexpr ((f == Family::H1    && op == DerivedQuantity::VALUE) ||
                (f == Family::DG    && op == DerivedQuantity::VALUE) ||
                (f == Family::Hdiv  && op == DerivedQuantity::DIV)) {
    return det_dX_dxi;
  }
}

// the transformations above, from the per-component jacobian rows
// dX_dxi_q(c, q) = d(X_c)/d(xi) that the kernels compute.  When the element is
// embedded in a higher-dimensional space (a boundary domain, or a surface mesh:
// X has more components than the element has reference coordinates) dX_dxi is
// (sdim x gdim) and has no inverse; only its measure sqrt(det(JᵀJ)) is
// defined, which is all the scalar-valued transformations (H1/DG values) use.
// ponytail: spatial derivatives and vector-valued traces on such elements
// need the pseudo-inverse (JᵀJ)⁻¹Jᵀ and sdim-wide outputs, not built yet
template < uint32_t sdim, uint32_t gdim >
__host__ __device__ double facet_measure(const mat<sdim, gdim> & J) {
  return sqrt(det(dot(transpose(J), J)));
}

// a facet's jacobian J (sdim x gdim, sdim > gdim) has no inverse: store its
// pseudo-inverse (JᵀJ)⁻¹Jᵀ and the facet measure sqrt(det(JᵀJ)) instead
template < uint32_t sdim, uint32_t gdim, memory::space mem_space >
__host__ __device__ void facet_jacobian(nd::view<double, 3, mem_space> dxi_dX,
                                        nd::view<double, 1, mem_space> det_dX_dxi,
                                        nd::view<const double, 3, mem_space> J,
                                        uint32_t q) {
  mat<sdim, gdim> A;
  for (uint32_t i = 0; i < sdim; i++)
    for (uint32_t j = 0; j < gdim; j++) { A(i, j) = J(q, i, j); }
  mat<gdim, gdim> G = dot(transpose(A), A);
  det_dX_dxi(q) = sqrt(det(G));
  mat<gdim, sdim> P = dot(inv(G), transpose(A));
  for (uint32_t i = 0; i < gdim; i++)
    for (uint32_t j = 0; j < sdim; j++) { dxi_dX(q, i, j) = P(i, j); }
}

template < uint32_t gdim >
double facet_measure(const nd::view<vec<gdim>, 2> & dX_dxi_q, uint32_t q) {
  mat<gdim, gdim> G{};
  for (uint32_t c = 0; c < dX_dxi_q.shape[0]; c++) {
    G = G + outer(dX_dxi_q(c, q), dX_dxi_q(c, q));
  }
  return sqrt(det(G));
}

template < Family f, DerivedQuantity op, uint32_t gdim >
auto piola_transformation(const nd::view<vec<gdim>, 2> & dX_dxi_q, uint32_t q) {
  using A_type = decltype(piola_transformation<f, op>(mat<gdim, gdim>{}));
  if (dX_dxi_q.shape[0] == gdim) {
    mat<gdim, gdim> dX_dxi;
    for (uint32_t c = 0; c < gdim; c++) { dX_dxi[c] = dX_dxi_q(c, q); }
    return piola_transformation<f, op>(dX_dxi);
  }
  if constexpr (is_scalar_valued(f) && op == DerivedQuantity::VALUE) {
    return 1.0;
  } else {
    FEMTO_ASSERT(false, "spatial derivatives and vector-valued fields are not supported on facets embedded in a higher-dimensional space");
    return A_type{};
  }
}

template < Family f, DerivedQuantity op, uint32_t gdim >
auto weighted_piola_transformation(const nd::view<vec<gdim>, 2> & dX_dxi_q, uint32_t q) {
  using A_type = decltype(weighted_piola_transformation<f, op>(mat<gdim, gdim>{}));
  if (dX_dxi_q.shape[0] == gdim) {
    mat<gdim, gdim> dX_dxi;
    for (uint32_t c = 0; c < gdim; c++) { dX_dxi[c] = dX_dxi_q(c, q); }
    return weighted_piola_transformation<f, op>(dX_dxi);
  }
  if constexpr (is_scalar_valued(f) && op == DerivedQuantity::VALUE) {
    return facet_measure(dX_dxi_q, q);
  } else {
    FEMTO_ASSERT(false, "spatial derivatives and vector-valued fields are not supported on facets embedded in a higher-dimensional space");
    return A_type{};
  }
}

template < Geometry geom, Family test_family, Family trial_family>
void jacobian_rows_and_columns(nd::view<int,5> rows,
                               nd::view<int,5> cols,
                               FunctionSpace trial_space,
                               FunctionSpace test_space,
                               GeometryInfo trial_offsets,
                               GeometryInfo test_offsets,
                               nd::view<const int> elements,
                               nd::view<const Connection, 2> connectivity) {

  FiniteElement< geom, test_family > test_el{test_space.degree};
  FiniteElement< geom, trial_family > trial_el{trial_space.degree};

  // allocate storage for an element's nodal forces
  constexpr uint32_t gdim = dimension(geom);

  uint32_t num_elements = rows.shape[0];
  uint32_t test_components = test_space.components;
  uint32_t trial_components = trial_space.components;
  uint32_t nodes_per_test_element = test_el.num_nodes();
  uint32_t nodes_per_trial_element = trial_el.num_nodes();

  nd::array<uint32_t, 1, memory::space::cpu> test_ids({nodes_per_test_element});
  nd::array<uint32_t, 1, memory::space::cpu> trial_ids({nodes_per_trial_element});

  // for each element of this geometry in the domain
  for (uint32_t e = 0; e < num_elements; e++) {

    // get the ids of nodes for that element
    test_el.indices(test_offsets, connectivity(elements(e)).data(), test_ids.data());
    trial_el.indices(trial_offsets, connectivity(elements(e)).data(), trial_ids.data());

    // populate the row/column entries for the element jacobian
    for (uint32_t J = 0; J < nodes_per_trial_element; J++) {
      for (uint32_t j = 0; j < trial_space.components; j++) {
        for (uint32_t I = 0; I < nodes_per_test_element; I++) {
          for (uint32_t i = 0; i < test_space.components; i++) {
            rows(e, J, j, I, i) = int(test_ids(I) * test_space.components + i);
            cols(e, J, j, I, i) = int(trial_ids(J) * trial_space.components + j);
          }
        }
      }
    }

  }

}

template < Geometry geom, typename view_t >
__host__ __device__ auto quadrature_point(uint32_t q, const view_t & xi) {

  if constexpr (Geometry::Quadrilateral == geom) {
    uint32_t q1D = xi.shape[0];
    uint32_t qx = q % q1D;
    uint32_t qy = q / q1D;
    return vec2{xi(qx, 0), xi(qy, 0)};
  }

  if constexpr (Geometry::Hexahedron == geom) {
    uint32_t q1D = xi.shape[0];
    uint32_t qx = q % q1D;
    uint32_t qy = (q % (q1D * q1D)) / q1D;
    uint32_t qz = q / (q1D * q1D);
    return vec3{xi(qx, 0), xi(qy, 0), xi(qz, 0)};
  }

  // all other geometries
  constexpr int gdim = dimension(geom);
  vec< gdim, double > xi_q;
  for (uint32_t c = 0; c < gdim; c++) {
    xi_q(c) = xi(q, c);
  }
  return xi_q;

}

}

namespace nd {

template < typename T, uint32_t n >
struct printer< fm::vec<n,T> >{ 
  static __host__ __device__ void print(const fm::vec<n,T> & v) {
    printf("{");
    printer<T>::print(v(0));
    for (int i = 1; i < n; i++) {
      printf(",");
      printer<T>::print(v(i));
    }
    printf("}");
  }
};

}
