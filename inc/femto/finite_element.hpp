#pragma once

#include "femto/geometry.hpp"
#include "femto/quadrature.hpp"
#include "fm/types/vec.hpp"
#include "fm/types/matrix.hpp"

namespace femto {

enum class Family { 
  H1, 
  Hcurl, 
  Hdiv, 
  DG,
  MITC  // shell elements with mixed interpolation of tensorial components (see elements/mitc_*.hpp)
};

__host__ __device__ constexpr bool is_scalar_valued(Family f) {
  return (f == Family::H1) || (f == Family::DG);
}

__host__ __device__ constexpr bool is_vector_valued(Family f) {
  return (f == Family::Hcurl) || (f == Family::Hdiv);
}

struct FunctionSpace {
  Family family;
  uint32_t degree;
  uint32_t components;
  FunctionSpace() : family{}, degree{}, components{} {}
  FunctionSpace(Family f, uint32_t d = 2, uint32_t c = 1) : family{f}, degree{d}, components{c}{}

  bool operator==(const FunctionSpace & other) const {
    return (components == other.components) && 
           (degree == other.degree) &&
           (family == other.family);
  }
};

GeometryInfo nodes_per_geom(FunctionSpace space, uint32_t gdim);
GeometryInfo interior_nodes_per_geom(FunctionSpace space, uint32_t gdim);

GeometryInfo dofs_per_geom(FunctionSpace space, uint32_t gdim);
GeometryInfo interior_dofs_per_geom(FunctionSpace space, uint32_t gdim);

enum class TransformationType {
  PhysicalToParent,
  TransposePhysicalToParent,
};

template < Geometry g, Family f >
struct FiniteElement;

template < Geometry geom, Family family >
auto shape_function_derivatives(FiniteElement< geom, family > element,
                                const nd::view<const double, 2> xi) {
  if constexpr (family == Family::H1) {
    return element.evaluate_shape_function_gradients(xi);
  } 

  if constexpr (family == Family::Hcurl) {
    return element.evaluate_shape_function_curls(xi);
  } 
}

template < Geometry geom, Family family >
auto weighted_shape_function_derivatives(FiniteElement< geom, family > element,
                                         const nd::view<const double, 2> xi,
                                         const nd::view<const double, 1> weights) {
  if constexpr (family == Family::H1) {
    return element.evaluate_weighted_shape_function_gradients(xi, weights);
  } 

  if constexpr (family == Family::Hcurl) {
    return element.evaluate_weighted_shape_function_curls(xi, weights);
  } 
}

// not constexpr: a constexpr function may not hold a static variable before
// C++23, and without static storage the table is rebuilt in GPU local memory
// on every call (it is dynamically indexed, so it cannot be folded away)
__host__ __device__ inline fm::mat2 face_transformation(int8_t id) {
  static constexpr fm::mat2 matrices[5] = {
    {{{0, 1}, {1, 0}}},
    {{{-1, 0}, {-1, 1}}},
    {{{1, -1}, {0, -1}}},
    {{{-1, -1}, {0, 1}}}, 
    {{{1, 0}, {-1, -1}}}
  };

  return matrices[id-1];
}

__host__ __device__ constexpr int8_t face_transformation_id(TransformationType type, int orientation) {
  // arithmetic rather than a local LUT[3]: a function-local array indexed by
  // a runtime orientation is materialized in (and re-stored to) GPU local
  // memory at every call site
  if (type == TransformationType::PhysicalToParent) {
    return int8_t(orientation + 1);                        // {1, 2, 3}
  } else {
    return int8_t(orientation == 0 ? 1 : orientation + 3); // {1, 4, 5}
  }
}

}

#include "femto/elements/h1_vertex.hpp"
#include "femto/elements/h1_edge.hpp"
#include "femto/elements/h1_triangle.hpp"
#include "femto/elements/h1_quadrilateral.hpp"
#include "femto/elements/h1_tetrahedron.hpp"
#include "femto/elements/h1_hexahedron.hpp"

#include "femto/elements/hcurl_vertex.hpp"
#include "femto/elements/hcurl_edge.hpp"
#include "femto/elements/hcurl_triangle.hpp"
#include "femto/elements/hcurl_quadrilateral.hpp"
#include "femto/elements/hcurl_tetrahedron.hpp"
#include "femto/elements/hcurl_hexahedron.hpp"

#include "femto/elements/dg_vertex.hpp"
#include "femto/elements/dg_edge.hpp"
#include "femto/elements/dg_triangle.hpp"
#include "femto/elements/dg_quadrilateral.hpp"
#include "femto/elements/dg_tetrahedron.hpp"
#include "femto/elements/dg_hexahedron.hpp"

#include "femto/elements/mitc_common.hpp"
#include "femto/elements/mitc_stubs.hpp"
#include "femto/elements/mitc_triangle.hpp"
#include "femto/elements/mitc_quadrilateral.hpp"
