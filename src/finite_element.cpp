#include "femto/finite_element.hpp"
#include "femto/interpolation.hpp"

namespace femto {

template < Family family >
GeometryInfo nodes_per_geom(uint32_t p, uint32_t gdim) {
  GeometryInfo nodes_per{};
  nodes_per.vert = FiniteElement<Geometry::Vertex, family>{p}.num_nodes();
  nodes_per.edge = FiniteElement<Geometry::Edge, family>(p).num_nodes();
  nodes_per.tri  = FiniteElement<Geometry::Triangle, family>{p}.num_nodes();
  nodes_per.quad = FiniteElement<Geometry::Quadrilateral, family>{p}.num_nodes();
  nodes_per.tet  = FiniteElement<Geometry::Tetrahedron, family>{p}.num_nodes();
  nodes_per.hex  = FiniteElement<Geometry::Hexahedron, family>{p}.num_nodes();

  // DG elements only have nodes on cells with the highest geometric dimension
  if (family == Family::DG) {
    nodes_per.vert *= 0;

    nodes_per.edge *= (gdim == 1);

    nodes_per.tri  *= (gdim == 2);
    nodes_per.quad *= (gdim == 2);

    nodes_per.tet *= (gdim == 3);
    nodes_per.hex *= (gdim == 3);
  } 

  return nodes_per;
}
GeometryInfo nodes_per_geom(FunctionSpace space, uint32_t gdim) {
  if (space.family == Family::H1) { 
    return nodes_per_geom< Family::H1 >(space.degree, gdim); 
  }
  if (space.family == Family::Hcurl) { 
    return nodes_per_geom< Family::Hcurl >(space.degree, gdim); 
  }
  if (space.family == Family::DG) { 
    return nodes_per_geom< Family::DG >(space.degree, gdim); 
  }
  if (space.family == Family::MITC) { 
    return nodes_per_geom< Family::MITC >(space.degree, gdim); 
  }
  return {};
}

template < Family family >
GeometryInfo interior_nodes_per_geom(uint32_t p, uint32_t gdim) {
  GeometryInfo interior_nodes_per{};
  interior_nodes_per.vert = FiniteElement<Geometry::Vertex, family>{p}.num_interior_nodes();
  interior_nodes_per.edge = FiniteElement<Geometry::Edge, family>{p}.num_interior_nodes();
  interior_nodes_per.tri  = FiniteElement<Geometry::Triangle, family>{p}.num_interior_nodes();
  interior_nodes_per.quad = FiniteElement<Geometry::Quadrilateral, family>{p}.num_interior_nodes();
  interior_nodes_per.tet  = FiniteElement<Geometry::Tetrahedron, family>{p}.num_interior_nodes();
  interior_nodes_per.hex  = FiniteElement<Geometry::Hexahedron, family>{p}.num_interior_nodes();

  // DG elements only have nodes on cells with the highest geometric dimension
  if (family == Family::DG) {
    interior_nodes_per.vert *= 0;

    interior_nodes_per.edge *= (gdim == 1);

    interior_nodes_per.tri  *= (gdim == 2);
    interior_nodes_per.quad *= (gdim == 2);

    interior_nodes_per.tet *= (gdim == 3);
    interior_nodes_per.hex *= (gdim == 3);
  } 

  return interior_nodes_per;
}

GeometryInfo interior_nodes_per_geom(FunctionSpace space, uint32_t gdim) {
  if (space.family == Family::H1) { 
    return interior_nodes_per_geom< Family::H1 >(space.degree, gdim); 
  }
  if (space.family == Family::Hcurl) { 
    return interior_nodes_per_geom< Family::Hcurl >(space.degree, gdim); 
  }
  if (space.family == Family::DG) { 
    return interior_nodes_per_geom< Family::DG >(space.degree, gdim); 
  }
  if (space.family == Family::MITC) { 
    return interior_nodes_per_geom< Family::MITC >(space.degree, gdim); 
  }
  return {};
}

GeometryInfo dofs_per_geom(FunctionSpace space, uint32_t gdim) {
  return nodes_per_geom(space, gdim) * space.components;
}

GeometryInfo interior_dofs_per_geom(FunctionSpace space, uint32_t gdim) {
  return interior_nodes_per_geom(space, gdim) * space.components;
}

} // namespace femto