#pragma once

#include <set>

#include "containers/ndarray.hpp"

#include "femto/field.hpp"
#include "femto/geometry.hpp"
#include "femto/quadrature.hpp"
#include "femto/finite_element.hpp"
#include "femto/connection.hpp"

#include "linear_algebra/sparse_matrix.hpp"

namespace femto {

template < Family f, memory::space mem_space >
struct Field;

template < memory::space mem_space >
struct Mesh : public GeometryData< nd::array< Connection, 2, mem_space > > {
  static_assert(mem_space != memory::UNIFIED, "Mesh<memory::UNIFIED> is not defined");

  Mesh() {}

  template < memory::space other_space >
  Mesh(const Mesh<other_space> & other) {
    *this = other;
  }

  template < memory::space other_space >
  Mesh & operator=(const Mesh<other_space> & other) {
    this->vert = other.vert;
    this->edge = other.edge;
    this->tri = other.tri;
    this->quad = other.quad;
    this->tet = other.tet;
    this->hex = other.hex;
    X = other.X;
    spatial_dimension = other.spatial_dimension;
    geometry_dimension = other.geometry_dimension;
    return *this;
  }

  static Mesh load(std::string filename);
  static Mesh import_from_json_string(std::string json_string);

  static Mesh create_1D(const nd::array< double, 2, memory::space::cpu > & nodes, uint32_t degree,
                        const nd::array< uint32_t, 2, memory::space::cpu > & edges);

  static Mesh create_2D(const nd::array< double, 2, memory::space::cpu > & nodes, uint32_t degree,
                        const nd::array< uint32_t, 2, memory::space::cpu > & tris, const nd::array< uint32_t, 2, memory::space::cpu > & quads);

  static Mesh create_3D(const nd::array< double, 2, memory::space::cpu > & nodes, uint32_t degree,
                        const nd::array< uint32_t, 2, memory::space::cpu > & tets, const nd::array< uint32_t, 2, memory::space::cpu > & hexes);

  static Mesh cuboid(stack::array< uint32_t, 2 > num_elements, vec2 dimensions);
  static Mesh cuboid(stack::array< uint32_t, 3 > num_elements, vec3 dimensions);

  static Mesh disk(vec2 center, double r, double h, int p, Geometry element_type);
  static Mesh ball(vec3 center, double r, double h, int p, Geometry element_type);

  static Mesh coffee_mug(double base_radius, double rim_radius, double height, double thickness, int handles, int p);

  // the air inside and around a cylindrically symmetric bottle (see src/mesh/bottle.cpp)
  static Mesh bottle(double body_radius, double body_height, double shoulder_height,
                     double neck_radius, double neck_length, double thickness,
                     double air_margin, double air_below, double air_above, double h);

  // its planar counterpart: the air inside and around the cross section of a
  // bottle-shaped channel (mirror symmetric about x = 0, opening in +y), as a
  // quadrilateral mesh with the same parameters (radii become half-widths)
  static Mesh bottle_2d(double body_half_width, double body_height, double shoulder_height,
                        double neck_half_width, double neck_length, double thickness,
                        double air_margin, double air_below, double air_above, double h);

  GeometryInfo geometry_counts() const {
    return GeometryInfo{
      this->vert.shape[0],
      this->edge.shape[0],
      this->tri.shape[0],
      this->quad.shape[0],
      this->tet.shape[0],
      this->hex.shape[0]
    };
  }

  nd::view< const Connection, 2, mem_space > connectivity(Geometry g) const { return (*this)[g]; }

  // information about the mesh shape
  Field<Family::H1, mem_space> X;

  uint32_t spatial_dimension;
  uint32_t geometry_dimension;

};

template < memory::space target_space, memory::space source_space >
Mesh<target_space> copy_to(const Mesh<source_space> & input) {
  return Mesh<target_space>(input);
}

template < Family f, memory::space residual_space >
template < memory::space mesh_space >
void Residual<f, residual_space>::reset(FunctionSpace space_, const Mesh<mesh_space>& mesh) {
  FEMTO_ASSERT(space_.family == f, "mismatched family");
  space = space_;
  uint32_t gdim = mesh.geometry_dimension;
  GeometryInfo nodes_per = interior_nodes_per_geom(FunctionSpace{space.family, space.degree}, gdim);
  GeometryInfo counts = mesh.geometry_counts();

  // resize is a no-op when the size already matches; integrate_residual
  // accumulates, so the values have to be zeroed either way
  data.resize({total(nodes_per * counts), space.components});
  nd::zero(data);
  offsets = scan(nodes_per * counts);
}

template < Family f, memory::space residual_space >
template < memory::space mesh_space >
Residual<f, residual_space>::Residual(FunctionSpace space, const Mesh<mesh_space>& mesh) {
  reset(space, mesh);
}

void save(const Mesh<> & mesh, std::string filename);

template < memory::space mem_space = memory::space::cpu >
struct SubMesh : public GeometryData< nd::array< uint32_t, 1, mem_space > >{

  SubMesh() : parent{nullptr} {}
  SubMesh(const Mesh<mem_space> * ptr) : parent{ptr} {}

  // the mesh this object refers to
  const Mesh<mem_space> * parent;

  int spatial_dimension;
  int geometry_dimension;

};

Mesh<> merge(std::vector< Mesh<> > meshes, double r);

bool is_manifold(const Mesh<> & mesh, bool print_report = false);

SubMesh<> boundary_of(const Mesh<> & mesh);

#ifdef NDARRAY_ENABLE_CUDA
// found on the host, only the entity id lists live on the device
SubMesh<memory::space::gpu> boundary_of(const Mesh<memory::space::gpu> & mesh);
#endif

Mesh<> import_stl(std::string filename);
Mesh<> import_vtk(std::string filename);
Mesh<> import_gmsh(std::string filename);

void export_stl(const Mesh<> & mesh, std::string filename);
void export_vtk(const Mesh<> & mesh, std::string filename);
void export_gmsh(const Mesh<> & mesh, std::string filename);

Mesh<> convert_to_simplices(const Mesh<> & mesh);
Mesh<> convert_to_tensor_product(const Mesh<> & mesh);

Mesh<> coarsen(const Mesh<> & mesh, float angles[2]);

}
