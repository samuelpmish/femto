#include "femto/mesh.hpp"

#ifdef NDARRAY_ENABLE_CUDA

namespace femto {

template <>
Mesh<memory::space::gpu> Mesh<memory::space::gpu>::create_1D(
  const nd::array< double, 2, memory::space::cpu > & nodes,
  uint32_t degree,
  const nd::array< uint32_t, 2, memory::space::cpu > & edges) {
  return Mesh<memory::space::gpu>{Mesh<memory::space::cpu>::create_1D(nodes, degree, edges)};
}

template <>
Mesh<memory::space::gpu> Mesh<memory::space::gpu>::create_2D(
  const nd::array< double, 2, memory::space::cpu > & nodes,
  uint32_t degree,
  const nd::array< uint32_t, 2, memory::space::cpu > & tris,
  const nd::array< uint32_t, 2, memory::space::cpu > & quads) {
  return Mesh<memory::space::gpu>{Mesh<memory::space::cpu>::create_2D(nodes, degree, tris, quads)};
}

template <>
Mesh<memory::space::gpu> Mesh<memory::space::gpu>::create_3D(
  const nd::array< double, 2, memory::space::cpu > & nodes,
  uint32_t degree,
  const nd::array< uint32_t, 2, memory::space::cpu > & tets,
  const nd::array< uint32_t, 2, memory::space::cpu > & hexes) {
  return Mesh<memory::space::gpu>{Mesh<memory::space::cpu>::create_3D(nodes, degree, tets, hexes)};
}

template <>
Mesh<memory::space::gpu> Mesh<memory::space::gpu>::load(std::string filename) {
  return Mesh<memory::space::gpu>{Mesh<memory::space::cpu>::load(filename)};
}

template <>
Mesh<memory::space::gpu> Mesh<memory::space::gpu>::import_from_json_string(std::string json_string) {
  return Mesh<memory::space::gpu>{Mesh<memory::space::cpu>::import_from_json_string(json_string)};
}

template <>
Mesh<memory::space::gpu> Mesh<memory::space::gpu>::cuboid(stack::array< uint32_t, 2 > num_elements, vec2 dimensions) {
  return Mesh<memory::space::gpu>{Mesh<memory::space::cpu>::cuboid(num_elements, dimensions)};
}

template <>
Mesh<memory::space::gpu> Mesh<memory::space::gpu>::cuboid(stack::array< uint32_t, 3 > num_elements, vec3 dimensions) {
  return Mesh<memory::space::gpu>{Mesh<memory::space::cpu>::cuboid(num_elements, dimensions)};
}

SubMesh<memory::space::gpu> boundary_of(const Mesh<memory::space::gpu> & mesh) {
  Mesh<> mesh_cpu = copy_to<memory::space::cpu>(mesh);
  SubMesh<> bdr_cpu = boundary_of(mesh_cpu);
  SubMesh<memory::space::gpu> bdr{&mesh};
  bdr.spatial_dimension = bdr_cpu.spatial_dimension;
  bdr.geometry_dimension = bdr_cpu.geometry_dimension;
  bdr.vert = bdr_cpu.vert;
  bdr.edge = bdr_cpu.edge;
  bdr.tri = bdr_cpu.tri;
  bdr.quad = bdr_cpu.quad;
  bdr.tet = bdr_cpu.tet;
  bdr.hex = bdr_cpu.hex;
  return bdr;
}

}

#endif
