#include "femto/mesh.hpp"

namespace femto {

template <>
Mesh<> Mesh<>::cuboid(stack::array< uint32_t, 2 > e, vec2 dimensions) {

  uint32_t ex = e[0];
  uint32_t ey = e[1];

  uint32_t nx = e[0] + 1;
  uint32_t ny = e[1] + 1;

  uint32_t nnodes = nx * ny;
  uint32_t nelems = ex * ey;

  uint32_t dx = 1;
  uint32_t dy = nx;

  vec2 scale{dimensions[0] / e[0], dimensions[1] / e[1]};

  nd::array< double, 2, memory::space::cpu > nodes({nnodes, 2});
  for (uint32_t y = 0; y < ny; y++) {
    for (uint32_t x = 0; x < nx; x++) {
      uint32_t node_id = x * dx + y * dy;
      nodes(node_id, 0) = x * scale[0];
      nodes(node_id, 1) = y * scale[1];
    }
  }

  nd::array< uint32_t, 2, memory::space::cpu > tris({0, 0});
  nd::array< uint32_t, 2, memory::space::cpu > quads({nelems, 4});
  for (uint32_t y = 0; y < e[1]; y++) {
    for (uint32_t x = 0; x < e[0]; x++) {
      uint32_t elem_id = x + y * ex;
      uint32_t base_id = x * dx + y * dy;
      quads(elem_id, 0) = base_id;
      quads(elem_id, 1) = base_id + dx;
      quads(elem_id, 2) = base_id + dx + dy;
      quads(elem_id, 3) = base_id      + dy;
    }
  }

  uint32_t degree = 1;
  return Mesh<>::create_2D(nodes, degree, tris, quads);

}

template <>
Mesh<> Mesh<>::cuboid(stack::array< uint32_t, 3 > e, vec3 dimensions) {

  uint32_t ex = e[0];
  uint32_t ey = e[1];
  uint32_t ez = e[2];

  uint32_t nx = ex + 1;
  uint32_t ny = ey + 1;
  uint32_t nz = ez + 1;

  uint32_t nnodes = nx * ny * nz;
  uint32_t nelems = ex * ey * ez;

  uint32_t dx = 1;
  uint32_t dy = nx;
  uint32_t dz = nx * ny;

  vec3 scale{dimensions[0] / e[0], dimensions[1] / e[1], dimensions[2] / e[2]};

  nd::array< double, 2, memory::space::cpu > nodes({nnodes, 3});
  for (uint32_t z = 0; z < nz; z++) {
    for (uint32_t y = 0; y < ny; y++) {
      for (uint32_t x = 0; x < nx; x++) {
        uint32_t node_id = x * dx + y * dy + z * dz;
        nodes(node_id, 0) = x * scale[0];
        nodes(node_id, 1) = y * scale[1];
        nodes(node_id, 2) = z * scale[2];
      }
    }
  }

  nd::array< uint32_t, 2, memory::space::cpu > tets({0, 0});
  nd::array< uint32_t, 2, memory::space::cpu > hexes({nelems, 8});
  for (uint32_t z = 0; z < e[2]; z++) {
    for (uint32_t y = 0; y < e[1]; y++) {
      for (uint32_t x = 0; x < e[0]; x++) {
        uint32_t elem_id = x + y * ex + z * ex * ey;
        uint32_t base_id = x * dx + y * dy + z * dz;
        hexes(elem_id, 0) = base_id;
        hexes(elem_id, 1) = base_id + dx;
        hexes(elem_id, 2) = base_id + dx + dy;
        hexes(elem_id, 3) = base_id      + dy;
        hexes(elem_id, 4) = base_id           + dz;
        hexes(elem_id, 5) = base_id + dx      + dz;
        hexes(elem_id, 6) = base_id + dx + dy + dz;
        hexes(elem_id, 7) = base_id      + dy + dz;
      }
    }
  }

  uint32_t degree = 1;
  return Mesh<>::create_3D(nodes, degree, tets, hexes);

}

}
