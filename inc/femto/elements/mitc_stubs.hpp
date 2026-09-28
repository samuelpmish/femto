#pragma once

// The MITC family (MITC3+ and MITC4+, p = 1 only) places its nodes on the vertices, plus
// one interior node per triangle for the MITC3+ rotation bubble; it has no 3D elements.
// These stubs only report node counts so that nodes_per_geom() can size a Field<Family::MITC>.

namespace femto {

template <>
struct FiniteElement<Geometry::Vertex, Family::MITC> {
  __host__ __device__ uint32_t num_nodes() const { return 1; }
  __host__ __device__ uint32_t num_interior_nodes() const { return 1; }
  uint32_t p;
};

template <>
struct FiniteElement<Geometry::Edge, Family::MITC> {
  __host__ __device__ uint32_t num_nodes() const { return 2; }
  __host__ __device__ uint32_t num_interior_nodes() const { return 0; }
  uint32_t p;
};

template <>
struct FiniteElement<Geometry::Tetrahedron, Family::MITC> {
  __host__ __device__ uint32_t num_nodes() const { return 0; }
  __host__ __device__ uint32_t num_interior_nodes() const { return 0; }
  uint32_t p;
};

template <>
struct FiniteElement<Geometry::Hexahedron, Family::MITC> {
  __host__ __device__ uint32_t num_nodes() const { return 0; }
  __host__ __device__ uint32_t num_interior_nodes() const { return 0; }
  uint32_t p;
};

} // namespace femto
