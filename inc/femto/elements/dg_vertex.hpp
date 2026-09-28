#pragma once

#include "femto/connection.hpp" 

namespace femto {

template <>
struct FiniteElement<Geometry::Vertex, Family::DG> {

  using source_type = double;
  using flux_type = double;

  __host__ __device__ uint32_t num_nodes() const { return 0; }
  __host__ __device__ uint32_t num_interior_nodes() const { return 0; }

  uint32_t p;

};

} // namespace femto