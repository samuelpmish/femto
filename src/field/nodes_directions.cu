#include "femto/mesh.hpp"

#include "misc/macros.hpp"

#ifdef NDARRAY_ENABLE_CUDA

namespace femto {

namespace {

template < typename T, uint32_t rank >
nd::array<T, rank, memory::space::cpu> copy_to_cpu(const nd::array<T, rank, memory::space::gpu> & input) {
  nd::array<T, rank, memory::space::cpu> output(input.shape);
  if (input.sz > 0) {
    CUDA_CHECK(cudaMemcpy(output.data(), input.data(), input.sz * sizeof(T), cudaMemcpyDefault));
  }
  return output;
}

template < Family family >
Field<family> metadata_on_cpu(const Field<family, memory::space::gpu> & input) {
  Field<family> output;
  output.degree = input.degree;
  output.offsets = input.offsets;
  output.data = nd::array<double, 2, memory::space::cpu>(input.data.shape);
  return output;
}

template < Family family >
Field<family> copy_to_cpu(const Field<family, memory::space::gpu> & input) {
  Field<family> output;
  output.degree = input.degree;
  output.offsets = input.offsets;
  output.data = copy_to_cpu(input.data);
  return output;
}

Mesh<> copy_to_cpu(const Mesh<memory::space::gpu> & input) {
  Mesh<> output;
  output.vert = copy_to_cpu(input.vert);
  output.edge = copy_to_cpu(input.edge);
  output.tri = copy_to_cpu(input.tri);
  output.quad = copy_to_cpu(input.quad);
  output.tet = copy_to_cpu(input.tet);
  output.hex = copy_to_cpu(input.hex);
  output.X = copy_to_cpu(input.X);
  output.spatial_dimension = input.spatial_dimension;
  output.geometry_dimension = input.geometry_dimension;
  return output;
}

} // namespace

template < Family family >
nd::array<double,2,memory::space::gpu> nodes_for(
  const Field<family, memory::space::gpu> & u,
  const Mesh<memory::space::gpu> & mesh) {

  nd::array<double, 2, memory::space::cpu> output = nodes_for(metadata_on_cpu(u), copy_to_cpu(mesh));
  return nd::array<double, 2, memory::space::gpu>(output);
}

template nd::array<double,2,memory::space::gpu> nodes_for<Family::H1>(const Field<Family::H1, memory::space::gpu> &, const Mesh<memory::space::gpu> &);
template nd::array<double,2,memory::space::gpu> nodes_for<Family::Hcurl>(const Field<Family::Hcurl, memory::space::gpu> &, const Mesh<memory::space::gpu> &);
template nd::array<double,2,memory::space::gpu> nodes_for<Family::DG>(const Field<Family::DG, memory::space::gpu> &, const Mesh<memory::space::gpu> &);

template < Family family >
nd::array<double,2,memory::space::gpu> directions_for(
  const Field<family, memory::space::gpu> & u,
  const Mesh<memory::space::gpu> & mesh) {

  nd::array<double, 2, memory::space::cpu> output = directions_for(metadata_on_cpu(u), copy_to_cpu(mesh));
  return nd::array<double, 2, memory::space::gpu>(output);
}

template nd::array<double,2,memory::space::gpu> directions_for<Family::Hcurl>(const Field<Family::Hcurl, memory::space::gpu> &, const Mesh<memory::space::gpu> &);

} // namespace femto

#endif
