#pragma once

#include "femto/domain.hpp"

#ifdef FEMTO_ENABLE_CUDA

namespace femto {

namespace impl {

namespace stiffness_cuda {

bool can_integrate_h1_tet_spmat(
  BasisFunctionOp<DerivedQuantity::GRAD, Family::H1> test,
  BasisFunctionOp<DerivedQuantity::GRAD, Family::H1> trial,
  const Domain<memory::space::gpu> & domain);

void integrate_h1_tet_spmat(
  sparse_matrix<memory::space::gpu> & A,
  BasisFunctionOp<DerivedQuantity::GRAD, Family::H1> test,
  nd::view<const double, 5, memory::space::gpu> qdata,
  BasisFunctionOp<DerivedQuantity::GRAD, Family::H1> trial,
  const Domain<memory::space::gpu> & domain,
  DomainType type);

void integrate_h1_tet_emat(
  nd::array<double, 5, memory::space::gpu> & element_matrices,
  BasisFunctionOp<DerivedQuantity::GRAD, Family::H1> test,
  nd::view<const double, 5, memory::space::gpu> qdata,
  BasisFunctionOp<DerivedQuantity::GRAD, Family::H1> trial,
  const Domain<memory::space::gpu> & domain,
  DomainType type);

} // namespace stiffness_cuda

} // namespace impl

} // namespace femto

#endif
