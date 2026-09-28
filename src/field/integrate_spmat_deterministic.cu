#include "integrate_spmat_deterministic.cuh"

namespace femto {

namespace impl {

#define FEMTO_INSTANTIATE_ELEMENT_MATRICES(TF, TOP, UF, UOP) \
  template ElementMatrices<memory::space::gpu> element_matrices<TF, TOP, UF, UOP>( \
    FunctionSpace, const nd::view<const double, 5, memory::space::gpu>, FunctionSpace, \
    const Domain<memory::space::gpu> &, const DomainType); \
  template void element_matrices_into<TF, TOP, UF, UOP>( \
    FunctionSpace, const nd::view<const double, 5, memory::space::gpu>, FunctionSpace, \
    const Domain<memory::space::gpu> &, const DomainType, \
    nd::view<double, 3, memory::space::gpu>, \
    nd::view<uint32_t, 2, memory::space::gpu>, \
    nd::view<uint32_t, 2, memory::space::gpu>);

FEMTO_INSTANTIATE_ELEMENT_MATRICES(Family::H1, DerivedQuantity::VALUE, Family::H1, DerivedQuantity::VALUE)
FEMTO_INSTANTIATE_ELEMENT_MATRICES(Family::H1, DerivedQuantity::VALUE, Family::H1, DerivedQuantity::GRAD)
FEMTO_INSTANTIATE_ELEMENT_MATRICES(Family::H1, DerivedQuantity::GRAD, Family::H1, DerivedQuantity::VALUE)
FEMTO_INSTANTIATE_ELEMENT_MATRICES(Family::H1, DerivedQuantity::GRAD, Family::H1, DerivedQuantity::GRAD)
FEMTO_INSTANTIATE_ELEMENT_MATRICES(Family::Hcurl, DerivedQuantity::VALUE, Family::Hcurl, DerivedQuantity::VALUE)
FEMTO_INSTANTIATE_ELEMENT_MATRICES(Family::Hcurl, DerivedQuantity::VALUE, Family::Hcurl, DerivedQuantity::CURL)
FEMTO_INSTANTIATE_ELEMENT_MATRICES(Family::Hcurl, DerivedQuantity::CURL, Family::Hcurl, DerivedQuantity::VALUE)
FEMTO_INSTANTIATE_ELEMENT_MATRICES(Family::Hcurl, DerivedQuantity::CURL, Family::Hcurl, DerivedQuantity::CURL)

#undef FEMTO_INSTANTIATE_ELEMENT_MATRICES

} // namespace impl

} // namespace femto
