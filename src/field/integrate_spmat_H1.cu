#include "integrate_spmat.cuh"

namespace femto {

namespace impl {

template std::function< void(femto::sparse_matrix<memory::space::gpu>&) > integrate_sparse_matrix<Family::H1, DerivedQuantity::VALUE, Family::H1, DerivedQuantity::VALUE>(FunctionSpace, const nd::view<const double, 5, memory::space::gpu>, FunctionSpace, const Domain<memory::space::gpu> &, const DomainType);
template std::function< void(femto::sparse_matrix<memory::space::gpu>&) > integrate_sparse_matrix<Family::H1, DerivedQuantity::VALUE, Family::H1, DerivedQuantity::GRAD>(FunctionSpace, const nd::view<const double, 5, memory::space::gpu>, FunctionSpace, const Domain<memory::space::gpu> &, const DomainType);
template std::function< void(femto::sparse_matrix<memory::space::gpu>&) > integrate_sparse_matrix<Family::H1, DerivedQuantity::VALUE, Family::Hcurl, DerivedQuantity::VALUE>(FunctionSpace, const nd::view<const double, 5, memory::space::gpu>, FunctionSpace, const Domain<memory::space::gpu> &, const DomainType);
template std::function< void(femto::sparse_matrix<memory::space::gpu>&) > integrate_sparse_matrix<Family::H1, DerivedQuantity::VALUE, Family::Hcurl, DerivedQuantity::CURL>(FunctionSpace, const nd::view<const double, 5, memory::space::gpu>, FunctionSpace, const Domain<memory::space::gpu> &, const DomainType);
template std::function< void(femto::sparse_matrix<memory::space::gpu>&) > integrate_sparse_matrix<Family::H1, DerivedQuantity::VALUE, Family::DG, DerivedQuantity::VALUE>(FunctionSpace, const nd::view<const double, 5, memory::space::gpu>, FunctionSpace, const Domain<memory::space::gpu> &, const DomainType);
template std::function< void(femto::sparse_matrix<memory::space::gpu>&) > integrate_sparse_matrix<Family::H1, DerivedQuantity::VALUE, Family::DG, DerivedQuantity::GRAD>(FunctionSpace, const nd::view<const double, 5, memory::space::gpu>, FunctionSpace, const Domain<memory::space::gpu> &, const DomainType);
template std::function< void(femto::sparse_matrix<memory::space::gpu>&) > integrate_sparse_matrix<Family::H1, DerivedQuantity::GRAD, Family::H1, DerivedQuantity::VALUE>(FunctionSpace, const nd::view<const double, 5, memory::space::gpu>, FunctionSpace, const Domain<memory::space::gpu> &, const DomainType);
template std::function< void(femto::sparse_matrix<memory::space::gpu>&) > integrate_sparse_matrix<Family::H1, DerivedQuantity::GRAD, Family::H1, DerivedQuantity::GRAD>(FunctionSpace, const nd::view<const double, 5, memory::space::gpu>, FunctionSpace, const Domain<memory::space::gpu> &, const DomainType);
template std::function< void(femto::sparse_matrix<memory::space::gpu>&) > integrate_sparse_matrix<Family::H1, DerivedQuantity::GRAD, Family::Hcurl, DerivedQuantity::VALUE>(FunctionSpace, const nd::view<const double, 5, memory::space::gpu>, FunctionSpace, const Domain<memory::space::gpu> &, const DomainType);
template std::function< void(femto::sparse_matrix<memory::space::gpu>&) > integrate_sparse_matrix<Family::H1, DerivedQuantity::GRAD, Family::Hcurl, DerivedQuantity::CURL>(FunctionSpace, const nd::view<const double, 5, memory::space::gpu>, FunctionSpace, const Domain<memory::space::gpu> &, const DomainType);
template std::function< void(femto::sparse_matrix<memory::space::gpu>&) > integrate_sparse_matrix<Family::H1, DerivedQuantity::GRAD, Family::DG, DerivedQuantity::VALUE>(FunctionSpace, const nd::view<const double, 5, memory::space::gpu>, FunctionSpace, const Domain<memory::space::gpu> &, const DomainType);
template std::function< void(femto::sparse_matrix<memory::space::gpu>&) > integrate_sparse_matrix<Family::H1, DerivedQuantity::GRAD, Family::DG, DerivedQuantity::GRAD>(FunctionSpace, const nd::view<const double, 5, memory::space::gpu>, FunctionSpace, const Domain<memory::space::gpu> &, const DomainType);

} // namespace impl

} // namespace femto
