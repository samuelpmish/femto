#include "integrate_spmat.hpp"

namespace femto {

namespace impl {

template std::function< void(femto::sparse_matrix<>&) > integrate_sparse_matrix<Family::DG, DerivedQuantity::VALUE, Family::H1, DerivedQuantity::VALUE>(FunctionSpace, const nd::view<const double, 5>, FunctionSpace, const Domain<> &, const DomainType);
template std::function< void(femto::sparse_matrix<>&) > integrate_sparse_matrix<Family::DG, DerivedQuantity::VALUE, Family::H1, DerivedQuantity::GRAD>(FunctionSpace, const nd::view<const double, 5>, FunctionSpace, const Domain<> &, const DomainType);
template std::function< void(femto::sparse_matrix<>&) > integrate_sparse_matrix<Family::DG, DerivedQuantity::VALUE, Family::Hcurl, DerivedQuantity::VALUE>(FunctionSpace, const nd::view<const double, 5>, FunctionSpace, const Domain<> &, const DomainType);
template std::function< void(femto::sparse_matrix<>&) > integrate_sparse_matrix<Family::DG, DerivedQuantity::VALUE, Family::Hcurl, DerivedQuantity::CURL>(FunctionSpace, const nd::view<const double, 5>, FunctionSpace, const Domain<> &, const DomainType);
template std::function< void(femto::sparse_matrix<>&) > integrate_sparse_matrix<Family::DG, DerivedQuantity::VALUE, Family::DG, DerivedQuantity::VALUE>(FunctionSpace, const nd::view<const double, 5>, FunctionSpace, const Domain<> &, const DomainType);
template std::function< void(femto::sparse_matrix<>&) > integrate_sparse_matrix<Family::DG, DerivedQuantity::VALUE, Family::DG, DerivedQuantity::GRAD>(FunctionSpace, const nd::view<const double, 5>, FunctionSpace, const Domain<> &, const DomainType);
template std::function< void(femto::sparse_matrix<>&) > integrate_sparse_matrix<Family::DG, DerivedQuantity::GRAD, Family::H1, DerivedQuantity::VALUE>(FunctionSpace, const nd::view<const double, 5>, FunctionSpace, const Domain<> &, const DomainType);
template std::function< void(femto::sparse_matrix<>&) > integrate_sparse_matrix<Family::DG, DerivedQuantity::GRAD, Family::H1, DerivedQuantity::GRAD>(FunctionSpace, const nd::view<const double, 5>, FunctionSpace, const Domain<> &, const DomainType);
template std::function< void(femto::sparse_matrix<>&) > integrate_sparse_matrix<Family::DG, DerivedQuantity::GRAD, Family::Hcurl, DerivedQuantity::VALUE>(FunctionSpace, const nd::view<const double, 5>, FunctionSpace, const Domain<> &, const DomainType);
template std::function< void(femto::sparse_matrix<>&) > integrate_sparse_matrix<Family::DG, DerivedQuantity::GRAD, Family::Hcurl, DerivedQuantity::CURL>(FunctionSpace, const nd::view<const double, 5>, FunctionSpace, const Domain<> &, const DomainType);
template std::function< void(femto::sparse_matrix<>&) > integrate_sparse_matrix<Family::DG, DerivedQuantity::GRAD, Family::DG, DerivedQuantity::VALUE>(FunctionSpace, const nd::view<const double, 5>, FunctionSpace, const Domain<> &, const DomainType);
template std::function< void(femto::sparse_matrix<>&) > integrate_sparse_matrix<Family::DG, DerivedQuantity::GRAD, Family::DG, DerivedQuantity::GRAD>(FunctionSpace, const nd::view<const double, 5>, FunctionSpace, const Domain<> &, const DomainType);

} // namespace impl

} // namespace femto
