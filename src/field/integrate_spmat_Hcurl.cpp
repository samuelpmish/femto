#include "integrate_spmat.hpp"

namespace femto {

namespace impl {

template std::function< void(femto::sparse_matrix<>&) > integrate_sparse_matrix<Family::Hcurl, DerivedQuantity::VALUE, Family::H1, DerivedQuantity::VALUE>(FunctionSpace, const nd::view<const double, 5>, FunctionSpace, const Domain<> &, const DomainType);
template std::function< void(femto::sparse_matrix<>&) > integrate_sparse_matrix<Family::Hcurl, DerivedQuantity::VALUE, Family::H1, DerivedQuantity::GRAD>(FunctionSpace, const nd::view<const double, 5>, FunctionSpace, const Domain<> &, const DomainType);
template std::function< void(femto::sparse_matrix<>&) > integrate_sparse_matrix<Family::Hcurl, DerivedQuantity::VALUE, Family::Hcurl, DerivedQuantity::VALUE>(FunctionSpace, const nd::view<const double, 5>, FunctionSpace, const Domain<> &, const DomainType);
template std::function< void(femto::sparse_matrix<>&) > integrate_sparse_matrix<Family::Hcurl, DerivedQuantity::VALUE, Family::Hcurl, DerivedQuantity::CURL>(FunctionSpace, const nd::view<const double, 5>, FunctionSpace, const Domain<> &, const DomainType);
template std::function< void(femto::sparse_matrix<>&) > integrate_sparse_matrix<Family::Hcurl, DerivedQuantity::VALUE, Family::DG, DerivedQuantity::VALUE>(FunctionSpace, const nd::view<const double, 5>, FunctionSpace, const Domain<> &, const DomainType);
template std::function< void(femto::sparse_matrix<>&) > integrate_sparse_matrix<Family::Hcurl, DerivedQuantity::VALUE, Family::DG, DerivedQuantity::GRAD>(FunctionSpace, const nd::view<const double, 5>, FunctionSpace, const Domain<> &, const DomainType);
template std::function< void(femto::sparse_matrix<>&) > integrate_sparse_matrix<Family::Hcurl, DerivedQuantity::CURL, Family::H1, DerivedQuantity::VALUE>(FunctionSpace, const nd::view<const double, 5>, FunctionSpace, const Domain<> &, const DomainType);
template std::function< void(femto::sparse_matrix<>&) > integrate_sparse_matrix<Family::Hcurl, DerivedQuantity::CURL, Family::H1, DerivedQuantity::GRAD>(FunctionSpace, const nd::view<const double, 5>, FunctionSpace, const Domain<> &, const DomainType);
template std::function< void(femto::sparse_matrix<>&) > integrate_sparse_matrix<Family::Hcurl, DerivedQuantity::CURL, Family::Hcurl, DerivedQuantity::VALUE>(FunctionSpace, const nd::view<const double, 5>, FunctionSpace, const Domain<> &, const DomainType);
template std::function< void(femto::sparse_matrix<>&) > integrate_sparse_matrix<Family::Hcurl, DerivedQuantity::CURL, Family::Hcurl, DerivedQuantity::CURL>(FunctionSpace, const nd::view<const double, 5>, FunctionSpace, const Domain<> &, const DomainType);
template std::function< void(femto::sparse_matrix<>&) > integrate_sparse_matrix<Family::Hcurl, DerivedQuantity::CURL, Family::DG, DerivedQuantity::VALUE>(FunctionSpace, const nd::view<const double, 5>, FunctionSpace, const Domain<> &, const DomainType);
template std::function< void(femto::sparse_matrix<>&) > integrate_sparse_matrix<Family::Hcurl, DerivedQuantity::CURL, Family::DG, DerivedQuantity::GRAD>(FunctionSpace, const nd::view<const double, 5>, FunctionSpace, const Domain<> &, const DomainType);

} // namespace impl

} // namespace femto
