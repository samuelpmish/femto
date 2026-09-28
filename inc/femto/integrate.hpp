#pragma once

#include <functional>

#include "femto/domain.hpp"
#include "femto/type_traits.hpp"

#include "misc/for_constexpr.hpp"

namespace femto {

namespace impl {

template < typename return_type, typename T >
uint32_t dimension_of_first_argument(return_type (*f)(T)) {
  return dimension(T{});
}

template < typename return_type, typename T >
uint32_t dimension_of_first_argument(std::function< return_type(T) > integrand) {
  return dimension(T{});
}

////////////////////////////////////////////////////////////////////////////////

template < typename T, memory::space mem_space >
T integrate_ndarray(const nd::array< T, 1, mem_space > & integrand, const Domain<>& domain, const DomainType & type);

////////////////////////////////////////////////////////////////////////////////

template < typename return_type >
return_type integrate_function_pointer(return_type (*f)(vec2), const Domain<>& domain, const DomainType & type);

template < typename return_type >
return_type integrate_function_pointer(return_type (*f)(vec3), const Domain<>& domain, const DomainType & type);

////////////////////////////////////////////////////////////////////////////////

template < typename return_type >
return_type integrate_stdfunction(std::function< return_type(vec2) > integrand, const Domain<>& domain, const DomainType & type);

template < typename return_type >
return_type integrate_stdfunction(std::function< return_type(vec3) > integrand, const Domain<>& domain, const DomainType & type);

////////////////////////////////////////////////////////////////////////////////

template < Family family, DerivedQuantity op, memory::space mem_space >
void integrate_residual(Residual<family, mem_space> & r, FunctionSpace space, const nd::view<const double, 3> f, const Domain<> & domain, const DomainType type);

#ifdef NDARRAY_ENABLE_CUDA
template < Family family, DerivedQuantity op >
void integrate_residual(
  Residual<family, memory::space::gpu> & r,
  FunctionSpace space,
  const nd::view<const double, 3, memory::space::gpu> f,
  const Domain<memory::space::gpu> & domain,
  const DomainType type);
#endif

// like the sparse matrix overload below, this returns a writer instead of a
// Residual, so `r = integrate(...)` refills r rather than allocating a new dof
// vector per call.  integrand is a two-word handle ({BasisFunctionOp, const
// array &}) and is captured by value; domain is captured by reference, so the
// writer must be run in the full-expression that created it.
template < uint32_t n, memory::space mem_space, memory::space domain_space, DerivedQuantity dq, Family f >
typename Residual<f, mem_space>::writer integrate_weighted_integrand(
  const WeightedIntegrand< BasisFunctionOp<dq, f>, nd::array<double, n, mem_space>, void > & integrand,
  const Domain<domain_space> & domain,
  const DomainType type) {

  static_assert(
    (domain_space == memory::space::cpu && mem_space != memory::space::gpu) ||
    (domain_space == memory::space::gpu && mem_space == memory::space::gpu),
    "integrand data and domain memory spaces are incompatible"
  );

  return typename Residual<f, mem_space>::writer([integrand, &domain, type](Residual<f, mem_space> & r) {

    FunctionSpace space = integrand.test.function.space;
    r.reset(space, domain.mesh);

    uint32_t gdim = domain.geometry_dimension;
    const nd::array<double, n, mem_space> & qdata = integrand.qdata;

    stack::array<uint32_t, 3> shape3D{
      qdata.shape[0],
      space.components,
      qshape(f, dq, gdim)
    };

    FEMTO_ASSERT(compatible_shapes(qdata.shape, shape3D), "incompatible array shapes");

    constexpr memory::space view_space = domain_space == memory::space::gpu ? memory::space::gpu : memory::space::cpu;
    nd::view<const double, 3, view_space> q3D{qdata.data(), shape3D};
    integrate_residual<f, dq>(r, space, q3D, domain, type);

  });
}

////////////////////////////////////////////////////////////////////////////////

template < Family test_family, DerivedQuantity test_op, Family trial_family, DerivedQuantity trial_op >
std::function< void(femto::sparse_matrix<>&) > integrate_sparse_matrix(FunctionSpace phi, const nd::view<const double, 5> f, FunctionSpace psi, const Domain<> &domain, const DomainType type);

#ifdef NDARRAY_ENABLE_CUDA
template < Family test_family, DerivedQuantity test_op, Family trial_family, DerivedQuantity trial_op >
std::function< void(femto::sparse_matrix<memory::space::gpu>&) > integrate_sparse_matrix(
  FunctionSpace phi,
  const nd::view<const double, 5, memory::space::gpu> f,
  FunctionSpace psi,
  const Domain<memory::space::gpu> &domain,
  const DomainType type);
#endif

} // namespace impl

// element stiffness matrices, the product of element_integrate() below:
// K(e, ci * NC + cj, I * trial_nodes + J) holds the (ci, cj) component block
// of test node I against trial node J on element e (component-major,
// pair-minor, so the assembling pass reads contiguous runs), and
// test_ids/trial_ids record each element's dof node ids for that assembly
template < memory::space mem_space >
struct ElementMatrices {
  nd::array<double, 3, mem_space> K;
  nd::array<uint32_t, 2, mem_space> test_ids;
  nd::array<uint32_t, 2, mem_space> trial_ids;
};

namespace impl {

#ifdef NDARRAY_ENABLE_CUDA
template < Family test_family, DerivedQuantity test_op, Family trial_family, DerivedQuantity trial_op >
ElementMatrices<memory::space::gpu> element_matrices(
  FunctionSpace phi,
  const nd::view<const double, 5, memory::space::gpu> qdata,
  FunctionSpace psi,
  const Domain<memory::space::gpu> & domain,
  const DomainType type);

template < Family test_family, DerivedQuantity test_op, Family trial_family, DerivedQuantity trial_op >
void element_matrices_into(
  FunctionSpace phi,
  const nd::view<const double, 5, memory::space::gpu> qdata,
  FunctionSpace psi,
  const Domain<memory::space::gpu> & domain,
  const DomainType type,
  nd::view<double, 3, memory::space::gpu> K,
  nd::view<uint32_t, 2, memory::space::gpu> test_ids,
  nd::view<uint32_t, 2, memory::space::gpu> trial_ids);
#endif

template < uint32_t n, memory::space mem_space, DerivedQuantity test_dq, Family test_f, DerivedQuantity trial_dq, Family trial_f >
std::function< void(femto::sparse_matrix<mem_space>&) > integrate_weighted_integrand(
  const WeightedIntegrand< BasisFunctionOp<test_dq, test_f>, nd::array<double, n, mem_space>, BasisFunctionOp<trial_dq, trial_f> > & integrand,
  const Domain<> & domain,
  const DomainType type) {

  FunctionSpace test_space = integrand.test.function.space;
  FunctionSpace trial_space = integrand.trial.function.space;
  const nd::array<double, n, mem_space> & qdata = integrand.qdata;
  uint32_t gdim = domain.geometry_dimension;

  stack::array<uint32_t, 5> shape5D = {
    qdata.shape[0],
    test_space.components, qshape(test_f, test_dq, gdim),
    trial_space.components, qshape(trial_f, trial_dq, gdim)
  };

  FEMTO_ASSERT(compatible_shapes(qdata.shape, shape5D), "incompatible array shapes");

  nd::view<const double, 5> q5D{qdata.data(), shape5D};
  return integrate_sparse_matrix< test_f, test_dq, trial_f, trial_dq >(test_space, q5D, trial_space, domain, type);

}

#ifdef NDARRAY_ENABLE_CUDA
template < uint32_t n, DerivedQuantity test_dq, Family test_f, DerivedQuantity trial_dq, Family trial_f >
std::function< void(femto::sparse_matrix<memory::space::gpu>&) > integrate_weighted_integrand(
  const WeightedIntegrand< BasisFunctionOp<test_dq, test_f>, nd::array<double, n, memory::space::gpu>, BasisFunctionOp<trial_dq, trial_f> > & integrand,
  const Domain<memory::space::gpu> & domain,
  const DomainType type) {

  FunctionSpace test_space = integrand.test.function.space;
  FunctionSpace trial_space = integrand.trial.function.space;
  const nd::array<double, n, memory::space::gpu> & qdata = integrand.qdata;
  uint32_t gdim = domain.geometry_dimension;

  stack::array<uint32_t, 5> shape5D = {
    qdata.shape[0],
    test_space.components, qshape(test_f, test_dq, gdim),
    trial_space.components, qshape(trial_f, trial_dq, gdim)
  };

  FEMTO_ASSERT(compatible_shapes(qdata.shape, shape5D), "incompatible array shapes");

  nd::view<const double, 5, memory::space::gpu> q5D{qdata.data(), shape5D};
  return integrate_sparse_matrix< test_f, test_dq, trial_f, trial_dq >(test_space, q5D, trial_space, domain, type);

}
#endif

////////////////////////////////////////////////////////////////////////////////

template < Family family, DerivedQuantity test_op, DerivedQuantity trial_op, memory::space mem_space >
void integrate_sparse_matrix_diagonal(Residual<family, mem_space> & r, FunctionSpace phi, const nd::view<const double, 5> f, FunctionSpace psi, const Domain<> &domain, const DomainType type);

template < uint32_t n, memory::space mem_space, DerivedQuantity test_dq, Family f, DerivedQuantity trial_dq >
nd::array<double,2,mem_space> integrate_weighted_integrand(
  const DiagonalOnly < WeightedIntegrand< BasisFunctionOp<test_dq, f>, nd::array<double, n, mem_space>, BasisFunctionOp<trial_dq, f> > > & integrand,
  const Domain<> & domain,
  const DomainType type) {

  FunctionSpace test_space = integrand.test.function.space;
  FunctionSpace trial_space = integrand.trial.function.space;
  const nd::array<double, n, mem_space> & qdata = integrand.qdata;

  FEMTO_ASSERT(test_space == trial_space, "must have matching test and trial spaces for diag(...)");

  uint32_t gdim = domain.geometry_dimension;

  stack::array<uint32_t, 5> shape5D = {
    qdata.shape[0],
    test_space.components, qshape(f, test_dq, gdim),
    trial_space.components, qshape(f, trial_dq, gdim)
  };

  FEMTO_ASSERT(compatible_shapes(qdata.shape, shape5D), "incompatible array shapes");

  Residual<f, mem_space> output(test_space, domain.mesh);

  nd::view<const double,5> q5D{qdata.data(), shape5D};
  integrate_sparse_matrix_diagonal< f, test_dq, trial_dq, mem_space >(output, test_space, q5D, trial_space, domain, type);

  return output.data;

}

} // namespace impl

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template < typename integrand_t >
auto integrate(const integrand_t & integrand, const DomainWithType<> & d) {

  if constexpr (is_a_1D_ndarray<integrand_t>::value) {
    return impl::integrate_ndarray(integrand, d.domain, d.type);
  } 

  if constexpr (is_a_function_pointer<integrand_t>::value) {
    uint32_t sdim = impl::dimension_of_first_argument(integrand);
    FEMTO_ASSERT(d.domain.mesh.X.data.shape[1] == sdim, "integration function has wrong signature");
    return impl::integrate_function_pointer(integrand, d.domain, d.type);
  } 

  if constexpr (is_a_stdfunction<integrand_t>::value) {
    uint32_t sdim = impl::dimension_of_first_argument(integrand);
    FEMTO_ASSERT(d.domain.mesh.X.data.shape[1] == sdim, "integration function has wrong signature");
    return impl::integrate_stdfunction(integrand, d.domain, d.type);
  } 

  if constexpr (is_a_weighted_integrand<integrand_t>::value) {
    return impl::integrate_weighted_integrand(integrand, d.domain, d.type);
  }

  static_assert(always_true<integrand_t>::value, "error: unsupported integrand type");

}

#ifdef NDARRAY_ENABLE_CUDA
template < typename integrand_t >
auto integrate(const integrand_t & integrand, const DomainWithType<memory::space::gpu> & d) {

  if constexpr (is_a_weighted_integrand<integrand_t>::value) {
    return impl::integrate_weighted_integrand(integrand, d.domain, d.type);
  }

  static_assert(always_true<integrand_t>::value, "error: unsupported integrand type for GPU domain");

}

// like the sparse matrix form of integrate(), but returns the element
// stiffness matrices instead of scatter-adding into a CSR matrix: no row
// search, no atomics, bitwise reproducible.  See ElementMatrices for the
// layout.  Isoparametric domains only, for now.
template < uint32_t n, DerivedQuantity test_dq, Family test_f, DerivedQuantity trial_dq, Family trial_f >
ElementMatrices<memory::space::gpu> element_integrate(
  const WeightedIntegrand< BasisFunctionOp<test_dq, test_f>, nd::array<double, n, memory::space::gpu>, BasisFunctionOp<trial_dq, trial_f> > & integrand,
  const DomainWithType<memory::space::gpu> & d) {

  FunctionSpace test_space = integrand.test.function.space;
  FunctionSpace trial_space = integrand.trial.function.space;
  const nd::array<double, n, memory::space::gpu> & qdata = integrand.qdata;
  uint32_t gdim = d.domain.geometry_dimension;

  stack::array<uint32_t, 5> shape5D = {
    qdata.shape[0],
    test_space.components, qshape(test_f, test_dq, gdim),
    trial_space.components, qshape(trial_f, trial_dq, gdim)
  };

  FEMTO_ASSERT(compatible_shapes(qdata.shape, shape5D), "incompatible array shapes");

  nd::view<const double, 5, memory::space::gpu> q5D{qdata.data(), shape5D};
  return impl::element_matrices< test_f, test_dq, trial_f, trial_dq >(test_space, q5D, trial_space, d.domain, d.type);

}
#endif

}
