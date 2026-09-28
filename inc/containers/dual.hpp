#pragma once

#include <cmath>

#define FEMTO_DUAL

template <typename value_type, typename gradient_type>
struct dual {
  value_type    value;
  gradient_type gradient;
};

template <typename S, typename T>
dual(S, T) -> dual<S,T>;

/** @brief addition of a dual number and a non-dual number */
template <typename value_type, typename gradient_type>
constexpr auto operator+(dual<value_type, gradient_type> a, value_type b)
{
  return dual{a.value + b, a.gradient};
}

/** @brief addition of a dual number and a non-dual number */
template <typename value_type, typename gradient_type>
constexpr auto operator+(value_type a, dual<value_type, gradient_type> b)
{
  return dual{a + b.value, b.gradient};
}

/** @brief addition of two dual numbers */
template <typename VA, typename GA, typename VB, typename GB>
constexpr auto operator+(dual<VA, GA> a, dual<VB, GB> b)
{
  return dual{a.value + b.value, a.gradient + b.gradient};
}

/** @brief unary negation of a dual number */
template <typename value_type, typename gradient_type>
constexpr auto operator-(dual<value_type, gradient_type> x)
{
  return dual{-x.value, -x.gradient};
}

/** @brief subtraction of a non-dual number from a dual number */
template <typename value_type, typename gradient_type>
constexpr auto operator-(dual<value_type, gradient_type> a, value_type b)
{
  return dual{a.value - b, a.gradient};
}

/** @brief subtraction of a dual number from a non-dual number */
template <typename value_type, typename gradient_type>
constexpr auto operator-(value_type a, dual<value_type, gradient_type> b)
{
  return dual{a - b.value, -b.gradient};
}

/** @brief subtraction of two dual numbers */
template <typename VA, typename GA, typename VB, typename GB>
constexpr auto operator-(dual<VA, GA> a, dual<VB, GB> b)
{
  return dual{a.value - b.value, a.gradient - b.gradient};
}

/** @brief multiplication of a dual number and a non-dual number */
template <typename value_type, typename gradient_type>
constexpr auto operator*(const dual<value_type, gradient_type>& a, double b)
{
  return dual{a.value * b, a.gradient * b};
}

/** @brief multiplication of a dual number and a non-dual number */
template <typename value_type, typename gradient_type>
constexpr auto operator*(double a, const dual<value_type, gradient_type>& b)
{
  return dual{a * b.value, a * b.gradient};
}

/** @brief multiplication of two dual numbers */
template <typename VA, typename GA, typename VB, typename GB>
constexpr auto operator*(dual<VA, GA> a, dual<VB, GB> b)
{
  return dual{a.value * b.value, b.value * a.gradient + a.value * b.gradient};
}

/** @brief division of a dual number by a non-dual number */
template <typename value_type, typename gradient_type>
constexpr auto operator/(const dual<value_type, gradient_type>& a, double b)
{
  return dual{a.value / b, a.gradient / b};
}

/** @brief division of a non-dual number by a dual number */
template <typename value_type, typename gradient_type>
constexpr auto operator/(double a, const dual<value_type, gradient_type>& b)
{
  return dual{a / b.value, -(a / (b.value * b.value)) * b.gradient};
}

/** @brief division of two dual numbers */
template <typename VA, typename GA, typename VB, typename GB>
constexpr auto operator/(dual<VA, GA> a, dual<VB, GB> b)
{
  return dual{a.value / b.value, (a.gradient / b.value) - (a.value * b.gradient) / (b.value * b.value)};
}

/**
 * @brief Generates const + non-const overloads for a binary comparison operator
 * Comparisons are conducted against the "value" part of the dual number
 * @param[in] x The comparison operator to overload
 */
#define binary_comparator_overload(x)                                   \
  template <typename S, typename T>                                     \
  constexpr bool operator x(const dual<S, T>& a, double b) {            \
    return a.value x b;                                                 \
  }                                                                     \
                                                                        \
  template <typename S, typename T>                                     \
  constexpr bool operator x(double a, const dual<S, T>& b) {            \
    return a x b.value;                                                 \
  };                                                                    \
                                                                        \
  template <typename S, typename T, typename U>                         \
  constexpr bool operator x(const dual<S, T>& a, const dual<S, U>& b) { \
    return a.value x b.value;                                           \
  };

binary_comparator_overload(<);   ///< implement operator<  for dual numbers
binary_comparator_overload(<=);  ///< implement operator<= for dual numbers
binary_comparator_overload(==);  ///< implement operator== for dual numbers
binary_comparator_overload(>=);  ///< implement operator>= for dual numbers
binary_comparator_overload(>);   ///< implement operator>  for dual numbers

#undef binary_comparator_overload

/** @brief compound assignment (+) for dual numbers */
template <typename scalar_type, typename gradient_type>
constexpr auto& operator+=(dual<scalar_type, gradient_type>& a, const dual<scalar_type, gradient_type>& b)
{
  a.value += b.value;
  a.gradient += b.gradient;
  return a;
}

/** @brief compound assignment (-) for dual numbers */
template <typename scalar_type, typename gradient_type>
constexpr auto& operator-=(dual<scalar_type, gradient_type>& a, const dual<scalar_type, gradient_type>& b)
{
  a.value -= b.value;
  a.gradient -= b.gradient;
  return a;
}

/** @brief compound assignment (+) for dual numbers with `double` righthand side */
template <typename scalar_type, typename gradient_type>
constexpr auto& operator+=(dual<scalar_type, gradient_type>& a, double b)
{
  a.value += b;
  return a;
}

/** @brief compound assignment (-) for dual numbers with `double` righthand side */
template <typename scalar_type, typename gradient_type>
constexpr auto& operator-=(dual<scalar_type, gradient_type>& a, double b)
{
  a.value -= b;
  return a;
}


/** @brief implementation of cosine for dual numbers */
template <typename scalar_type, typename gradient_type>
auto sin(dual<scalar_type, gradient_type> a)
{
  using std::cos, std::sin;
  return dual<scalar_type, gradient_type>{sin(a.value), a.gradient * cos(a.value)};
}

/** @brief implementation of cosine for dual numbers */
template <typename scalar_type, typename gradient_type>
auto cos(dual<scalar_type, gradient_type> a)
{
  using std::cos, std::sin;
  return dual<scalar_type, gradient_type>{cos(a.value), -a.gradient * sin(a.value)};
}

/** @brief implementation of exponential function for dual numbers */
template <typename scalar_type, typename gradient_type>
auto exp(dual<scalar_type, gradient_type> a)
{
  using std::exp;
  return dual<scalar_type, gradient_type>{exp(a.value), exp(a.value) * a.gradient};
}

/** @brief implementation of the natural logarithm function for dual numbers */
template <typename scalar_type, typename gradient_type>
auto log(dual<scalar_type, gradient_type> a)
{
  using std::log;
  return dual<scalar_type, gradient_type>{log(a.value), a.gradient / a.value};
}

/** @brief implementation of absolute value function for dual numbers */
template <typename scalar_type, typename gradient_type>
auto abs(dual<scalar_type, gradient_type> x)
{
  return (x.value >= 0) ? x : -x;
}

/** @brief implementation of square root for dual numbers */
template <typename scalar_type, typename gradient_type>
auto sqrt(dual<scalar_type, gradient_type> x)
{
  using std::sqrt;
  return dual<scalar_type, gradient_type>{sqrt(x.value), x.gradient / (2.0 * sqrt(x.value))};
}

/** @brief implementation of `a` (dual) raised to the `b` (dual) power */
template <typename scalar_type, typename gradient_type>
auto pow(dual<scalar_type, gradient_type> a, dual<scalar_type, gradient_type> b)
{
  using std::pow, std::log;
  double value = pow(a.value, b.value);
  return dual<scalar_type, gradient_type>{value, value * (a.gradient * (b.value / a.value) + b.gradient * log(a.value))};
}

/** @brief implementation of `a` (non-dual) raised to the `b` (dual) power */
template <typename scalar_type, typename gradient_type>
auto pow(double a, dual<scalar_type, gradient_type> b)
{
  using std::pow, std::log;
  double value = pow(a, b.value);
  return dual<scalar_type, gradient_type>{value, value * b.gradient * log(a)};
}

/** @brief implementation of `a` (dual) raised to the `b` (non-dual) power */
template <typename scalar_type, typename gradient_type>
auto pow(dual<scalar_type, gradient_type> a, double b)
{
  using std::pow;
  double value = pow(a.value, b);
  return dual<scalar_type, gradient_type>{value, value * a.gradient * b / a.value};
}

/** @brief overload of operator<< for `dual` to work with `std::cout` and other `std::ostream`s */
template <typename scalar_type, typename gradient_type>
auto& operator<<(std::ostream& out, dual<scalar_type, gradient_type> A)
{
  out << '(' << A.value << ' ' << A.gradient << ')';
  return out;
}

/** @brief promote a value to a dual number of the appropriate type */
constexpr auto make_dual(double x) { return dual{x, 1.0}; }

/** @brief return the "value" part from a given type. For non-dual types, this is just the identity function */
template <typename T>
 auto get_value(const T& arg)
{
  return arg;
}

/** @brief return the "value" part from a dual number type */
template <typename scalar_type, typename gradient_type>
 auto get_value(dual<scalar_type, gradient_type> arg)
{
  return arg.value;
}

/** @brief return the "gradient" part from a dual number type */
template <typename scalar_type, typename gradient_type>
 auto get_gradient(dual<scalar_type, gradient_type> arg)
{
  return arg.gradient;
}

constexpr auto gradient_wrt(double x) { return dual{x, 1.0}; }

auto hessian_wrt(double x) {
  return dual{dual{x, 1.0}, dual{1.0, 0.0}};
}

#if defined FEMTO_VEC
  #include "linear_algebra/dual_vec.hpp"
#endif

#if defined FEMTO_MAT
  #include "linear_algebra/dual_mat.hpp"
#endif
