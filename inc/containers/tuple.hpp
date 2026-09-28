#pragma once

#include <utility>
#include <iostream>

#include "containers/dual.hpp"

namespace femto {

template <int i>
struct Index {};

namespace index_aliases {
  constexpr Index<0> _0;
  constexpr Index<1> _1;
  constexpr Index<2> _2;
  constexpr Index<3> _3;
  constexpr Index<4> _4;
  constexpr Index<5> _5;
}  // namespace IndexAliases

template <int i, typename T>
struct value_at_position {
  constexpr T& operator[](Index<i>) & { return value; }
  constexpr const T& operator[](Index<i>) const& { return value; }
  constexpr T&& operator[](Index<i>) && { return std::move(value); }
  constexpr const T&& operator[](Index<i>) const&& { return std::move(value); }
  constexpr static T type_of(Index<i>){ return {}; }
  T value;
};

template <typename S, typename... T>
struct tuple_base;

template <int... i, typename... T>
struct tuple_base<std::integer_sequence<int, i...>, T...>
    : public value_at_position<i, T>... {
  using value_at_position<i, T>::operator[]...;
  using value_at_position<i, T>::type_of...;
};

#define FEMTO_TUPLE

template <typename... T>
struct tuple : public tuple_base<std::make_integer_sequence<int, sizeof...(T)>, T...> {};

template < typename ... T >
tuple(T ... ) -> tuple < T ... >;

template < typename T >
constexpr bool is_tuple(T) { return false; }

template < typename ... T >
constexpr bool is_tuple(tuple < T ...  >) { return true; }

template < int i, typename ... T>
constexpr auto & get(tuple< T... > & x) { return x[Index<i>{}]; }

template < int i, typename ... T>
constexpr const auto & get(const tuple< T... > & x) { return x[Index<i>{}]; }

template <int i, typename... T>
constexpr auto&& get(tuple<T...>&& x) { return std::move(x)[Index<i>{}]; }

template <int i, typename... T>
constexpr const auto&& get(const tuple<T...>&& x) { return std::move(x)[Index<i>{}]; }

// elementwise operators: +, -, *, /
template <typename... S, typename... T, int... i>
constexpr auto plus_helper(const tuple<S...>& x, const tuple<T...>& y, std::integer_sequence<int, i...>) {
  return tuple{get<i>(x) + get<i>(y)...};
}
template <typename... S, typename... T>
constexpr auto operator+(const tuple<S...>& x, const tuple<T...>& y) {
  static_assert(sizeof...(S) == sizeof...(T));
  return plus_helper(x, y, std::make_integer_sequence<int, static_cast<int>(sizeof...(S))>());
}

template <typename... S, typename... T, int... i>
constexpr auto minus_helper(const tuple<S...>& x, const tuple<T...>& y, std::integer_sequence<int, i...>) {
  return tuple{get<i>(x) - get<i>(y)...};
}
template <typename... S, typename... T>
constexpr auto operator-(const tuple<S...>& x, const tuple<T...>& y) {
  static_assert(sizeof...(S) == sizeof...(T));
  return minus_helper(x, y, std::make_integer_sequence<int, static_cast<int>(sizeof...(S))>());
}

template <typename... S, typename... T, int... i>
constexpr auto multiply_helper(const tuple<S...>& x, const tuple<T...>& y, std::integer_sequence<int, i...>) {
  return tuple{get<i>(x) * get<i>(y)...};
}
template <typename... S, typename... T>
constexpr auto operator*(const tuple<S...>& x, const tuple<T...>& y) {
  static_assert(sizeof...(S) == sizeof...(T));
  return multiply_helper(x, y, std::make_integer_sequence<int, static_cast<int>(sizeof...(S))>());
}

template <typename... S, typename... T, int... i>
constexpr auto divide_helper(const tuple<S...>& x, const tuple<T...>& y, std::integer_sequence<int, i...>) {
  return tuple{get<i>(x) / get<i>(y)...};
}
template <typename... S, typename... T>
constexpr auto operator/(const tuple<S...>& x, const tuple<T...>& y) {
  static_assert(sizeof...(S) == sizeof...(T));
  return divide_helper(x, y, std::make_integer_sequence<int, static_cast<int>(sizeof...(S))>());
}


// multiplication / division with scalars
template <typename... S, int... i>
constexpr auto multiply_helper(const tuple<S...>& x, const double & y, std::integer_sequence<int, i...>) {
  return tuple{(get<i>(x) * y)...};
}
template <typename... S >
constexpr auto operator*(const tuple<S...>& x, double y) {
  return multiply_helper(x, y, std::make_integer_sequence<int, static_cast<int>(sizeof...(S))>());
}
template <typename... S >
constexpr auto operator*(double y, const tuple<S...>& x) {
  return multiply_helper(x, y, std::make_integer_sequence<int, static_cast<int>(sizeof...(S))>());
}

template <typename... S, int... i>
constexpr auto divide_helper(const tuple<S...>& x, const double & y, std::integer_sequence<int, i...>) {
  return tuple{(get<i>(x) / y)...};
}
template <typename... S, int... i>
constexpr auto divide_helper(const double & x, const tuple<S...>& y, std::integer_sequence<int, i...>) {
  return tuple{(x / get<i>(y))...};
}

template <typename... S >
constexpr auto operator/(const tuple<S...>& x, double y) {
  return divide_helper(x, y, std::make_integer_sequence<int, static_cast<int>(sizeof...(S))>());
}
template <typename... S >
constexpr auto operator/(double x, const tuple<S...>& y) {
  return divide_helper(x, y, std::make_integer_sequence<int, static_cast<int>(sizeof...(S))>());
}

//template <typename... S, typename T, int... i>
//constexpr auto divide_helper(const tuple<S...>& x, const dual<T>& y, std::integer_sequence<int, i...>) {
//  return tuple{(get<i>(x) / y)...};
//}
//
//template <typename... S, typename T>
//constexpr auto operator/(const tuple<S...>& x, const dual<T>& y) {
//  return divide_helper(x, y, std::make_integer_sequence<int, static_cast<int>(sizeof...(S))>());
//}

template <typename... T, std::size_t... i>
auto& print_helper(std::ostream& out, const tuple<T...>& A, std::integer_sequence<size_t, i...>) {
  out << "tuple{"; (..., (out << (i == 0 ? "" : ", ") << get<i>(A))); out << "}";
  return out;
}

template <typename... T>
auto& operator<<(std::ostream& out, const tuple<T...>& A) {
  return print_helper(out, A, std::make_integer_sequence<size_t, sizeof...(T)>());
}

}  // namespace femto

namespace std {
  template <typename ... T> struct tuple_size< ::femto::tuple<T ... > > : std::integral_constant<size_t, sizeof ... (T)> { };

//  template <size_t I, typename Head, typename ... Tail>
//  struct tuple_element<I, ::femto::tuple<Head, Tail...> > : tuple_element<I - 1, tuple<Tail...>> {
//  };
//  
//  template <typename Head, typename... Tail>
//  struct tuple_element<0, ::femto::tuple<Head, Tail...>> {
//    using type = Head;  ///< the type at the specified index
//  };

  template <size_t I, typename ... T>
  struct tuple_element<I, ::femto::tuple< T ... > > {
    using type = decltype(::femto::tuple<T ... >{}.type_of(::femto::Index<I>{}));
  };

}

#if defined FEMTO_DUAL

#endif