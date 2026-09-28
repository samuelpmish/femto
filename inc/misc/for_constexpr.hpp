#pragma once

#include <utility>      // for std::integer_sequence

namespace impl {
  template < auto x >
  struct value{
    constexpr operator decltype(x)() { return x; }
  };

  template <int... i, typename lambda>
  constexpr void for_constexpr(lambda&& f, std::integer_sequence<int, i...>) {
      (f(value<i>{}), ...);
  }
}

template < auto ... args, typename T >
void foreach_constexpr(T && function) {
  (function(impl::value<args>{}), ...);
}

template <int n, typename lambda>
constexpr void for_constexpr(lambda&& f) {
  impl::for_constexpr(f, std::make_integer_sequence<int, n>{});
}

