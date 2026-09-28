#pragma once

#include "tuple.hpp"

int enzyme_dup;
int enzyme_dupnoneed;
int enzyme_out;
int enzyme_const;

template < typename return_type, typename ... T >
extern return_type __enzyme_fwddiff(void*, T ... );

template < typename return_type, typename ... T >
extern return_type __enzyme_autodiff(void*, T ... );

namespace femto {

template < auto function, typename T > 
auto jvp(T arg) {
  using output_type = decltype(function(arg));
  return [=](T darg) {
    return __enzyme_fwddiff<output_type>((void*)function, enzyme_dup, arg, darg);   
  };
}

template < auto function, typename T0, typename T1 > 
auto jvp(T0 arg0, T1 arg1) {
  using output_type = decltype(function(arg0, arg1));
  return [=](T0 darg0, T1 darg1) {
    return __enzyme_fwddiff<output_type>((void*)function, enzyme_dup, arg0, darg0,
                                                          enzyme_dup, arg1, darg1);   
  };
}

template < auto function, typename T0, typename T1, typename T2 > 
auto jvp(T0 arg0, T1 arg1, T2 arg2) {
  using output_type = decltype(function(arg0, arg1, arg2));
  return [=](T0 darg0, T1 darg1, T2 darg2) {
    return __enzyme_fwddiff<output_type>((void*)function, enzyme_dup, arg0, darg0,
                                                          enzyme_dup, arg1, darg1,
                                                          enzyme_dup, arg2, darg2);   
  };
}

// "return-by-value" to "return-by-reference" transformation
template < auto function, typename ... T >
constexpr auto rbv_to_rbr() {
  using output_type = decltype(function(T{} ...));
  return [](output_type & output, T ... args){
    output = function(args ...);
  };
}

namespace enzyme {

  // note: we define this tuple instead of using `std::tuple` for the return type from jvp
  // because some std::tuple implementations actually pack the entries backwards, which messes
  // with the way enzyme aliases memory in its outputs.
  template < typename ... T >
  struct tuple;

  template < typename T0 >
  struct tuple<T0> { T0 value0; };

  template < typename T0, typename T1 >
  struct tuple<T0,T1> { T0 value0; T1 value1; };

  template < typename T0, typename T1, typename T2 >
  struct tuple<T0, T1, T2> { T0 value0; T1 value1; T2 value2; };

  template < typename ... T >
  tuple(T...) -> tuple<T...>;

}

template < auto function, typename T > 
auto vjp(const T & arg) {
  using output_type = decltype(function(arg));
  constexpr auto rbr_function = rbv_to_rbr<function, T>();
  return [=](output_type dv) {
    output_type unused{};
    return __enzyme_autodiff< T >((void*)+rbr_function, enzyme_dupnoneed, &unused, &dv,
                                                                            enzyme_out, arg); 
  };
}

template < auto function, typename T0, typename T1 > 
auto vjp(const T0 & arg0, const T1 & arg1) {
  using output_type = decltype(function(arg0, arg1));
  constexpr auto rbr_function = rbv_to_rbr<function, T0, T1>();
  return [=](output_type dv) {
    output_type unused{};
    return __enzyme_autodiff< enzyme::tuple<T0, T1> >((void*)+rbr_function, enzyme_dupnoneed, &unused, &dv,
                                                                            enzyme_out, arg0,
                                                                            enzyme_out, arg1); 
  };
}

template < auto function, typename T0, typename T1, typename T2 > 
auto vjp(const T0 & arg0, const T1 & arg1, const T2 & arg2) {
  using output_type = decltype(function(arg0, arg1, arg2));
  constexpr auto rbr_function = rbv_to_rbr<function, T0, T1, T2>();
  return [=](output_type dv) {
    output_type unused{};
    return __enzyme_autodiff< enzyme::tuple<T0, T1, T2> >((void*)+rbr_function, enzyme_dupnoneed, &unused, &dv,
                                                                            enzyme_out, arg0,
                                                                            enzyme_out, arg1,
                                                                            enzyme_out, arg2); 
  };
}

template < auto function >
struct jacobian_of;

template < typename return_type, 
           typename arg0_type,
           return_type (*func_ptr)(arg0_type) >
struct jacobian_of< func_ptr > {
  using functor_type = struct functor {

    // jvp
    friend return_type operator*(functor f, arg0_type darg) {
      return __enzyme_fwddiff<return_type>((void*)func_ptr, enzyme_dup, f.arg0, darg);   
    };

    // vjp
    friend arg0_type operator*(return_type dout, functor f) {
      constexpr auto rbr_func = rbv_to_rbr<func_ptr, arg0_type>();
      return_type unused{};
      return __enzyme_autodiff< arg0_type >((void*)+rbr_func, 
          enzyme_dupnoneed, &unused, &dout,
          enzyme_out, f.arg0);
    };

    arg0_type arg0;
  };

  static constexpr auto factory = [](arg0_type arg0) {
    return functor_type{arg0}; 
  };
};

template < typename return_type, 
           typename arg0_type,
           typename arg1_type,
           return_type (*func_ptr)(arg0_type, arg1_type) >
struct jacobian_of< func_ptr > {

  using functor_type = struct functor {

    // jvp
    friend return_type operator*(functor f, tuple<arg0_type, arg1_type> dargs) {
      return __enzyme_fwddiff<return_type>((void*)func_ptr, enzyme_dup, f.arg0, dargs[Index<0>{}],
                                                            enzyme_dup, f.arg1, dargs[Index<1>{}]);   
    };

    // vjp
    friend tuple<arg0_type, arg1_type> operator*(return_type dout, functor f) {
      constexpr auto rbr_func = rbv_to_rbr<func_ptr, arg0_type, arg1_type>();
      return_type unused{};
      return __enzyme_autodiff< tuple<arg0_type, arg1_type> >((void*)+rbr_func, 
          enzyme_dupnoneed, &unused, &dout,
          enzyme_out, f.arg0,
          enzyme_out, f.arg1);
    };

    arg0_type arg0;
    arg0_type arg1;
  };

  static constexpr auto factory = [](arg0_type arg0, arg1_type arg1) {
    return functor_type{arg0, arg1}; 
  };
};

template < auto func >
auto jacobian() { return jacobian_of<func>::factory; };

}


#if 0

namespace impl {

  constexpr int vsize(const double &) { return 1; }

  template < int n >
  constexpr int vsize(const vec<n> &) { return n; }

  template < int m, int n >
  constexpr int vsize(const mat<m, n> &) { return m * n; }

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

  template < typename T0, typename T1 >
  struct nested;

  template < typename T >
  struct nested< double, T >{ using type = T; };

  template < int n, typename T > 
  struct nested< vec<n>, T >{ using type = vec<n, T>; };

  template < int m, int n, typename T > 
  struct nested< mat<m,n>, T >{ using type = mat<m, n, T>; };

////////////////////////////////////////////////////////////////////////////////

  template< typename output_type, typename function, typename ... arg_types >
  void wrapper(output_type & output, const function & f, const arg_types & ... args) {
      output = f(args...);
  }

////////////////////////////////////////////////////////////////////////////////

  template < typename function, typename input_type > 
  __attribute__((always_inline))
  auto jvp(const function & f, const input_type & x) {
    using output_type = decltype(f(x));
    void * func_ptr = reinterpret_cast<void*>(wrapper< output_type, function, input_type >);
    return [=](const input_type & dx) {
      output_type unused{};
      output_type df{};
      __enzyme_fwddiff<void>(func_ptr,
        enzyme_dupnoneed, &unused, &df,
        enzyme_const, reinterpret_cast<const void*>(&f), 
        enzyme_dup, &x, &dx
      );
      return df;
    };
  }

////////////////////////////////////////////////////////////////////////////////

  template < typename function, typename input_type > 
  __attribute__((always_inline))
  auto jacfwd(const function & f, const input_type & x) {
    using output_type = decltype(f(x));
    using jac_type = typename impl::nested<output_type, input_type>::type;
    void * func_ptr = reinterpret_cast<void*>(wrapper< output_type, function, input_type >);

    jac_type J{};
    input_type dx{};

    double * dx_ptr = reinterpret_cast<double *>(&dx);

    // tuple input and tuple output
    if constexpr (is_tuple<output_type>{} && is_tuple<input_type>{}) {

      static_assert((tuple_size<output_type>{} == 2) && (tuple_size<input_type>{} == 2), "error: jacfwd() currently only supports tuples of 2 values");

      constexpr int m0 = impl::vsize(get<0>(output_type{}));
      constexpr int m1 = impl::vsize(get<1>(output_type{}));
      constexpr int n0 = impl::vsize(get<0>(input_type{}));
      constexpr int n1 = impl::vsize(get<1>(input_type{}));
 
      for (int j = 0; j < (n0 + n1); j++) {
        dx_ptr[j] = 1.0;
  
        output_type unused{};
        output_type df_dxj{};
        __enzyme_fwddiff<void>(func_ptr,
          enzyme_dupnoneed, &unused, &df_dxj,
          enzyme_const, reinterpret_cast<const void*>(&f), 
          enzyme_dup, &x, &dx
        );

        double * df0_dxj_ptr = reinterpret_cast<double *>(&get<0>(df_dxj));
        double * df1_dxj_ptr = reinterpret_cast<double *>(&get<1>(df_dxj));

        if (j < n0) {
          int j0 = j;
          double * J00_ptr = reinterpret_cast<double *>(&get<0>(get<0>(J)));
          double * J10_ptr = reinterpret_cast<double *>(&get<0>(get<1>(J)));
          for (int i0 = 0; i0 < m0; i0++) {
            J00_ptr[i0 * n0 + j0] = df0_dxj_ptr[i0];
          }

          for (int i1 = 0; i1 < m1; i1++) {
            J10_ptr[i1 * n0 + j0] = df1_dxj_ptr[i1];
          }
        } else {
          int j1 = j - n0;
          double * J01_ptr = reinterpret_cast<double *>(&get<1>(get<0>(J)));
          double * J11_ptr = reinterpret_cast<double *>(&get<1>(get<1>(J)));
          for (int i0 = 0; i0 < m0; i0++) {
            J01_ptr[i0 * n1 + j1] = df0_dxj_ptr[i0];
          }

          for (int i1 = 0; i1 < m1; i1++) {
            J11_ptr[i1 * n1 + j1] = df1_dxj_ptr[i1];
          }
        }

        dx_ptr[j] = 0.0;
      }
  
      return J;

    } else {

      constexpr int m = impl::vsize(output_type{});
      constexpr int n = impl::vsize(input_type{});

      double * J_ptr = reinterpret_cast<double *>(&J);

      for (int j = 0; j < n; j++) {
        dx_ptr[j] = 1.0;
  
        output_type unused{};
        output_type df_dxj{};
        __enzyme_fwddiff<void>(func_ptr,
          enzyme_dupnoneed, &unused, &df_dxj,
          enzyme_const, reinterpret_cast<const void*>(&f), 
          enzyme_dup, &x, &dx
        );
  
        double * df_dxj_ptr = reinterpret_cast<double *>(&df_dxj);
        for (int i = 0; i < m; i++) {
          J_ptr[i * n + j] = df_dxj_ptr[i];
        }

        dx_ptr[j] = 0.0;
      }
  
      return J;

    }
  

  }
  
}

////////////////////////////////////////////////////////////////////////////////

template < int i, typename function, typename T0 > 
__attribute__((always_inline))
auto forall_jvp(const function & f, const T0 & arg0) {
  if constexpr (i == 0) { return impl::jvp(f, arg0); }
}

template < int i, typename function, typename T0, typename T1 > 
__attribute__((always_inline))
auto jvp(const function & f, const T0 & arg0, const T1 & arg1) {
  if constexpr (i == 0) { return impl::jvp([&](T0 x){ return f(x, arg1); }, arg0); }
  if constexpr (i == 1) { return impl::jvp([&](T1 x){ return f(arg0, x); }, arg1); }
}

template < int i, typename function, typename T0, typename T1, typename T2 > 
__attribute__((always_inline))
auto jvp(const function & f, const T0 & arg0, const T1 & arg1, const T2 & arg2) {
  if constexpr (i == 0) { return impl::jvp([&](const T0 & x){ return f(x, arg1, arg2); }, arg0); }
  if constexpr (i == 1) { return impl::jvp([&](const T1 & x){ return f(arg0, x, arg2); }, arg1); }
  if constexpr (i == 2) { return impl::jvp([&](const T2 & x){ return f(arg0, arg1, x); }, arg2); }
}

template < int i, typename function, typename T0, typename T1, typename T2, typename T3 > 
__attribute__((always_inline))
auto jvp(const function & f, const T0 & arg0, const T1 & arg1, const T2 & arg2, const T3 & arg3) {
  if constexpr (i == 0) { return impl::jvp([&](const T0 & x){ return f(x, arg1, arg2, arg3); }, arg0); }
  if constexpr (i == 1) { return impl::jvp([&](const T1 & x){ return f(arg0, x, arg2, arg3); }, arg1); }
  if constexpr (i == 2) { return impl::jvp([&](const T2 & x){ return f(arg0, arg1, x, arg3); }, arg2); }
  if constexpr (i == 3) { return impl::jvp([&](const T3 & x){ return f(arg0, arg1, arg2, x); }, arg3); }
}

template < int i, typename function, typename T0, typename T1, typename T2, typename T3, typename T4 > 
__attribute__((always_inline))
auto jvp(const function & f, const T0 & arg0, const T1 & arg1, const T2 & arg2, const T3 & arg3, const T4 & arg4) {
  if constexpr (i == 0) { return impl::jvp([&](const T0 & x){ return f(x, arg1, arg2, arg3, arg4); }, arg0); }
  if constexpr (i == 1) { return impl::jvp([&](const T1 & x){ return f(arg0, x, arg2, arg3, arg4); }, arg1); }
  if constexpr (i == 2) { return impl::jvp([&](const T2 & x){ return f(arg0, arg1, x, arg3, arg4); }, arg2); }
  if constexpr (i == 3) { return impl::jvp([&](const T3 & x){ return f(arg0, arg1, arg2, x, arg4); }, arg3); }
  if constexpr (i == 4) { return impl::jvp([&](const T4 & x){ return f(arg0, arg1, arg2, arg3, x); }, arg4); }
}

template < int i, typename function, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5 > 
__attribute__((always_inline))
auto jvp(const function & f, const T0 & arg0, const T1 & arg1, const T2 & arg2, const T3 & arg3, const T4 & arg4, const T5 & arg5) {
  if constexpr (i == 0) { return impl::jvp([&](const T0 & x){ return f(x, arg1, arg2, arg3, arg4, arg5); }, arg0); }
  if constexpr (i == 1) { return impl::jvp([&](const T1 & x){ return f(arg0, x, arg2, arg3, arg4, arg5); }, arg1); }
  if constexpr (i == 2) { return impl::jvp([&](const T2 & x){ return f(arg0, arg1, x, arg3, arg4, arg5); }, arg2); }
  if constexpr (i == 3) { return impl::jvp([&](const T3 & x){ return f(arg0, arg1, arg2, x, arg4, arg5); }, arg3); }
  if constexpr (i == 4) { return impl::jvp([&](const T4 & x){ return f(arg0, arg1, arg2, arg3, x, arg5); }, arg4); }
  if constexpr (i == 5) { return impl::jvp([&](const T5 & x){ return f(arg0, arg1, arg2, arg3, arg4, x); }, arg5); }
}

////////////////////////////////////////////////////////////////////////////////

template < int i, typename function, typename T0 > 
__attribute__((always_inline))
auto jacfwd(const function & f, const T0 & arg0) {
  if constexpr (i == 0) { return impl::jacfwd(f, arg0); }
}

template < int i, typename function, typename T0, typename T1 > 
__attribute__((always_inline))
auto jacfwd(const function & f, const T0 & arg0, const T1 & arg1) {
  if constexpr (i == 0) { return impl::jacfwd([&](T0 x){ return f(x, arg1); }, arg0); }
  if constexpr (i == 1) { return impl::jacfwd([&](T1 x){ return f(arg0, x); }, arg1); }
}

template < int i, typename function, typename T0, typename T1, typename T2 > 
__attribute__((always_inline))
auto jacfwd(const function & f, const T0 & arg0, const T1 & arg1, const T2 & arg2) {
  if constexpr (i == 0) { return impl::jacfwd([&](T0 x){ return f(x, arg1, arg2); }, arg0); }
  if constexpr (i == 1) { return impl::jacfwd([&](T1 x){ return f(arg0, x, arg2); }, arg1); }
  if constexpr (i == 2) { return impl::jacfwd([&](T2 x){ return f(arg0, arg1, x); }, arg2); }
}

template < int i, typename function, typename T0, typename T1, typename T2, typename T3 > 
__attribute__((always_inline))
auto jacfwd(const function & f, const T0 & arg0, const T1 & arg1, const T2 & arg2, const T3 & arg3) {
  if constexpr (i == 0) { return impl::jacfwd([&](T0 x){ return f(x, arg1, arg2, arg3); }, arg0); }
  if constexpr (i == 1) { return impl::jacfwd([&](T1 x){ return f(arg0, x, arg2, arg3); }, arg1); }
  if constexpr (i == 2) { return impl::jacfwd([&](T2 x){ return f(arg0, arg1, x, arg3); }, arg2); }
  if constexpr (i == 3) { return impl::jacfwd([&](T3 x){ return f(arg0, arg1, arg2, x); }, arg3); }
}

template < int i, typename function, typename T0, typename T1, typename T2, typename T3, typename T4 > 
__attribute__((always_inline))
auto jacfwd(const function & f, const T0 & arg0, const T1 & arg1, const T2 & arg2, const T3 & arg3, const T4 & arg4) {
  if constexpr (i == 0) { return impl::jacfwd([&](T0 x){ return f(x, arg1, arg2, arg3, arg4); }, arg0); }
  if constexpr (i == 1) { return impl::jacfwd([&](T1 x){ return f(arg0, x, arg2, arg3, arg4); }, arg1); }
  if constexpr (i == 2) { return impl::jacfwd([&](T2 x){ return f(arg0, arg1, x, arg3, arg4); }, arg2); }
  if constexpr (i == 3) { return impl::jacfwd([&](T3 x){ return f(arg0, arg1, arg2, x, arg4); }, arg3); }
  if constexpr (i == 4) { return impl::jacfwd([&](T4 x){ return f(arg0, arg1, arg2, arg3, x); }, arg4); }
}

template < int i, typename function, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5 > 
__attribute__((always_inline))
auto jacfwd(const function & f, const T0 & arg0, const T1 & arg1, const T2 & arg2, const T3 & arg3, const T4 & arg4, const T5 & arg5) {
  if constexpr (i == 0) { return impl::jacfwd([&](T0 x){ return f(x, arg1, arg2, arg3, arg4, arg5); }, arg0); }
  if constexpr (i == 1) { return impl::jacfwd([&](T1 x){ return f(arg0, x, arg2, arg3, arg4, arg5); }, arg1); }
  if constexpr (i == 2) { return impl::jacfwd([&](T2 x){ return f(arg0, arg1, x, arg3, arg4, arg5); }, arg2); }
  if constexpr (i == 3) { return impl::jacfwd([&](T3 x){ return f(arg0, arg1, arg2, x, arg4, arg5); }, arg3); }
  if constexpr (i == 4) { return impl::jacfwd([&](T4 x){ return f(arg0, arg1, arg2, arg3, x, arg5); }, arg4); }
  if constexpr (i == 5) { return impl::jacfwd([&](T5 x){ return f(arg0, arg1, arg2, arg3, arg4, x); }, arg5); }
}

#endif
