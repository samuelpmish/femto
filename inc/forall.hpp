#pragma once

#include "containers/ndarray.hpp"
#include "misc/timer.hpp"
#include "fm/types/vec.hpp"
#include "fm/types/matrix.hpp"

#include "femto/assert.hpp"
#include "femto/threadpool.hpp"

#include <functional>
#include <type_traits>

namespace femto {

template < typename ... T >
struct type_list{};

template < typename T >
struct FunctionSignature;

template < typename ret_t, typename ... arg_t >
struct FunctionSignature< ret_t (*)(arg_t ...) > {
    using return_type = ret_t;
    using argument_types = type_list< arg_t ... >;
};

template < typename obj_t, typename ret_t, typename ... arg_t >
struct FunctionSignature< ret_t (obj_t::*)(arg_t ...) const > {
    using return_type = ret_t;
    using argument_types = type_list< arg_t ... >;
};

// the signature of a functor or of a plain function pointer
template < typename T >
struct signature_of { using type = FunctionSignature< decltype(&T::operator()) >; };

template < typename ret_t, typename ... arg_t >
struct signature_of< ret_t (*)(arg_t ...) > { using type = FunctionSignature< ret_t (*)(arg_t ...) >; };

using namespace fm;

template < typename T >
struct array_memory_space;

template < typename T, uint32_t rank, memory::space mem_space >
struct array_memory_space< nd::array< T, rank, mem_space > > {
  static constexpr memory::space value = mem_space;
};

template < typename T >
inline constexpr memory::space array_memory_space_v = array_memory_space< std::remove_cvref_t<T> >::value;

template < typename first_arg, typename ... arg_types >
constexpr memory::space common_memory_space() {
  constexpr memory::space mem_space = array_memory_space_v<first_arg>;
  static_assert(((array_memory_space_v<arg_types> == mem_space) && ...), "forall(...) requires all inputs to use the same memory space");
  return mem_space;
}

template < memory::space mem_space, typename T >
auto make_ndarray_in(uint32_t length, T) {
  return nd::array<T,1,mem_space>({length});
}

template < memory::space mem_space >
auto make_ndarray_in(uint32_t length, double) {
  return nd::array<double,2,mem_space>({length, 1});
}

template < memory::space mem_space, uint32_t n >
auto make_ndarray_in(uint32_t length, vec<n>) {
  return nd::array<double,2,mem_space>({length, uint32_t(n)});
}

template < memory::space mem_space, uint32_t m, uint32_t n >
auto make_ndarray_in(uint32_t length, mat<m, n>) {
  return nd::array<double,3,mem_space>({length, uint32_t(m), uint32_t(n)});
}

template < memory::space mem_space, uint32_t m, uint32_t n, uint32_t p, uint32_t q >
auto make_ndarray_in(uint32_t length, mat<m, n, mat<p, q> >) {
  return nd::array<double,5,mem_space>({length, m, n, p, q});
}

template < typename T >
static constexpr bool is_vec(T) { return false; }

template < typename T, uint32_t n >
static constexpr bool is_vec(vec<n,T> v) { return true; }

template < typename T >
static constexpr bool is_mat(T) { return false; }

template < typename T, uint32_t m, uint32_t n >
static constexpr bool is_mat(mat<m,n,T> v) { return true; }

template < typename T, uint32_t n, uint32_t rank, memory::space mem_space >
auto load_vec(const nd::array< T, rank, mem_space > & arr, int i, vec<n,T>) {
  vec<n,T> output;
  for (uint32_t j = 0; j < n; j++) {
    if constexpr (rank == 2) {
      output[j] = arr(i,j);
    }
    if constexpr (rank == 3) {
      output[j] = arr(i,0,j);
    }
  }
  return output;
}

template < typename T, uint32_t m, uint32_t n, memory::space mem_space >
auto load_mat(const nd::array< T, 3, mem_space > & arr, int i, mat<m,n,T>) {
  mat<m,n,T> output;
  for (uint32_t j = 0; j < m; j++) {
    for (uint32_t k = 0; k < n; k++) {
      output(j,k) = arr(i,j,k);
    }
  }
  return output;
}

template < typename T, typename arr_type, uint32_t rank, memory::space mem_space >
auto load(const nd::array< arr_type, rank, mem_space > & arr, int i) {
  if constexpr (std::is_same<T, float>::value || 
                std::is_same<T, double>::value) {
    static_assert(rank == 1 || rank == 2 || rank == 3);
    if constexpr (rank == 1) { return arr(i); }
    if constexpr (rank == 2) { return arr(i,0); }
    if constexpr (rank == 3) { return arr(i,0,0); }
  }

  if constexpr (is_vec(T{})) {
    static_assert(rank == 2 || rank == 3);
    return load_vec(arr, i, T{});
  }

  if constexpr (is_mat(T{})) {
    static_assert(rank == 3);
    return load_mat(arr, i, T{});
  }
}

template < typename T, uint32_t n, uint32_t rank, memory::space mem_space >
void save_vec(nd::array< T, rank, mem_space > & arr, int i, const vec<n,T> & value) {
  for (uint32_t j = 0; j < n; j++) {
    if constexpr (rank == 2) {
      arr(i,j) = value[j];
    }
    if constexpr (rank == 3) {
      arr(i,0,j) = value[j];
    }
  }
}

template < typename T, uint32_t m, uint32_t n, memory::space mem_space >
void save_mat(nd::array< T, 3, mem_space > & arr, uint32_t i, const mat<m,n,T> & value) {
  for (uint32_t j = 0; j < m; j++) {
    for (uint32_t k = 0; k < n; k++) {
      arr(i,j,k) = value(j,k);
    }
  }
}

template < typename T, typename arr_type, uint32_t rank, memory::space mem_space >
auto save(nd::array< arr_type, rank, mem_space > & arr, uint32_t i, const T & value) {
  if constexpr (std::is_same<T, float>::value || 
                std::is_same<T, double>::value) {
    if constexpr (rank == 1) arr(i) = value;
    if constexpr (rank == 2) arr(i, 0) = value;
  }

  if constexpr (is_vec(T{})) {
    static_assert(rank == 2 || rank == 3);
    save_vec(arr, i, value);
  }

  if constexpr (is_mat(T{})) {
    static_assert(rank == 3);
    save_mat(arr, i, value);
  }
}

#ifdef __CUDACC__

template < typename functor, typename return_type, typename ... input_types >
__global__ void forall_kernel(
  uint32_t n, 
  functor f, 
  return_type * output, 
  const input_types * ... inputs) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n) {
    output[tid] = f(inputs[tid] ...); 
  }
}

template < typename functor, typename ... parameter_types >
__global__ void forall_void_kernel(uint32_t n, functor f, parameter_types * ... args) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n) {
    f(args[tid] ...);
  }
}

#endif

namespace impl {

// where a forall() over arrays in this memory space runs.  unified memory is
// reachable from both sides, so it follows the compiler: a .cu translation
// unit gets the kernel, a host-only one gets the threadpool loop.
template < memory::space mem_space >
inline constexpr bool on_device =
#ifdef __CUDACC__
  mem_space != memory::space::cpu;
#else
  false;
#endif

// a gpu array has nowhere to run outside a .cu file, since both the kernel
// launch and the device side of the q-function only exist when nvcc compiles
// this header.  spelled as a function so the message names forall(), not the
// branch it came from.
template < memory::space mem_space >
void assert_reachable_from_host() {
  static_assert(mem_space != memory::space::gpu,
                "forall(): gpu arrays require a .cu translation unit and a __device__ q-function");
}

inline constexpr uint32_t blocksize = 128;

// ponytail: below this many entries the pool's fork-join (about 30 us on a
// 32-thread pool) costs more than the loop itself for a cheap q-function, so
// the loop runs on the calling thread.  A fixed count, not cost-aware: a
// q-function costing much more than a few ns per entry would profit from the
// pool earlier.  Upgrade path: a cost hint per call or per functor.
inline constexpr uint32_t serial_below = 16384;

template < typename callable >
void host_for(uint32_t n, const callable & body) {
  if (n < serial_below) {
    for (uint32_t i = 0; i < n; i++) { body(i); }
  } else {
    threadpool::parallel_for(n, body);
  }
}

template < memory::space mem_space, typename callable, typename return_type, typename ... input_types >
void launch(uint32_t n, const callable & f, return_type * output, const input_types * ... inputs) {
  if constexpr (on_device<mem_space>) {
#ifdef __CUDACC__
    forall_kernel<<< (n + blocksize - 1) / blocksize, blocksize >>>(n, f, output, inputs ...);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
#endif
  } else {
    assert_reachable_from_host<mem_space>();
    host_for(n, [&](uint32_t i) {
      output[i] = f(inputs[i] ... );
    });
  }
}

template < memory::space mem_space, typename callable, typename ... parameter_types >
void launch_void(uint32_t n, const callable & f, parameter_types * ... args) {
  if constexpr (on_device<mem_space>) {
#ifdef __CUDACC__
    forall_void_kernel<<< (n + blocksize - 1) / blocksize, blocksize >>>(n, f, args ...);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
#endif
  } else {
    assert_reachable_from_host<mem_space>();
    host_for(n, [&](uint32_t i) {
      f(args[i] ... );
    });
  }
}

// a std::function or a plain function pointer is a host address, so it can
// only be called from the host loop no matter which space the data lives in
template < typename callable, typename return_type, typename ... input_types >
void host_loop(uint32_t n, const callable & f, return_type * output, const input_types * ... inputs) {
  host_for(n, [&](uint32_t i) {
    output[i] = f(inputs[i] ... );
  });
}

// A q-function that returns void writes through its non-const reference
// arguments instead.  Those output arrays are sized here if they are not
// already the right shape, so a caller that keeps them alive across calls
// stops reallocating -- see forall_void_tests.cpp.
template < typename T >
inline constexpr bool is_output = std::is_lvalue_reference_v<T> &&
                                  !std::is_const_v< std::remove_reference_t<T> >;

// resize() reallocates only when the total size changes, so an output that is
// already the right shape keeps its buffer.  make_ndarray_in with length 0
// allocates nothing; it is here only to spell out the shape for this type.
template < memory::space mem_space, typename param_t, typename array_t >
void size_if_output(array_t & arr, uint32_t n) {
  if constexpr (is_output<param_t>) {
    auto shape = make_ndarray_in<mem_space>(0u, std::remove_reference_t<param_t>{}).shape;
    shape[0] = n;
    arr.resize(shape);
  }
}

// check the argument list, take the loop length from the inputs, and size any
// output that is not already the right shape
template < typename ... parameter_types, typename ... arg_types >
uint32_t size_void_outputs(type_list< parameter_types ... >, arg_types & ... args) {

  static_assert(sizeof...(parameter_types) == sizeof...(arg_types),
                "forall(): one array per q-function argument");
  static_assert(( is_output<parameter_types> || ... ),
                "forall(): a void q-function needs a non-const reference argument to write to");
  static_assert(( !is_output<parameter_types> || ... ),
                "forall(): a void q-function needs an input to take its length from");

  constexpr memory::space mem_space = common_memory_space<arg_types...>();

  // the loop length comes from the inputs, since an output may still be empty
  uint32_t lengths[] = { (is_output<parameter_types> ? 0u : args.shape[0]) ... };
  uint32_t n = 0;
  for (uint32_t length : lengths) {
    if (length == 0) { continue; }
    if (n == 0) { n = length; }
    FEMTO_ASSERT(length == n, "forall(): inputs have different lengths");
  }

  ( size_if_output<mem_space, parameter_types>(args, n), ... );

  return n;
}

template < typename ... parameter_types, typename callable, typename ... arg_types >
void forall_void(type_list< parameter_types ... > parameters, callable func, arg_types & ... args) {

  const uint32_t n = size_void_outputs(parameters, args ...);

  launch_void< common_memory_space<arg_types...>() >(
    n,
    func,
    reinterpret_cast< std::remove_reference_t<parameter_types> * >(args.data()) ...
  );

}

template < typename output_type, typename ... input_types, typename callable, typename ... arg_types >
auto forall(output_type, type_list< input_types ... >, callable func, const arg_types & ... args) {

  constexpr memory::space mem_space = common_memory_space<arg_types...>();

  uint32_t leading_dimensions[] = {args.shape[0] ...};
  uint32_t n = leading_dimensions[0];
  auto output = make_ndarray_in<mem_space>(n, output_type{});

  launch<mem_space>(
    n, 
    func,
    reinterpret_cast< output_type * >(output.data()), 
    reinterpret_cast< const std::remove_cvref_t<input_types> * >(args.data()) ... 
  );

  return output;
}

}

template < typename return_type, typename ... parameter_types, typename ... arg_types >
auto forall(std::function< return_type(const parameter_types & ...) > f, const arg_types & ... args) {

  constexpr memory::space mem_space = common_memory_space<arg_types...>();
  static_assert(mem_space != memory::space::gpu,
                "forall(): a std::function cannot be called on the device");

  uint32_t leading_dimensions[] = {args.shape[0] ...};
  uint32_t n = leading_dimensions[0];
  auto output = make_ndarray_in<mem_space>(n, return_type{});

  impl::host_loop(
    n, 
    f,
    reinterpret_cast< return_type * >(output.data()), 
    reinterpret_cast< const parameter_types * >(args.data()) ... 
  );

  return output;
}

template < typename return_type, typename ... parameter_types, typename ... arg_types >
auto forall(return_type (*f)(const parameter_types & ...), const arg_types & ... args) {

  constexpr memory::space mem_space = common_memory_space<arg_types...>();
  static_assert(mem_space != memory::space::gpu,
                "forall(): a function pointer is a host address; pass a __host__ __device__ functor instead");

  uint32_t leading_dimensions[] = {args.shape[0] ...};
  uint32_t n = leading_dimensions[0];
  auto output = make_ndarray_in<mem_space>(n, return_type{});

  impl::host_loop(
    n, 
    f,
    reinterpret_cast< return_type * >(output.data()), 
    reinterpret_cast< const parameter_types * >(args.data()) ... 
  );

  return output;
}

// void q-function: outputs are the non-const reference arguments, and are
// filled in place when they are already the right size.  reusing the output
// matters most on the device, where cudaFree synchronizes.
template < typename callable, typename ... arg_types,
           typename = std::enable_if_t< std::is_void_v< typename signature_of<callable>::type::return_type > > >
void forall(callable func, arg_types & ... args) {

  impl::forall_void(
    typename signature_of<callable>::type::argument_types{},
    func,
    args ...
  );

}

template < typename callable, typename ... arg_types >
auto forall(callable func, const arg_types & ... args) {

  using signature = FunctionSignature< decltype(&callable::operator()) >;

  return impl::forall(
    typename signature::return_type{}, 
    typename signature::argument_types{}, 
    func, 
    args ...
  );

}

} // namespace femto
