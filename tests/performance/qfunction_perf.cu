#include "femto/mesh.hpp"
#include "femto/field.hpp"
#include "femto/domain.hpp"

#include "containers/tuple.hpp"

#include "forall.hpp"

#include "misc/timer.hpp"

#include <functional>

namespace compiler {
static void please_do_not_optimize_away([[maybe_unused]] void* p) { asm volatile("" : : "g"(p) : "memory"); }
}

using namespace femto;

timer stopwatch;

struct record { std::string name; double time; };


namespace impl {

template < typename functor, typename return_type, typename ... input_types >
__global__ void cuda_forall_kernel_reinterpret(
  uint32_t n, 
  functor f, 
  return_type * outputs, 
  const input_types * ... inputs) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n) {
    return_type output = f(inputs[tid] ...); 
  }
}


template < typename output_type, typename ... input_types, typename callable, typename ... arg_types >
auto cuda_forall_reinterpret(output_type, type_list< input_types ... >, callable func, const arg_types & ... args) {

  uint32_t leading_dimensions[] = {args.shape[0] ...};
  int n = leading_dimensions[0];
  auto output = make_ndarray_in<memory::UNIFIED>(n, output_type{});

  {
    int blocksize = 128;
    int gridsize = (n + blocksize - 1) / blocksize;
    forall_kernel<<< gridsize, blocksize >>>(
      n, 
      func,
      reinterpret_cast< output_type * >(output.data()), 
      reinterpret_cast< const input_types * >(args.data()) ... 
    );
    cudaDeviceSynchronize();
  }

  return output;
}

}

template < typename callable, typename ... arg_types >
auto cuda_forall_reinterpret(callable func, const arg_types & ... args) {

  using signature = FunctionSignature< decltype(&callable::operator()) >;

  return ::impl::cuda_forall_reinterpret(
    typename signature::return_type{}, 
    typename signature::argument_types{}, 
    func, 
    args ...
  );

}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

namespace impl {

template < typename T >
struct loader;

template < typename S, typename T >
__device__ void cuda_load(uint32_t i, uint32_t n, S * array, T & value) {
  static constexpr uint32_t components = sizeof(T) / sizeof(S); 
  S * value_ptr = reinterpret_cast<S*>(&value);
  for (int j = 0; j < components; j++) {
    value_ptr[j] = array[j * n + i];
  }
};

template < typename S, typename T >
__device__ void cuda_store(uint32_t i, uint32_t n, S * array, const T & value) {
  static constexpr uint32_t components = sizeof(T) / sizeof(S); 
  S * value_ptr = reinterpret_cast<S*>(&value);
  for (int j = 0; j < components; j++) {
    array[j * n + i] = value_ptr[j];
  }
};

template < typename functor, typename return_type, typename ... input_types >
__global__ void cuda_forall_kernel_load(
  uint32_t n, 
  functor f, 
  return_type * outputs, 
  const input_types * ... inputs) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n) {
    return_type output = f(inputs[tid] ...); 
  }
}

template < typename output_type, typename ... input_types, typename callable, typename ... arg_types >
auto cuda_forall_load(output_type, type_list< input_types ... >, callable func, const arg_types & ... args) {

  uint32_t leading_dimensions[] = {args.shape[0] ...};
  int n = leading_dimensions[0];
  auto output = make_ndarray_in<memory::UNIFIED>(n, output_type{});

  {
    int blocksize = 128;
    int gridsize = (n + blocksize - 1) / blocksize;
    forall_kernel<<< gridsize, blocksize >>>(
      n, 
      func,
      reinterpret_cast< output_type * >(output.data()), 
      reinterpret_cast< const input_types * >(args.data()) ... 
    );
    cudaDeviceSynchronize();
  }

  return output;
}

}

template < typename callable, typename ... arg_types >
auto cuda_forall_load(callable func, const arg_types & ... args) {

  using signature = FunctionSignature< decltype(&callable::operator()) >;

  return ::impl::cuda_forall_load(
    typename signature::return_type{}, 
    typename signature::argument_types{}, 
    func, 
    args ...
  );

}

template < uint32_t dim >
__host__ __device__ vec<dim> qfunction(vec<dim> du_dxi, mat<dim,dim> dX_dxi) {
  mat<dim,dim> dxi_dX = inv(dX_dxi);
  vec<dim> du_dX = dot(du_dxi, dxi_dX);
  vec<dim> heat_flux = 3.0 * du_dX;
  return dot(heat_flux, transpose(dxi_dX)) * det(dX_dxi);
}

template < uint32_t dim >
double baseline(const nd::array<double, 3, memory::UNIFIED> & du_dxi_q, const nd::array<double, 3, memory::UNIFIED> & dX_dxi_q) {
  stopwatch.start();
  nd::array<double, 2, memory::UNIFIED> f_q = forall(qfunction<dim>, du_dxi_q, dX_dxi_q);
  stopwatch.stop();
  compiler::please_do_not_optimize_away(f_q.data());
  return stopwatch.elapsed();
}

template < uint32_t dim >
void run_test_suite(uint32_t n) {

  nd::array<double, 3, memory::UNIFIED> du_dxi_q({n, 1u, dim});
  nd::array<double, 3, memory::UNIFIED> dX_dxi_q({n, dim, dim});

  nd::array<double, 3, memory::UNIFIED> du_dxi_q_T({n, 1u, dim});
  nd::array<double, 3, memory::UNIFIED> dX_dxi_q_T({n, dim, dim});

  du_dxi_q_T.stride = nd::compute_strides(du_dxi_q_T.shape, nd::ordering::col_major);
  dX_dxi_q_T.stride = nd::compute_strides(dX_dxi_q_T.shape, nd::ordering::col_major);

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < dim; j++) {
      for (int k = 0; k < dim; k++) {
        double a = (j==k) + 0.5f * random();
        dX_dxi_q(i, j, k) = a;
        dX_dxi_q_T(i, j, k) = a;
      }
      double b = random();
      du_dxi_q(i, 0, j) = b;
      du_dxi_q_T(i, 0, j) = b;
    }
  }

  std::vector< record > timings;

//  timings.push_back({"data allocation only", data_allocation_only<dim>(du_dxi_q, dX_dxi_q)});
//  timings.push_back({"data movement only", data_movement_only<dim>(du_dxi_q, dX_dxi_q)});
//  timings.push_back({"baseline", baseline<dim>(du_dxi_q, dX_dxi_q)});
//  timings.push_back({"baseline (no inv)", baseline_no_inv<dim>(du_dxi_q, dX_dxi_q)});
//  timings.push_back({"baseline (no inv no transpose)", baseline_no_inv_no_transpose<dim>(du_dxi_q, dX_dxi_q)});
//  timings.push_back({"baseline (col major)", baseline<dim>(du_dxi_q_T, dX_dxi_q_T)});
//  timings.push_back({"raw_ptr_access", raw_ptr_access<dim>(du_dxi_q, dX_dxi_q)});
//  timings.push_back({"raw_ptr_access_no_inv", raw_ptr_access_no_inv<dim>(du_dxi_q, dX_dxi_q)});

  std::cout << "dim = " << dim << ", n = " << n << std::endl;
  for (auto rec : timings) {
    std::cout << rec.name << ": " << rec.time * 1000.0 << std::endl;
  }

};

int main(int argc, char **argv) {
  run_test_suite<2>(1 << 14);
  run_test_suite<2>(1 << 17);
  run_test_suite<2>(1 << 20);
}
