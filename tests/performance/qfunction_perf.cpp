#include "femto/mesh.hpp"
#include "femto/field.hpp"
#include "femto/domain.hpp"

#include "forall.hpp"

#include "misc/timer.hpp"

#include <functional>

namespace compiler {
static void please_do_not_optimize_away([[maybe_unused]] void* p) { asm volatile("" : : "g"(p) : "memory"); }
}

using namespace femto;

timer stopwatch;

struct record { std::string name; double time; };

using vec4 = vec<4>;

template < uint32_t dim >
vec<dim, vec4> qfunction_SIMD(vec<dim, vec4> du_dxi, mat<dim,dim, vec4> dX_dxi) {
  mat<dim,dim> dxi_dX = inv(dX_dxi);
  vec<dim> du_dX = dot(du_dxi, dxi_dX);
  vec<dim> heat_flux = 3.0 * du_dX;
  return dot(heat_flux, transpose(dxi_dX)) * det(dX_dxi);
}

template < uint32_t dim >
vec<dim> qfunction(const vec<dim> & du_dxi, const mat<dim,dim> & dX_dxi) {
  mat<dim,dim> dxi_dX = inv(dX_dxi);
  vec<dim> du_dX = dot(du_dxi, dxi_dX);
  vec<dim> heat_flux = 3.0 * du_dX;
  return dot(heat_flux, transpose(dxi_dX)) * det(dX_dxi);
}

template < uint32_t dim >
vec<dim> qfunction_no_inv(const vec<dim> & du_dxi, const mat<dim,dim> & dxi_dX) {
  vec<dim> heat_flux = 3.0 * dot(du_dxi, dxi_dX);
  return dot(heat_flux, transpose(dxi_dX));
}

template < uint32_t dim >
vec<dim> qfunction_no_inv_no_transpose(const vec<dim> & du_dxi, const mat<dim,dim> & dxi_dX) {
  vec<dim> heat_flux = 3.0 * dot(du_dxi, dxi_dX);
  return dot(dxi_dX, heat_flux);
}

template < uint32_t dim >
double data_allocation_only(const nd::cpu_array<double, 3> & du_dxi_q, const nd::cpu_array<double, 3> & dX_dxi_q) {
  stopwatch.start();
  nd::cpu_array<double, 3> output(du_dxi_q.shape);
  compiler::please_do_not_optimize_away(output.data());
  stopwatch.stop();

  return stopwatch.elapsed();
}

template < uint32_t dim >
double data_movement_only(const nd::cpu_array<double, 3> & du_dxi_q, const nd::cpu_array<double, 3> & dX_dxi_q) {
  vec<dim> * du_dxi_v = (vec<dim> *)du_dxi_q.data();
  std::vector< vec<dim> > f_q(du_dxi_q.shape[0]);
  uint32_t n = du_dxi_q.shape[0];

  stopwatch.start();
  for (uint32_t i = 0; i < n; i++) {
    f_q[i] = du_dxi_v[i];
  }
  stopwatch.stop();
  compiler::please_do_not_optimize_away(f_q.data());
  return stopwatch.elapsed();
}

template < uint32_t dim >
double baseline(const nd::cpu_array<double, 3> & du_dxi_q, const nd::cpu_array<double, 3> & dX_dxi_q) {
  stopwatch.start();
  nd::cpu_array<double, 2> f_q = forall(qfunction<dim>, du_dxi_q, dX_dxi_q);
  stopwatch.stop();
  compiler::please_do_not_optimize_away(f_q.data());
  return stopwatch.elapsed();
}

template < uint32_t dim >
double baseline_no_inv(const nd::cpu_array<double, 3> & du_dxi_q, const nd::cpu_array<double, 3> & dX_dxi_q) {
  stopwatch.start();
  nd::cpu_array<double, 2> f_q = forall(qfunction_no_inv<dim>, du_dxi_q, dX_dxi_q);
  stopwatch.stop();
  compiler::please_do_not_optimize_away(f_q.data());
  return stopwatch.elapsed();
}

template < uint32_t dim >
double baseline_no_inv_no_transpose(const nd::cpu_array<double, 3> & du_dxi_q, const nd::cpu_array<double, 3> & dX_dxi_q) {
  stopwatch.start();
  nd::cpu_array<double, 2> f_q = forall(qfunction_no_inv_no_transpose<dim>, du_dxi_q, dX_dxi_q);
  stopwatch.stop();
  compiler::please_do_not_optimize_away(f_q.data());
  return stopwatch.elapsed();
}

//template < uint32_t dim >
//double blocked(const nd::cpu_array<double, 3> & du_dxi_q, const nd::cpu_array<double, 3> & dX_dxi_q) {
//  vec<dim> * du_dxi_v = (vec<dim> *)du_dxi_q.data();
//  mat<dim,dim> * dX_dxi_v = (mat<dim,dim> *)dX_dxi_q.data();
//  std::vector< vec<dim> > f_q(du_dxi_q.shape[0]);
//  uint32_t n = du_dxi_q.shape[0];
//
//  stopwatch.start();
//  for (uint32_t i = 0; i < n/4; i++) {
//    vec<dim, vec4> du_dxi;
//    mat<dim,dim, vec4> dX_dxi;
//    vec<dim, vec4> tmp = qfunction_SIMD<dim>(du_dxi, dX_dxi);
//    compiler::please_do_not_optimize_away(tmp.data);
//  }
//  stopwatch.stop();
//  compiler::please_do_not_optimize_away(f_q.data());
//  return stopwatch.elapsed();
//}

template < uint32_t dim >
double raw_ptr_access(const nd::cpu_array<double, 3> & du_dxi_q, const nd::cpu_array<double, 3> & dX_dxi_q) {
  stopwatch.start();
  vec<dim> * du_dxi_v = (vec<dim> *)du_dxi_q.data();
  mat<dim,dim> * dX_dxi_v = (mat<dim,dim> *)dX_dxi_q.data();
  std::vector< vec<dim> > f_q(du_dxi_q.shape[0]);
  uint32_t n = du_dxi_q.shape[0];

  for (uint32_t i = 0; i < n; i++) {
    f_q[i] = qfunction<dim>(du_dxi_v[i], dX_dxi_v[i]);
  }
  stopwatch.stop();
  compiler::please_do_not_optimize_away(f_q.data());
  return stopwatch.elapsed();
}

template < uint32_t dim >
double raw_ptr_access_no_inv(const nd::cpu_array<double, 3> & du_dxi_q, const nd::cpu_array<double, 3> & dX_dxi_q) {
  stopwatch.start();
  vec<dim> * du_dxi_v = (vec<dim> *)du_dxi_q.data();
  mat<dim,dim> * dX_dxi_v = (mat<dim,dim> *)dX_dxi_q.data();
  std::vector< vec<dim> > f_q(du_dxi_q.shape[0]);
  uint32_t n = du_dxi_q.shape[0];

  for (uint32_t i = 0; i < n; i++) {
    f_q[i] = qfunction_no_inv<dim>(du_dxi_v[i], dX_dxi_v[i]);
  }
  stopwatch.stop();
  compiler::please_do_not_optimize_away(f_q.data());
  return stopwatch.elapsed();
}

template < uint32_t dim >
void run_test_suite(uint32_t n) {

  nd::cpu_array<double, 3> du_dxi_q({n, 1u, dim});
  nd::cpu_array<double, 3> dX_dxi_q({n, dim, dim});

  nd::cpu_array<double, 3> du_dxi_q_T({n, 1u, dim});
  nd::cpu_array<double, 3> dX_dxi_q_T({n, dim, dim});

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

  timings.push_back({"data allocation only", data_allocation_only<dim>(du_dxi_q, dX_dxi_q)});
  timings.push_back({"data movement only", data_movement_only<dim>(du_dxi_q, dX_dxi_q)});
  timings.push_back({"baseline", baseline<dim>(du_dxi_q, dX_dxi_q)});
  timings.push_back({"baseline (no inv)", baseline_no_inv<dim>(du_dxi_q, dX_dxi_q)});
  timings.push_back({"baseline (no inv no transpose)", baseline_no_inv_no_transpose<dim>(du_dxi_q, dX_dxi_q)});
  timings.push_back({"baseline (col major)", baseline<dim>(du_dxi_q_T, dX_dxi_q_T)});
  timings.push_back({"raw_ptr_access", raw_ptr_access<dim>(du_dxi_q, dX_dxi_q)});
  timings.push_back({"raw_ptr_access_no_inv", raw_ptr_access_no_inv<dim>(du_dxi_q, dX_dxi_q)});

  std::cout << "dim = " << dim << ", n = " << n << std::endl;
  for (auto rec : timings) {
    std::cout << rec.name << ": " << rec.time * 1000.0 << std::endl;
  }

};

int main(int argc, char **argv) {
  run_test_suite<2>(1 << 14);
  run_test_suite<2>(1 << 17);
  run_test_suite<2>(1 << 20);

  run_test_suite<3>(1 << 14);
  run_test_suite<3>(1 << 17);
  run_test_suite<3>(1 << 20);
}
