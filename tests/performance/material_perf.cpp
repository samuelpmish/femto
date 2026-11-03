// SPDX-FileCopyrightText: Copyright (c) 2023-2026 Sam Mish
// SPDX-License-Identifier: MIT

// The cost of the q-function for the solid mechanics models in inc/materials,
// next to the interpolation and integration it sits between, on the same mesh.
//
// Only the q-function depends on the model: interpolation and integration see
// the same quadrature data whatever produced it, so they are timed once per
// mesh, and each model times only its q-function, over the same du_dX:
//   residual : du_dX -> P, and the new state for a model that has one (the
//              viscous strain, or the corotational models' Cauchy stress)
//   tangent  : du_dX -> dP/dF
// Each q-function runs through forall() and, on the gpu, forall_staged(),
// which writes each block's outputs out together (on the host the two are the
// same loop).  The totals are interpolate + q-function + integrate, and the
// "1 core" columns are the q-function's ns/qpt on cache-resident data on a
// single core (cpu only): the arithmetic, without the memory traffic the
// full-size runs mostly wait on.
//
// Besides corotational.hpp's HughesWinget, RashidEig and RashidNewton, the
// rows "hughes-winget, 1st" (the notebook's first order update) and
// "rashid, newton+eig" (Newton's R, then the log by an eigendecomposition) are
// for comparison, and the "polar alone" rows are the polar decomposition of
// each step's relative deformation gradient by itself, both ways.
//
// Every phase writes into arrays allocated ahead of time, as a simulation
// reusing its work arrays would, so first-touch page faults stay out of it.
// Each time is the best of a few repetitions after a warm up.
//
// The displacement is the last of 32 steps to a 30 degree rotation on top of a
// smooth strain field (the amplitude argument), so each step's relative
// deformation gradient turns about 1 degree, with a strain of about
// amplitude / 32.  Newton's polar iteration count depends on that strain, not
// on the rotation.
//
// material_perf.cu builds this same file for the gpu.
//
// usage: material_perf [amplitude = 0.05] [repetitions = 5]

#include "femto/mesh.hpp"
#include "femto/field.hpp"
#include "femto/domain.hpp"

#include "forall.hpp"

#include "materials/linear_elasticity.hpp"
#include "materials/neohookean.hpp"
#include "materials/viscoelasticity.hpp"
#include "materials/corotational.hpp"
#include "linear_algebra/polar_decomp.hpp"

#include "fusion_common.hpp"   // tet_cuboid

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#ifdef __linux__
#include <sched.h>
#endif

using namespace femto;

#ifdef __CUDACC__
static constexpr memory::space mem = memory::space::gpu;
static const char * device = "gpu";
#else
static constexpr memory::space mem = memory::space::cpu;
static const char * device = "cpu";
#endif

// a model's q-functions, writing into arrays allocated ahead of time
template < typename model_t >
struct Stress {
  model_t model;
  __host__ __device__ void operator()(const mat3 & du_dX, mat3 & P) const { P = model(du_dX); }
};

template < typename model_t >
struct Tangent {
  model_t model;
  __host__ __device__ void operator()(const mat3 & du_dX, mat< 3, 3, mat3 > & dP_dF) const { dP_dF = model.jac(du_dX); }
};

// for comparison with corotational.hpp's kinematics: Hughes-Winget as the
// notebook writes it, first order, and Rashid with Newton's R and the log by
// an eigendecomposition of U, the dearest of the three ways
struct HughesWingetFirstOrder {
  __host__ __device__ Increment operator()(const mat3 & f) const {
    auto I = Identity<3>();
    mat3 l = 2.0 * I - 4.0 * inv(f + I);
    mat3 w = 0.5 * (l - transpose(l));
    return {2.0 * inv(I - 0.5 * w) - I, 0.5 * (l + transpose(l))};
  }
};

struct RashidNewtonEigLog {
  __host__ __device__ Increment operator()(const mat3 & f) const {
    auto [R, U] = polar_newton(f);
    return {R, logm(U)};
  }
};

// the polar decomposition of each step's relative deformation gradient
// alone, as Rashid's update starts with it
template < bool use_eig >
struct StepRotation {
  __host__ __device__ void operator()(const mat3 & du_dX, const mat3 & du_dX_old, mat3 & R) const {
    auto I = Identity<3>();
    mat3 f = dot(I + du_dX, inv(I + du_dX_old));
    if constexpr (use_eig) { R = polar_eig(f).R; } else { R = polar_newton(f).R; }
  }
};

// what the q-functions read and write at n points: state is the viscous
// strain or the Cauchy stress (or R, for StepRotation), state_old the last step's
template < memory::space space >
struct QuadratureData {
  nd::array<double, 3, space> du_dX, du_dX_old, P, state_old, state;
  nd::array<double, 5, space> dP_dF;
  QuadratureData(uint32_t n) : du_dX({n, 3u, 3u}), du_dX_old({n, 3u, 3u}), P({n, 3u, 3u}),
                               state_old({n, 3u, 3u}), state({n, 3u, 3u}), dP_dF({n, 3u, 3u, 3u, 3u}) {
    zero(state_old);
  }
};

// the best of reps runs, after one to warm up
template < typename callable >
double best_of(int reps, const callable & f) {
  double best = 1.0e30;
  for (int k = 0; k <= reps; k++) {
    double t = femto::time([&]() {
      f();
#ifdef __CUDACC__
      CUDA_CHECK(cudaDeviceSynchronize());
#endif
    });
    if (k > 0) { best = std::min(best, t); }
  }
  return best;
}

// u = Q (X + w(X)) - X: a rotation by angle about (1, 1, 1) of a smooth
// displacement w, whose gradient has entries of order amplitude
void fill_displacement(Field<Family::H1> & u, const Mesh<> & mesh, double amplitude, double angle) {
  mat3 Q = RotationMatrix(normalize(vec3{1.0, 1.0, 1.0}) * angle);
  nd::cpu_array<double, 2> X = nodes_for(u, mesh);
  for (uint32_t i = 0; i < X.shape[0]; i++) {
    vec3 Xi = load<vec3>(X, i);
    vec3 w;
    for (int c = 0; c < 3; c++) {
      w[c] = amplitude * std::sin(1.7 * Xi[0] + 0.4 * c) * std::cos(1.3 * Xi[1] - 0.2 * c) * (0.5 + 0.5 * Xi[2]);
    }
    save(u.data, i, vec3(dot(Q, Xi + w) - Xi));
  }
}

template < uint32_t p >
void run_case(const char * name, const Mesh<> & host_mesh, double amplitude, int reps) {

  constexpr double lambda = 10.0, mu = 10.0;
  LinearElasticModel elastic{lambda, mu};
  NeoHookeanModel neohookean{lambda, mu};
  LinearViscoelasticModel viscoelastic{lambda, mu, 2.0 * mu, 1.0, 0.1};
  CorotationalModel< HughesWingetFirstOrder > hughes_winget_1{lambda, mu};
  CorotationalModel< HughesWinget > hughes_winget{lambda, mu};
  CorotationalModel< RashidNewtonEigLog > rashid_newton_eig{lambda, mu};
  CorotationalModel< RashidEig > rashid_eig{lambda, mu};
  CorotationalModel< RashidNewton > rashid_newton{lambda, mu};

  // the last step of 32 to a 30 degree rotation, so a step turns about 1 degree
  Field host_u = create_field<Family::H1>(host_mesh, p, 3), host_u_old = host_u;
  fill_displacement(host_u, host_mesh, amplitude, M_PI / 6.0);
  fill_displacement(host_u_old, host_mesh, amplitude * 31.0 / 32.0, (M_PI / 6.0) * 31.0 / 32.0);

  Mesh<mem> mesh = host_mesh;
  Field<Family::H1, mem> u = host_u, u_old = host_u_old;
  Domain<mem> domain(mesh, MeshQuadratureRule(p + 1));
  BasisFunction<Family::H1> phi(p, 3);

  nd::array<double, 3, mem> du_q = evaluate(grad(u), domain);
  uint32_t nq = du_q.shape[0];
  QuadratureData<mem> full(nq);
  full.du_dX = du_q;
  full.du_dX_old = evaluate(grad(u_old), domain);

  // the model-independent phases, integrating the neo-hookean model's data
  forall(Stress<NeoHookeanModel>{neohookean}, full.du_dX, full.P);
  forall(Tangent<NeoHookeanModel>{neohookean}, full.du_dX, full.dP_dF);
  Residual<Family::H1, mem> r = integrate(dot(full.P, grad(phi)), domain);
  sparse_matrix<mem> K = blank_sparse_matrix(grad(phi), grad(phi), domain);

  double interpolate = best_of(reps, [&]() { du_q = evaluate(grad(u), domain); });
  double integrate_residual = best_of(reps, [&]() { r = integrate(dot(full.P, grad(phi)), domain); });
  double integrate_stiffness = best_of(reps, [&]() { integrate(dot(grad(phi), full.dP_dF, grad(phi)), domain)(K); });

  // each q-function's {residual, tangent} time over the given data, launched
  // through forall() or forall_staged(), -1 for a tangent a model doesn't have
  const char * names[] = {"linear elastic", "neo-hookean", "linear viscoelastic",
                          "hughes-winget, 1st", "hughes-winget", "rashid, newton+eig",
                          "rashid, eig", "rashid, newton", "polar alone, newton", "polar alone, eig"};
  constexpr int rows = 10, models = 8;
  auto time_qfunctions = [&](QuadratureData<mem> & q, int n, auto launch) {
    auto stateful = [&](auto model) { return best_of(n, [&]() { launch(model, q.du_dX, q.du_dX_old, q.state_old, q.P, q.state); }); };
    return std::array< std::array< double, 2 >, rows >{{
      {best_of(n, [&]() { launch(Stress<LinearElasticModel>{elastic}, q.du_dX, q.P); }),
       best_of(n, [&]() { launch(Tangent<LinearElasticModel>{elastic}, q.du_dX, q.dP_dF); })},
      {best_of(n, [&]() { launch(Stress<NeoHookeanModel>{neohookean}, q.du_dX, q.P); }),
       best_of(n, [&]() { launch(Tangent<NeoHookeanModel>{neohookean}, q.du_dX, q.dP_dF); })},
      {best_of(n, [&]() { launch(viscoelastic, q.du_dX, q.state_old, q.P, q.state); }),
       best_of(n, [&]() { launch(Tangent<LinearViscoelasticModel>{viscoelastic}, q.du_dX, q.dP_dF); })},
      {stateful(hughes_winget_1), -1.0},
      {stateful(hughes_winget), -1.0},
      {stateful(rashid_newton_eig), -1.0},
      {stateful(rashid_eig), -1.0},
      {stateful(rashid_newton), -1.0},
      {best_of(n, [&]() { launch(StepRotation<false>{}, q.du_dX, q.du_dX_old, q.state); }), -1.0},
      {best_of(n, [&]() { launch(StepRotation<true>{}, q.du_dX, q.du_dX_old, q.state); }), -1.0},
    }};
  };
  auto plain = [](auto f, auto & ... arrays) { forall(f, arrays ...); };
  auto staged = [](auto f, auto & ... arrays) { forall_staged(f, arrays ...); };

  auto times = time_qfunctions(full, reps, plain);

  // forall_staged() is forall() on the host, so it only has its own times on the gpu
  std::array< std::array< double, 2 >, rows > staged_times;
  for (auto & row : staged_times) { row = {-1.0, -1.0}; }
  if constexpr (mem != memory::space::cpu) {
    staged_times = time_qfunctions(full, reps, staged);
  }

  // the arithmetic alone: one core, on a block of points small enough that
  // forall() runs it on the calling thread and it stays in that core's cache
  constexpr uint32_t nb = 1024;
  std::array< std::array< double, 2 >, rows > block;
  for (auto & row : block) { row = {-1.0, -1.0}; }
  if constexpr (mem == memory::space::cpu) {
    QuadratureData<mem> q(nb);
    for (uint32_t i = 0; i < q.du_dX.sz; i++) {
      q.du_dX.data()[i] = full.du_dX.data()[i];
      q.du_dX_old.data()[i] = full.du_dX_old.data()[i];
    }
#ifdef __linux__
    // on a hybrid cpu the calling thread can otherwise sit on an efficiency
    // core for the whole measurement and come out 2x slow; cpu 0 is a
    // performance core on Intel's.  Only this thread moves, not the pool's.
    cpu_set_t was, cpu0;
    sched_getaffinity(0, sizeof(was), &was);
    CPU_ZERO(&cpu0);
    CPU_SET(0, &cpu0);
    sched_setaffinity(0, sizeof(cpu0), &cpu0);
#endif
    block = time_qfunctions(q, 20 * reps, plain);
#ifdef __linux__
    sched_setaffinity(0, sizeof(was), &was);
#endif
  }

  // a value with the given format, or "-" where there is none
  auto cell = [](const char * format, double value) {
    char buffer[32] = "-";
    if (value >= 0.0) { snprintf(buffer, 32, format, value); }
    return std::string(buffer);
  };
  auto ms = [&](double t) { return cell("%.2f", 1e3 * t); };
  auto total = [&](bool has, double t, double others) { return cell("%.2f", has && t >= 0.0 ? 1e3 * (t + others) : -1.0); };

  printf("\n%s, %s, p = %u: %u elements, %u quadrature points, %u unknowns\n",
         device, name, p, host_mesh.hex.shape[0] + host_mesh.tet.shape[0], nq, host_u.size());
  printf("  interpolate du_dX    %9.3f ms\n", 1e3 * interpolate);
  printf("  integrate residual   %9.3f ms\n", 1e3 * integrate_residual);
  printf("  integrate stiffness  %9.3f ms\n", 1e3 * integrate_stiffness);
  printf("                                residual (ms)                      stiffness (ms)            1 core (ns/qpt)\n");
  printf("                           q-function       total            q-function        total\n");
  printf("  model                 forall  staged  forall  staged     forall  staged   forall   staged  residual tangent\n");
  for (int m = 0; m < rows; m++) {
    bool model = (m < models);
    double residual_others = interpolate + integrate_residual, tangent_others = interpolate + integrate_stiffness;
    printf("  %-20s %7s %7s %7s %7s    %7s %7s %8s %8s  %8s %7s\n", names[m],
           ms(times[m][0]).c_str(), ms(staged_times[m][0]).c_str(),
           total(model, times[m][0], residual_others).c_str(), total(model, staged_times[m][0], residual_others).c_str(),
           ms(times[m][1]).c_str(), ms(staged_times[m][1]).c_str(),
           total(model, times[m][1], tangent_others).c_str(), total(model, staged_times[m][1], tangent_others).c_str(),
           cell("%.1f", block[m][0] < 0.0 ? -1.0 : 1e9 * block[m][0] / nb).c_str(),
           cell("%.1f", block[m][1] < 0.0 ? -1.0 : 1e9 * block[m][1] / nb).c_str());
  }
  fflush(stdout);
}

int main(int argc, char ** argv) {

  double amplitude = (argc > 1) ? atof(argv[1]) : 0.05;
  int reps = (argc > 2) ? atoi(argv[2]) : 5;

  printf("displacement amplitude %g, best of %d\n", amplitude, reps);

  // about 1-2 million quadrature points each, far more data than any cache holds
  {
    Mesh<> mesh = Mesh<>::cuboid({64, 64, 64}, vec3{1.0, 1.0, 1.0});
    run_case<1>("hex", mesh, amplitude, reps);
  }
  {
    Mesh<> mesh = Mesh<>::cuboid({36, 36, 36}, vec3{1.0, 1.0, 1.0});
    run_case<2>("hex", mesh, amplitude, reps);
  }
  {
    Mesh<> mesh = fusion::tet_cuboid(40);
    run_case<1>("tet", mesh, amplitude, reps);
  }
  {
    Mesh<> mesh = fusion::tet_cuboid(30);
    run_case<2>("tet", mesh, amplitude, reps);
  }

}
