// performance breakdown of the coffee mug solid mechanics analysis, matching
// what the web demo runs for the default slider settings:
//
//   Mesh::coffee_mug(3, 4, 9, 0.5, 1 handle, p = 1)
//   analyze:   assemble M and K, Cholesky factorization of M
//              (rho = 2.4 g/cm^3, E = 100 GPa, nu = 0.3, CGS units)
//   transient: 1000 central difference steps, 5 m/s rim strike
//   modal:     first 10 non-rigid eigenpairs of (K, M)
//
// run directly (not part of ctest):  ./tests/mug_analysis_perf_tests
//
// the timings come from the same instrumentation the wasm build uses
// (examples/cpp/mug_analysis.hpp), so hot spots found here map 1:1 onto the browser.

#include "gtest/gtest.h"

#include "../../examples/cpp/mug_analysis.hpp"

#include "femto/threadpool.hpp"

#include <cstdio>
#include <thread>

using namespace femto;

static constexpr double rho = 2.4;
static constexpr double E = 1.0e12;
static constexpr double nu = 0.3;

static void row(const char * label, double ms, double total_ms) {
  printf("    %-28s %8.1f ms  (%4.1f%%)\n", label, ms, 100.0 * ms / total_ms);
}

// cost of an (almost) empty parallel region: fork + join synchronization
static double region_overhead_us(int reps) {
  static double sink[64];
  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < reps; r++) {
    threadpool::block_parallel_for(64, [&](uint32_t begin, uint32_t end) {
      for (uint32_t i = begin; i < end; i++) { sink[i] += 1.0; }
    });
  }
  return 1000.0 * mug::milliseconds_since(t0) / reps;
}

static void run_breakdown(const char * label, uint32_t num_threads) {

  threadpool::set_num_threads(num_threads);
  printf("\n===== %s, %u thread%s =====\n", label, num_threads, num_threads == 1 ? "" : "s");
  printf("  fork/join overhead: %.1f us per region\n", region_overhead_us(500));

  auto t0 = std::chrono::steady_clock::now();
  Mesh<> mesh = Mesh<>::coffee_mug(3.0, 4.0, 9.0, 0.5, 1, 1);
  double mesh_ms = mug::milliseconds_since(t0);
  printf("  mesh generation: %.1f ms  (%u hexes, %u vertices)\n",
         mesh_ms, mesh.hex.shape[0], mesh.vert.shape[0]);

  auto a = std::make_shared<mug::Analysis>(mug::analyze(mesh, rho, E, nu));
  double analyze_total = a->assemble_ms;
  printf("  analyze: %.1f ms  (%u dofs, %u nonzeros)\n", analyze_total, a->ndof, uint32_t(a->K.nnz));
  row("field/domain setup", a->setup_ms, analyze_total);
  row("K assembly", a->K_ms, analyze_total);
  row("M assembly", a->M_ms, analyze_total);

  mug::TransientSim sim(a, mesh, 1.0e6);
  sim.advance(1000);
  printf("  transient: setup %.1f ms, %u steps in %.1f ms  (dt = %.1f ns)\n",
         sim.setup_ms, sim.step_count, sim.step_ms, sim.dt * 1e9);

  mug::ModalResult mo = mug::modal(*a, 10);
  double mo_other = mo.solve_ms - mo.factor_ms - mo.apply_ms - mo.mgs_ms - mo.ritz_ms;
  printf("  modal: %.1f ms  (%u Lanczos iterations, f1 = %.0f Hz)\n",
         mo.solve_ms, mo.iterations, std::sqrt(mo.eigenvalues[0]) / (2 * M_PI));
  row("Cholesky(K + sigma M)", mo.factor_ms, mo.solve_ms);
  row("operator applies", mo.apply_ms, mo.solve_ms);
  row("M-orthonormalization", mo.mgs_ms, mo.solve_ms);
  row("Rayleigh-Ritz", mo.ritz_ms, mo.solve_ms);
  row("convergence checks + misc", mo_other, mo.solve_ms);

  // light correctness checks so a broken build can't silently "pass"
  EXPECT_GT(mo.eigenvalues[0], 1e8);
  EXPECT_LT(mo.eigenvalues[0], 1e9);
  EXPECT_GT(sim.max_displacement, 0.0);

}

TEST(mug_analysis_perf, breakdown) {

  uint32_t hw = std::thread::hardware_concurrency();

  run_breakdown("thread pool", 1);
  run_breakdown("thread pool", hw);

  // fork/join overhead across thread counts
  printf("\n===== fork/join overhead (us per region) =====\n");
  printf("  %8s %12s\n", "threads", "overhead");
  for (uint32_t t : {1u, 2u, 4u, 8u, 16u, hw}) {
    threadpool::set_num_threads(t);
    printf("  %8u %10.1f\n", t, region_overhead_us(500));
  }

  threadpool::set_num_threads(hw);

}
