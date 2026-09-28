#pragma once

#include <cinttypes>
#include <functional>

// FEMTO_SINGLE_THREADED replaces the thread pool with serial loops, for
// platforms without threads (e.g. webassembly builds without pthreads)
#ifndef FEMTO_SINGLE_THREADED
#include "misc/timer.hpp"
#include "misc/BS_thread_pool.hpp"
#endif

namespace femto {

namespace threadpool {

#ifdef FEMTO_SINGLE_THREADED

  void set_num_threads(uint32_t num_threads);

  template < typename callable >
  void parallel_for(uint32_t n, const callable & func) {
    for (uint32_t i = 0; i < n; i++) { func(i); }
  }

  template < typename callable >
  void block_parallel_for(uint32_t n, const callable & func) {
    func(uint32_t(0), n);
  }

#else

  void set_num_threads(uint32_t num_threads);

  BS::thread_pool & bs_pool();  // created on first use

  template < typename callable >
  void parallel_for(uint32_t n, const callable & func) {
    if (n == 0) { return; }
    bs_pool().detach_loop<uint32_t>(0, n, func);
    bs_pool().wait();
  }

  template < typename callable >
  void block_parallel_for(uint32_t n, const callable & func) {
    if (n == 0) { return; }
    bs_pool().detach_blocks<uint32_t>(0, n, func);
    bs_pool().wait();
  }

#endif

}

}
