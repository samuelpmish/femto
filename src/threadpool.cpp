#include "femto/threadpool.hpp"

namespace femto {

namespace threadpool {

#ifdef FEMTO_SINGLE_THREADED

  void set_num_threads(uint32_t) {}

#else

BS::thread_pool & bs_pool() {
  static BS::thread_pool pool;
  return pool;
}

void set_num_threads(uint32_t num_threads) {
  bs_pool().reset(num_threads);
}

#endif

}

}
