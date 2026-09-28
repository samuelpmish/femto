#pragma once

#if defined(FEMTO_ENABLE_CUDA) && __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define FEMTO_NVTX_ENABLED 1
#else
#define FEMTO_NVTX_ENABLED 0
#endif

namespace femto::nvtx {

struct scoped_range {
  explicit scoped_range(const char * name) {
#if FEMTO_NVTX_ENABLED
    nvtxRangePushA(name);
#else
    (void)name;
#endif
  }

  ~scoped_range() {
#if FEMTO_NVTX_ENABLED
    nvtxRangePop();
#endif
  }

  scoped_range(const scoped_range &) = delete;
  scoped_range & operator=(const scoped_range &) = delete;
};

} // namespace femto::nvtx

#undef FEMTO_NVTX_ENABLED
