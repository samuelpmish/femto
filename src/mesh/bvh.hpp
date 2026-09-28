#pragma once

// femto uses cuBQL BVHs built over float3 boxes. The builder implementations
// are compiled exactly once: bvh.cpp holds the host builder, bvh.cu holds the
// device builders. Include this header (instead of defining
// CUBQL_*_BUILDER_IMPLEMENTATION yourself) to use them.
#include "cuBQL/bvh.h"
#include "cuBQL/traversal/fixedBoxQuery.h"

namespace cuBQL {

  namespace cpu {
    extern template void spatialMedian(BinaryBVH<float, 3> &, const box_t<float, 3> *, uint32_t, BuildConfig);
  }

#ifdef __CUDACC__
  extern template void gpuBuilder(BinaryBVH<float, 3> &, const box_t<float, 3> *, uint32_t, BuildConfig, cudaStream_t, GpuMemoryResource &);

  namespace cuda {
    extern template void radixBuilder<float, 3>(BinaryBVH<float, 3> &, const box_t<float, 3> *, uint32_t, BuildConfig, cudaStream_t, GpuMemoryResource &);
    extern template void free(BinaryBVH<float, 3> &, cudaStream_t, GpuMemoryResource &);
  }
#endif

}
