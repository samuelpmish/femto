// instantiate cuBQL's device builders for the BVH type femto uses (see
// bvh.hpp). Only the builders femto needs are instantiated here; if another
// one (sahBuilder, rebinRadixBuilder, ...) is needed later, add it to this
// list and to the extern templates in bvh.hpp
#define CUBQL_GPU_BUILDER_IMPLEMENTATION 1
#include "cuBQL/bvh.h"

namespace cuBQL {

  template void gpuBuilder(BinaryBVH<float, 3> &, const box_t<float, 3> *, uint32_t, BuildConfig, cudaStream_t, GpuMemoryResource &);

  namespace cuda {
    template void radixBuilder<float, 3>(BinaryBVH<float, 3> &, const box_t<float, 3> *, uint32_t, BuildConfig, cudaStream_t, GpuMemoryResource &);
    template void free(BinaryBVH<float, 3> &, cudaStream_t, GpuMemoryResource &);
  }

}
