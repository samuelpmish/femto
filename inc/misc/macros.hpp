#pragma once

#ifdef __CUDACC__

#include <cstdio>
#include <cuda.h>

namespace impl {

    // not intended to be used directly, use the CUDA_CHECK macro instead
    inline void cuda_check(cudaError_t code, const char *file, int line) {
        if (code != cudaSuccess) {
            fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(code), file, line);
            exit(code);
        }
    }

}

#define CUDA_CHECK(call) { ::impl::cuda_check((call), __FILE__, __LINE__); }

#else

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#ifndef __forceinline__
#define __forceinline__
#endif

#endif

#define HOSTDEV __host__ __device__
