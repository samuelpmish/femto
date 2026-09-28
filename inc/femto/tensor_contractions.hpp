#pragma once

#include <cassert>
#include "containers/ndarray.hpp"

inline void contract(const nd::view<const double,3> & A, nd::view<double,2> & B, nd::view<double,3> C, bool accumulate = false) {
  //  A1(qx, iz, iy) = B(qx, ix) * in(iz, iy, ix) 
  //  A2(qy, qx, iz) = B(qy, iy) * A1(qx, iz, iy) 
  // out(qz, qy, qx) = B(qz, iz) * A2(qy, qx, iz) 
  uint32_t n0 = C.shape[0];
  uint32_t n1 = C.shape[1];
  uint32_t n2 = C.shape[2];
  uint32_t m = A.shape[2];
  for (uint32_t i0 = 0; i0 < n0; i0++) {
    for (uint32_t i1 = 0; i1 < n1; i1++) {
      for (uint32_t i2 = 0; i2 < n2; i2++) {
        double sum = (accumulate) ? C(i0, i1, i2) : 0.0;
        for (int j = 0; j < m; j++) {
          sum += B(i0, j) * A(i1, i2, j);
        }
        C(i0, i1, i2) = sum;
      }
    }
  }
}

/// C(qx, iy) {=,+=} sum_{ix} A(qx, ix) * B(iy, ix)  
inline void _contract(nd::view<double,2> C, const nd::view<const double,2> & A, const nd::view<const double,2> & B, bool accumulate = false) {
  uint32_t n0 = C.shape[0];
  uint32_t n1 = C.shape[1];
  uint32_t m = A.shape[1];
  assert(A.shape[0] == C.shape[0]);
  assert(B.shape[0] == C.shape[1]);
  assert(A.shape[1] == B.shape[1]);
  for (uint32_t i0 = 0; i0 < n0; i0++) {
    for (uint32_t i1 = 0; i1 < n1; i1++) {
      double sum = (accumulate) ? C(i0, i1) : 0.0;
      for (int j = 0; j < m; j++) {
        sum += A(i0, j) * B(i1, j);
      }
      C(i0, i1) = sum;
    }
  }
}

template < int i >
void contract(const nd::view<const double,2> & A, nd::view<double,2> & B, nd::view<double,2> C, bool accumulate = false) {
  uint32_t n0 = C.shape[0];
  uint32_t n1 = C.shape[1];
  uint32_t m = A.shape[i];
  assert(C.shape[1] == B.shape[0]);
  assert(A.shape[i] == B.shape[1]);
  assert(A.shape[1-i] == C.shape[0]);
  for (uint32_t i0 = 0; i0 < n0; i0++) {
    for (uint32_t i1 = 0; i1 < n1; i1++) {
      double sum = (accumulate) ? C(i0, i1) : 0.0;
      for (int j = 0; j < m; j++) {
        if constexpr (i == 0) { sum += A(j, i0) * B(i1, j); }
        if constexpr (i == 1) { sum += A(i0, j) * B(i1, j); }
      }
      C(i0, i1) = sum;
    }
  }
}

inline void contract(const nd::view<const double> & A, nd::view<double,2> & B, const nd::view<double> C, bool accumulate = false) {
  uint32_t n = C.shape[0];
  uint32_t m = A.shape[0];
  for (uint32_t i = 0; i < n; i++) {
    double sum = (accumulate) ? C(i) : 0.0;
    for (int j = 0; j < m; j++) {
      sum += A(j) * B(i, j);
    }
    C(i) = sum;
  }
}

// out = (Tz ⊗ Ty ⊗ Tx)(in): contract the fastest axis of `in` with the last
// axis of Tx, then the next-fastest with Ty, then the slowest with Tz. The
// output ordering is {Tz-rows, Ty-rows, Tx-rows}. scratch must hold
// Tx.shape[0]*in.shape[0]*in.shape[1] + Ty.shape[0]*Tx.shape[0]*in.shape[0] doubles
inline void tp_apply(const nd::view<const double,3> & in,
                     nd::view<double,2> & Tx, nd::view<double,2> & Ty, nd::view<double,2> & Tz,
                     nd::view<double,3> out, double * scratch, bool accumulate = false) {
  uint32_t a = Tx.shape[0];
  uint32_t b = in.shape[0];
  uint32_t c = in.shape[1];
  uint32_t d = Ty.shape[0];
  contract(in, Tx, nd::view<double,3>(scratch, {a, b, c}));
  contract(nd::view<const double,3>(scratch, {a, b, c}), Ty, nd::view<double,3>(scratch + a*b*c, {d, a, b}));
  contract(nd::view<const double,3>(scratch + a*b*c, {d, a, b}), Tz, out, accumulate);
}

// 2D version: out ordering is {Ty-rows, Tx-rows}.
// scratch must hold Tx.shape[0]*in.shape[0] doubles
inline void tp_apply(const nd::view<const double,2> & in,
                     const nd::view<const double,2> & Tx, const nd::view<const double,2> & Ty,
                     nd::view<double,2> out, double * scratch, bool accumulate = false) {
  uint32_t a = Tx.shape[0];
  uint32_t b = in.shape[0];
  _contract(nd::view<double,2>(scratch, {a, b}), Tx, in);
  _contract(out, Ty, nd::view<const double,2>(scratch, {a, b}), accumulate);
}

#ifdef __CUDACC__
struct threadid {
  int x;
  int y;
  int z;
  int stride;
};

__device__ __forceinline__ void cuda_contract(threadid tid, const nd::view<double,3> & A, const nd::view<double,2> & B, nd::view<double,3> & C, bool accumulate = false) {
  uint32_t n0 = C.shape[0];
  uint32_t n1 = C.shape[1];
  uint32_t n2 = C.shape[2];
  uint32_t m = A.shape[2];

  for (uint32_t i0 = tid.z; i0 < n0; i0 += tid.stride) {
    for (uint32_t i1 = tid.y; i1 < n1; i1 += tid.stride) {
      for (uint32_t i2 = tid.x; i2 < n2; i2 += tid.stride) {
        double sum = (accumulate) ? C(i0, i1, i2) : 0.0;
        for (int j = 0; j < m; j++) {
          sum += B(i0, j) * A(i1, i2, j);
        }
        C(i0, i1, i2) = sum;
      }
    }
  }
  __syncthreads();
}

template < int i >
__device__ __forceinline__ void cuda_contract(threadid tid, const nd::view<double,2> & A, const nd::view<double,2> & B, nd::view<double,2> C, bool accumulate = false) {
  uint32_t n0 = C.shape[0];
  uint32_t n1 = C.shape[1];
  uint32_t m = A.shape[i];
  for (uint32_t i0 = tid.y; i0 < n0; i0 += tid.stride) {
    for (uint32_t i1 = tid.x; i1 < n1; i1 += tid.stride) {
      double sum = (accumulate) ? C(i0, i1) : 0.0;
      for (int j = 0; j < m; j++) {
        if constexpr (i == 0) { sum += A(j, i0) * B(i1, j); }
        if constexpr (i == 1) { sum += A(i0, j) * B(i1, j); }
      }
      C(i0, i1) = sum;
    }
  }
  __syncthreads();
}

// device analog of the 3D tp_apply above (all threads of the block must call this)
__device__ __forceinline__ void cuda_tp_apply(threadid tid, const nd::view<double,3> & in,
                     const nd::view<double,2> & Tx, const nd::view<double,2> & Ty, const nd::view<double,2> & Tz,
                     nd::view<double,3> & out, double * scratch, bool accumulate = false) {
  uint32_t a = Tx.shape[0];
  uint32_t b = in.shape[0];
  uint32_t c = in.shape[1];
  uint32_t d = Ty.shape[0];
  nd::view<double,3> A1(scratch, {a, b, c});
  nd::view<double,3> A2(scratch + a*b*c, {d, a, b});
  cuda_contract(tid, in, Tx, A1);
  cuda_contract(tid, A1, Ty, A2);
  cuda_contract(tid, A2, Tz, out, accumulate);
}

// device analog of the 2D tp_apply above (all threads of the block must call this)
__device__ __forceinline__ void cuda_tp_apply(threadid tid, const nd::view<double,2> & in,
                     const nd::view<double,2> & Tx, const nd::view<double,2> & Ty,
                     nd::view<double,2> out, double * scratch, bool accumulate = false) {
  uint32_t a = Tx.shape[0];
  uint32_t b = in.shape[0];
  nd::view<double,2> A1(scratch, {a, b});
  cuda_contract<1>(tid, Tx, in, A1);
  cuda_contract<1>(tid, Ty, A1, out, accumulate);
}

///// C(qx, iy) {=,+=} sum_{ix} A(qx, ix) * B(iy, ix)
//inline void _contract(nd::view<double,2> C, const nd::view<const double,2> & A, const nd::view<const double,2> & B, bool accumulate = false) {
//  uint32_t n0 = C.shape[0];
//  uint32_t n1 = C.shape[1];
//  uint32_t m = A.shape[1];
//  assert(A.shape[0] == C.shape[0]);
//  assert(B.shape[0] == C.shape[1]);
//  assert(A.shape[1] == B.shape[1]);
//  for (uint32_t i0 = 0; i0 < n0; i0++) {
//    for (uint32_t i1 = 0; i1 < n1; i1++) {
//      double sum = (accumulate) ? C(i0, i1) : 0.0;
//      for (int j = 0; j < m; j++) {
//        sum += A(i0, j) * B(i1, j);
//      }
//      C(i0, i1) = sum;
//    }
//  }
//}
//
//template < int i >
//void contract(const nd::view<const double,2> & A, nd::view<double,2> & B, nd::view<double,2> C, bool accumulate = false) {
//  uint32_t n0 = C.shape[0];
//  uint32_t n1 = C.shape[1];
//  uint32_t m = A.shape[i];
//  assert(C.shape[1] == B.shape[0]);
//  assert(A.shape[i] == B.shape[1]);
//  assert(A.shape[1-i] == C.shape[0]);
//  for (uint32_t i0 = 0; i0 < n0; i0++) {
//    for (uint32_t i1 = 0; i1 < n1; i1++) {
//      double sum = (accumulate) ? C(i0, i1) : 0.0;
//      for (int j = 0; j < m; j++) {
//        if constexpr (i == 0) { sum += A(j, i0) * B(i1, j); }
//        if constexpr (i == 1) { sum += A(i0, j) * B(i1, j); }
//      }
//      C(i0, i1) = sum;
//    }
//  }
//}
//
//inline void contract(const nd::view<const double> & A, nd::view<double,2> & B, const nd::view<double> C, bool accumulate = false) {
//  uint32_t n = C.shape[0];
//  uint32_t m = A.shape[0];
//  for (uint32_t i = 0; i < n; i++) {
//    double sum = (accumulate) ? C(i) : 0.0;
//    for (int j = 0; j < m; j++) {
//      sum += A(j) * B(i, j);
//    }
//    C(i) = sum;
//  }
//}
#endif
