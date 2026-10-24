#pragma once

#include "fm/types/matrix.hpp"

// the deviatoric part of a square matrix, A - tr(A) / n I.  Nothing here but
// fm, so it can move there.
namespace femto {

using namespace fm;

template < Kind kind, u32 n, typename T >
__host__ __device__ constexpr auto dev(const matrix< kind, n, n, T > & A) {
  return A - (tr(A) / n) * Identity< n, T >();
}

} // namespace femto
