#include "containers/ndarray.hpp"

#include "gtest/gtest.h"

#include <iostream>

template < typename T, uint32_t n >
stack::array<T, n> remove_ones(const stack::array<T, n> & x) { 
  uint32_t rank = 0;
  stack::array<T, n> copy{};
  for (int i = 0; i < n; i++) {
    if (x[i] > 1) copy[rank++] = x[i];
  }
  return copy;
}

template < typename T, uint32_t m, uint32_t n >
bool compatible_shapes(const stack::array<T, m> & x, 
                       const stack::array<T, n> & y) {
  auto x_filtered = remove_ones(x);
  auto y_filtered = remove_ones(y);
  for (int i = 0; i < std::max(m, n); i++) {
    auto xval = (i >= m) ? 0 : x_filtered[i];
    auto yval = (i >= n) ? 0 : y_filtered[i];
    if (xval != yval) { return false; }
  }
  return true;
}

TEST(remove_ones_tests, no_ones_2D) {
  EXPECT_EQ((stack::array{3, 2}), (stack::array{3, 2}));
}

TEST(remove_ones_tests, no_ones_3D) {
  EXPECT_EQ((stack::array{3, 2, 4}), (stack::array{3, 2, 4}));
}

#define COMPATIBLE(name, x, y)          \
TEST(ReshapeCompatibility, name) {      \
  EXPECT_TRUE(compatible_shapes(x, y)); \
}

#define INCOMPATIBLE(name, x, y)         \
TEST(ReshapeIncompatibility, name) {     \
  EXPECT_FALSE(compatible_shapes(x, y)); \
}

COMPATIBLE(Trivial2D, (stack::array{2, 3}),          (stack::array{2, 3}));
COMPATIBLE(Trivial3D, (stack::array{2, 3, 4}),       (stack::array{2, 3, 4}));
COMPATIBLE(Trivial4D, (stack::array{2, 3, 4, 5}),    (stack::array{2, 3, 4, 5}));
COMPATIBLE(Trivial5D, (stack::array{2, 3, 4, 5, 6}), (stack::array{2, 3, 4, 5, 6}));

COMPATIBLE(Degenerate3D1, (stack::array{50, 3}),          (stack::array{50, 1, 3}));
COMPATIBLE(Degenerate3D2, (stack::array{50, 3}),          (stack::array{50, 3, 1}));

COMPATIBLE(Degenerate4D1, (stack::array{50, 3, 2}), (stack::array{50, 1, 3, 2}));
COMPATIBLE(Degenerate4D2, (stack::array{50, 3, 2}), (stack::array{50, 3, 1, 2}));
COMPATIBLE(Degenerate4D3, (stack::array{50, 3, 2}), (stack::array{50, 3, 2, 1}));

COMPATIBLE(Degenerate5D1, (stack::array{50, 3, 2, 4}), (stack::array{50, 1, 3, 2, 4}));
COMPATIBLE(Degenerate5D2, (stack::array{50, 3, 2, 4}), (stack::array{50, 3, 1, 2, 4}));
COMPATIBLE(Degenerate5D3, (stack::array{50, 3, 2, 4}), (stack::array{50, 3, 2, 1, 4}));
COMPATIBLE(Degenerate5D4, (stack::array{50, 3, 2, 4}), (stack::array{50, 3, 2, 4, 1}));

////////////////////////////////////////////////////////////////////////////////

INCOMPATIBLE(Trivial2D, (stack::array{3, 2}),          (stack::array{2, 3}));
INCOMPATIBLE(Trivial3D, (stack::array{3, 2, 4}),       (stack::array{2, 3, 4}));
INCOMPATIBLE(Trivial4D, (stack::array{3, 2, 4, 5}),    (stack::array{2, 3, 4, 5}));
INCOMPATIBLE(Trivial5D, (stack::array{3, 2, 4, 5, 6}), (stack::array{2, 3, 4, 5, 6}));

INCOMPATIBLE(WrongRank3D, (stack::array{50, 2}), (stack::array{50, 2, 2}));
INCOMPATIBLE(Degenerate5D, (stack::array{50, 1}), (stack::array{50, 1, 2, 1, 2}));