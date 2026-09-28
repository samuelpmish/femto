#include <iostream>

#include "containers/ndarray.hpp"

#include <gtest/gtest.h>

TEST(numpy_test, save_and_load) { 

  nd::cpu_array<uint64_t, 2> original_data(std::vector<uint64_t>{1,2,3,4,5,6,7,8,9,10}, {5, 2});
  save("test_data.npy", original_data);

  auto imported_data = load< uint64_t, 2 >(std::string("test_data.npy"));

  EXPECT_TRUE(original_data.shape[0] == imported_data.shape[0]);
  EXPECT_TRUE(original_data.shape[1] == imported_data.shape[1]);
  EXPECT_TRUE(original_data.strides[0] == imported_data.strides[0]);
  EXPECT_TRUE(original_data.strides[1] == imported_data.strides[1]);

  for (int i = 0; i < original_data.shape[0]; i++) {
    for (int j = 0; j < original_data.shape[1]; j++) {
      EXPECT_TRUE(original_data(i,j) == imported_data(i,j));
    }
  }
  
}