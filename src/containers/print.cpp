#include "containers/ndarray.hpp"

#include <iomanip>
#include <iostream>

#if 0
template < typename T, uint32_t dim >
void print_(nd::view< const T, dim > arr, int depth = 0) {
  if constexpr (dim == 1) {
    std::cout << std::string(2 * depth, ' ');
    std::cout << "{";
    for (int i = 0; i < arr.shape[0]; i++) {
      std::cout << arr(i);
      if (i != arr.shape[0] - 1) { std::cout << ","; }
    }
    std::cout << "}";
  } else {
    const T * ptr = arr.data();
    stack::array< uint32_t, dim - 1 > shape;
    stack::array< uint32_t, dim - 1 > stride;
    for (int i = 0; i < dim - 1; i++) {
      shape[i] = arr.shape[i+1];
      stride[i] = arr.stride[i+1];
    }
    nd::view<const T, dim - 1> slice{ptr, shape, stride};

    std::cout << std::string(2 * depth, ' ');
    std::cout << "{";
    if (arr.shape[0] == 1) {
      print_(slice, 0);
    } else {
      std::cout << std::endl;
      for (int i = 0; i < arr.shape[0]; i++) {
        print_(slice, depth+1);
        if (i != arr.shape[0] - 1) { std::cout << ","; }
        std::cout << std::endl;
        slice.values += arr.stride[0];
      }
      std::cout << std::string(2 * depth, ' ');
    }
    std::cout << "}";
  }
}

void print(nd::view<const uint64_t, 1> arr) { print_(arr); std::cout << std::endl; };
void print(nd::view<const uint64_t, 2> arr) { print_(arr); std::cout << std::endl; };
void print(nd::view<const uint64_t, 3> arr) { print_(arr); std::cout << std::endl; };

void print(nd::view<const double, 1> arr) { print_(arr); std::cout << std::endl; };
void print(nd::view<const double, 2> arr) { print_(arr); std::cout << std::endl; };
void print(nd::view<const double, 3> arr) { print_(arr); std::cout << std::endl; };
void print(nd::view<const double, 4> arr) { print_(arr); std::cout << std::endl; };
#endif