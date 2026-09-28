#pragma once

#include <string>
#include <iostream>

namespace femto {
  inline void error(std::string message) {
    std::cerr << message << std::endl;
    exit(1);
  }
}

#define CHECK_FOR_SIZE_MISMATCH(m, n) \
if (m != n) { \
  femto::error(std::string("size mismatch at ") + __FILE__ + ":" + std::to_string(__LINE__)); \
}

// TODO: prefer std::source_location when adopting c++20
// (implemented since clang-16, g++-11, msvc-19.29)
#define FEMTO_ASSERT(condition, message) \
if (!(condition)) { \
  femto::error(std::string("Assertion failed at ") + __FILE__ + ":" + std::to_string(__LINE__) + ": " + message); \
}

#define FEMTO_ERROR(message) \
femto::error(std::string("error at ") + __FILE__ + ":" + std::to_string(__LINE__) + ": " + message);
