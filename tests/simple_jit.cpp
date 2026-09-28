#if FEMTO_ENABLE_JIT
#include <iostream>

#include "JIT.hpp"

int main() {
  JIT jit({"-O3"});

  jit.compile(R"(
    void foo() {}
  )");

  return 0;
}
#endif