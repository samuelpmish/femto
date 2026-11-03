// SPDX-FileCopyrightText: Copyright (c) 2023-2026 Sam Mish
// SPDX-License-Identifier: MIT

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