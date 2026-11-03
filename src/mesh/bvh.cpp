// SPDX-FileCopyrightText: Copyright (c) 2023-2026 Sam Mish
// SPDX-License-Identifier: MIT

// instantiate cuBQL's host builder for the BVH type femto uses (see bvh.hpp)
#include "cuBQL/bvh.h"

CUBQL_CPU_INSTANTIATE_BINARY_BVH(float, 3)
