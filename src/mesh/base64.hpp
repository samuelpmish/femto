// SPDX-FileCopyrightText: Copyright (c) 2023-2026 Sam Mish
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <string>
#include <cinttypes>

namespace Base64 {
  std::string Encode(const uint32_t & data);
  std::string Encode(const std::vector<uint8_t> & data);
  std::vector<uint8_t> Decode(const std::string & input);
  std::vector<uint8_t> Decode(const std::vector < std::string > & inputs);
};