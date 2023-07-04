
#pragma once
#include <cstdint>

using dim_t = int32_t;

// NOLINTBEGIN
// clang-format off
enum class Device : std::int8_t{ 
  CPU, 
  GPU 
};
// clang-format on
// NOLINTEND
