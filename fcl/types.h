
#pragma once
#include <cstdint>
#include <iostream>

using dim_t = int32_t;

// NOLINTBEGIN
// clang-format off
enum class Device : std::int8_t{ 
  CPU, 
  GPU 
};
// clang-format on
// NOLINTEND
//
struct Point {
  int x;
  int y;
};

std::ostream &operator<<(std::ostream &out, const Point &point);
