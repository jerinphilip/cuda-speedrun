
#include "fcl/types.h"

#include <iostream>

std::ostream &operator<<(std::ostream &out, const Point &point) {
  out << "(" << point.x << " " << point.y << ")";
  return out;
}
