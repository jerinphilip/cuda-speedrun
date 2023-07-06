
#include "fcl/types.h"

#include <iostream>

std::ostream &operator<<(std::ostream &out, const Point &point) {
  int *x = const_cast<int *>(&point.x);
  int *y = const_cast<int *>(&point.y);
  out << "(" << *x << " " << *y << ")";
  return out;
}
