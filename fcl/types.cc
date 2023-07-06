
#include "fcl/types.h"

#include <iostream>

std::ostream &operator<<(std::ostream &out, const Point &point) {
  // Compiler refuses to work with p.x, as it is int and not const int.
  //
  // So some ninja techniques (const_cast) are applied here to cast away the
  // requirement.
  //
  // This is printing code, and point.x is not modified in the process, so
  // should be okay.
  int *x = const_cast<int *>(&point.x);
  int *y = const_cast<int *>(&point.y);
  out << "(" << *x << " " << *y << ")";
  return out;
}
