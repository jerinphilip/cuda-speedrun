#include "3rd-party/CLI11.hpp"
#include "fcl/examples.h"

int main(int argc, char **argv) {
  CLI::App app{"App description"};

  std::string example;
  app.add_option("-e,--fn", example, "Example to execute")->required();
  CLI11_PARSE(app, argc, argv);

// Macro abuse.
#define ADD_FN(fn)            \
  do {                        \
    if (example == #fn) fn(); \
  } while (0)

  ADD_FN(compare_fused_separate);
  ADD_FN(occupancy_info);
  ADD_FN(hello_world);
  ADD_FN(matrix_init);

  return 0;
}
