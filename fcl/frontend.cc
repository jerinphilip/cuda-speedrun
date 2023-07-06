#include <unordered_set>

#include "3rd-party/CLI11.hpp"
#include "fcl/examples.h"

int main(int argc, char **argv) {
  CLI::App app{"App description"};

  std::string example;
  std::unordered_set<std::string> fns;
  app.add_option("-e,--fn", example, "Example to execute")->required();
  CLI11_PARSE(app, argc, argv);

// Macro abuse.
#define ADD_FN(fn)        \
  fns.insert(#fn);        \
  do {                    \
    if (example == #fn) { \
      fn();               \
      return 0;           \
    }                     \
  } while (0)

  ADD_FN(compare_fused_separate);
  ADD_FN(occupancy_info);
  ADD_FN(hello_world);
  ADD_FN(matrix_init);
  ADD_FN(matrix_squaring);
  ADD_FN(aos_vs_soa);
  ADD_FN(maximum_in_a_large_array);
  ADD_FN(find_element);
  ADD_FN(identifiers);
  ADD_FN(hw_runtime_info);
  ADD_FN(add_nearby_shared_mem);
  ADD_FN(dynamic_shared_mem);
  ADD_FN(constant_memory_example);
  ADD_FN(avg_classwork);

  fprintf(stderr,
          "Unknown example %s called. Please choose from the following.\n",
          example.c_str());
  for (const auto &fn : fns) {
    fprintf(stderr, " - %s\n", fn.c_str());
  }

  return 1;
}
