#include "3rd-party/CLI11.hpp"
#include "fcl/examples.h"

int main(int argc, char **argv) {
  CLI::App app{"App description"};

  std::string example;
  app.add_option("-e,--fn", example, "Example to execute")->required();
  CLI11_PARSE(app, argc, argv);

  if (example == "compare_fused_separate") {
    compare_fused_separate();
  }

  return 0;
}
