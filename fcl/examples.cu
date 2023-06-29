#include <cuda.h>

#include <cstdio>
#include <numeric>
#include <vector>

#include "fcl/buffer.h"
#include "fcl/kernels.h"

void compare_fused_separate() {
  constexpr size_t size = 1000;  // NOLINT
  Buffer<int> a(size, Device::CPU);
  std::iota(a.data(), a.data() + a.size(), 0);
  auto ga = a.to(Device::GPU);

  Buffer<int> b(size, Device::CPU);
  std::iota(b.data(), b.data() + b.size(), 0);
  auto gb = b.to(Device::GPU);

  auto pipelined = [&]() {
    Buffer<int> c(size, Device::GPU);
    vsqr_<<<1, size>>>(ga.data());
    vcube_<<<1, size>>>(gb.data());
    vadd<<<1, size>>>(ga.data(), gb.data(), c.data());
    return c;
  };

  auto fused = [&]() {
    Buffer<int> c(size, Device::GPU);
    fused_sqr_cub_add<<<1, size>>>(ga.data(), gb.data(), c.data());
    return c;
  };

  Buffer<int> pc = pipelined();
  Buffer<int> fc = fused();

  auto validate = [&](Buffer<int> &gc) {
    Buffer<int> c = gc.to(Device::CPU);
    std::cout << c << "\n";

    int *px = a.data(), *py = b.data(), *pz = c.data();  // NOLINT
    for (size_t i = 0; i < c.size(); i++) {              // NOLINT
      int x = *px, y = *py, z = *pz;                     // NOLINT
      int expected = x * x + y * y * y;
      if (z != expected) {
        printf("computed %d != %d expected\n", z, expected);
        fprintf(stderr, "Mismatch found.\n");
      };
      ++px, ++py, ++pz;
    }
  };

  validate(fc);
  validate(pc);
}
