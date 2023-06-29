#include <cuda.h>

#include <cstdio>
#include <vector>

#include "fcl/buffer.h"
#include "fcl/kernels.h"

void compare_fused_separate() {
  constexpr size_t size = 1000;  // NOLINT
  Buffer<int> a(size, Device::CPU);
  auto ga = a.to(Device::GPU);

  Buffer<int> b(size, Device::CPU);
  auto gb = b.to(Device::GPU);

  auto pipelined = [&]() {
    Buffer<int> c(size, Device::GPU);
    vsqr_<<<1, size>>>(ga.data());
    vcube_<<<1, size>>>(gb.data());
    vadd<<<1, size>>>(ga.data(), gb.data(), c.data());
    return c;
  };

  auto fused = [&]() {
    Buffer<int> result(size, Device::GPU);
    fused_sqr_cub_add<<<1, size>>>(ga.data(), gb.data(), result.data());
    return result;
  };

  // Buffer<int> g_result = pipelined();
  Buffer<int> gc = fused();

  Buffer<int> c = gc.to(Device::CPU);

  int *px = a.data(), *py = b.data(), *pz = c.data();  // NOLINT
  for (size_t i = 0; i < c.size(); i++) {              // NOLINT
    int x = *px, y = *py, z = *pz;                     // NOLINT
    if (z != x * x + y * y * y) {
      fprintf(stderr, "Mismatch found.\n");
      std::abort();
    };
    printf("%d\n", z);
    ++px, ++py, ++pz;
  }
}
