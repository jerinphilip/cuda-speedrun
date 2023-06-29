#include <cuda.h>

#include <cstdio>
#include <numeric>
#include <vector>

#include "fcl/buffer.h"
#include "fcl/kernels.h"

void compare_fused_separate() {
  constexpr size_t size = 100;  // NOLINT
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

  // Important to run fused after pipelined, because we're using in place vsqr_
  // and vcube_, which will affect outputs if pipelined is ran first.

  Buffer<int> fc = fused();
  Buffer<int> pc = pipelined();

  auto validate = [&](const Buffer<int> &gc) -> bool {
    bool flag = true;
    Buffer<int> c = gc.to(Device::CPU);
    std::cout << "a: " << a << "\n\n";
    std::cout << "b: " << b << "\n\n";
    std::cout << "c: " << c << "\n\n";

    int *px = a.data(), *py = b.data(), *pz = c.data();  // NOLINT
    for (size_t i = 0; i < c.size(); i++) {              // NOLINT
      int x = *px, y = *py, z = *pz;                     // NOLINT
      int expected = x * x + y * y * y;
      if (z != expected) {
        // fprintf(stderr, "computed %d != %d expected (%d, %d)\n", z, expected,
        // x,
        //         y);
        // fprintf(stderr, "Mismatch found.\n");
        flag = false;
      };
      ++px, ++py, ++pz;
    }
    return flag;
  };

  bool pipeline_ret = validate(pc);
  bool fused_ret = validate(fc);

  std::cout << "Pipelined: " << (pipeline_ret ? "success" : "failure") << "\n";
  std::cout << "Fused: " << (fused_ret ? "success" : "failure") << "\n";
}
