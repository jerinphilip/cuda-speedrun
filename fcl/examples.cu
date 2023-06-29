#include <cuda.h>

#include <cstdio>
#include <numeric>
#include <vector>

#include "fcl/buffer.h"
#include "fcl/kernels.h"
#include "fcl/timer.h"

#define TRACE(x)                          \
  do {                                    \
    std::cerr << #x << ": " << x << "\n"; \
  } while (0)

void compare_fused_separate() {
  // The following exercise is in 2-computation, slide 19.
  // http://www.cse.iitm.ac.in/~rupesh/teaching/gpu/jan23/2-computation.pdf
  //
  //  1. Write a CUDA program to initialize an array of size 32 to all zeros in
  //  parallel.
  //  2. Change the array size to 1024.
  //  3. Create another kernel that adds i to array[i].
  //  4. Change the array size to 8000.
  //  5. Check if answer to problem 3 still works.

  // This does not work for size larger than 1024.
  // https://stackoverflow.com/questions/28928632/cuda-program-not-working-for-more-than-1024-threads
  // gpuErrchk(cudaPeekAtLastError());

  constexpr size_t size = 1024;  // NOLINT
  Buffer<int> a(size, Device::CPU);
  std::fill(a.data(), a.data() + a.size(), 1);
  auto ga = a.to(Device::GPU);

  Buffer<int> b(size, Device::CPU);
  std::fill(b.data(), b.data() + b.size(), 1);
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

  Timer ft;
  Buffer<int> fc = fused();
  double fc_runtime = ft.elapsed() * 1000;

  Timer fp;
  Buffer<int> pc = pipelined();
  double pc_runtime = fp.elapsed() * 1000;

  auto validate = [&](const Buffer<int>& gc) -> bool {
    bool flag = true;
    Buffer<int> c = gc.to(Device::CPU);

    // std::cout << "a: " << a << "\n\n";
    // std::cout << "b: " << b << "\n\n";
    // std::cout << "c: " << c << "\n\n";

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

  std::cout << "Pipelined: " << (pipeline_ret ? "success" : "failure") << " "
            << pc_runtime << "\n";
  std::cout << "Fused: " << (fused_ret ? "success" : "failure") << " "
            << fc_runtime << "\n";
}

void matmul() {
  const size_t M = 1000, N = 1000, P = 1000;
  Buffer<int> A(M * N, Device::CPU);
  Buffer<int> B(N * P, Device::CPU);
}

void occupancy_info() {
  // cudaOccupancyMaxPotentialBlockSizeVariableSMem(
  //     int* minGridSize, int* blockSize, T func,
  //     UnaryFunction blockSizeToDynamicSMemSize, int blockSizeLimit = 0)

  int min_grid_size, block_size;

#define ESTIMATE_KERNEL(kernel)                                                \
  do {                                                                         \
    std::cerr << "kernel: " << #kernel << "\n";                                \
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel, 0, \
                                       0);                                     \
    TRACE(min_grid_size);                                                      \
    TRACE(block_size);                                                         \
    std::cerr << "\n";                                                         \
  } while (0)

  ESTIMATE_KERNEL(fused_sqr_cub_add);
  ESTIMATE_KERNEL(vsqr_);
  ESTIMATE_KERNEL(vsqr);
  ESTIMATE_KERNEL(vcube_);
  ESTIMATE_KERNEL(vadd);

#undef ESTIMATE_KERNEL
}
