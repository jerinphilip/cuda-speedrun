#include <cuda.h>

#include <cstdio>
#include <numeric>
#include <vector>

#include "fcl/buffer.h"
#include "fcl/error.h"
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
  //
  // This can be detected using the following:
  //
  //    gpuErrchk(cudaPeekAtLastError());

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
  gpuErrchk(cudaPeekAtLastError());

  Timer fp;
  Buffer<int> pc = pipelined();
  double pc_runtime = fp.elapsed() * 1000;
  gpuErrchk(cudaPeekAtLastError());

  auto validate = [&](const Buffer<int> &gc) -> bool {
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

void matrix_init() {
  // Setting this to 32 because 32*32 = 2 * 1024
  // Doing larger than 2 does not work because of the 1024 threads restriction.

  constexpr size_t M = 32, N = 32, P = 32;
  Buffer<int> A(M * N, Device::CPU);
  Buffer<int> B(N * P, Device::CPU);

  auto gA = A.to(Device::GPU);
  auto gB = A.to(Device::GPU);

  if (false) {
    scalar_init<<<M, N>>>(gA.data());
    scalar_init<<<N, P>>>(gB.data());
  } else {
    // If I want the launch configuration to be <<<2, X>>>, what is X?  The rest
    // of the code should be intact.

    // Some factor math of the following appears to be working.
    scalar_init<<<2, N * M / 2>>>(gA.data());
    scalar_init<<<2, P * N / 2>>>(gB.data());
  }

  auto iA = gA.to(Device::CPU);
  auto iB = gA.to(Device::CPU);

  std::cout << iA << "\n";
  std::cout << iB << "\n";
}

void hello_world() {
  print_hello_world<<<1, 1>>>();
  cudaDeviceSynchronize();
}

void matrix_squaring() {
  auto matrix_square_cpu = [](const int *A, size_t N, int *B) {
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < N; ++j) {
        for (size_t k = 0; k < N; ++k) {
          B[i * N + j] += A[i * N + k] * A[k * N + j];
        }
      }
    }
  };

  constexpr size_t N = 64;

  Buffer<int> A(N * N, Device::CPU);

  // Initialize.
  // std::iota(A.data(), A.data() + A.size(), 1);
  std::fill(A.data(), A.data() + A.size(), 1);

  auto gA = A.to(Device::GPU);

  Buffer<int> B(N * N, Device::CPU);
  std::fill(B.data(), B.data() + B.size(), 0);

  auto gB_v1 = B.to(Device::GPU);
  auto gB_v2 = B.to(Device::GPU);

  Timer cpu_timer;
  matrix_square_cpu(A.data(), N, B.data());
  double cpu_time = cpu_timer.elapsed() * 1000;

  Timer gpu_timer_v1;
  matrix_square_v1<<<1, N>>>(gA.data(), N, gB_v1.data());
  double gpu_time_v1 = gpu_timer_v1.elapsed() * 1000;

  Timer gpu_timer_v2;
  matrix_square_v2<<<N, N>>>(gA.data(), N, gB_v2.data());
  double gpu_time_v2 = gpu_timer_v2.elapsed() * 1000;

  auto C_v1 = gB_v1.to(Device::CPU);
  auto C_v2 = gB_v2.to(Device::CPU);

  int *pb = B.data(), *pc_v1 = C_v1.data(), *pc_v2 = C_v2.data();  // NOLINT
  for (size_t i = 0; i < N * N; i++) {
    if (*pb != *pc_v1 || *pb != *pc_v2) {
      fprintf(stderr, "mismatch at %u: (b: %d, c_v1: %d, c_v2: %d)\n", i, *pb,
              *pc_v1, *pc_v2);
    }
    ++pb, ++pc_v1, ++pc_v2;
  }
  fprintf(stderr,
          "completed matrix_squaring: cpu %lfms gpu_v1 %lfms gpu_v2 %lfms\n",
          cpu_time, gpu_time_v1, gpu_time_v2);
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
