#include <cuda.h>

#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

#include "fcl/buffer.h"
#include "fcl/error.h"
#include "fcl/kernels.h"
#include "fcl/timer.h"

#define TRACE(x)                          \
  do {                                    \
    std::cerr << #x << ": " << x << "\n"; \
  } while (0)

void fill_random_int(int *A, dim_t size, int max_value = 1e9) {
  std::mt19937_64 generator(/*seed=*/42);
  for (size_t i = 0; i < size; i++) {
    A[i] = generator() % max_value;
  }
}

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

  constexpr dim_t size = 1024;  // NOLINT
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

  Timer<Device::GPU> ft;
  Buffer<int> fc = fused();
  double fc_runtime = ft.elapsed() * 1000;
  gpuErrchk(cudaPeekAtLastError());

  Timer<Device::GPU> fp;
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
    for (dim_t i = 0; i < c.size(); i++) {               // NOLINT
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

  constexpr dim_t M = 32, N = 32, P = 32;
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
  auto matrix_square_cpu = [](const int *A, dim_t N, int *B) {
    for (dim_t i = 0; i < N; ++i) {
      for (dim_t j = 0; j < N; ++j) {
        for (dim_t k = 0; k < N; ++k) {
          B[i * N + j] += A[i * N + k] * A[k * N + j];
        }
      }
    }
  };

  constexpr dim_t N = 64;

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

  // Version 1 is 1xN dispatch.
  // Each of the 1xN parallelizes the outer loop with index i.
  Timer<Device::GPU> gpu_timer_v1;
  matrix_square_v1<<<1, N>>>(gA.data(), N, gB_v1.data());
  double gpu_time_v1 = gpu_timer_v1.elapsed() * 1000;

  // Version 2 is NxN dispatch.
  // Each of the NxN retrieves i, j as idx/N and idx%N
  // And executes the loop over k.
  Timer<Device::GPU> gpu_timer_v2;
  matrix_square_v2<<<N, N>>>(gA.data(), N, gB_v2.data());
  double gpu_time_v2 = gpu_timer_v2.elapsed() * 1000;

  auto C_v1 = gB_v1.to(Device::CPU);
  auto C_v2 = gB_v2.to(Device::CPU);

  int *pb = B.data(), *pc_v1 = C_v1.data(), *pc_v2 = C_v2.data();  // NOLINT
  for (dim_t i = 0; i < N * N; i++) {
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

void warp_with_conditions() {}

void aos_vs_soa() {
  constexpr dim_t N = 1024;

  // Array of structures (AoS)
  Timer<Device::GPU> aos_timer;
  Buffer<Node> aos_nodes(N, Device::GPU);
  aos_pass<<<1, N>>>(aos_nodes.data(), aos_nodes.size());
  double aos_time = aos_timer.elapsed() * 1000;

  // Structure of Arrays (SoA)
  Timer<Device::GPU> soa_timer;
  Buffer<int> is(N, Device::GPU);
  Buffer<double> ds(N, Device::GPU);
  Buffer<char> cs(N, Device::GPU);
  Nodes soa_nodes = {
      .is = is.data(),  //
      .ds = ds.data(),  //
      .cs = cs.data()   //
  };

  double soa_time = soa_timer.elapsed() * 1000;
  soa_pass<<<1, N>>>(soa_nodes.is, soa_nodes.ds, soa_nodes.cs, N);

  fprintf(stderr, "Time aos = %lf, soa = %lf\n", aos_time, soa_time);
}

void maximum_in_a_large_array() {
  constexpr dim_t N = 1024;
  constexpr dim_t K = 32;

  assert(N % K == 0);
  constexpr dim_t partitions = N / K;

  Buffer<int> cx(N, Device::CPU);
  fill_random_int(cx.data(), cx.size());

  std::cout << "xs: " << cx << "\n";

  auto xs = cx.to(Device::GPU);
  Buffer<int> ys(partitions, Device::GPU);
  maximum_in_a_large_array_kernel<<<1, partitions>>>(xs.data(), xs.size(), K,
                                                     ys.data());

  Buffer<int> z(1, Device::GPU);
  maximum_in_a_large_array_kernel<<<1, 1>>>(ys.data(), ys.size(), K, z.data());

  std::cout << "ys: " << ys << "\n";
  std::cout << "z: " << z << "\n";
}

void find_element() {
  constexpr dim_t N = 1024;
  constexpr dim_t K = 32;

  assert(N % K == 0);
  constexpr dim_t partitions = N / K;
  Buffer<int> cx(N, Device::CPU);
  std::iota(cx.data(), cx.data() + cx.size(), 0);

  auto xs = cx.to(Device::GPU);
  Buffer<int> out(1, Device::GPU);

  const int query = 54;
  find_element_kernel<<<1, partitions>>>(xs.data(), xs.size(), K, query,
                                         out.data());

  std::cout << "out: " << out << "\n";
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
