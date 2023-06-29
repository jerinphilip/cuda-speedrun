#include <cuda.h>

#include <cstdio>
#include <vector>

#include "buffer.h"

#define N 1000

__global__ void vsqr_(int *A) {  // NOLINT
  // Kernel computes vsqr for one  data-item.
  int x = A[threadIdx.x];
  A[threadIdx.x] = x * x;
}

__global__ void vcube_(int *A) {  // NOLINT
  // Kernel computes vsqr for one  data-item.
  int x = A[threadIdx.x];
  A[threadIdx.x] = x * x * x;
}

__global__ void vadd(const int *A, const int *B, int *C) {
  // Kernel computes vsqr for one  data-item.
  int i = threadIdx.x;
  C[i] = A[i] + B[i];
}

__global__ void fused_sqr_cub_add(const int *A, const int *B, int *C) {
  int i = threadIdx.x;
  int x = A[i], y = B[i];  // NOLINT
  C[i] = x * x + y * y * y;
}

std::vector<int> generate(size_t size) {
  std::vector<int> data(size);
  for (size_t i = 0; i < size; i++) {
    data[i] = i;
  }
  return data;
}

int main() {
  // Copy data to a GPU memory buffer.
  std::vector<int> first = generate(N);
  std::vector<int> second = generate(N);

  Buffer<int> g_first(first.data(), first.size());
  Buffer<int> g_second(second.data(), second.size());

  auto pipelined = [&]() {
    Buffer<int> g_result(N);
    vsqr_<<<1, N>>>(g_first.data());
    vcube_<<<1, N>>>(g_second.data());
    vadd<<<1, N>>>(g_first.data(), g_second.data(), g_result.data());
    return g_result;
  };

  auto fused = [&]() {
    Buffer<int> g_result(N);
    fused_sqr_cub_add<<<1, N>>>(g_first.data(), g_second.data(),
                                g_result.data());
    return g_result;
  };

  // Buffer<int> g_result = pipelined();
  Buffer<int> g_result = fused();

  Buffer<int> result = g_result.cpu();
  for (size_t i = 0; i < result.size(); i++) {  // NOLINT
    int x = first[i], y = second[i];            // NOLINT
    if (result[i] != x * x + y * y * y) {
      fprintf(stderr, "Mismatch found.\n");
      std::abort();
    };
    printf("%d\n", result[i]);
  }
  return 0;
}
