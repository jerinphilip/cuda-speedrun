#include <cuda.h>

#include <cstdio>

__global__ void vsqr(int *A) {
  // Kernel computes vsqr for one  data-item.
  A[threadIdx.x] = threadIdx.x * threadIdx.x;
}

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
  int x = A[i];
  int y = B[i];
  C[i] = (x * x) + (y * y * y);
}

__global__ void print_hello_world() {
  // Print hello world.
  // Unsure how this is working, because printf is code that will run on host.
  // Does this mean device can call functions that execute on the host?
  printf("Hello World.\n");
}

__global__ void scalar_init(int *A) {
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  A[id] = static_cast<int>(id);
}
