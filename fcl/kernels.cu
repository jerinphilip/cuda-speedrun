#include <cuda.h>

#include <cassert>
#include <cstdio>

#include "fcl/kernels.h"
#include "fcl/utils.h"

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
  dim_t id = blockIdx.x * blockDim.x + threadIdx.x;
  A[id] = static_cast<int>(id);
}

__global__ void matrix_square_v1(const int *A, dim_t N, int *B) {
  dim_t id = blockIdx.x * blockDim.x + threadIdx.x;
  for (dim_t j = 0; j < N; ++j) {
    for (dim_t k = 0; k < N; ++k) {
      B[id * N + j] += A[id * N + k] * A[k * N + j];
    }
  }
}

__global__ void matrix_square_v2(const int *A, dim_t N, int *B) {
  dim_t id = blockIdx.x * blockDim.x + threadIdx.x;
  dim_t i = id / N;
  dim_t j = id % N;
  for (dim_t k = 0; k < N; ++k) {
    B[i * N + j] += A[i * N + k] * A[k * N + j];
  }
}

__global__ void warp_branch_paths(int *A, dim_t size) {
  dim_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id % 2) {
    A[id] = id;
  } else {
    A[id] = size * size;
  }
  A[id]++;
}

__global__ void aos_pass(Node *nodes, dim_t size) {
  dim_t id = blockIdx.x * blockDim.x + threadIdx.x;
  nodes[id].i = id;
  nodes[id].d = 0.0F;
  nodes[id].c = 'c';
}

__global__ void soa_pass(int *is, double *ds, char *cs, dim_t size) {
  dim_t id = blockIdx.x * blockDim.x + threadIdx.x;
  is[id] = id;
  ds[id] = 0.0f;
  cs[id] = 'd';
}

__global__ void maximum_in_a_large_array_kernel(const int *xs, dim_t size,
                                                dim_t partition_size, int *ys) {
  dim_t id = blockIdx.x * blockDim.x + threadIdx.x;

  // threadId i computes for i*32, (i+1)*32
  dim_t start = id * partition_size;
  dim_t end = start + partition_size;

  // Precondition, not checked in non-debug builds.
  assert(end <= size);

  int x_max = xs[start];
  for (dim_t i = start + 1; i < end; i++) {
    x_max = xs[i] > x_max ? xs[i] : x_max;
  }

  // Write output.
  ys[id] = x_max;
}

__global__ void find_element_kernel(const int *xs, dim_t size,
                                    dim_t partition_size, int query, int *out) {
  dim_t id = blockIdx.x * blockDim.x + threadIdx.x;
  dim_t start = id * partition_size;
  dim_t end = start + partition_size;
  for (dim_t i = start; i < end; i++) {
    if (xs[i] == query) {
      *out = i;
      // We're done, return.
      return;
    }
  }
}

__global__ void block_thread_dispatch_identifier() {
  if (threadIdx.x == 0 && blockIdx.x == 0 && threadIdx.y == 0 &&
      blockIdx.y == 0 && threadIdx.z == 0 && blockIdx.z == 0) {
    printf("[narrow] %d %d %d %d %d %d.\n", gridDim.x, gridDim.y, gridDim.z,
           blockDim.x, blockDim.y, blockDim.z);
  }
  printf("[all] threadId = (%d %d %d), blockId =(%d %d %d).\n", threadIdx.x,
         threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);
}

__global__ void add_nearby(int *A, dim_t size) {
  // Exploit shared memory?
  constexpr dim_t SIZE = 32;
  __shared__ int buffer[SIZE];
  dim_t id = blockIdx.x * blockDim.x + threadIdx.x;

  // Each row is assigned to a thread block.
  // Each thread is assigned a matrix element M[i][j].

  // Where do I zero initialize?
  // Does this look efficient enough?
  buffer[id] = 0;
  __syncthreads();

  buffer[id] += A[id];
  if (id + 1 < size) {
    // This condition will be true for most threads, so no need to worry about
    // divergence.
    buffer[id] += A[id + 1];
  }
  __syncthreads();
  A[id] = buffer[id];
}

__global__ void hw_exec_info() {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  printf("(thread: %d, SM: %d, warp-id: %d, warp-lane: %d)\n", idx, __smid(),
         __warpid(), __laneid());
}
