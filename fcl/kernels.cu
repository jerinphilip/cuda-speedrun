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

__global__ void add_nearby(int *A, dim_t M, dim_t N) {
  // A is an M x N matrix, assuming row-major indexing for convenience.

  // Each block corresponds to a row.
  // Each thread is assigned a matrix element M[i, j]

  // Buffer holds the copy of a row M[i, :]
  // There are N items in the row.

  // Shared Memory:
  // * Programmable L1 cache / Scratchpad memory
  // * Accessible only in a thread block
  // * Useful for repeated small data or coordination
  constexpr dim_t SIZE = 1024;
  __shared__ int buffer[SIZE];

  dim_t id = blockIdx.x * blockDim.x + threadIdx.x;

  dim_t i = id / N;
  dim_t j = id % N;

  // printf(
  //     " id = blockIdx.x * blockDim.x + threadIdx.x\n"
  //     " %d = %d * %d + %d\n"
  //     "A[%d, %d] = %d\n",  //
  //     id, blockIdx.x, blockDim.x, threadIdx.x, i, j, A[i * N + j]);

  // Copy over value to buffer.
  buffer[j] = A[i * N + j];

  if (j + 1 < N) {
    // This condition will be true for most threads, so no need to worry about
    // divergence.
    buffer[j] += A[i * N + (j + 1)];
    // printf("Adding A[%d, %d] = %d\n", i, j + 1, A[i * N + (j + 1)]);
  }

  //  Synchronizes all threads within a block
  // – Used to prevent RAW / WAR / WAW hazards
  __syncthreads();

  A[i * N + j] = buffer[j];
}

__global__ void hw_exec_info() {
  int idx = blockIdx.x + blockDim.x * threadIdx.x;
  // warp-id is sus, see warp-id function for more information.
  printf("thread: %d\tSM: %d\twarp-id*: %d\t warp-lane: %d\n", idx, __smid(),
         __warpid(), __laneid());
}

__global__ void dynshared() {
  extern __shared__ int s[];
  dim_t i = threadIdx.x;
  s[i] = i;
  __syncthreads();

  // For even threads, print.
  if (i % 2) {
    printf("%d\n", s[i]);
  }
}

__global__ void constant_memory_kernel(int *A, dim_t size) {
  int i = threadIdx.x;
  int *source = reinterpret_cast<int *>(&constant_buffer[0]);
  A[i] = source[i];
}

__global__ void avg_classwork_kernel(Point *A, dim_t size, int *global_sum) {
  // Write CUDA code for the following functionality.
  // – Assume following data type, filled with some values.
  //   struct Point { int x, y; } arr[N];
  // – Each thread should operate on 4 elements.
  // – Find the average AVG of x values.
  // – If a thread sees y value above the average, it replaces all 4 y values
  //   with AVG.
  // – Otherwise, it adds y values to a global sum.
  // – Host prints the number of elements set to AVG

  dim_t id = blockIdx.x * blockDim.x + threadIdx.x;

  // Each thread operates on 4 elements.
  // [4*id, 4*id + 4]
  const dim_t batch_size = 4;
  dim_t start = batch_size * id;
  dim_t end = start + batch_size;

  // Find average of x values.
  int sum = 0;
  for (dim_t i = start; i < end; i++) {
    sum += A[i].x;
  }

  int average = sum / batch_size;

  // Does the thread see y value above average?
  // No divergence below.
  bool y_above_avg_spotted = false;
  for (dim_t i = start; i < end; i++) {
    if (A[i].y > average) {
      y_above_avg_spotted = true;
    }
  }

  // The if-else's are a function of data, and warps may diverge in which branch
  // among if/else gets executed.

  // There is perhaps room for optimizations here.
  for (dim_t i = start; i < end; i++) {
    if (y_above_avg_spotted) {
      A[i].y = average;
    } else {
      *global_sum += A[i].y;
    }
  }
}
