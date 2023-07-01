#include <cuda.h>

#include "fcl/types.h"

__global__ void no_divergence(int *A, dim_t size) {
  (void)size;
  dim_t id = blockIdx.x * blockDim.x + threadIdx.x;
  for (dim_t i = 0; i < id; ++i) {
    A[id] += i;
  }
}

__global__ void divergence(int *A, dim_t size) {
  (void)size;
  dim_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id % 2) {
    A[id] = id;
  } else if (A[id] % 2) {
    A[id] = id / 2;
  } else {
    A[id] = id * 2;
  }
}

__global__ void classwork_sample_xyz_branched(int *A, dim_t size) {
  // assert(x == y || x == z);
  dim_t id = blockIdx.x * blockDim.x + threadIdx.x;
  int x = A[id];
  int y = (A[id] / 2) * 2;  // This will create odd/even branches.
  int z = 0;
  if (x == y) {
    x = z;
  } else {
    x = y;
  }
}

__global__ void classwork_sample_xyz_no_branches(int *A, dim_t size) {
  // assert(x == y || x == z);
  dim_t id = blockIdx.x * blockDim.x + threadIdx.x;
  int x = A[id];
  int y = (A[id] / 2) * 2;  // This will create odd/even branches.
  int z = 0;
  int gate = (x == y);
  x = (gate)*z + (1 - gate) * y;
  A[id] = x;
}
