#include <cuda.h>

#include <cstdio>

__global__ void dkernel() {
  if (threadIdx.x == 0 && blockIdx.x == 0 && threadIdx.y == 0 &&
      blockIdx.y == 0 && threadIdx.z == 0 && blockIdx.z == 0) {
    printf("[narrow] %d %d %d %d %d %d.\n", gridDim.x, gridDim.y, gridDim.z,
           blockDim.x, blockDim.y, blockDim.z);
  }

  printf("[all] %d %d %d %d %d %d.\n", threadIdx.x, threadIdx.y, threadIdx.z,
         blockIdx.x, blockIdx.y, blockIdx.z);
}

int main() {
  dim3 grid(2, 3, 4);
  dim3 block(5, 6, 7);
  dkernel<<<grid, block>>>();
  cudaDeviceSynchronize();
  return 0;
}
