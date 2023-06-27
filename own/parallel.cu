#include <cuda.h>

#include <cstdio>
#define N 100

__global__ void vsqr(int *A) {
  // Kernel computes vsqr for one  data-item.
  A[threadIdx.x] = threadIdx.x * threadIdx.x;
}

int main() {
  int host_buffer[N];

  int *device_buffer;
  cudaMalloc(&device_buffer, N * sizeof(int));

  vsqr<<<1, N>>>(device_buffer);
  cudaMemcpy(host_buffer, device_buffer, N * sizeof(int),
             cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();
  for (int i = 0; i < N; i++) {  // NOLINT
    printf("%d\n", host_buffer[i]);
  }
  return 0;
}
