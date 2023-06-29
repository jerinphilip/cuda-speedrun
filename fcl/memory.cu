#include <cuda.h>

#include <cstdio>

const char *msg = "Hello World.\n";

__global__ void dkernel() {
  // Memory is shared, should not compile.
  printf(msg);  // NOLINT
}
int main() {
  dkernel<<<1, 32>>>();
  cudaDeviceSynchronize();
  return 0;
}
