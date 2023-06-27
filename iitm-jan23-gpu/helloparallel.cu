#include <cuda.h>
#include <stdio.h>
__global__ void dkernel() { printf("Hello World.\n"); }
int main() {
  dkernel<<<1, 32>>>();
  cudaThreadSynchronize();
  return 0;
}
