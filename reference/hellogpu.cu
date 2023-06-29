#include <cuda.h>
#include <stdio.h>
__global__ void dkernel() { printf("Hello World.\n"); }
int main() {
  dkernel<<<1, 1>>>();
  return 0;
}
