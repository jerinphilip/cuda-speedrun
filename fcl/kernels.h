#include <cuda.h>

__global__ void vsqr(int *A);
__global__ void vadd(const int *A, const int *B, int *C);

// _ suffixes represent an in-place operation, something @jerinphilip picked up
// from PyTorch conventions.
__global__ void vsqr_(int *A);   // NOLINT
__global__ void vcube_(int *A);  // NOLINT
                                 //

__global__ void fused_sqr_cub_add(const int *A, const int *B, int *C);

__global__ void print_hello_world();
