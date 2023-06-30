#pragma once
#include <cuda.h>

using dim_t = int32_t;

__global__ void vsqr(int *A);
__global__ void vadd(const int *A, const int *B, int *C);

// _ suffixes represent an in-place operation, something @jerinphilip picked up
// from PyTorch conventions.
__global__ void vsqr_(int *A);   // NOLINT
__global__ void vcube_(int *A);  // NOLINT
                                 //

__global__ void fused_sqr_cub_add(const int *A, const int *B, int *C);

__global__ void print_hello_world();

__global__ void scalar_init(int *A);

__global__ void matrix_square_v1(const int *A, dim_t N, int *B);
__global__ void matrix_square_v2(const int *A, dim_t N, int *B);

__global__ void warp_branch_paths(int *A, dim_t size);
