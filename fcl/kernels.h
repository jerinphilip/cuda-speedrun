#pragma once
#include <cuda.h>

#include "fcl/types.h"

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

// structure of Arrays vs array of structures.

// Node in an array of nodes (structures).
struct Node {
  int i;
  double d;
  char c;
};

// struct holding node constituent arrays.
// binding by index.
struct Nodes {
  int *is;
  double *ds;
  char *cs;
};

__global__ void aos_pass(Node *nodes, dim_t size);
__global__ void soa_pass(int *is, double *ds, char *cs, dim_t size);

__global__ void maximum_in_a_large_array_kernel(const int *xs, dim_t size,
                                                dim_t partition_size, int *ys);

__global__ void find_element_kernel(const int *xs, dim_t size,
                                    dim_t partition_size, int query, int *out);

__global__ void block_thread_dispatch_identifier();

__global__ void add_nearby(int *A);

__global__ void hw_exec_info();
