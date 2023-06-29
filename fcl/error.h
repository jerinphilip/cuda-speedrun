#pragma once

// Adapted from https://stackoverflow.com/a/14038590/4565794
//
// Normal GPU asserts.
void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true);

// In case of dynamic parallelism.
// Note that when using CUDA Dynamic Parallelism, a very similar methodology can
// and should be applied to any usage of the CUDA runtime API in device kernels,
// as well as after any device kernel launches:
__device__ void cdpAssert(cudaError_t code, const char *file, int line,
                          bool abort = true);

#define gpuErrchk(ans)                    \
  do {                                    \
    gpuAssert((ans), __FILE__, __LINE__); \
  } while (0)

#define cdpErrchk(ans)                    \
  do {                                    \
    cdpAssert((ans), __FILE__, __LINE__); \
  } while (0)
