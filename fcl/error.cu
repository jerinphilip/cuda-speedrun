#include <cuda.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cstdio>

#include "fcl/error.h"

void gpuAssert(cudaError_t code, const char *file, int line,
               bool abort /* = true*/) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}

__device__ void cdpAssert(cudaError_t code, const char *file, int line,
                          bool abort /* = true*/) {
  if (code != cudaSuccess) {
    printf("GPU kernel assert: %s %s %d\n", cudaGetErrorString(code), file,
           line);
    if (abort) assert(0);
  }
}
