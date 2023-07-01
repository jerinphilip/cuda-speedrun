#pragma once
#include "fcl/types.h"

__global__ void no_divergence(int *A, dim_t size);
__global__ void divergence(int *A, dim_t size);
