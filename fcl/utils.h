#include <cstdint>

static __device__ __inline__ uint32_t __smid() {
  uint32_t smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  return smid;
}

static __device__ __inline__ uint32_t __warpid() {
  // https://stackoverflow.com/a/44337310/4565794
  //  The first problem - as @Patwie suggests - is that %warp_id does not give
  //  you what you actually want - it's not the index of the warp in the context
  //  of the grid, but rather in the context of the physical SM (which can hold
  //  so many warps resident at a time), and those two are not the same. So
  //  don't use %warp_id.
  //
  uint32_t warpid;
  asm volatile("mov.u32 %0, %%warpid;" : "=r"(warpid));
  return warpid;
}

static __device__ __inline__ uint32_t __laneid() {
  uint32_t laneid;
  asm volatile("mov.u32 %0, %%laneid;" : "=r"(laneid));
  return laneid;
}
