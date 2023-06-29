#include <cuda.h>

#include <cstdio>

template <class Scalar>
class GPUBuffer {
 public:
  GPUBuffer(Scalar *host_buffer, size_t size) : size_(size) {
    size_t mem_size = size * sizeof(Scalar);
    cudaMalloc(&buffer_, mem_size);
    cudaMemcpy(buffer_, host_buffer, mem_size, cudaMemcpyHostToDevice);
  }

  explicit GPUBuffer(size_t size) : size_(size) {
    size_t mem_size = size * sizeof(Scalar);
    cudaMalloc(&buffer_, mem_size);
  }

  Scalar *data() { return buffer_; }
  size_t size() const { return size_; }

  std::vector<Scalar> cpu() {
    std::vector<Scalar> result(size_);
    size_t mem_size = size_ * sizeof(Scalar);
    cudaMemcpy(result.data(), buffer_, mem_size, cudaMemcpyDeviceToHost);
    return result;
  }

  ~GPUBuffer() {
    // Free device memory.
    gpuErrchk(cudaFree(buffer_));
  }

 private:
  size_t size_;
  Scalar *buffer_;
};
