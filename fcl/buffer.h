#include <cuda.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

// NOLINTBEGIN
enum class Device { CPU, GPU };
// NOLINTEND
// https://stackoverflow.com/a/14038590/4565794
#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}

template <class Scalar>
class BaseBuffer {
 public:
  virtual Scalar *data() = 0;
  virtual size_t size() = 0;
};

template <class Scalar>
class GPUBuffer : BaseBuffer<Scalar> {
 public:
  explicit GPUBuffer(size_t size) : size_(size) {
    size_t mem_size = size * sizeof(Scalar);
    cudaMalloc(&buffer_, mem_size);
  }

  ~GPUBuffer() {
    // cudaFree, and some check.
    gpuErrchk(cudaFree(buffer_));
  }

 private:
  size_t size_ = 0;
  Scalar *buffer_ = nullptr;
};

template <class Scalar>
class CPUBuffer {
  explicit CPUBuffer(size_t size) : size_(size) {
    size_t mem_size = size * sizeof(Scalar);
    buffer_ = static_cast<Scalar *>(malloc(mem_size));
  }

  ~CPUBuffer() { free(buffer_); }

 private:
  size_t size_ = 0;
  Scalar *buffer_ = nullptr;
};

template <class Scalar>
class Buffer {
 public:
  Buffer(size_t size, Device device) : size_(size), device_(device) {}

  const Device &device() const { return device_; }
  Scalar *data() { return buffer_; }
  size_t size() const { return size_; }

  Buffer<Scalar> to(Device device) {
    if (device == Device::CPU and device_ == Device::GPU) {
      Buffer<Scalar> result(size_, device);
      size_t mem_size = size_ * sizeof(Scalar);
      cudaMemcpy(result.data(), buffer_->data(), mem_size,
                 cudaMemcpyDeviceToHost);
      return result;
    }
    if (device == Device::GPU and device_ == Device::CPU) {
      Buffer<Scalar> result(size_, device);
      size_t mem_size = size_ * sizeof(Scalar);
      cudaMemcpy(buffer_, result.data(), mem_size, cudaMemcpyHostToDevice);
      return result;
    }

    std::cerr << "Unsupported conversion between devices\n";
    std::abort();
  }

 private:
  size_t size_;
  std::unique_ptr<BaseBuffer<Scalar>> buffer_;
  Device device_;
};
