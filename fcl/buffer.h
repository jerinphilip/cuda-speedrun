#include <cuda.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

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

// NOLINTBEGIN
enum class Device { CPU, GPU };
// NOLINTEND

template <class Scalar>
class BaseBuffer {
 public:
  virtual Scalar *data() = 0;
  virtual size_t size() = 0;
  ~BaseBuffer() = default;
};

template <class Scalar>
class GPUBuffer : public BaseBuffer<Scalar> {
 public:
  explicit GPUBuffer(size_t size) : size_(size) {
    size_t mem_size = size * sizeof(Scalar);
    cudaMalloc(&buffer_, mem_size);
    cudaMemset(buffer_, 0, mem_size);
  }
  Scalar *data() final { return buffer_; };
  size_t size() final { return size_; };

  ~GPUBuffer() {
    // cudaFree, and some check.
    gpuErrchk(cudaFree(buffer_));
  }

 private:
  size_t size_ = 0;
  Scalar *buffer_ = nullptr;
};

template <class Scalar>
class CPUBuffer : public BaseBuffer<Scalar> {
 public:
  explicit CPUBuffer(size_t size) : size_(size) {
    size_t mem_size = size * sizeof(Scalar);
    buffer_ = static_cast<Scalar *>(malloc(mem_size));
  }

  Scalar *data() final { return buffer_; };
  size_t size() final { return size_; };

  ~CPUBuffer() { free(buffer_); }

 private:
  size_t size_ = 0;
  Scalar *buffer_ = nullptr;
};

template <class Scalar>
class Buffer {
 public:
  using GPUType = GPUBuffer<Scalar>;
  using CPUType = CPUBuffer<Scalar>;
  using OpaqueType = BaseBuffer<Scalar>;

  static std::unique_ptr<OpaqueType> factory(size_t size, Device device) {
    switch (device) {
      case Device::CPU:
        return std::make_unique<CPUType>(size);
      case Device::GPU:
        return std::make_unique<GPUType>(size);
      default:
        std::abort();
    }
    return nullptr;
  }

  Buffer(size_t size, Device device)
      : size_(size), device_(device), buffer_(factory(size, device)) {}

  const Device &device() const { return device_; }
  Scalar *data() { return buffer_->data(); }
  size_t size() const { return buffer_->size(); }

  Buffer<Scalar> to(Device device) {
    if (device == Device::CPU and device_ == Device::GPU) {
      Buffer<Scalar> target(size_, device);
      size_t mem_size = size_ * sizeof(Scalar);
      cudaMemcpy(target.data(), buffer_->data(), mem_size,
                 cudaMemcpyDeviceToHost);
      return target;
    }
    if (device == Device::GPU and device_ == Device::CPU) {
      Buffer<Scalar> target(size_, device);
      size_t mem_size = size_ * sizeof(Scalar);
      cudaMemcpy(target.data(), buffer_->data(), mem_size,
                 cudaMemcpyHostToDevice);
      return target;
    }

    std::cerr << "Unsupported conversion between devices\n";
    std::abort();
  }

 private:
  size_t size_;
  std::unique_ptr<OpaqueType> buffer_;
  Device device_;
};
