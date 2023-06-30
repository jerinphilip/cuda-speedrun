#pragma once
#include <cuda.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

#include "fcl/error.h"

using dim_t = int32_t;

// NOLINTBEGIN
enum class Device { CPU, GPU };
// NOLINTEND

template <class Scalar>
class BaseBuffer {
 public:
  virtual Scalar *data() = 0;
  virtual dim_t size() = 0;
  ~BaseBuffer() = default;
};

template <class Scalar>
class GPUBuffer : public BaseBuffer<Scalar> {
 public:
  explicit GPUBuffer(dim_t size) : size_(size) {
    dim_t mem_size = size * sizeof(Scalar);
    cudaMalloc(&buffer_, mem_size);
    cudaMemset(buffer_, 0, mem_size);
  }
  Scalar *data() final { return buffer_; };
  dim_t size() final { return size_; };

  ~GPUBuffer() {
    // cudaFree, and some check.
    gpuErrchk(cudaFree(buffer_));
  }

 private:
  dim_t size_ = 0;
  Scalar *buffer_ = nullptr;
};

template <class Scalar>
class CPUBuffer : public BaseBuffer<Scalar> {
 public:
  explicit CPUBuffer(dim_t size) : size_(size) {
    dim_t mem_size = size * sizeof(Scalar);
    buffer_ = static_cast<Scalar *>(malloc(mem_size));
  }

  Scalar *data() final { return buffer_; };
  dim_t size() final { return size_; };

  ~CPUBuffer() { free(buffer_); }

 private:
  dim_t size_ = 0;
  Scalar *buffer_ = nullptr;
};

template <class Scalar>
class Buffer {
 public:
  using GPUType = GPUBuffer<Scalar>;
  using CPUType = CPUBuffer<Scalar>;
  using OpaqueType = BaseBuffer<Scalar>;

  static std::unique_ptr<OpaqueType> factory(dim_t size, Device device) {
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

  Buffer(dim_t size, Device device)
      : size_(size), device_(device), buffer_(factory(size, device)) {}

  const Device &device() const { return device_; }
  Scalar *data() { return buffer_->data(); }
  const Scalar *data() const { return buffer_->data(); }
  dim_t size() const { return buffer_->size(); }

  Buffer<Scalar> to(Device device) const {
    if (device == Device::CPU and device_ == Device::GPU) {
      Buffer<Scalar> target(size_, device);
      dim_t mem_size = size_ * sizeof(Scalar);
      cudaMemcpy(target.data(), buffer_->data(), mem_size,
                 cudaMemcpyDeviceToHost);
      return target;
    }
    if (device == Device::GPU and device_ == Device::CPU) {
      Buffer<Scalar> target(size_, device);
      dim_t mem_size = size_ * sizeof(Scalar);
      cudaMemcpy(target.data(), buffer_->data(), mem_size,
                 cudaMemcpyHostToDevice);
      return target;
    }

    std::cerr << "Unsupported conversion between devices\n";
    std::abort();
  }

  friend std::ostream &operator<<(std::ostream &out,
                                  const Buffer<Scalar> &buffer) {
    if (buffer.device() == Device::GPU) {
      auto host_buffer = buffer.to(Device::CPU);
      out << host_buffer;
    } else {
      const Scalar *start = buffer.data();
      const Scalar *end = start + buffer.size();
      out << "Buffer(device = cpu, [";
      for (const Scalar *p = start; p != end; ++p) {
        if (p != start) {
          out << " ";
        }
        out << *p;
      }
      out << "]";
    }
    return out;
  }

 private:
  dim_t size_;
  std::unique_ptr<OpaqueType> buffer_;
  Device device_;
};
