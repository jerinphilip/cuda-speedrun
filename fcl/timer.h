#pragma once

#include <chrono>
#include <ctime>
#include <string>

#include "fcl/error.h"
#include "fcl/types.h"

// Helper function to get the current date and time
std::string date();

// Adapted from
// https://github.com/marian-nmt/marian-dev/blob/master/src/common/timer.h
//
// Timer measures elapsed time.
template <enum Device = Device::CPU>
class Timer {
 public:
  // Create and start the timer
  Timer() : start_(clock::now()) {}

  Timer(const Timer &timer) = delete;
  Timer &operator=(const Timer &timer) = delete;

  // Get the time elapsed without stopping the timer.  If the template type is
  // not specified, it returns the time counts as represented by
  // std::chrono::seconds
  template <class Duration = std::chrono::seconds>
  double elapsed() {
    using duration_double =
        std::chrono::duration<double, typename Duration::period>;
    return std::chrono::duration_cast<duration_double>(clock::now() - start_)
        .count();
  }

 protected:
  using clock = std::chrono::steady_clock;
  using time_point = std::chrono::time_point<clock>;
  using duration = std::chrono::nanoseconds;

  time_point start_;     // Starting time point
  bool stopped_{false};  // Indicator if the timer has been stopped
  duration time_;        // Time duration from start() to stop()
};

template <>
class Timer<Device::GPU> {
 public:
  Timer() {
    gpuErrchk(cudaEventCreate(&start_));
    gpuErrchk(cudaEventCreate(&stop_));
    gpuErrchk(cudaEventRecord(start_, 0));
  }

  double elapsed() {
    gpuErrchk(cudaEventRecord(stop_, 0));
    gpuErrchk(cudaEventSynchronize(stop_));
    gpuErrchk(cudaEventElapsedTime(&elapsed_, start_, stop_));
    return elapsed_ / 1000;
  }

  ~Timer() {
    gpuErrchk(cudaEventDestroy(start_));
    gpuErrchk(cudaEventDestroy(stop_));
  }

 private:
  float elapsed_ = 0;
  cudaEvent_t start_;
  cudaEvent_t stop_;
};
