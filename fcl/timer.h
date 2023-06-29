#pragma once

#include <chrono>
#include <ctime>
#include <string>

// Helper function to get the current date and time
std::string date();

// Adapted from
// https://github.com/marian-nmt/marian-dev/blob/master/src/common/timer.h
//
// Timer measures elapsed time.
class Timer {
 protected:
  using clock = std::chrono::steady_clock;
  using time_point = std::chrono::time_point<clock>;
  using duration = std::chrono::nanoseconds;

  time_point start_;     // Starting time point
  bool stopped_{false};  // Indicator if the timer has been stopped
  duration time_;        // Time duration from start() to stop()

 public:
  // Create and start the timer
  Timer() : start_(clock::now()) {}

  // Get the time elapsed without stopping the timer.  If the template type is
  // not specified, it returns the time counts as represented by
  // std::chrono::seconds
  template <class Duration = std::chrono::seconds>
  double elapsed() const {
    using duration_double =
        std::chrono::duration<double, typename Duration::period>;
    return std::chrono::duration_cast<duration_double>(clock::now() - start_)
        .count();
  }
};
