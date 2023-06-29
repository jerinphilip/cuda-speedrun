#include "fcl/timer.h"

#include <string>

std::string date() {
  std::time_t now =
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  char date[100] = {};
  std::strftime(date, sizeof(date), "%F %X %z", std::localtime(&now));
  return date;
}
