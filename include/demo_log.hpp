#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace demo {

inline spdlog::logger& log(std::string_view subsystem = "demo") {
  static std::mutex mutex;
  static std::unordered_map<std::string, std::shared_ptr<spdlog::logger>> loggers;
  const std::string name(subsystem);
  std::lock_guard<std::mutex> lock(mutex);
  auto& logger = loggers[name];
  if (!logger) {
    logger = spdlog::stdout_color_mt("g1-h2." + name);
  }
  return *logger;
}

}  // namespace demo
