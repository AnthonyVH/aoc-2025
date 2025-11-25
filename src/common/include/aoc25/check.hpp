#pragma once

#include <fmt/format.h>
#include <spdlog/spdlog.h>

namespace aoc25 {

  /// Exception type thrown by the `check` function.
  struct check_failure : public std::runtime_error {
    using std::runtime_error::runtime_error;
  };

  /// Check if condition is true. If not, logs an error message and throws an exception of type
  /// `check_failure`.
  template <class... Args>
  void check(bool condition, fmt::format_string<Args...> fmt, Args &&... args) {
    if (!condition) {
      auto msg = fmt::format(std::move(fmt), std::forward<Args>(args)...);
      spdlog::error(msg);
      throw check_failure(std::move(msg));
    }
  }

}  // namespace aoc25
