#pragma once

#include <concepts>
#include <cstdint>

namespace aoc25 {

  /// @brief Calculate the non-negative mod(value, modulus).
  template <std::integral T, std::integral U>
  constexpr T mod(T value, U modulus);

  constexpr uint64_t num_combinations(int8_t n, int8_t k);

  constexpr unsigned num_digits(uint64_t x);

}  // namespace aoc25

#include "aoc25/math.tpp"
