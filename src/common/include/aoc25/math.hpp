#pragma once

#include <cassert>
#include <concepts>
#include <type_traits>

namespace aoc25 {

  /// @brief Calculate the non-negative mod(value, modulus).
  template <std::integral T, std::integral U>
  constexpr T mod(T value, U modulus) {
    assert(modulus > 0);
    [[assume(modulus > 0)]];
    auto const cast_mod = static_cast<T>(modulus);
    return ((value % cast_mod) + cast_mod) % cast_mod;
  }

  unsigned num_digits(uint64_t x);

}  // namespace aoc25
