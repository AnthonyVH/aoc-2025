#pragma once

#include "aoc25/day.hpp"
#include "aoc25/simd.hpp"

#include <cstdint>

/*

 */

namespace aoc25 {

  template <>
  struct day_t<10> {
    uint64_t solve(part_t<1>, version_t<0>, simd_string_view_t input);

    uint64_t solve(part_t<2>, version_t<0>, simd_string_view_t input);
    uint64_t solve(part_t<2>, version_t<1>, simd_string_view_t input);
  };

}  // namespace aoc25
