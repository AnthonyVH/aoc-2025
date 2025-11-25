#pragma once

#include "aoc25/simd.hpp"

#include <concepts>

namespace aoc25 {

  template <class StringLike>
    requires requires(StringLike s) {
      { s.substr(0, 1) } -> std::same_as<StringLike>;
    }
  StringLike trim(StringLike s);

  template <class T>
  T to_int(std::string_view str);

  std::span<simd_aligned_span_t<char const>> split_lines(
      simd_aligned_span_t<char const> input,
      std::span<simd_aligned_span_t<char const>> output,
      char splitter = '\n');

}  // namespace aoc25

#include "aoc25/string.tpp"
