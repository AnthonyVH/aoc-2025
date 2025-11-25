#pragma once

#include "aoc25/memory.hpp"

#include <cstddef>
#include <string>

namespace aoc25 {

  static constexpr size_t simd_alignment_bytes = 512 / 8;

  template <size_t Alignment>
  using aligned_string_t =
      std::basic_string<char, std::char_traits<char>, aligned_allocator<char, Alignment>>;

  /** @brief A string whose buffer is aligned to aoc25::simd_alignment_bytes. This allows using it
   * with unaligned SIMD loads/stores.
   */
  using simd_string_t = aligned_string_t<simd_alignment_bytes>;

  template <class T>
  using simd_vector_t = std::vector<T, aligned_allocator<T, simd_alignment_bytes>>;

  template <class T>
  using simd_aligned_span_t = aligned_span<simd_alignment_bytes, T>;

}  // namespace aoc25
