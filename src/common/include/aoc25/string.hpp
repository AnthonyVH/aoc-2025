#pragma once

#include "aoc25/simd.hpp"

#include <concepts>

namespace aoc25 {

  namespace detail {

    template <class CallbackFn>
    concept is_split_callback = requires(CallbackFn cb, simd_string_view_t sv) {
      { cb(sv) } -> std::same_as<bool>;
    };

  }  // namespace detail

  template <class StringLike>
    requires requires(StringLike s) {
      { s.substr(0, 1) } -> std::same_as<StringLike>;
    }
  StringLike trim(StringLike s);

  template <class T>
  T to_int(std::string_view str);

  std::span<simd_span_t<char const>> split_lines(simd_span_t<char const> input,
                                                 std::span<simd_span_t<char const>> output,
                                                 char splitter = '\n');

  /** @brief Split an input string into substrings by a splitter.
   *
   * If multiple splitters are adjacent, empty substrings are produced. If the input starts with a
   * splitter, an empty substring is produced for the start. If the input ends with a splitter, no
   * substring is produced for the end. If there are characters remaining after the last splitter,
   * a final substring is produced for them.
   */
  template <detail::is_split_callback CallbackFn>
  simd_string_view_t split(simd_string_view_t input, CallbackFn && callback, char splitter = '\n');

  template <std::integral T, class CallbackFn>
    requires std::is_unsigned_v<T> && std::invocable<CallbackFn, T>
  void convert_single_int(simd_string_view_t input, CallbackFn && callback);

}  // namespace aoc25

#include "aoc25/string-hwy.hpp"
#include "aoc25/string.tpp"
