
#include <type_traits>
#if defined(AOC25_SIMD_HWY_H_TARGET) == defined(HWY_TARGET_TOGGLE)
#  ifdef AOC25_SIMD_HWY_H_TARGET
#    undef AOC25_SIMD_HWY_H_TARGET
#  else
#    define AOC25_SIMD_HWY_H_TARGET
#  endif

#  include "aoc25/simd.hpp"

#  include <fmt/ranges.h>
#  include <spdlog/spdlog.h>

#  include <bit>
#  include <cassert>
#  include <concepts>
#  include <cstdint>

#  undef HWY_TARGET_INCLUDE
#  define HWY_TARGET_INCLUDE "src/simd-hwy.hpp"

// clang-format off
#include <hwy/foreach_target.h>
// clang-format on

#  include <hwy/base.h>
#  include <hwy/highway.h>
#  include <hwy/highway_export.h>

HWY_BEFORE_NAMESPACE();

namespace aoc25 {

  namespace HWY_NAMESPACE {

    namespace hn = hwy::HWY_NAMESPACE;

    template <std::integral T>
    uint64_t count(simd_span_t<T const> input, T const & value) {
      // No support for char, must use uint8_t instead.
      using input_t = std::conditional_t<std::is_same_v<T, char>, uint8_t, T>;
      using unsigned_t = std::make_unsigned_t<T>;
      using signed_t = std::make_signed_t<T>;

      static constexpr hn::ScalableTag<input_t> word_tag{};
      static constexpr hn::ScalableTag<signed_t> signed_word_tag{};
      static constexpr hn::ScalableTag<unsigned_t> accum_tag{};
      static constexpr hn::ScalableTag<uint64_t> combine_tag{};

      static constexpr size_t lanes = hn::Lanes(word_tag);

      auto const targets = hn::Set(word_tag, static_cast<input_t>(value));
      auto const * const HWY_RESTRICT data = reinterpret_cast<input_t const *>(input.data());

      SPDLOG_TRACE("Counting {:?} occurences in {} element span", value, input.size());

      size_t input_word_pos = 0;
      auto accumulator = hn::Zero(accum_tag);
      auto combined_accumulator = hn::Zero(combine_tag);

      auto const combine_accumulators = [&]() {
        static constexpr auto num_widens = sizeof(uint64_t) / sizeof(T);
        decltype(combined_accumulator) widened_sum;

        if constexpr (num_widens == 1) {
          widened_sum = accumulator;
        } else if constexpr (num_widens == 2) {
          widened_sum = hn::SumsOf2(accumulator);
        } else if constexpr (num_widens == 4) {
          widened_sum = hn::SumsOf4(accumulator);
        } else if constexpr (num_widens == 8) {
          widened_sum = hn::SumsOf8(accumulator);
        } else {
          static_assert(false, "Unsupported widening factor");
        }

        combined_accumulator = hn::Add(combined_accumulator, widened_sum);
        accumulator = hn::Zero(accum_tag);
      };

      // Process initial unaligned bytes.
      size_t const unaligned_bytes = reinterpret_cast<uintptr_t>(data) % lanes;
      size_t const unaligned_words = unaligned_bytes / sizeof(T);
      assert(unaligned_bytes % sizeof(T) == 0);

      auto const get_inc_words = [](auto const equals) {
        // Convert equality mask to increment vector by subtracting from zero.
        return hn::BitCast(accum_tag,
                           hn::Sub(hn::Zero(signed_word_tag),
                                   hn::BitCast(signed_word_tag, hn::VecFromMask(equals))));
      };

      if (unaligned_words != 0) {
        size_t const initial_words = std::min(lanes - unaligned_words, input.size());
        SPDLOG_TRACE("Processing {} unaligned elements", initial_words);

        auto const chunk = hn::LoadU(word_tag, data);
        auto const eq = hn::MaskedEq(hn::FirstN(word_tag, initial_words), chunk, targets);
        accumulator = hn::Add(accumulator, get_inc_words(eq));

        // Always combine, so we can combine at fixed intervals in the main loop.
        combine_accumulators();

        input_word_pos += initial_words;  // Update input position for aligned loads afterwards.
      }

      // Process inputs in chunks of 'lanes' words.
      [[maybe_unused]] size_t const num_chunks = (input.size() - input_word_pos) / lanes;
      if (num_chunks > 0) {
        assert((reinterpret_cast<uintptr_t>(data) + input_word_pos * sizeof(T)) % lanes == 0);
        SPDLOG_TRACE("Processing {} words ({} chunks)", num_chunks * lanes, num_chunks);
      }

      static constexpr size_t num_accums_before_combine = std::numeric_limits<input_t>::max();
      while (input_word_pos + lanes <= input.size()) {
        // Loop the maximum amount while still avoiding accumulator overflow.
        size_t const num_loops =
            std::min(num_accums_before_combine, (input.size() - input_word_pos) / lanes);
        size_t const loops_end = input_word_pos + num_loops * lanes;

        for (; input_word_pos + lanes <= loops_end; input_word_pos += lanes) {
          auto const chunk = hn::Load(word_tag, data + input_word_pos);
          auto const eq = hn::Eq(chunk, targets);
          accumulator = hn::Add(accumulator, get_inc_words(eq));
        }

        // Prevent accumulators from overflowing.
        SPDLOG_TRACE("Combining accumulators after {} chunks", num_loops);
        combine_accumulators();
      }

      // Process remaining words. This amount is smaller than a lane, so no need for a loop.
      // We also know that the input is padded, so no risk of reading out of bounds.
      if (input_word_pos < input.size()) {
        size_t const remaining_words = input.size() - input_word_pos;
        SPDLOG_TRACE("Processing {} remaining words", remaining_words);
        assert((reinterpret_cast<uintptr_t>(data) + input_word_pos * sizeof(T)) % lanes == 0);

        auto const chunk = hn::Load(word_tag, data + input_word_pos);
        auto const eq = hn::MaskedEq(hn::FirstN(word_tag, remaining_words), chunk, targets);
        accumulator = hn::Add(accumulator, get_inc_words(eq));
      }

      // One last combine, then horizontal sum of the combined accumulator.
      combine_accumulators();
      return hn::ReduceSum(combine_tag, combined_accumulator);
    }

    template <class T>
      requires std::is_arithmetic_v<T>
    size_t find_minimum_pos(simd_span_t<T const> input) {
      static constexpr auto tag = hn::ScalableTag<T>{};
      static constexpr size_t lanes = hn::Lanes(tag);

      auto const * const HWY_RESTRICT data = reinterpret_cast<T const *>(input.data());

      T min_value = std::numeric_limits<T>::max();
      size_t min_pos = 0;

      auto min_values = hn::Set(tag, min_value);
      size_t pos = 0;

      auto const process_minimums = [&](hn::Mask<decltype(tag)> const & is_less,
                                        bool update_min_values) {
        if (uint32_t match_bits = hn::BitsFromMask(tag, is_less); match_bits != 0) [[unlikely]] {
          while (match_bits != 0) {
            // Process in order, since there might be multiple "temporary" minimums.
            uint32_t const lane = std::countr_zero(match_bits);
            match_bits &= match_bits - 1;  // Clear lowest set bit.

            auto const value = data[pos + lane];
            if (value < min_value) {
              min_value = value;
              min_pos = pos + lane;
            }
          }

          if (update_min_values) {
            min_values = hn::Set(tag, min_value);
          }
        }
      };

      // Process initial unaligned bytes.
      size_t const unaligned_bytes = reinterpret_cast<uintptr_t>(data) % (lanes * sizeof(T));
      size_t const unaligned_words = unaligned_bytes / sizeof(T);
      assert(unaligned_bytes % sizeof(T) == 0);

      if (unaligned_words != 0) {
        size_t const initial_words = std::min(lanes - unaligned_words, input.size());

        auto const chunk = hn::LoadU(tag, data + pos);
        auto const is_less = hn::MaskedLt(hn::FirstN(tag, initial_words), chunk, min_values);
        process_minimums(is_less, true);

        pos += initial_words;
      }

      for (; pos + lanes <= input.size(); pos += lanes) {  // Process aligned chunks.
        auto const chunk = hn::Load(tag, data + pos);
        auto const is_less = hn::Lt(chunk, min_values);
        process_minimums(is_less, true);
      }

      if (pos < input.size()) {  // Handle last partial chunk.
        size_t const lanes_remaining = input.size() - pos;
        assert(pos + lanes_remaining == input.size());

        auto const chunk = hn::LoadNOr(min_values, tag, data + pos, lanes_remaining);
        auto const is_less = hn::Lt(chunk, min_values);
        process_minimums(is_less, false);
      }

      return min_pos;
    }

  }  // namespace HWY_NAMESPACE

}  // namespace aoc25

HWY_AFTER_NAMESPACE();

namespace aoc25 {

  template <std::integral T>
  uint64_t count(simd_span_t<T const> input, T const & value) {
    HWY_EXPORT_T(table, count<T>);
    return HWY_DYNAMIC_DISPATCH_T(table)(input, value);
  }

  template <class T>
    requires std::is_arithmetic_v<T>
  size_t find_minimum_pos(simd_span_t<T const> input) {
    HWY_EXPORT_T(table, find_minimum_pos<T>);
    return HWY_DYNAMIC_DISPATCH_T(table)(input);
  }

}  // namespace aoc25

#endif  // AOC25_STRING_HWY_H_TARGET
