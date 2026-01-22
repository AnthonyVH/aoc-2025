#include "aoc25/day_02.hpp"

#include "aoc25/math.hpp"
#include "aoc25/simd.hpp"
#include "aoc25/string.hpp"

#include <fmt/ranges.h>
#include <spdlog/spdlog.h>
#include <sys/types.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <string_view>

namespace aoc25 {
  namespace {

    uint64_t power_of_10(uint8_t exponent) {
      static constexpr auto table = std::to_array<uint64_t>({
          1,
          10,
          100,
          1'000,
          10'000,
          100'000,
          1'000'000,
          10'000'000,
          100'000'000,
          1'000'000'000,
          10'000'000'000,
      });
      return table.at(exponent);
    }

    uint64_t slice_msb_digits(uint64_t value, uint8_t digits_to_remove) {
      return value / power_of_10(digits_to_remove);
    }

    uint64_t repeat_value(uint64_t value, uint8_t value_width, uint8_t target_width) {
      assert(target_width % value_width == 0);
      [[assume(target_width % value_width == 0)]];
      [[assume(value_width > 0 && value_width <= 5)]];
      [[assume(target_width > 0 && target_width <= 10)]];

      // Multiplier table for each # repeats per value width (up to 5 digits).
      static constexpr std::array<std::array<uint64_t, 11>, 6> multipliers = {{
          {},  // 0-width (unused)
          {
              0ULL,
              1ULL,
              11ULL,
              111ULL,
              1111ULL,
              11111ULL,
              111111ULL,
              1111111ULL,
              11111111ULL,
              111111111ULL,
              1111111111ULL,
          },  // 1-width, up to 10 repeats
          {
              0ULL,
              1ULL,
              101ULL,
              10101ULL,
              1010101ULL,
              101010101ULL,
          },  // 2-width, up to 5 repeats
          {
              0ULL,
              1ULL,
              1001ULL,
              1001001ULL,
              1001001001ULL,
          },  // 3-width, up to 4 repeats
          {
              0ULL,
              1ULL,
              10001ULL,
              100010001ULL,
          },  // 4-width, up to 3 repeats
          {
              0ULL,
              1ULL,
              100001ULL,
          },  // 5-width, up to 2 repeats
      }};

      uint8_t const num_repeats = target_width / value_width;
      return value * multipliers.at(value_width).at(num_repeats);
    }

    constexpr std::span<uint8_t const> divisors(uint8_t num_digits) {
      // clang-format off
      static constexpr std::array<uint8_t, 14> lut = std::to_array<uint8_t>({
        1,        //  2, 3, 5, 7 digits
        1, 2,     //  4 digits
        1, 2, 3,  //  6 digits
        1, 2, 4,  //  8 digits
        1, 3,     //  9 digits
        1, 2, 5,  // 10 digits
      });
      // clang-format on

      static constexpr std::array<std::span<uint8_t const>, 11> span_lut = {
          std::span<uint8_t const>{},             //  0 digits
          std::span<uint8_t const>{},             //  1 digit
          std::span<uint8_t const>{&lut[0], 1},   //  2 digits
          std::span<uint8_t const>{&lut[0], 1},   //  3 digits
          std::span<uint8_t const>{&lut[1], 2},   //  4 digits
          std::span<uint8_t const>{&lut[0], 1},   //  5 digits
          std::span<uint8_t const>{&lut[3], 3},   //  6 digits
          std::span<uint8_t const>{&lut[0], 1},   //  7 digits
          std::span<uint8_t const>{&lut[6], 3},   //  8 digits
          std::span<uint8_t const>{&lut[9], 2},   //  9 digits
          std::span<uint8_t const>{&lut[11], 3},  // 10 digits
      };

      assert(num_digits < span_lut.size());
      return span_lut[num_digits];
    }

    uint64_t any_repeated_digit_sum(uint64_t limit) {
      if (limit == 0) {
        return 0;
      }

      uint8_t const limit_digits = aoc25::num_digits(limit);
      SPDLOG_DEBUG("limit: {}, limit_digits: {}", limit, limit_digits);

      std::array<uint64_t, 10> slice_sums{};

      for (uint8_t slice_digits : divisors(limit_digits)) {
        assert(limit_digits % slice_digits == 0);

        uint64_t slice_upper = slice_msb_digits(limit, limit_digits - slice_digits);
        uint64_t repeated_upper = repeat_value(slice_upper, slice_digits, limit_digits);

        uint64_t const slice_lower = power_of_10(slice_digits - 1);
        uint64_t const repeated_lower = repeat_value(slice_lower, slice_digits, limit_digits);
        uint64_t const diff_between_repeats = repeat_value(1, slice_digits, limit_digits);

        bool const limit_exceeded = repeated_upper > limit;
        uint64_t const num_repeats = slice_upper - slice_lower + 1 - limit_exceeded;
        uint64_t slice_sum = repeated_lower * num_repeats +
                             (num_repeats * (num_repeats - 1) / 2) * diff_between_repeats;

        SPDLOG_DEBUG(
            "slice_digits: {}, repeated_lower: {}, repeated_upper: {}, "
            "num_repeats: {}, diff_between_repeats: {}, slice_sum: {}",
            slice_digits, repeated_lower, repeated_upper, num_repeats, diff_between_repeats,
            slice_sum);

        // Subtract double-counted values due numbers already generated by smaller slice widths.
        for (uint8_t subslice_digits : divisors(slice_digits)) {
          slice_sum -= slice_sums.at(subslice_digits);
        }

        slice_sums.at(slice_digits) = slice_sum;
      }

      SPDLOG_DEBUG("slice_sums: {}", slice_sums);
      uint64_t sum = std::accumulate(slice_sums.begin(), slice_sums.end(), 0ULL);

      // Add sum for values with less digits. The array below contains the sum of any-repeated
      // digit numbers from 0 to the maximum value with n digits (i.e. for 0, 9, 99, 999, ...).
      static constexpr auto sum_for_max_value = std::to_array<uint64_t>({
          0ULL,
          0ULL,
          495ULL,
          5'490ULL,
          500'895ULL,
          1'000'890ULL,
          540'590'850ULL,
          590'590'845ULL,
          495'595'086'345ULL,
          990'640'130'895ULL,
      });

      uint64_t const & less_digits_sum = sum_for_max_value.at(limit_digits - 1);
      SPDLOG_DEBUG("[sum: {}, less_digits_sum: {}, result: {}]", sum, less_digits_sum,
                   sum + less_digits_sum);
      sum += less_digits_sum;

      return sum;
    }

    uint64_t twice_repeated_digit_sum(uint64_t limit) {
      if (limit == 0) {
        return 0;
      }

      uint8_t const limit_digits = aoc25::num_digits(limit);
      SPDLOG_DEBUG("limit: {}, limit_digits: {}", limit, limit_digits);

      uint64_t sum = 0;

      if (uint8_t const slice_digits = limit_digits / 2; slice_digits * 2 == limit_digits) {
        // Double repetition only possible when limit_digits is even.
        uint64_t slice_upper = slice_msb_digits(limit, slice_digits);
        uint64_t const repeated_upper = repeat_value(slice_upper, slice_digits, limit_digits);
        bool const limit_exceeded = repeated_upper > limit;

        uint64_t const slice_lower = power_of_10(slice_digits - 1);
        uint64_t const repeated_lower = repeat_value(slice_lower, slice_digits, limit_digits);
        uint64_t const diff_between_repeats = repeat_value(1, slice_digits, limit_digits);

        uint64_t const num_repeats = slice_upper - slice_lower + 1 - limit_exceeded;
        sum = repeated_lower * num_repeats +
              (num_repeats * (num_repeats - 1) / 2) * diff_between_repeats;

        SPDLOG_DEBUG(
            "slice_digits: {}, repeated_lower: {}, repeated_upper: {}, "
            "num_repeats: {}, diff_between_repeats: {}, sum: {}",
            slice_digits, repeated_lower, repeated_upper, num_repeats, diff_between_repeats, sum);
      }

      /* Add sum for values with less digits. The array below contains the sum of twice-repeated
       * digit numbers from 0 to the maximum value with n digits (i.e. for 0, 9, 99, 999, ...).
       *
       * Note: This array is compressed, because values with an odd number of digits don't
       * contribute to the sum.
       */
      static constexpr auto sum_for_max_value = std::to_array<uint64_t>({
          0ULL,                // 0 digits
          495ULL,              // 2 digits
          495'900ULL,          // 4 digits
          495'540'450ULL,      // 6 digits
          495'500'035'950ULL,  // 8 digits
      });

      uint64_t const & less_digits_sum = sum_for_max_value.at((limit_digits - 1) / 2);
      SPDLOG_DEBUG("sum: {}, less_digits_sum: {}, result: {}", sum, less_digits_sum,
                   sum + less_digits_sum);
      sum += less_digits_sum;

      SPDLOG_DEBUG("<limit: {}, sum: {}>", limit, sum);
      return sum;
    }

  }  // namespace

  uint64_t day_t<2>::solve(part_t<1>, version_t<0>, simd_string_view_t input) {
    uint64_t sum = 0;

    aoc25::split(
        input,
        [&](simd_string_view_t segment) {
          // Find next "-" separator.
          auto const dash = segment.find('-');
          assert(dash != segment.npos);
          SPDLOG_TRACE("dash @ {}", dash);

          // NOTE: No measurable speedup using convert_single_int here.
          uint64_t const lower = to_int<uint64_t>(segment.substr(0, dash));
          segment.remove_prefix(dash + 1);
          uint64_t const upper = to_int<uint64_t>(segment);
          SPDLOG_TRACE("parsed {}-{}", lower, upper);

          auto const entry_sum =
              twice_repeated_digit_sum(upper) - twice_repeated_digit_sum(lower - 1);
          sum += entry_sum;
          SPDLOG_DEBUG("[entry: {}-{}, entry_sum: {}, total_sum: {}]", lower, upper, entry_sum,
                       sum);
        },
        ',');

    return sum;
  }

  uint64_t day_t<2>::solve(part_t<2>, version_t<0>, simd_string_view_t input) {
    uint64_t sum = 0;

    aoc25::split(
        input,
        [&](simd_string_view_t segment) {
          // Find next "-" separator.
          auto const dash = segment.find('-');
          assert(dash != segment.npos);
          SPDLOG_TRACE("dash @ {}", dash);

          // NOTE: No measurable speedup using convert_single_int here.
          uint64_t const lower = to_int<uint64_t>(segment.substr(0, dash));
          segment.remove_prefix(dash + 1);
          uint64_t const upper = to_int<uint64_t>(segment);
          SPDLOG_TRACE("parsed {}-{}", lower, upper);

          auto const entry_sum = any_repeated_digit_sum(upper) - any_repeated_digit_sum(lower - 1);
          sum += entry_sum;
          SPDLOG_DEBUG("[entry: {}-{}, entry_sum: {}, total_sum: {}]", lower, upper, entry_sum,
                       sum);
        },
        ',');

    return sum;
  }

}  // namespace aoc25
