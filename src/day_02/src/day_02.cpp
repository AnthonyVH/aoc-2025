#include "aoc25/day_02.hpp"

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

    struct range_t {
      uint64_t lower;
      uint64_t upper;
    };

    [[maybe_unused]] std::string format_as(range_t const & obj) {
      return fmt::format("({}-{})", obj.lower, obj.upper);
    }

    simd_vector_t<range_t> parse_input(version_t<0>, std::string_view input) {
      simd_vector_t<range_t> result;
      result.reserve(100);

      while (!input.empty()) {
        // Find next "-" separator.
        auto const dash = input.find('-');
        if (dash == input.npos) {
          break;
        }
        SPDLOG_TRACE("dash @ {}", dash);

        range_t entry;
        entry.lower = to_int<uint64_t>(input.substr(0, dash));
        input.remove_prefix(dash + 1);

        // When at the end of the line, there's no comma.
        auto end = input.find(',');
        bool const has_next = end != input.npos;
        end = has_next ? end : input.size();
        SPDLOG_TRACE("comma @ {}, has_next: {}", end, has_next);

        entry.upper = to_int<uint64_t>(input.substr(0, end));
        input.remove_prefix(end + has_next);

        SPDLOG_TRACE("pushing {}", entry);
        result.push_back(entry);
      }

      return result;
    }

    template <std::integral T>
    uint8_t num_digits(T value) {
      // TODO: Implement something faster.
      return std::floor(std::log10(value)) + 1;
    }

    [[maybe_unused]] uint64_t power_of_10(uint8_t exponent) {
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

    [[maybe_unused]] uint64_t slice_msb_digits(uint64_t value, uint8_t digits_to_remove) {
      return value / power_of_10(digits_to_remove);
    }

    [[maybe_unused]] uint64_t repeat_value(uint64_t value,
                                           uint8_t value_width,
                                           uint8_t target_width) {
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

    [[maybe_unused]] uint64_t any_repeated_digit_sum(uint64_t limit) {
      if (limit == 0) {
        return 0;
      }

      uint8_t const limit_digits = num_digits(limit);
      SPDLOG_DEBUG("limit: {}, limit_digits: {}", limit, limit_digits);

      std::array<uint64_t, 10> slice_sums{};

      for (uint8_t slice_digits = 1; slice_digits <= (limit_digits / 2); ++slice_digits) {
        if (limit_digits % slice_digits != 0) {  // Slices must fit evenly.
          continue;
        }

        uint64_t slice_upper = slice_msb_digits(limit, limit_digits - slice_digits);
        uint64_t repeated_upper = repeat_value(slice_upper, slice_digits, limit_digits);
        if (repeated_upper > limit) {
          repeated_upper = repeat_value(slice_upper - 1, slice_digits, limit_digits);
          slice_upper -= 1;
        }

        uint64_t const slice_lower = power_of_10(slice_digits - 1);
        uint64_t const repeated_lower = repeat_value(slice_lower, slice_digits, limit_digits);
        uint64_t const diff_between_repeats = repeat_value(1, slice_digits, limit_digits);

        uint64_t const num_repeats = slice_upper - slice_lower + 1;
        uint64_t slice_sum = repeated_lower * num_repeats +
                             (num_repeats * (num_repeats - 1) / 2) * diff_between_repeats;

        SPDLOG_DEBUG(
            "slice_digits: {}, repeated_lower: {}, repeated_upper: {}, "
            "num_repeats: {}, diff_between_repeats: {}, slice_sum: {}",
            slice_digits, repeated_lower, repeated_upper, num_repeats, diff_between_repeats,
            slice_sum);

        // Subtract double-counted values due numbers already generated by smaller slice widths.
        for (uint8_t subslice_digits = 1; subslice_digits <= (slice_digits / 2);
             ++subslice_digits) {
          if (slice_digits % subslice_digits == 0) {  // Slices must fit evenly.
            slice_sum -= slice_sums.at(subslice_digits);
          }
        }

        slice_sums.at(slice_digits) = slice_sum;
      }

      SPDLOG_DEBUG("slice_sums: {}", slice_sums);
      uint64_t sum = std::accumulate(slice_sums.begin(), slice_sums.end(), 0ULL);

      // Add sum for values with less digits. The array below contains the sum of any-repeated
      // digit numbers from 0 to the maximum value with n digits (i.e. for 0, 0, 99, 999, ...).
      if (limit_digits >= 3) {
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
      }

      return sum;
    }

    [[maybe_unused]] uint64_t twice_repeated_digit_sum(uint64_t limit) {
      if (limit == 0) {
        return 0;
      }

      uint8_t const limit_digits = num_digits(limit);
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

      // Add sum for values with less digits. The array below contains the sum of twice-repeated
      // digit numbers from 0 to the maximum value with n digits (i.e. for 0, 0, 99, 999, ...).
      if (limit_digits >= 3) {
        // TODO: Compress, since entries with odd number of digits don't contribute to the sum.
        static constexpr auto sum_for_max_value = std::to_array<uint64_t>({
            0ULL,
            0ULL,
            495ULL,
            495ULL,
            495'900ULL,
            495'900ULL,
            495'540'450ULL,
            495'540'450ULL,
            495'500'035'950ULL,
            495'500'035'950ULL,
        });

        uint64_t const & less_digits_sum = sum_for_max_value.at(limit_digits - 1);
        SPDLOG_DEBUG("sum: {}, less_digits_sum: {}, result: {}", sum, less_digits_sum,
                     sum + less_digits_sum);
        sum += less_digits_sum;
      }

      SPDLOG_DEBUG("<limit: {}, sum: {}>", limit, sum);
      return sum;
    }

  }  // namespace

  uint64_t day_t<2>::solve(part_t<1>, std::string_view input) {
    auto const entries = parse_input(version<0>, input);

    uint64_t sum = 0;

    for (auto const & entry : entries) {
      // Skip if both bounds have same number of odd number of digits.
      uint8_t const lower_digits = num_digits(entry.lower);
      uint8_t const upper_digits = num_digits(entry.upper);
      if (lower_digits == upper_digits && (lower_digits % 2) == 1) {
        continue;
      }

      auto const entry_sum =
          twice_repeated_digit_sum(entry.upper) - twice_repeated_digit_sum(entry.lower - 1);
      sum += entry_sum;
      SPDLOG_DEBUG("[entry: {}, entry_sum: {}, total_sum: {}]", entry, entry_sum, sum);
    }

    return sum;
  }

  uint64_t day_t<2>::solve(part_t<2>, std::string_view input) {
    auto const entries = parse_input(version<0>, input);

    uint64_t sum = 0;

    for (auto const & entry : entries) {
      auto const entry_sum =
          any_repeated_digit_sum(entry.upper) - any_repeated_digit_sum(entry.lower - 1);
      sum += entry_sum;
      SPDLOG_DEBUG("[entry: {}, entry_sum: {}, total_sum: {}]", entry, entry_sum, sum);
    }

    return sum;
  }

}  // namespace aoc25
