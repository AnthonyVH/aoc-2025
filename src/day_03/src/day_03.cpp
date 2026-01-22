#include "aoc25/day_03.hpp"

#include "aoc25/algorithm.hpp"
#include "aoc25/simd.hpp"
#include "aoc25/string.hpp"

#include <fmt/ranges.h>
#include <spdlog/spdlog.h>

#include <cassert>
#include <cstdint>
#include <numeric>
#include <ranges>
#include <span>
#include <string_view>

namespace aoc25 {
  namespace {

    static constexpr uint8_t max_line_length = 100;
    static constexpr uint8_t max_lines = 200;

    std::vector<simd_string_view_t> split_lines(simd_string_view_t input) {
      // Each line in the input has the same length.
      size_t const line_length = input.find('\n') + 1;
      size_t const num_lines = input.size() / line_length;

      std::vector<simd_string_view_t> result;
      result.reserve(num_lines);

      for (size_t line_idx = 0; line_idx < num_lines; ++line_idx) {
        auto const line = input.substr(line_idx * line_length, line_length - 1);  // No newline.
        assert(input[line_idx * line_length + line_length - 1] == '\n');
        result.push_back(line);
      }

      return result;
    }

    // Note: No initialization of the members, since the algorithms below initialize them.
    struct entry_t {
      char max_digit;
      uint8_t position;
    };

    [[maybe_unused]] std::string format_as(entry_t const & obj) {
      return fmt::format("<{} @ {}>", obj.max_digit, obj.position);
    }

    /** The idea behind this structure is that given an input of length N, and needing to generate a
     * number of D digits, the first digit must be selected from the first N - D entries of the
     * input. For each subsequent digit, the available entries extend to the right by one position.
     * Furthermore, if digit[i] used input entry at position P, then digit[i + 1] can only select
     * from entries at position > P. Note that this doesn't the end of the range from which an input
     * entry can be selected.
     *
     * To efficiently find the maximum digit for each position, we keep track of the maximum digit
     * that is available starting from each position in the input. E.g. if input[i + 1] = '5', and
     * input[i] = '3', then we know that we could select '5' as the maximum digit for position i.
     *
     * We can calculate this information by iterating backwards over the input once, and keeping
     * track of the maximum seen so far, as well as its position. If we see a value equal to the
     * current maximum the earlier position (i.e. lower index in the input) is preferred, since it
     * will allow for a later digit to use the maximum from a later position.
     *
     * Now we can select the maximum value available for digit 0 by simply looking at entry[0]. This
     * also tells us where this maximum digit is found in the input. Hence, we can directly skip to
     * that position + 1 as the entry to look at for the next digit.
     *
     * Whenever we want to select the value for the next digit, we must also update the existing
     * entries, because one extra input entry is now available to select from. However, there is no
     * point in propagating updates past the entry from which the current digit will be selected,
     * since those earlier entries will not be used anymore.
     *
     * By iteratively doing all of this, we can select the maximum possible digit for each position
     * in O(N) time, walking the entire input only once.
     */
    struct max_digit_info_t {
      explicit max_digit_info_t(std::span<char const> input)
          : entries_{}, next_digit_pos{0} {
        assert(input.size() <= entries_.size());
        initialize(input);
      }

      void update(uint8_t pos, char data) {
        // Updating the entry at pos is not required for correctness, however it allows the update
        // loop to exit earlier in many cases and thus speeds up the overall algorithm.
        entries_[pos] = {
            .max_digit = data,
            .position = pos,
        };

        // Propagate the update backwards to the front. But not past the earliest possible position
        // for the next digit. This was benchmarked to be faster than:
        //   - a linear search from the front, then filling all entries up to pos.
        //   - a binary search, then filling all entries up to pos.
        //   - a "skip walk" jumping from each best position to the next and only updating a single
        //     entry.
        uint8_t update_idx = pos;
        for (; update_idx-- > next_digit_pos;) {
          if (entries_[update_idx].max_digit >= data) {
            break;  // If current entry is correct, the next ones are too.
          }
        }

        // Positions between the update_idx one and pos will be jumped over if ever used, so no
        // need to update them. Note that the loop exited when update_idx was past the element to
        // update, so increment it again.
        entries_[++update_idx] = entry_t{
            .max_digit = data,
            .position = pos,
        };

        SPDLOG_TRACE("next_digit_pos: {}, update: {}, update_idx @ {}, entries: {}", next_digit_pos,
                     entries_[update_idx], update_idx,
                     std::span{&entries_[next_digit_pos], entries_.data() + entries_.size()});
      }

      char pop_next_digit() {
        auto const & entry = entries_[next_digit_pos];
        next_digit_pos = entry.position + 1;
        return entry.max_digit;
      }

      [[maybe_unused]] friend std::string format_as(max_digit_info_t const & obj) {
        return fmt::format("[next_digit_pos: {}, entries: {::s}]", obj.next_digit_pos,
                           obj.entries_);
      }

     private:
      std::array<entry_t, max_line_length> entries_;
      uint8_t next_digit_pos;

      void initialize(std::span<char const> input) {
        assert(!input.empty());

        // Prime loop by preparing data for the last entry.
        auto best_entry = entry_t{
            .max_digit = input.back(),
            .position = static_cast<uint8_t>(input.size() - 1),
        };

        // Iterate backwards, but skip the last position which it is already primed.
        for (uint8_t pos = input.size() - 1; pos-- > 0;) {
          char const data = input[pos];
          [[assume(data >= '0' && data <= '9')]];

          // Current entry comes before previous entry, since we're iterating backwards. So if it's
          // higher or equal, we should select it instead of a later one in the input. If not, then
          // it will never be used, so just skip it entirely.
          if (data >= best_entry.max_digit) {
            // Store best entry in the last position where it's the best, i.e. the previously
            // inspected position.
            entries_[pos + 1] = best_entry;
            entries_[best_entry.position] = entry_t{
                .max_digit = best_entry.max_digit,
                .position = static_cast<uint8_t>(pos + 1),
            };

            best_entry = {.max_digit = data, .position = pos};
          }
        }

        // Set the last best entry at position 0.
        entries_[0] = best_entry;
        entries_[best_entry.position] = entry_t{
            .max_digit = best_entry.max_digit,
            .position = 0,
        };

        SPDLOG_TRACE("entries: {}", std::span{entries_.data(), entries_.data() + input.size()});
      }
    };

  }  // namespace

  uint32_t day_t<3>::solve(part_t<1>, version_t<0>, simd_string_view_t input) {
    auto const lines = split_lines(input);

    uint32_t sum = 0;

#pragma omp parallel for reduction(+ : sum) schedule(static)
    for (auto const & line : lines) {
      // The first digit is the first largest in the range [0, N - 2). The second one is the
      // largest in the remaining range. This could also be done in a single pass, but it's faster
      // to do two passes with SIMD.
      size_t const first_digit_pos = find_maximum_pos(line.substr(0, line.size() - 1).as_span());
      char const second_digit = find_maximum(line.substr(first_digit_pos + 1).as_span());

      std::array<char, 2> const max_digits = {line[first_digit_pos], second_digit};
      SPDLOG_DEBUG("line: {:s}, max_digits: {}", line, max_digits);
      sum += 10 * (max_digits[0] - '0') + (max_digits[1] - '0');
    }

    return sum;
  }

  uint64_t day_t<3>::solve(part_t<2>, version_t<0>, simd_string_view_t input) {
    auto const lines = split_lines(input);

    static constexpr uint8_t digits_to_use = 12;

    uint64_t sum = 0;

#pragma omp parallel for reduction(+ : sum) schedule(static)
    for (auto const & line : lines) {
      // Prepare info for the first digit. Note that we already process all the entries required to
      // decide the first digit here, even though we'll reprocess the last entry again in the main
      // loop. This avoids a potential extra iteration over the entries at the cost of a single
      // useless update-iteration in the first iteration of the main loop.
      uint8_t const last_entry_for_first_digit_pos = line.size() - digits_to_use;
      auto max_digits =
          max_digit_info_t(std::span<char const>(line.data(), last_entry_for_first_digit_pos));
      SPDLOG_TRACE("Prepared max_digits: {}", max_digits);

      // Calculate jolts and update the max_digits entries as we go.
      uint64_t jolts = 0;

      // Note that the initial value of entry_idx_max will cause the first update to process an
      // entry we already prepared. However, since updating entries with already seen values is
      // idempotent. Furthermore, the update will exit almost immediately, so this won't cause a
      // measurable slowdown. This way we avoid conditional updates inside the loop.
      uint8_t entry_idx_max = last_entry_for_first_digit_pos - 1;

      for (uint8_t digit_idx = 0; digit_idx < digits_to_use; ++digit_idx) {
        // Update relevant max_digits entries with the next value in the input.
        ++entry_idx_max;
        assert(entry_idx_max < line.size());
        max_digits.update(entry_idx_max, line[entry_idx_max]);
        SPDLOG_TRACE("digit {}, max_digits: {}", digit_idx, max_digits);

        jolts *= 10;
        jolts += max_digits.pop_next_digit() - '0';
      }

      SPDLOG_DEBUG("line: {:s}, jolts: {}", line, jolts);
      sum += jolts;
    }

    return sum;
  }

}  // namespace aoc25
