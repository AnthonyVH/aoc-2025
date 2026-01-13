#include "aoc25/day_03.hpp"

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

    std::span<simd_string_view_t> split_lines(simd_string_view_t input,
                                              std::span<simd_string_view_t> output) {
      size_t begin = 0;
      size_t line_idx = 0;

      while (begin < input.size()) {
        assert(line_idx < output.size());

        size_t const line_end = input.find('\n', begin);
        assert(line_end != input.npos);

        output[line_idx++] = input.substr(begin, line_end - begin);
        begin = line_end + 1;
      }

      return output.first(line_idx);
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

      void update_alt_a(uint8_t pos, char data) {
        // Find the last position that is less than the new value. Don't inspect entries which won't
        // be used anyway, by jumping over them.
        uint8_t update_idx = next_digit_pos;
        while ((update_idx < pos) && (entries_[update_idx].max_digit >= data)) {
          update_idx = entries_[update_idx].position + 1;
        }

        // Update the found position. If we ever use this for a digit, then we will jump to the
        // position after pos for the next digit. If that next position is less than the current
        // value, then it will contain info, otherwise it will be empty (since its info will also be
        // stored elsewhere), but that must mean that it was a higher value than this one, in which
        // case the entry at pos won't be used anyway.
        entries_[update_idx] = entry_t{
            .max_digit = data,
            .position = pos,
        };

        SPDLOG_TRACE("[update_alt_a] next_digit_pos: {}, update: {}, update_it @ {}, entries: {}",
                     next_digit_pos, entries_[update_idx], update_idx,
                     std::span{&entries_[next_digit_pos], entries_.data() + entries_.size()});
      }

      void update_alt_b(uint8_t pos, char data) {
        // Find the last position that is less than the new value. All entries from that position
        // to the end need to be updated to match the new entry. Just doing a lineair search here,
        // because the range to search is small, so this is faster than a binary search.
        auto const update_it =
            std::find_if(&entries_[next_digit_pos], &entries_[pos],
                         [data](entry_t const & entry) { return entry.max_digit < data; });

        // Update all the entries from the found position up to the updated position. This is
        // necessary because otherwise the search won't work properly.
        std::fill(update_it, &entries_[pos + 1],
                  entry_t{
                      .max_digit = data,
                      .position = pos,
                  });

        SPDLOG_TRACE("[update_alt_b] next_digit_pos: {}, update: {}, update @ {}, entries: {}",
                     next_digit_pos, entries_[pos], std::distance(entries_.data(), update_it),
                     std::span{&entries_[next_digit_pos], entries_.data() + entries_.size()});
      }

      void update_alt_c(uint8_t pos, char data) {
        // Find the last position that is less than the new value. All entries from that position
        // to the end need to be updated to match the new entry.
        auto const update_it = std::lower_bound(
            &entries_[next_digit_pos], &entries_[pos], data,
            [](entry_t const & entry, char value) { return entry.max_digit >= value; });

        // Update all the entries from the found position up to the updated position. This is
        // necessary because otherwise the binary search won't work properly.
        std::fill(update_it, &entries_[pos + 1],
                  entry_t{
                      .max_digit = data,
                      .position = pos,
                  });

        SPDLOG_TRACE("[update_alt_c] next_digit_pos: {}, update: {}, update @ {}, entries: {}",
                     next_digit_pos, entries_[pos], std::distance(entries_.data(), update_it),
                     std::span{&entries_[next_digit_pos], entries_.data() + entries_.size()});
      }

      void update_alt_d(uint8_t pos, char data) {
        // Walk the list backwards to find the earliest position at which the current update is the
        // best choice for a digit. This walk happens in two steps: first we inspect an entry, and
        // if its value is less than the current value, we use its position field to jump to the
        // place where that entry is used. We then repeat this process by looking at the entry in
        // front of that, etc.

        // This was benchmarked to be faster than:
        //   - a linear iteration backwards up to maximum next_digit_pos.
        //   - a linear search from the front, then filling all entries up to pos.
        //   - a binary search, then filling all entries up to pos.
        //   - a "skip" jumping from each best position to the next and updating a single entry.
        uint8_t prev_update_idx = pos;
        int8_t update_idx = pos - 1;  // Signed so no wrap around if we end up at 0 in the loop.
        assert(entries_.size() <= std::numeric_limits<uint8_t>::max() / 2);

        SPDLOG_TRACE("[update_alt_d] start update for {}, next_digit_pos {}, update_idx: {}",
                     entry_t{.max_digit = data, .position = pos}, next_digit_pos, update_idx);

        while (update_idx >= next_digit_pos) {
          SPDLOG_TRACE("[update_alt_d] update_idx: {}", update_idx);

          assert(update_idx < static_cast<ssize_t>(entries_.size()));
          auto const & entry = entries_[update_idx];

          if (entry.max_digit >= data) {
            break;
          }

          assert(entry.position <= std::numeric_limits<uint8_t>::max() / 2);
          prev_update_idx = entry.position;  // Keep track of last valid update index.
          update_idx = entry.position - 1;   // Inspect entry in front of current one next.
        }

        // The update_idx is now pointing at an entry this is either out of bounds, or larger than
        // the current value. However, the prev_update_idx contains the position that should
        // actually be updated.
        entries_[prev_update_idx] = entry_t{
            .max_digit = data,
            .position = pos,
        };

        // Set current position to point to where the value is used.
        entries_[pos] = {
            .max_digit = data,
            .position = prev_update_idx,
        };

        SPDLOG_TRACE("[update_alt_d] next_digit_pos: {}, update: {}, update @ {}, entries: {}",
                     next_digit_pos, entries_[pos], prev_update_idx,
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
    // This benchmarks significantly faster than using aoc25::split().
    std::array<simd_string_view_t, max_lines> all_lines;
    std::span<simd_string_view_t> const lines = split_lines(input, all_lines);

    uint32_t sum = 0;

#pragma omp parallel for reduction(+ : sum)
    for (auto const & line : lines) {
      std::array<char, 2> max_digits = {};

      // Iterate backwards and keep track of the largest two values seen. Skip the very last
      // element, because that can only be used for the LSB.
      for (size_t pos = line.size() - 1; pos-- > 0;) {
        if (auto const data = line[pos]; data > max_digits[0]) {
          max_digits = {data, max_digits[0]};
        }
      }

      // Take last digit into account for LSB.
      assert(!line.empty());
      max_digits[1] = std::max(max_digits[1], line.back());

      SPDLOG_DEBUG("line: '{}', max_digits: {}", line, max_digits);
      sum += 10 * (max_digits[0] - '0') + (max_digits[1] - '0');
    }

    return sum;
  }

  uint64_t day_t<3>::solve(part_t<2>, version_t<0>, simd_string_view_t input) {
    std::array<simd_string_view_t, max_lines> all_lines;
    std::span<simd_string_view_t> const lines = split_lines(input, all_lines);

    static constexpr uint8_t digits_to_use = 12;

    uint64_t sum = 0;

#pragma omp parallel for reduction(+ : sum)
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

      SPDLOG_DEBUG("line: '{}', jolts: {}", line, jolts);
      sum += jolts;
    }

    return sum;
  }

}  // namespace aoc25
