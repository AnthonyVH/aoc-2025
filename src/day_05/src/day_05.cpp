#include "aoc25/day_05.hpp"

#include "aoc25/algorithm.hpp"
#include "aoc25/simd.hpp"
#include "aoc25/string.hpp"

#include <fmt/ranges.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <ranges>
#include <string_view>

namespace aoc25 {

  namespace {

    // NOTE: Ranges are exclusive!
    struct range_t {
      range_t()
          : range_t(0, 0) {}

      range_t(uint64_t b, uint64_t e)
          : begin(b), end(e) {
        assert(begin <= end);
      }

      uint64_t begin;
      uint64_t end;
    };

    [[maybe_unused]] std::string format_as(range_t const & obj) {
      return fmt::format("[{}-{})", obj.begin, obj.end);
    }

    struct ranges_t {
      void insert(range_t const & range) {
        SPDLOG_TRACE("inserting {}, ranges: {}", range, *this);

        // Find insertion point for the start of the range, where the found range's end is >= new
        // range's begin. We then check if the new range's begin overlaps with the previous range.
        auto const begin_it = aoc25::lower_bound(end_, range.begin);
        size_t const begin_pos = std::ranges::distance(end_.begin(), begin_it);

        // Find insertion point for the end of the range, where the found range's begin is > new
        // range's end. We then check if the new range's end overlaps with the previous range.
        auto const end_it = aoc25::upper_bound(begin_.begin() + begin_pos, begin_.end(), range.end);
        size_t const end_pos = std::ranges::distance(begin_.begin(), end_it);

        SPDLOG_TRACE("begin_pos @ {}, end_pos @ {}, merge: {}", begin_pos, end_pos,
                     begin_pos != end_pos);

        if (begin_pos != end_pos) {  // Update existing range and erase merged ones.
          assert(begin_pos < begin_.size());
          assert(end_pos - 1 < end_.size());

          begin_[begin_pos] = std::min(begin_[begin_pos], range.begin);
          end_[begin_pos] = std::max(end_[end_pos - 1], range.end);

          begin_.erase(std::next(begin_.begin() + begin_pos), begin_.begin() + end_pos);
          end_.erase(std::next(end_.begin() + begin_pos), end_.begin() + end_pos);
        } else {  // No merge, just insert.
          begin_.insert(begin_.begin() + begin_pos, range.begin);
          end_.insert(end_.begin() + begin_pos, range.end);
        }

        assert(begin_.size() == end_.size());
        SPDLOG_TRACE("after insert, ranges: {}", *this);
      }

      bool contains(uint64_t value) const {
        auto const end_it = aoc25::upper_bound(end_, value);
        size_t const pos = std::ranges::distance(end_.begin(), end_it);

        // End is exclusive, so no need to check for equality, it is larger than value.
        assert((end_it == end_.end()) || (value < *end_it));
        return (pos != end_.size()) && (begin_[pos] <= value);
      }

      std::span<uint64_t const> begins() const { return begin_; }
      std::span<uint64_t const> ends() const { return end_; }

      size_t size() const { return begin_.size(); }

     private:
      std::vector<uint64_t> begin_;
      std::vector<uint64_t> end_;
    };

    [[maybe_unused]] std::string format_as(ranges_t const & obj) {
      return fmt::format(
          "{::s}",
          std::views::zip(obj.begins(), obj.ends()) | std::views::transform([](auto const & e) {
            return fmt::format("[{}-{})", std::get<0>(e), std::get<1>(e));
          }));
    }

    std::pair<simd_string_view_t, ranges_t> parse_ranges(simd_string_view_t input) {
      ranges_t ranges;

      input = aoc25::split(input, [&](simd_string_view_t line) {
        if (line.empty()) {
          return false;  // Stop parsing.
        }

        auto const dash_pos = line.find('-');
        assert(dash_pos != line.npos);

        // Parse range begin and end.
        range_t range;

        convert_single_int<uint64_t>(line.substr(0, dash_pos),
                                     [&](uint64_t value) { range.begin = value; });
        convert_single_int<uint64_t>(line.substr(dash_pos + 1), [&](uint64_t value) {
          // Increment end of range by 1 to make it exclusive.
          range.end = value + 1;
        });

        ranges.insert(range);
        return true;  // Continue parsing.
      });

      return {input, std::move(ranges)};
    }

  }  // namespace

  uint32_t day_t<5>::solve(part_t<1>, version_t<0>, simd_string_view_t input) {
    auto const parse_result = parse_ranges(input);
    auto ingredient_lines = parse_result.first;
    auto const & fresh_ranges = parse_result.second;
    SPDLOG_DEBUG("Parsed fresh ranges: {}", fresh_ranges);

    // Iterate over ingredients and count how many are fresh.
    uint16_t num_fresh = 0;

    /* This benchmarks faster than:
     *    - Using aoc25::split() with a callback to process each line.
     *    - First splitting all lines and then processing them in parallel.
     *    - Splitting the ingredient_lines in approximately half and processing each in parallel.
     *    - Parsing the values, sorting them, and then doing a single O(N) walk though both the
     *      sorted values and the (sorted) ranges simultaneously.
     */
    while (!ingredient_lines.empty()) {
      auto const line_end = ingredient_lines.find('\n');
      assert(line_end != ingredient_lines.npos);

      auto const line = ingredient_lines.substr(0, line_end);
      ingredient_lines.remove_prefix(line_end + 1);

      convert_single_int<uint64_t>(line, [&](uint64_t ingredient_id) {
        bool const is_fresh = fresh_ranges.contains(ingredient_id);
        SPDLOG_TRACE("Checking ingredient ID {:15d}, fresh: {}", ingredient_id, is_fresh);
        num_fresh += is_fresh;
      });
    }

    return num_fresh;
  }

  uint64_t day_t<5>::solve(part_t<2>, version_t<0>, simd_string_view_t input) {
    // Note that we need to parse the ranges, because they overlap. I.e. we can't simply sum values
    // as we parse them.
    auto const [_, fresh_ranges] = parse_ranges(input);
    SPDLOG_DEBUG("Parsed fresh ranges: {}", fresh_ranges);

    // Just count the length of each range.
    // Note: The number of ranges is so small that multi-threading makes things slower.
    return std::ranges::fold_left(fresh_ranges.ends(), uint64_t{0}, std::plus<>{}) -
           std::ranges::fold_left(fresh_ranges.begins(), uint64_t{0}, std::plus<>{});
  }

}  // namespace aoc25
