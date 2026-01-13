#include "aoc25/day_05.hpp"

#include "aoc25/simd.hpp"
#include "aoc25/string.hpp"

#include <fmt/ranges.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <numeric>
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
        SPDLOG_TRACE("inserting {}, ranges: {::s}", range, ranges_);

        // Find insertion point for the start of the range, where the found range's end is >= new
        // range's begin. We then check if the new range's begin overlaps with the previous range.
        auto const begin_it = std::ranges::lower_bound(ranges_, range.begin, {}, &range_t::end);

        // Find insertion point for the end of the range, where the found range's begin is > new
        // range's end. We then check if the new range's end overlaps with the previous range.
        auto const end_it =
            std::upper_bound(begin_it, ranges_.end(), range.end,
                             [](auto value, auto const & rhs) { return value < rhs.begin; });

        SPDLOG_TRACE("begin_it @ {}, end_it @ {}, merge: {}",
                     std::distance(ranges_.begin(), begin_it),
                     std::distance(ranges_.begin(), end_it), begin_it != end_it);

        if (begin_it != end_it) {  // Update existing range and erase merged ones.
          begin_it->begin = std::min(begin_it->begin, range.begin);
          begin_it->end = std::max(std::prev(end_it)->end, range.end);
          ranges_.erase(std::next(begin_it), end_it);
        } else {  // No merge, just insert.
          ranges_.insert(begin_it, range);
        }

        SPDLOG_TRACE("after insert, ranges: {::s}", ranges_);
      }

      bool contains(uint64_t value) const {
        auto it = std::ranges::upper_bound(ranges_, value, {}, &range_t::end);
        // End is exclusive, so no need to check for equality, it is larger than value.
        assert((it == ranges_.end()) || (value < it->end));
        return (it != ranges_.end()) && (it->begin <= value);
      }

      // Iterators so FMT can format this.
      auto begin() const { return ranges_.begin(); }

      auto end() const { return ranges_.end(); }

     private:
      // TODO: Speedup if we keep two separate vectors for begins and ends? That allows the
      // contains() binary search to be done on just one vector.
      std::vector<range_t> ranges_;
    };

    std::pair<simd_string_view_t, ranges_t> parse_ranges(simd_string_view_t input) {
      ranges_t ranges;

      while (true) {
        auto const line_end = input.find('\n');
        assert(line_end != input.npos);

        auto const line = input.substr(0, line_end);
        input.remove_prefix(line_end + 1);

        if (line.empty()) {
          break;
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
      }

      return {input, std::move(ranges)};
    }

  }  // namespace

  uint32_t day_t<5>::solve(part_t<1>, version_t<0>, simd_string_view_t input) {
    auto const parse_result = parse_ranges(input);
    auto ingredient_lines = parse_result.first;
    auto const & fresh_ranges = parse_result.second;
    SPDLOG_DEBUG("Parsed fresh ranges: {::s}", fresh_ranges);

    // Iterate over ingredients and count how many are fresh.
    uint16_t num_fresh = 0;

    // This basic loop benchmarks faster than using SIMD split() with callbacks. It is also
    // faster than first splitting all lines and then processing them in parallel. It is also
    // faster than splitting the ingredient_lines in approximately half and processing each
    // in a separate thread.
    while (!ingredient_lines.empty()) {
      auto const line_end = ingredient_lines.find('\n');
      assert(line_end != ingredient_lines.npos);

      auto const line = ingredient_lines.substr(0, line_end);
      ingredient_lines.remove_prefix(line_end + 1);

      convert_single_int<uint64_t>(line, [&](uint64_t ingredient_id) {
        bool const is_fresh = fresh_ranges.contains(ingredient_id);
        SPDLOG_DEBUG("Checking ingredient ID {:15d}, fresh: {}", ingredient_id, is_fresh);
        num_fresh += is_fresh;
      });
    }

    return num_fresh;
  }

  uint64_t day_t<5>::solve(part_t<2>, version_t<0>, simd_string_view_t input) {
    // Can't use structured bindings, because then OMP won't compile when looping over the ranges.
    auto const parse_result = parse_ranges(input);
    auto const & fresh_ranges = parse_result.second;
    SPDLOG_DEBUG("Parsed fresh ranges: {::s}", fresh_ranges);

    // Just count the length of each range.
    uint64_t sum = 0;

    // Note: The number of ranges is so small that multi-threading makes things slower.
    for (auto const & range : fresh_ranges) {
      sum += range.end - range.begin;
    }

    return sum;
  }

}  // namespace aoc25
