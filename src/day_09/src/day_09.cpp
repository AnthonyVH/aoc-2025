#include "aoc25/day_09.hpp"

#include "aoc25/day.hpp"
#include "aoc25/simd.hpp"
#include "aoc25/string.hpp"

#include <spdlog/common.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <map>
#include <numeric>
#include <optional>
#include <ranges>
#include <type_traits>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "src/day_09.cpp"

// clang-format off
#include <hwy/foreach_target.h>
// clang-format on

#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();

namespace aoc25 {
  namespace {
    namespace HWY_NAMESPACE {

      namespace hn = hwy::HWY_NAMESPACE;

      [[maybe_unused]] void compiler_stop_complaining() {}

    }  // namespace HWY_NAMESPACE
  }  // namespace
}  // namespace aoc25

HWY_AFTER_NAMESPACE();

#ifdef HWY_ONCE

namespace aoc25 {

  namespace {

    HWY_EXPORT(compiler_stop_complaining);

    [[maybe_unused]] void compiler_stop_complaining() {
      return HWY_DYNAMIC_DISPATCH(compiler_stop_complaining)();
    }

    struct point_t {
      int32_t x;
      int32_t y;

      [[maybe_unused]] friend auto operator<(point_t const & lhs, point_t const & rhs) {
        return std::tie(lhs.x, lhs.y) < std::tie(rhs.x, rhs.y);
      }

      [[maybe_unused]] friend auto operator==(point_t const & lhs, point_t const & rhs) {
        return std::tie(lhs.x, lhs.y) == std::tie(rhs.x, rhs.y);
      }
    };

    [[maybe_unused]] std::string format_as(point_t const & obj) {
      return fmt::format("<{}, {}>", obj.x, obj.y);
    }

    std::vector<point_t> parse_input(simd_string_view_t input) {
      std::vector<point_t> result;
      result.reserve(1'000);

      split(input, [&](simd_string_view_t line) {
        size_t const comma_pos = line.find(',');
        auto const x_token = line.substr(0, comma_pos);
        auto const y_token = line.substr(comma_pos + 1);

        point_t point;
        convert_single_int<uint64_t>(x_token, [&](uint64_t value) { point.x = value; });
        convert_single_int<uint64_t>(y_token, [&](uint64_t value) { point.y = value; });
        result.push_back(point);

        SPDLOG_TRACE("Parsed {:s} into {}", line, point);
      });

      SPDLOG_DEBUG("Parsed {} points: {::s}", result.size(), result);
      return result;
    }

    uint64_t calc_area(point_t const & a, point_t const & b) {
      return (std::abs(static_cast<int64_t>(a.x) - b.x) + 1) *
             (std::abs(static_cast<int64_t>(a.y) - b.y) + 1);
    }

  }  // namespace

  uint64_t day_t<9>::solve(part_t<1>, version_t<0>, simd_string_view_t input) {
    std::vector<point_t> const points = parse_input(input);

    int64_t max_area = 0;
    size_t num_points = points.size();

    for (size_t idx_a = 0; idx_a < num_points - 1; ++idx_a) {
      auto const & point_a = points.at(idx_a);

      for (size_t idx_b = idx_a + 1; idx_b < num_points; ++idx_b) {
        auto const & point_b = points.at(idx_b);
        int64_t const area = calc_area(point_a, point_b);
        SPDLOG_DEBUG("Area between {} and {}: {}", point_a, point_b, area);
        max_area = std::max(max_area, area);
      }
    }

    return max_area;
  }

  /* TODO: The proper way for part 1 in O(N log N):
   * - Figure out direction of polygon (clockwise / counter-clockwise).
   * - Calculate convex hull (using e.g. Andrew's monotone chain algorithm).
   * - Use SMAWK to find the largest rectangle formed by any two points on the hull.
   */

  uint64_t day_t<9>::solve(part_t<2>, version_t<0>, simd_string_view_t input) {
    std::vector<point_t> points = parse_input(input);

    // Create map of vertical/horizontal edges for each x/y-coordinate.
    std::map<uint32_t, std::vector<std::array<uint32_t, 2>>> vertical_edges;
    std::map<uint32_t, std::vector<std::array<uint32_t, 2>>> horizontal_edges;

    {
      point_t prev_point = points.back();

      auto const to_array = [](uint32_t lhs, uint32_t rhs) -> std::array<uint32_t, 2> {
        auto const [a, b] = std::minmax(lhs, rhs);
        return {a, b};
      };

      for (auto const & point : points) {
        if (prev_point.x == point.x) {
          vertical_edges[point.x].emplace_back(to_array(prev_point.y, point.y));
        } else {
          horizontal_edges[point.y].emplace_back(to_array(prev_point.x, point.x));
        }

        prev_point = point;
      }
    }

    SPDLOG_DEBUG("Vertical edges: {}", vertical_edges);
    SPDLOG_DEBUG("Horizontal edges: {}", horizontal_edges);

    // Create active horizontal and vertical ranges. These are basically between two coordinates,
    // i.e. if there's vertical edges at x = 1 and x = 3, then the first "active" vertical range is
    // between x = [1, 3).
    auto const merge_ranges = [] [[nodiscard]] (
                                  std::span<std::array<uint32_t, 2> const> active_range,
                                  std::span<std::array<uint32_t, 2> const> new_edges) {
      std::vector<std::array<uint32_t, 2>> result;

      /* Assuming ranges are sorted. Pointing to a range in the active range and one in the new
       * edges one, loop the following:
       *
       * - As long as range in active range has earlier or equal starting point than range from new
       *   edges, add it to the result ranges. Increment active range pointer.
       * - If new edge range overlaps with last active range in result, split last active range in
       *   order to remove the new edge range from it. Increment edge range pointer.
       * - Otherwise, insert new edge range. Increment edge range pointer.
       *
       * Whenever inserting a range into the result, check if end of last element matches start of
       * new one, and if so merge them.
       */
      auto const add_range = [&](std::array<uint32_t, 2> const & range) {
        if (!result.empty() && (result.back()[1] == range[0])) {
          result.back()[1] = range[1];  // Merge consecutive ranges.
        } else {
          result.emplace_back(range);
        }
      };

      auto const merge_range = [&](std::array<uint32_t, 2> const & range) {
        // Equal endpoints don't count as overlap.
        assert(result.empty() || (result.back()[0] < range[1]));
        bool const has_overlap = !result.empty() && (range[0] < result.back()[1]);

        if (has_overlap) {
          // Remove range from last element.
          auto & last = result.back();

          bool const begin_equal = last[0] == range[0];
          bool const end_equal = last[1] == range[1];

          if (begin_equal && end_equal) {
            result.pop_back();
          } else if (begin_equal) {
            last[0] = range[1];
          } else if (end_equal) {
            last[1] = range[0];
          } else {  // Split into two.
            auto const split_last = std::array{range[1], last[1]};
            last[1] = range[0];
            result.emplace_back(split_last);
          }
        } else {
          add_range(range);
        }
      };

      auto active_it = active_range.begin();
      auto new_it = new_edges.begin();

      while ((active_it != active_range.end()) && (new_it != new_edges.end())) {
        bool const starts_earlier = active_it->at(0) <= new_it->at(0);
        if (starts_earlier) {
          add_range(*active_it++);
        } else {
          merge_range(*new_it++);
        }
      }

      while (active_it != active_range.end()) {
        add_range(*active_it++);
      }

      while (new_it != new_edges.end()) {
        merge_range(*new_it++);
      }

      return result;
    };

    auto const edges_to_ranges =
        [&](std::map<uint32_t, std::vector<std::array<uint32_t, 2>>> const & edge_map) {
          static constexpr auto empty_range = std::vector<std::array<uint32_t, 2>>{};
          std::map<uint32_t, std::vector<std::array<uint32_t, 2>>> result;

          auto const * prev_range = &empty_range;
          for (auto const & [x, edges] : edge_map) {
            auto const [it, _] = result.emplace(x, merge_ranges(*prev_range, edges));
            prev_range = &(it->second);
          }

          assert(!result.empty());
          assert(prev_range->empty());

          return result;
        };

    auto const vertical_ranges = edges_to_ranges(vertical_edges);
    auto const horizontal_ranges = edges_to_ranges(horizontal_edges);

    SPDLOG_DEBUG("Vertical ranges: {}", vertical_ranges);
    SPDLOG_DEBUG("Horizontal ranges: {}", horizontal_ranges);

    /* Find maximum area rectangle the dumb way:
     *
     *   - Iterate over all possible pairs.
     *   - For each pair, construct 4 edges.
     *   - Check whether edges fall completely inside an active range.
     *   - If yes, update maximum area.
     */
    auto const in_range =
        [&](std::map<uint32_t, std::vector<std::array<uint32_t, 2>>> const & range_map,
            uint32_t key, bool key_is_min_not_max, std::span<uint32_t const, 2> range) -> bool {
      // If key is first one in range, use entry at that key. Otherwise, use entry for previous key.
      auto key_it = range_map.find(key);
      assert(key_it != range_map.end());
      std::advance(key_it, key_is_min_not_max ? 0 : -1);
      auto const & ranges = key_it->second;
      SPDLOG_DEBUG("  Using ranges at key {}: {}", key_it->first, ranges);

      // First range for which end > range.begin.
      auto const begin_it =
          std::ranges::upper_bound(ranges, range[0], {}, [](auto const & e) { return e[1]; });
      bool const begin_in_range = (begin_it != ranges.end()) && (begin_it->at(0) <= range[0]);

      // First range for which end >= range.end.
      auto const end_it = std::ranges::lower_bound(begin_it, ranges.end(), range[1], {},
                                                   [](auto const & e) { return e[1]; });

      bool const is_in_range = begin_in_range && (begin_it == end_it);
      SPDLOG_DEBUG("  <{}, {}> is in range: {}", key, range, is_in_range);
      return is_in_range;
    };

    uint64_t max_area = 0;

    for (size_t idx_a = 0; idx_a < points.size(); ++idx_a) {
      auto const & point_a = points.at(idx_a);

      for (size_t idx_b = idx_a + 1; idx_b < points.size(); ++idx_b) {
        auto const & point_b = points.at(idx_b);
        SPDLOG_DEBUG("Rectangle between {} and {}", point_a, point_b);

        auto const [x_min, x_max] =
            std::minmax({static_cast<uint32_t>(point_a.x), static_cast<uint32_t>(point_b.x)});
        auto const [y_min, y_max] =
            std::minmax({static_cast<uint32_t>(point_a.y), static_cast<uint32_t>(point_b.y)});

        // Check all rectangle edges
        bool const in_polygon =
            in_range(horizontal_ranges, y_min, true, std::array{x_min, x_max}) &&
            ((y_min == y_max) ||
             in_range(horizontal_ranges, y_max, false, std::array{x_min, x_max})) &&
            in_range(vertical_ranges, x_min, true, std::array{y_min, y_max}) &&
            ((x_min == x_max) || in_range(vertical_ranges, x_max, false, std::array{y_min, y_max}));

        if (in_polygon) {
          auto const area = calc_area(point_a, point_b);
          SPDLOG_DEBUG("  area: {}, max area: {}", area, max_area);
          max_area = std::max(max_area, area);
        }
      }
    }

    return max_area;
  }

}  // namespace aoc25

#endif  // HWY_ONCE
