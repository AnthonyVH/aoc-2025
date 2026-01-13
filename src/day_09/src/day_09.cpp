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
#include <utility>

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

    using edge_vector_t = std::vector<std::array<uint32_t, 2>>;

    struct edges_per_coord_t {
      uint32_t coord;
      edge_vector_t edges;
    };

    [[maybe_unused]] std::string format_as(edges_per_coord_t const & obj) {
      return fmt::format("<coord: {}, edges: {}>", obj.coord, obj.edges);
    }

    std::vector<edges_per_coord_t> build_vertical_edges(std::vector<point_t> const & points) {
      std::vector<edges_per_coord_t> result;
      std::unordered_map<uint32_t, uint16_t> coord_to_idx;

      point_t prev_point = points.back();

      auto const to_array = [](uint32_t lhs, uint32_t rhs) -> std::array<uint32_t, 2> {
        auto const [a, b] = std::minmax(lhs, rhs);
        return {a, b};
      };

      auto const get_edges_vector = [&](uint32_t coord) -> edge_vector_t & {
        auto const [it, inserted] =
            coord_to_idx.try_emplace(coord, static_cast<uint16_t>(coord_to_idx.size()));
        if (inserted) {
          result.emplace_back(edges_per_coord_t{
              .coord = coord,
              .edges = {},
          });
        }
        return result.at(it->second).edges;
      };

      for (auto const & point : points) {
        if (prev_point.x == point.x) {  // Vertical edge.
          auto & edges = get_edges_vector(point.x);
          edges.emplace_back(to_array(prev_point.y, point.y));
        } else {
          // No need to store horizontal edges.
        }

        prev_point = point;
      }

      // Sort by coordinate.
      std::ranges::sort(result, {}, &edges_per_coord_t::coord);

      return result;
    }

    using range_vector_t = std::vector<std::array<uint32_t, 2>>;

    // Create active horizontal and vertical ranges. These are basically between two coordinates,
    // i.e. if there's vertical edges at x = 1 and x = 3, then the first "active" vertical range is
    // between x = [1, 3).
    [[nodiscard]] range_vector_t merge_ranges(std::span<std::array<uint32_t, 2> const> active_range,
                                              std::span<std::array<uint32_t, 2> const> new_edges) {
      range_vector_t result;

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
    }

    struct ranges_per_coord_t {
      uint32_t coord;
      range_vector_t ranges;
    };

    [[maybe_unused]] std::string format_as(ranges_per_coord_t const & obj) {
      return fmt::format("<coord: {}, edges: {}>", obj.coord, obj.ranges);
    }

    /** Convert a range of edges into a range of active ranges at each coordinate. A range is
     * "active" if it lies inside the polygon.
     */
    std::vector<ranges_per_coord_t> edges_to_ranges(
        std::vector<edges_per_coord_t> const & edges_per_coord) {
      static constexpr auto empty_range = range_vector_t{};

      std::vector<ranges_per_coord_t> result;
      result.reserve(edges_per_coord.size());

      auto const * prev_range = &empty_range;
      for (auto const & edges : edges_per_coord) {
        auto const & edge_list = edges.edges;
        auto const & new_elem = result.emplace_back(ranges_per_coord_t{
            .coord = edges.coord,
            .ranges = merge_ranges(*prev_range, edge_list),
        });
        prev_range = &new_elem.ranges;
      }

      // Last range should be empty, because it's at the end of the polygon.
      assert(!result.empty());
      assert(prev_range->empty());

      return result;
    }

    struct active_range_t {
      std::array<uint32_t, 2> range;
      std::vector<uint32_t> coords;
    };

    [[maybe_unused]] std::string format_as(active_range_t const & obj) {
      return fmt::format("<range: {}, coords: {}>", obj.range, obj.coords);
    }

    bool contains(std::span<uint32_t const, 2> range, uint32_t coord) noexcept {
      return (coord >= range[0]) && (coord <= range[1]);
    }

    std::vector<active_range_t> build_active_ranges(
        std::vector<std::array<uint32_t, 2>> const & ranges,
        std::vector<uint32_t> const & coords) {
      SPDLOG_DEBUG("Building active ranges from ranges: {} and coords: {}", ranges, coords);
      std::vector<active_range_t> result;
      result.reserve(ranges.size());

      size_t coord_idx = 0;
      size_t range_idx = 0;
      active_range_t * active_range = nullptr;

      // We want to end up with a list of active ranges and the coordinates in them. If there's no
      // coordinates in range, then we shouldn't store that range. Both coord and ranges are sorted.
      // So iterate in lockstep over both, and add to the result while they're both in range.
      while ((coord_idx < coords.size()) && (range_idx < ranges.size())) {
        auto const coord = coords.at(coord_idx);

        if (active_range) {
          /* There's two options:
           *   - coord is in active range: add it, and advance
           *   - coord is after active range: close active range
           *
           * Due to the sorted inputs, the third option (coord before active range) is not possible.
           */
          if (contains(active_range->range, coord)) {
            active_range->coords.push_back(coord);
            ++coord_idx;
          } else {
            assert(coord > active_range->range[1]);
            ++range_idx;
            active_range = nullptr;
          }
        } else {
          /* Here there's three options:
           *   - coord is in range: create new active range, add coord, advance coord
           *   - coord is before range: advance coord, i.e. discard the current coord
           *   - coord is after range: advance range, i.e. discard the current range
           */
          if (contains(ranges.at(range_idx), coord)) {
            active_range = std::addressof(result.emplace_back(active_range_t{
                .range = ranges.at(range_idx),
                .coords = {coord},
            }));
            ++coord_idx;
          } else if (coord < ranges.at(range_idx)[0]) {
            ++coord_idx;
          } else if (coord > ranges.at(range_idx)[1]) {
            ++range_idx;
          }
        }
      }

      return result;
    }

    void update_active_ranges(std::vector<active_range_t> & active_ranges,
                              std::vector<std::array<uint32_t, 2>> const & new_ranges) {
      SPDLOG_TRACE("    Updating active ranges {::s} with ranges: {}", active_ranges, new_ranges);

      size_t active_idx = 0;
      size_t new_idx = 0;

      // New ranges can only make the active range smaller, never larger.
      while ((active_idx < active_ranges.size()) && (new_idx < new_ranges.size())) {
        auto & active_range = active_ranges.at(active_idx);
        auto const & new_range = new_ranges.at(new_idx);

        bool const start_before_range = new_range[0] < active_range.range[0];
        bool const end_before_range = new_range[1] <= active_range.range[0];

        bool const start_after_range = new_range[0] >= active_range.range[1];
        bool const end_after_range = new_range[1] > active_range.range[1];

        SPDLOG_TRACE(
            "    Comparing active range {} (# {}) with new range {} (# {}): "
            "start_before_range={}, "
            "end_before_range={}, start_after_range={}, end_after_range={}",
            active_range, active_idx, new_range, new_idx, start_before_range, end_before_range,
            start_after_range, end_after_range);

        if (end_before_range) {
          assert(start_before_range);
          ++new_idx;  // New range is completely before range, skip it.
        } else if (start_after_range) {
          assert(end_after_range);

          // New range is completely after range, to nothing can overlap active range anymore.
          // Hence, it should be deleted.
          active_range.coords.clear();
          ++active_idx;
        } else if (start_before_range & end_after_range) {
          // New range completely covers range, keep it.
          ++active_idx;
        } else {
          // (Part of) new range overlaps with active range, reduce or split from active range.
          bool const start_inside_range = !start_before_range && !start_after_range;
          bool const end_inside_range = !end_before_range && !end_after_range;

          if (start_inside_range && !end_inside_range) {
            assert(!end_before_range && end_after_range);
            active_range.range[0] = new_range[0];  // Begin active range at start of new range.

            // Remove coords < begin of resized range.
            auto & coords = active_range.coords;
            size_t coord_idx = 0;

            while ((coord_idx < coords.size()) && (coords.at(coord_idx) < active_range.range[0])) {
              ++coord_idx;
            }

            std::ranges::rotate(coords, coords.begin() + coord_idx);
            coords.resize(coords.size() - coord_idx);

            // New range could overlap more active ranges, so check next active range.
            ++active_idx;
          } else if (end_inside_range) {
            assert((start_before_range || start_inside_range) && !start_after_range);

            // Split active range at end of new range. The upper range starts after the new range.
            auto const insert_it = active_ranges.begin() + active_idx + 1;
            auto upper_range_it = active_ranges.insert(
                insert_it, active_range_t{
                               .range = {new_range[1] + 1, active_range.range[1]},
                               .coords = {},
                           });

            // Lower range begins either at the old beign, or at the new range begin.
            auto lower_range_it = std::prev(upper_range_it);
            lower_range_it->range[0] = start_inside_range ? new_range[0] : lower_range_it->range[0];
            lower_range_it->range[1] = new_range[1];

            // Move upper coords to new range.
            auto & lower_coords = lower_range_it->coords;
            auto & upper_coords = upper_range_it->coords;

            size_t coord_idx = 0;

            // Skip all coordinates that come before lower range.
            while ((coord_idx < lower_coords.size()) &&
                   (lower_coords.at(coord_idx) < lower_range_it->range[0])) {
              ++coord_idx;
            }

            size_t const lower_coord_idx_start = coord_idx;

            // Skip past all coordinates inside lower range.
            while ((coord_idx < lower_coords.size()) &&
                   (lower_coords.at(coord_idx) <= lower_range_it->range[1])) {
              ++coord_idx;
            }

            size_t const lower_coord_idx_end = coord_idx;

            // Skip all coordinates to come between lower and upper range.
            while ((coord_idx < lower_coords.size()) &&
                   (lower_coords.at(coord_idx) < upper_range_it->range[0])) {
              ++coord_idx;
            }

            // Add all matching coordinates inside upper range.
            while ((coord_idx < lower_coords.size()) &&
                   (lower_coords.at(coord_idx) <= upper_range_it->range[1])) {
              upper_coords.push_back(lower_coords.at(coord_idx));
              ++coord_idx;
            }

            // Remove moved coordinates from lower range.
            std::ranges::rotate(lower_coords, lower_coords.begin() + lower_coord_idx_start);
            lower_coords.resize(lower_coord_idx_end - lower_coord_idx_start);

            // Apply next merges to upper range. New range finished inside current one, so check
            // next new range.
            ++active_idx;
            ++new_idx;
          } else {
            assert(false && "Unhandled case in active range update.");
            std::unreachable();
          }
        }
      }

      // Mark remaining active ranges as empty.
      for (; active_idx < active_ranges.size(); ++active_idx) {
        auto & active_range = active_ranges.at(active_idx);
        active_range.coords.clear();
      }

      // Remove all empty active ranges.
      auto const [new_end_it, prev_end_it] = std::ranges::remove_if(
          active_ranges, [](auto const & active_range) { return active_range.coords.empty(); });
      active_ranges.erase(new_end_it, prev_end_it);

      SPDLOG_TRACE("    Updated active ranges: {::s}", active_ranges);

      assert(std::ranges::none_of(active_ranges, [](auto const & active_range) {
        return active_range.range[0] == active_range.range[1];
      }));
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

    // Create list of vertical edges for each x-coordinate.
    auto const vertical_edges = build_vertical_edges(points);
    SPDLOG_DEBUG("Vertical edges: {}", vertical_edges);

    auto const vertical_ranges = edges_to_ranges(vertical_edges);
    SPDLOG_DEBUG("Vertical ranges: {}", vertical_ranges);

    /* For each x-coordinate i:
     *
     *   - Create active vertical ranges, there are already known.
     *   - Loop over all x-coordinates j, j >= i, while there are active ranges.
     *     - For each active range, for each extremal y coordinate in that range, calculate area
     *       with extremal y coordinates from x_j that fall within that range. Store maximum area.
     *     - Update active vertical ranges using activate ranges at x_j.
     *     - Remove any of x_i's y coordinates that are not within active ranges anymore. If an
     *       active range is empty, remove it.
     */
    uint64_t max_area = 0;

    for (size_t idx_a = 0; idx_a < vertical_edges.size() - 1; ++idx_a) {
      auto const & edges_a = vertical_edges.at(idx_a);
      auto const & ranges_a = vertical_ranges.at(idx_a).ranges;
      auto const x_a = edges_a.coord;

      auto active_ranges = build_active_ranges(
          ranges_a, edges_a.edges | std::views::join | std::ranges::to<std::vector>());
      SPDLOG_DEBUG("Starting at x = {} with active ranges: {::s}", x_a, active_ranges);

      for (size_t idx_b = idx_a + 1; !active_ranges.empty() && (idx_b < vertical_edges.size());
           ++idx_b) {
        auto const & edges_b = vertical_edges.at(idx_b);
        auto const x_b = edges_b.coord;
        auto const x_length = x_b - x_a + 1;
        auto const & new_edges = edges_b.edges;

        SPDLOG_DEBUG("  Checking edges at x = {}: {}", x_b, new_edges);

        size_t active_range_idx = 0;

        auto const is_in_range = [&](uint32_t coord) {
          if (active_range_idx >= active_ranges.size()) {
            return false;
          }

          auto const & active_range = active_ranges.at(active_range_idx);
          return (coord >= active_range.range[0]) && (coord <= active_range.range[1]);
        };

        auto const max_y_diff = [&](uint32_t coord) {
          auto const & active_range = active_ranges.at(active_range_idx);
          return std::max(std::abs(static_cast<int64_t>(active_range.coords.front()) - coord),
                          std::abs(static_cast<int64_t>(active_range.coords.back()) - coord));
        };

        for (size_t edges_idx = 0;
             (active_range_idx < active_ranges.size()) && (edges_idx < new_edges.size());
             ++edges_idx) {
          auto const & edge = new_edges.at(edges_idx);

          // Find active range which this edge start lies in.
          auto const is_above_range_end = [&](uint32_t coord) {
            auto const & active_range = active_ranges.at(active_range_idx);
            return coord > active_range.range[1];
          };

          while ((active_range_idx < active_ranges.size()) && is_above_range_end(edge[0])) {
            ++active_range_idx;
          }

          if (is_in_range(edge[0])) {  // Calculate maximum area with this edge's begin.
            auto const max_area_with_begin = x_length * (max_y_diff(edge[0]) + 1);
            max_area = std::max<uint64_t>(max_area, max_area_with_begin);
          }

          // Next find active range which this edge end lies in.
          auto const is_below_range_begin = [&](uint32_t coord) {
            auto const & active_range = active_ranges.at(active_range_idx);
            return coord < active_range.range[0];
          };

          while ((active_range_idx < active_ranges.size()) && is_below_range_begin(edge[1])) {
            ++active_range_idx;
          }

          if (is_in_range(edge[1])) {  // Calculate maximum area with this edge's end.
            auto const max_area_with_end = x_length * (max_y_diff(edge[1]) + 1);
            max_area = std::max<uint64_t>(max_area, max_area_with_end);
          }
        }

        // Update the active ranges.
        auto const & ranges_b = vertical_ranges.at(idx_b).ranges;
        update_active_ranges(active_ranges, ranges_b);
        SPDLOG_TRACE("    After x = {}, active ranges: {::s}", x_b, active_ranges);
      }
    }

    return max_area;
  }

}  // namespace aoc25

#endif  // HWY_ONCE
