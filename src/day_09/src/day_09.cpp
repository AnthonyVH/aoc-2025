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
#include <vector>

namespace aoc25 {
  namespace {

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

      auto const add_edge = [&](point_t const & point_a, point_t const & point_b) {
        assert(point_a.x == point_b.x);  // Vertical edges only.
        auto & edges = get_edges_vector(point_a.x);
        edges.emplace_back(to_array(point_a.y, point_b.y));
      };

      size_t idx = 0;

      if (points[0].y == points[1].y) {
        // First edge is horizontal, so offset indices by 1, and handle wrap-around edge.
        idx = 1;
        add_edge(points.back(), points[0]);
      }

      for (; idx < points.size() - 1; idx += 2) {
        auto const & point_a = points[idx + 0];
        auto const & point_b = points[idx + 1];
        add_edge(point_a, point_b);
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

    [[maybe_unused]] int64_t polygon_signed_area(std::span<point_t const> points) {
      int64_t area = 0;

      auto const signed_area = [](point_t const & a, point_t const & b) {
        // Shoelace trapezoid formula simplified for horizontal edges.
        assert(a.y == b.y);  // Only support horizontal edges.
        return (static_cast<int64_t>(a.x) - b.x) * a.y;
      };

      size_t const num_points = points.size();
      size_t idx = 0;

      // The edges always alternate between horizontal and vertical. When using the trapezoid
      // formulate in the shoelace algorithm, vertical edges don't contribute to the area. So first
      // find out where the first horizontal edge is and only take those into account.
      if (points[0].y != points[1].y) {
        // First edge is vertical, so offset indices by 1, and add "wrap-around" edge manually.
        idx = 1;
        area += signed_area(points.back(), points[0]);
      }

      for (; idx < num_points - 1; idx += 2) {
        assert(idx + 1 < num_points);
        auto const & point_a = points[idx + 0];
        auto const & point_b = points[idx + 1];
        area += signed_area(point_a, point_b);
      }

      SPDLOG_DEBUG("Polygon signed area: {}", area);
      assert(area != 0);  // Assumed no degenerate polygons.
      return area;
    }

    bool is_left_turn(point_t const & a, point_t const & b, point_t const & c) {
      // Vector AB.
      int64_t const ab_x = static_cast<int64_t>(b.x) - a.x;
      int64_t const ab_y = static_cast<int64_t>(b.y) - a.y;

      // Vector AC.
      int64_t const ac_x = static_cast<int64_t>(c.x) - a.x;
      int64_t const ac_y = static_cast<int64_t>(c.y) - a.y;

      auto const cross_product = (ab_x * ac_y) - (ab_y * ac_x);
      assert(cross_product != 0);  // Assumed that there's no collinear points.

      bool const turns_left = cross_product > 0;
      SPDLOG_TRACE("Points {}, {}, {} make a {} turn (cross product {})", a, b, c,
                   turns_left ? "left" : "right", cross_product);

      return turns_left;
    }

    std::vector<point_t> convex_hull(std::span<point_t const> polygon) {
      // Points must be in clockwise order.
      assert(polygon_signed_area(polygon) < 0);

      // First point must be extremal in x.
      assert(polygon.front().x == std::ranges::min_element(polygon, {}, &point_t::x)->x);

      std::vector<point_t> hull;
      hull.reserve(polygon.size());

      hull.push_back(polygon[0]);
      hull.push_back(polygon[1]);

      for (size_t idx = 2; idx < polygon.size(); ++idx) {
        auto const & point = polygon[idx];

        while (hull.size() >= 2) {
          auto const hull_size = hull.size();
          auto const & point_a = hull[hull_size - 2];
          auto const & point_b = hull[hull_size - 1];

          // Check if the corner formed by the last three points is going left. If so, it's going
          // counter-clockwise (i.e. concave) and we need to remove the last point from the hull.
          if (is_left_turn(point_a, point_b, point)) {
            SPDLOG_TRACE("  Hull is concave at points [{}, {}, {}], removing {} from hull.",
                         point_a, point_b, point, point_b);
            hull.pop_back();
          } else {
            break;  // Right turn or collinear, keep last point.
          }
        }

        hull.push_back(point);
      }

      SPDLOG_DEBUG("Calculated convex hull with {} / {} points", hull.size(), polygon.size());
      return hull;
    }

    /** A list of points that are either contained within this struct, or reference
     * a span of external points.
     */
    struct chain_t {
      explicit chain_t(std::span<point_t const> span)
          : points_{}, span_{span} {}

      chain_t(std::span<point_t const> first, std::span<point_t const> second)
          : points_{}, span_{} {
        points_.reserve(first.size() + second.size());
        points_.insert(points_.end(), first.begin(), first.end());
        points_.insert(points_.end(), second.begin(), second.end());
        span_ = points_;
      }

      auto begin() const { return span_.begin(); }
      auto end() const { return span_.end(); }

      size_t size() const { return span_.size(); }

     private:
      std::vector<point_t> points_;
      std::span<point_t const> span_;
    };

  }  // namespace

  uint64_t day_t<9>::solve(part_t<1>, version_t<0>, simd_string_view_t input) {
    // NOTE: Also tried this with SoA and SIMD, but that actually didn't improve performance. Didn't
    // look into it further because this scalar version is already very fast.
    std::vector<point_t> points = parse_input(input);
    assert(points.size() >= 4);

    /* The largest possible rectangle is guaranteed to be between two points on the convex hull of
     * the polygon. So first calculate the convex hull (in O(N)). Then split the hull into 4
     * quadrants (top-left, top-right, bottom-left, bottom-right) in O(N). Finally brute-force check
     * all rectangles between points in opposite quadrants (top-left with bottom-right, top-right
     * with bottom-left) in O(N^2). The largest possible rectangle is one of the two found this way.
     */

    /* The complex hull algorithm needs to start at an extremal point, so find one first. Then
     * rotate it to the beginning, such that we don't need to use modulo arithmetic to iterate the
     * list of points.
     *
     * However, the convex hull algorithm also needs the points to be in clockwise order. Given that
     * the first point in the list is extremal, the turn direction between two edges indicates
     * whether the points are in clockwise or counter-clockwise order. A left turn indicates
     * counter-clockwise order, while a right turn indicates clockwise order.
     *
     * This means, that if it turns out that the points are in counter-clockwise order, we need to
     * reverse the order of the points. However, if we do that after rotating the extremal point to
     * the front, the extremal point then ends up at the end of the list, and we would need another
     * rotation. So, in that case, we instead rotate the extremal point to the back of the list, and
     * then reverse the entire list. This way, the extremal point ends up at the front as desired.
     */
    auto const min_x_it = std::ranges::min_element(points, {}, &point_t::x);
    size_t const min_x_idx = std::ranges::distance(points.begin(), min_x_it);

    auto const are_points_cw =
        !is_left_turn(points[(min_x_idx + points.size() - 1) % points.size()], points[min_x_idx],
                      points[(min_x_idx + 1) % points.size()]);

    if (are_points_cw) {  // Simply rotate to put min x at front.
      std::ranges::rotate(points, min_x_it);
    } else {  // Rotate to put min x at back, then reverse entire list.
      SPDLOG_DEBUG("Reversing point order to be clockwise.");
      std::ranges::rotate(points, min_x_it + 1);
      std::ranges::reverse(points);
      assert(polygon_signed_area(points) < 0);
    }

    // Calculate the convex hull.
    auto const hull = convex_hull(points);

    // Find chains between extremal points.
    auto const x_extrema_it = std::ranges::minmax_element(hull, {}, &point_t::x);
    auto const y_extrema_it = std::ranges::minmax_element(hull, {}, &point_t::y);
    size_t const x_min_pos = std::ranges::distance(hull.begin(), x_extrema_it.min);
    size_t const x_max_pos = std::ranges::distance(hull.begin(), x_extrema_it.max);
    size_t const y_min_pos = std::ranges::distance(hull.begin(), y_extrema_it.min);
    size_t const y_max_pos = std::ranges::distance(hull.begin(), y_extrema_it.max);

    // Create chains of points between extremal points. One of these chains crosses the 0 index, so
    // to avoid having to do modulo arithmetic, we create a new vector to hold points on that chain.
    auto const create_chain = [&](size_t start_pos, size_t end_pos) -> chain_t {
      if (start_pos < end_pos) {
        return chain_t(std::span<point_t const>(hull).subspan(start_pos, end_pos - start_pos + 1));
      } else {
        return chain_t(std::span<point_t const>(hull).subspan(start_pos, hull.size() - start_pos),
                       std::span<point_t const>(hull).subspan(0, end_pos + 1));
      }
    };

    auto const top_left_chain = create_chain(x_min_pos, y_max_pos);
    auto const top_right_chain = create_chain(y_max_pos, x_max_pos);
    auto const bottom_right_chain = create_chain(x_max_pos, y_min_pos);
    auto const bottom_left_chain = create_chain(y_min_pos, x_min_pos);

    // The largest rectangle must be between points in opposite chains. Brute-force check all
    // rectangles between points in these opposite chains, and then select the largest one.
    auto const brute_force_max_rectangle = [](chain_t const & chain_a,
                                              chain_t const & chain_b) -> int64_t {
      SPDLOG_DEBUG("Brute-forcing {} rectangles between chains of size {} and {}",
                   chain_a.size() * chain_b.size(), chain_a.size(), chain_b.size());

      int64_t max_area = 0;

      for (auto const & point_a : chain_a) {
        for (auto const & point_b : chain_b) {
          int64_t const area = calc_area(point_a, point_b);
          SPDLOG_TRACE("Area between {} and {}: {}", point_a, point_b, area);
          max_area = std::max(max_area, area);
        }
      }

      return max_area;
    };

    auto const max_area_top_left_bottom_right =
        brute_force_max_rectangle(top_left_chain, bottom_right_chain);
    auto const max_area_top_right_bottom_left =
        brute_force_max_rectangle(top_right_chain, bottom_left_chain);
    auto const max_area = std::max(max_area_top_left_bottom_right, max_area_top_right_bottom_left);
    SPDLOG_DEBUG("Checked {} areas", (top_left_chain.size() * bottom_right_chain.size()) +
                                         (top_right_chain.size() * bottom_left_chain.size()));

    return max_area;
  }

  uint64_t day_t<9>::solve(part_t<2>, version_t<0>, simd_string_view_t input) {
    std::vector<point_t> points = parse_input(input);

    // Create list of vertical edges for each x-coordinate.
    auto const vertical_edges = build_vertical_edges(points);
    SPDLOG_DEBUG("Vertical edges: {::s}", vertical_edges);

    auto const vertical_ranges = edges_to_ranges(vertical_edges);
    SPDLOG_DEBUG("Vertical ranges: {::s}", vertical_ranges);

    /* For each x-coordinate i:
     *
     *   - Create active vertical ranges, these are already calculated.
     *   - Loop over all x-coordinates j, j >= i, while there are active ranges.
     *     - For each active range, for each extremal y coordinate in that range, calculate area
     *       to extremal y coordinates from x_j that fall within that range. Store maximum area.
     *     - Update active vertical ranges using activate ranges at x_j.
     *     - Remove any of x_i's y coordinates that are not within active ranges anymore. If an
     *       active range is empty, remove it.
     */
    uint64_t max_area = 0;

#pragma omp parallel for reduction(max : max_area) schedule(dynamic) num_threads(8)
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

          // Find active range which this edge's begin lies in.
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

          // Next find active range which this edge's end lies in.
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
