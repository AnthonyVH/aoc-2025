#include "aoc25/day_08.hpp"

#include "aoc25/day.hpp"
#include "aoc25/simd.hpp"
#include "aoc25/string.hpp"

#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <fmt/ostream.h>
#include <fmt/std.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <ranges>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "src/day_08.cpp"

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
      uint32_t x;
      uint32_t y;
      uint32_t z;

      [[maybe_unused]] friend auto operator<(point_t const & lhs, point_t const & rhs) {
        return std::tie(lhs.x, lhs.y, lhs.z) < std::tie(rhs.x, rhs.y, rhs.z);
      }
    };

    [[maybe_unused]] std::string format_as(point_t const & obj) {
      return fmt::format("<x: {}, y: {}, z: {}>", obj.x, obj.y, obj.z);
    }

    struct distance_t {
      uint64_t distance;
      uint16_t idx_a;
      uint16_t idx_b;

      [[maybe_unused]] friend auto operator<(distance_t const & lhs, distance_t const & rhs) {
        return lhs.distance < rhs.distance;  // Don't care about indices.
      }

      [[maybe_unused]] friend auto operator>(distance_t const & lhs, distance_t const & rhs) {
        return lhs.distance > rhs.distance;  // Don't care about indices.
      }
    };

    [[maybe_unused]] std::string format_as(distance_t const & obj) {
      return fmt::format("<distance: {}, idx_a: {}, idx_b: {}>", obj.distance, obj.idx_a,
                         obj.idx_b);
    }

    [[maybe_unused]] distance_t dist(uint16_t idx_a,
                                     uint16_t idx_b,
                                     point_t const & lhs,
                                     point_t const & rhs) {
      int64_t const dx = static_cast<int64_t>(lhs.x) - static_cast<int64_t>(rhs.x);
      int64_t const dy = static_cast<int64_t>(lhs.y) - static_cast<int64_t>(rhs.y);
      int64_t const dz = static_cast<int64_t>(lhs.z) - static_cast<int64_t>(rhs.z);

      return distance_t{
          .distance = static_cast<uint64_t>(dx * dx + dy * dy + dz * dz),
          .idx_a = idx_a,
          .idx_b = idx_b,
      };
    }

    struct disjoint_set_t {
      disjoint_set_t(uint16_t size)
          : parent_(size), size_(size, 1) {
        for (uint16_t idx = 0; idx < size; ++idx) {
          parent_[idx] = idx;
        }
      }

      uint16_t find(uint16_t item) const {
        if (parent_.at(item) != item) {
          parent_.at(item) = find(parent_.at(item));
        }
        return parent_[item];
      }

      /** @brief Merge sets a and b.
       *
       * @return The size of the merged set, if a merge happened.
       */
      std::optional<uint16_t> merge(uint16_t a, uint16_t b) {
        uint16_t const root_a = find(a);
        uint16_t const root_b = find(b);

        if (root_a == root_b) {
          return std::nullopt;
        }

        auto [small_idx, large_idx] = std::minmax(root_a, root_b, [&](uint16_t lhs, uint16_t rhs) {
          return size_.at(lhs) < size_.at(rhs);
        });

        parent_.at(small_idx) = large_idx;
        size_.at(large_idx) += size_.at(small_idx);
        size_.at(small_idx) = 0;  // Clear, so we can find max values later.
        return size_.at(large_idx);
      }

      uint16_t size(uint16_t item) const { return size_.at(find(item)); }

      std::vector<uint16_t> sizes() const && { return std::move(size_); }

     private:
      mutable std::vector<uint16_t> parent_;
      std::vector<uint16_t> size_;
    };

    std::vector<distance_t> calc_pairwise_distances(std::span<point_t const> points) {
      std::vector<distance_t> result;
      result.reserve(points.size() * (points.size() - 1) / 2);

      for (uint16_t idx_a = 0; idx_a < points.size(); ++idx_a) {
        for (uint16_t idx_b = idx_a + 1; idx_b < points.size(); ++idx_b) {
          result.push_back(dist(idx_a, idx_b, points[idx_a], points[idx_b]));
        }
      }

      return result;
    }

    std::vector<point_t> parse_input(simd_string_view_t input) {
      std::vector<point_t> result;

      split(input, [&](simd_string_view_t line) {
        std::array<uint32_t, 3> coords;
        size_t coord_idx = 0;

        while (!line.empty()) {
          size_t const comma_pos = line.find(',');
          simd_string_view_t const token = line.substr(0, comma_pos);
          convert_single_int<uint64_t>(token, [&](uint64_t value) { coords[coord_idx++] = value; });
          line.remove_prefix(token.size() + (comma_pos == line.npos ? 0 : 1));
        }

        result.push_back(point_t{.x = coords[0], .y = coords[1], .z = coords[2]});
      });

      SPDLOG_TRACE("Parsed {} points: {}", result.size(), result);
      return result;
    }

    struct ext_point_t : public point_t {
      uint16_t idx;
    };

    [[maybe_unused]] std::string format_as(ext_point_t const & obj) {
      return fmt::format("<idx: {}, x: {}, y: {}, z: {}>", obj.idx, obj.x, obj.y, obj.z);
    }

    struct closest_pair_finder_t {
      closest_pair_finder_t(std::span<ext_point_t const> sorted_x,
                            std::span<ext_point_t const> sorted_y,
                            std::span<ext_point_t const> sorted_z)
          : sorted_x_(sorted_x), sorted_y_(sorted_y), sorted_z_(sorted_z), pairs_() {}

      std::vector<std::pair<uint16_t, uint16_t>> find_pairs(size_t num_pairs) {
        num_pairs_ = num_pairs;
        max_accepted_dist_ = std::numeric_limits<uint64_t>::max();
        num_dist_calcs_ = 0;

        // Must preallocate to ensure references stay valid.
        size_t const num_split_lists =
            std::ceil(std::log2(sorted_x_.size())) - std::log2(brute_force_threshold) + 1;

        auto const prepare_lists = [&](std::vector<split_list_t> & lists) {
          lists.clear();
          lists.resize(num_split_lists);

          // Add some extra buffer to account for uneven split along median list element.
          size_t num_entries = (1.5 * sorted_x_.size()) / 2;

          for (size_t idx = 0; idx < num_split_lists; ++idx, num_entries /= 2) {
            assert(num_entries > 0);
            lists[idx].lhs.reserve(num_entries);
            lists[idx].rhs.reserve(num_entries);
          }
        };

        prepare_lists(x_lists_);
        prepare_lists(y_lists_);
        prepare_lists(z_lists_);

        pairs_.clear();
        pairs_.reserve(num_pairs);

        find_pairs_recursive_3d(sorted_x_, sorted_y_, sorted_z_, 0);

        SPDLOG_DEBUG("Calculated {} distances to find {} closest pairs", num_dist_calcs_,
                     num_pairs_);

        return pairs_ | std::views::transform([](distance_t const & d) {
                 return std::make_pair(d.idx_a, d.idx_b);
               }) |
               std::ranges::to<std::vector>();
      }

     private:
      struct split_list_t {
        std::vector<ext_point_t> lhs;
        std::vector<ext_point_t> rhs;
      };

      static constexpr size_t brute_force_threshold = 16;

      std::span<ext_point_t const> sorted_x_;
      std::span<ext_point_t const> sorted_y_;
      std::span<ext_point_t const> sorted_z_;

      size_t num_pairs_;
      std::vector<distance_t> pairs_;
      uint64_t max_accepted_dist_;
      uint64_t num_dist_calcs_;

      std::vector<split_list_t> x_lists_;
      std::vector<split_list_t> y_lists_;
      std::vector<split_list_t> z_lists_;

      std::vector<ext_point_t> slab_points_y_;
      std::vector<ext_point_t> slab_points_z_;
      std::vector<ext_point_t> strip_points_z_;

      bool try_add_distance(distance_t const & distance) {
        ++num_dist_calcs_;

        // Insert into list of pairs if it's among the closest found so far.
        size_t const num_current_pairs = pairs_.size();

        if (num_current_pairs < num_pairs_) {
          // NOTE: Keep accepted max distance at the maximum value to allow filling up the list.
          pairs_.push_back(distance);

          if ((num_current_pairs + 1) == num_pairs_) {
            std::make_heap(pairs_.begin(), pairs_.end());
            max_accepted_dist_ = pairs_.front().distance;
          }

          return true;
        } else if (distance < pairs_.front()) {
          std::pop_heap(pairs_.begin(), pairs_.end());
          pairs_.back() = distance;
          std::push_heap(pairs_.begin(), pairs_.end());
          max_accepted_dist_ = pairs_.front().distance;

          return true;
        }

        return false;
      }

      void bruteforce_distances(std::span<ext_point_t const> points) {
        size_t const num_points = points.size();

        for (uint16_t idx_i = 0; idx_i < num_points - 1; ++idx_i) {
          auto const & point_i = points[idx_i];

          for (uint16_t idx_j = idx_i + 1; idx_j < num_points; ++idx_j) {
            auto const & point_j = points[idx_j];

            auto const dx = static_cast<int64_t>(point_i.x) - static_cast<int64_t>(point_j.x);
            if (static_cast<uint64_t>(dx * dx) > max_accepted_dist_) {
              break;  // Since points are sorted, we can't obtain any closer distances.
            }

            auto const distance = dist(point_i.idx, point_j.idx, point_i, point_j);
            try_add_distance(distance);
          }
        }
      }

      void bruteforce_split_distances(std::span<ext_point_t const> points, uint32_t x_split) {
        size_t const num_points = points.size();

        for (uint16_t idx_i = 0; idx_i < num_points - 1; ++idx_i) {
          auto const & point_i = points[idx_i];
          bool const lhs_left_of_split = point_i.x <= x_split;

          for (uint16_t idx_j = idx_i + 1; idx_j < num_points; ++idx_j) {
            auto const & point_j = points[idx_j];

            auto const dy = static_cast<int64_t>(point_i.y) - static_cast<int64_t>(point_j.y);
            if (static_cast<uint64_t>(dy * dy) > max_accepted_dist_) {
              break;  // Since points are sorted, we can't obtain any closer distances.
            }

            bool const rhs_left_of_split = point_j.x <= x_split;
            if (lhs_left_of_split == rhs_left_of_split) {
              continue;
            }

            auto const distance = dist(point_i.idx, point_j.idx, point_i, point_j);
            try_add_distance(distance);
          }
        }
      }

      template <class Accessor>
      split_list_t const & split_list(std::span<ext_point_t const> points,
                                      uint32_t axis_value,
                                      Accessor accessor,
                                      std::vector<split_list_t> & split_lists,
                                      uint8_t split_level) {
        auto & result = split_lists.at(split_level);
        assert(points.data() != result.lhs.data());
        assert(points.data() != result.rhs.data());
        result.lhs.clear();
        result.rhs.clear();

        for (auto const & point : points) {
          if (std::invoke(accessor, point) <= axis_value) {
            result.lhs.push_back(point);
          } else {
            result.rhs.push_back(point);
          }
        }

        return result;
      }

      void find_pairs_recursive_2d(std::span<ext_point_t const> sorted_y,
                                   std::span<ext_point_t const> sorted_z,
                                   uint32_t x_split,
                                   [[maybe_unused]] uint8_t split_level) {
        SPDLOG_DEBUG("{:{}}[slab recursion] num points: {}, level: {}", "", 2 * (split_level + 1),
                     sorted_y.size(), split_level);

        // Apply same approach as for 3D recursion, but now treat it as a 2D problem.
        if (sorted_y.size() <= brute_force_threshold) {
          bruteforce_split_distances(sorted_y, x_split);
        } else {
          // Else, split points at median Y value.
          uint32_t const median_y = sorted_y[sorted_y.size() / 2].y;
          auto const split_it = std::ranges::find_if(
              sorted_y.subspan(sorted_y.size() / 2), [&](auto const y) { return y > median_y; },
              &ext_point_t::y);
          size_t const split_size_lhs = std::ranges::distance(sorted_y.begin(), split_it);

          auto const y_split_lhs = sorted_y.subspan(0, split_size_lhs);
          auto const y_split_rhs = sorted_y.subspan(split_size_lhs);
          auto const & z_split =
              split_list(sorted_z, median_y, &ext_point_t::y, z_lists_, split_level);

          find_pairs_recursive_2d(y_split_lhs, z_split.lhs, x_split, split_level + 1);
          find_pairs_recursive_2d(y_split_rhs, z_split.rhs, x_split, split_level + 1);

          // Find points within strip of width "max accepted distance" from the median Y value.
          find_center_points(sorted_z, strip_points_z_, &ext_point_t::y, median_y, split_level + 2);
          find_strip_pairs(strip_points_z_, x_split, median_y, split_level + 1);
        }
      }

      void find_strip_pairs(std::span<ext_point_t const> sorted_z,
                            uint32_t x_split,
                            uint32_t y_split,
                            [[maybe_unused]] uint8_t split_level) {
        SPDLOG_DEBUG("{:{}}[strip search] num points: {}, level: {}", "", 2 * (split_level + 1),
                     sorted_z.size(), split_level);

        // NOTE: Because we search for N closest pairs, we can not limit the number of neighbors
        // checked for each point to 7. For a single closest pair search this is the maximum number
        // of neighbors to check for each point in 2D space.
        size_t const num_points = sorted_z.size();

        for (size_t idx_i = 0; idx_i < num_points - 1; ++idx_i) {
          auto const & point_i = sorted_z[idx_i];
          [[maybe_unused]] bool const lhs_left_of_x_split = point_i.x <= x_split;
          [[maybe_unused]] bool const lhs_left_of_y_split = point_i.y <= y_split;

          for (size_t idx_j = idx_i + 1; idx_j < num_points; ++idx_j) {
            auto const & point_j = sorted_z[idx_j];

            auto const dz = static_cast<int64_t>(point_i.z) - static_cast<int64_t>(point_j.z);
            if (static_cast<uint64_t>(dz * dz) > max_accepted_dist_) {
              break;  // Since points are sorted, we can't obtain any closer distances.
            }

            // Only check pairs that are on opposite sides of the split plane/line.
            bool const rhs_left_of_x_split = point_j.x <= x_split;
            bool const rhs_left_of_y_split = point_j.y <= y_split;

            bool const different_x_side = lhs_left_of_x_split != rhs_left_of_x_split;
            bool const different_y_side = lhs_left_of_y_split != rhs_left_of_y_split;

            if (!different_x_side || !different_y_side) {
              continue;
            }

            auto const distance = dist(point_i.idx, point_j.idx, point_i, point_j);
            try_add_distance(distance);
          }
        }
      }

      void find_pairs_recursive_3d(std::span<ext_point_t const> sorted_x,
                                   std::span<ext_point_t const> sorted_y,
                                   std::span<ext_point_t const> sorted_z,
                                   uint8_t split_level) {
        SPDLOG_DEBUG("{:{}}[recursion] num points: {}, level: {}", "", 2 * split_level,
                     sorted_x.size(), split_level);

        if (sorted_x.size() <= brute_force_threshold) {
          bruteforce_distances(sorted_x);
        } else {
          // Else, split points at median X value.
          uint32_t const median_x = sorted_x[sorted_x.size() / 2].x;
          auto const split_it = std::ranges::find_if(
              sorted_x.subspan(sorted_x.size() / 2), [&](auto const x) { return x > median_x; },
              &ext_point_t::x);
          size_t const split_size_lhs = std::ranges::distance(sorted_x.begin(), split_it);

          auto const x_split_lhs = sorted_x.subspan(0, split_size_lhs);
          auto const x_split_rhs = sorted_x.subspan(split_size_lhs);
          auto const & y_split =
              split_list(sorted_y, median_x, &ext_point_t::x, y_lists_, split_level);
          auto const & z_split =
              split_list(sorted_z, median_x, &ext_point_t::x, z_lists_, split_level);

          find_pairs_recursive_3d(x_split_lhs, y_split.lhs, z_split.lhs, split_level + 1);
          find_pairs_recursive_3d(x_split_rhs, y_split.rhs, z_split.rhs, split_level + 1);

          // Find points within slab of width "max accepted distance" from the median X value.
          // TODO: This always matches all points. Is there any optimization here? What if we just
          // don't do it?
          find_center_points(sorted_y, slab_points_y_, &ext_point_t::x, median_x, split_level + 1);
          find_center_points(sorted_z, slab_points_z_, &ext_point_t::x, median_x, split_level + 1);

          // Find all nearest pairs inside the slab.
          // NOTE: No need to increase split level, since y_split and z_split aren't used anymore.
          find_pairs_recursive_2d(slab_points_y_, slab_points_z_, median_x, split_level);
        }
      }

      template <class Accessor>
      void find_center_points(std::span<ext_point_t const> src,
                              std::vector<ext_point_t> & dest,
                              Accessor accessor,
                              uint32_t midpoint,
                              [[maybe_unused]] uint8_t split_level) {
        assert(src.data() != dest.data());
        dest.clear();

        [[maybe_unused]] uint64_t max_d = 0;

        for (auto const & point : src) {
          auto const d =
              static_cast<int64_t>(std::invoke(accessor, point)) - static_cast<int64_t>(midpoint);
          if (static_cast<uint64_t>(d * d) < max_accepted_dist_) {
            dest.push_back(point);
          }

          max_d = std::max(max_d, static_cast<uint64_t>(d * d));
        }

        SPDLOG_DEBUG("{:{}}[center points] {} / {} points within distance {}, max d: {}, level: {}",
                     "", 2 * split_level, dest.size(), src.size(),
                     (max_accepted_dist_ == std::numeric_limits<uint64_t>::max()
                          ? "∞"
                          : fmt::format("{}", max_accepted_dist_)),
                     max_d, split_level);
      }
    };

  }  // namespace

  uint64_t day_t<8>::solve(part_t<1>, version_t<0>, simd_string_view_t input) {
    std::vector<point_t> const points = parse_input(input);

    // Do a very dumb thing and calculate all pairwise distances.
    std::vector<distance_t> distances = calc_pairwise_distances(points);

    // NOTE: Sketchy code, because the parameters for the example differ from the "real" input.
    size_t const connections_to_make = (distances.size() < 1000) ? 10 : 1000;

    // Put the first 1000 distances at the beginning. No need to sort them.
    std::nth_element(distances.begin(), distances.begin() + connections_to_make, distances.end());

    // Connect the 1000 closest pairs using a disjoint set.
    auto circuit_sets = disjoint_set_t(points.size());

    for (size_t i = 0; i < connections_to_make; ++i) {
      auto const & distance = distances.at(i);
      [[maybe_unused]] auto const new_size = circuit_sets.merge(distance.idx_a, distance.idx_b);
      SPDLOG_DEBUG("Connected {} & {}, group size: {}, distance: {}", distance.idx_a,
                   distance.idx_b, new_size, distance);
    }

    // No need to sort, just find the largest 3 circuits.
    auto circuit_sizes = std::move(circuit_sets).sizes();
    std::nth_element(circuit_sizes.begin(), circuit_sizes.begin() + 3, circuit_sizes.end(),
                     std::greater{});
    auto const largest_circuits = std::span(circuit_sizes).first(3);

    SPDLOG_DEBUG("Largest circuits: {}", largest_circuits);
    return std::accumulate(largest_circuits.begin(), largest_circuits.end(), 1ULL,
                           std::multiplies<uint64_t>());
  }

  uint64_t day_t<8>::solve(part_t<1>, version_t<1>, simd_string_view_t input) {
    std::vector<point_t> const points = parse_input(input);

    // NOTE: Sketchy code, because the parameters for the example differ from the "real" input.
    size_t const connections_to_make = (points.size() < 1000) ? 10 : 1000;

    // Calculate all pairwise distances, and keep track of the smallest N ones.
    std::vector<distance_t> distances;
    distances.reserve(connections_to_make);

    uint16_t num_pushed = 0;
    uint16_t const num_points = points.size();  // Because this doesn't seem to get cached.

    for (uint16_t idx_a = 0; idx_a < num_points - 1; ++idx_a) {
      for (uint16_t idx_b = idx_a + 1; idx_b < num_points; ++idx_b) {
        auto const distance = dist(idx_a, idx_b, points[idx_a], points[idx_b]);

        if (num_pushed < connections_to_make) {
          distances.push_back(distance);
          if (++num_pushed == connections_to_make) {
            std::make_heap(distances.begin(), distances.end());
          }
        } else if (distance < distances.front()) {
          // Note: Using a combined pop + push (i.e. replace) doesn't really speed things up.
          std::pop_heap(distances.begin(), distances.end());
          distances.back() = distance;
          std::push_heap(distances.begin(), distances.end());
        }
      }
    }

    // Connect the nearest N pairs.
    auto circuit_sets = disjoint_set_t(points.size());

    for (auto const & distance : distances) {
      [[maybe_unused]] auto const new_size = circuit_sets.merge(distance.idx_a, distance.idx_b);
      SPDLOG_DEBUG("Connected {:3d} & {:3d}, group size: {}, distance: {}", distance.idx_a,
                   distance.idx_b, new_size, distance);
    }

    // No need to sort, just find the largest 3 circuits.
    auto circuit_sizes = std::move(circuit_sets).sizes();
    std::ranges::nth_element(circuit_sizes, circuit_sizes.begin() + 3, std::greater{});
    auto const largest_circuits = std::span(circuit_sizes).first(3);

    SPDLOG_DEBUG("Largest circuits: {}", largest_circuits);
    return std::accumulate(largest_circuits.begin(), largest_circuits.end(), 1ULL,
                           std::multiplies<uint64_t>());
  }

  uint64_t day_t<8>::solve(part_t<1>, version_t<2>, simd_string_view_t input) {
    std::vector<point_t> const points = parse_input(input);

    // First create a list sorted per coordinate.
    // TODO: Try speed up by storing sorted indices instead.
    auto const create_sorted_copy = [&](auto sort_fn) {
      auto result = std::views::iota(size_t{0}, points.size()) |
                    std::views::transform([&points](size_t idx) -> ext_point_t {
                      auto const & point = points.at(idx);
                      ext_point_t ext;
                      ext.x = point.x;
                      ext.y = point.y;
                      ext.z = point.z;
                      ext.idx = static_cast<uint16_t>(idx);
                      return ext;
                    }) |
                    std::ranges::to<std::vector>();
      std::ranges::sort(result, sort_fn);
      return result;
    };

    auto const sorted_x = create_sorted_copy([](auto const & lhs, auto const & rhs) {
      return std::tie(lhs.x, lhs.y, lhs.z) < std::tie(rhs.x, rhs.y, rhs.z);
    });
    auto const sorted_y = create_sorted_copy([](auto const & lhs, auto const & rhs) {
      return std::tie(lhs.y, lhs.z, lhs.x) < std::tie(rhs.y, rhs.z, rhs.x);
    });
    auto const sorted_z = create_sorted_copy([](auto const & lhs, auto const & rhs) {
      return std::tie(lhs.z, lhs.x, lhs.y) < std::tie(rhs.z, rhs.x, rhs.y);
    });

    // NOTE: Sketchy code, because the parameters for the example differ from the "real" input.
    size_t const num_pairs = (points.size() < 1000) ? 10 : 1000;

    // Now apply a divide and conquer approach to find the K closest pairs.
    auto pair_finder = closest_pair_finder_t(sorted_x, sorted_y, sorted_z);
    auto const closest_pairs = pair_finder.find_pairs(num_pairs);

    // Connect the nearest N pairs.
    auto circuit_sets = disjoint_set_t(points.size());

    for (auto const & [idx_a, idx_b] : closest_pairs) {
      [[maybe_unused]] auto const new_size = circuit_sets.merge(idx_a, idx_b);
      SPDLOG_DEBUG("Connected {:3d} & {:3d}, group size: {}, distance: {}", idx_a, idx_b, new_size,
                   dist(idx_a, idx_b, points[idx_a], points[idx_b]));
    }

    // Extract the largest 3 circuits.
    auto circuit_sizes = std::move(circuit_sets).sizes();
    std::ranges::nth_element(circuit_sizes, circuit_sizes.begin() + 3, std::greater{});
    auto const largest_circuits = std::span(circuit_sizes).first(3);

    SPDLOG_DEBUG("Largest circuits: {}", largest_circuits);
    return std::accumulate(largest_circuits.begin(), largest_circuits.end(), 1ULL,
                           std::multiplies<uint64_t>());
  }

  uint64_t day_t<8>::solve(part_t<1>, version_t<3>, simd_string_view_t input) {
    std::vector<point_t> points = parse_input(input);

    // NOTE: Sketchy code, because the parameters for the example differ from the "real" input.
    size_t const connections_to_make = (points.size() < 1000) ? 10 : 1000;

    // Sort points first, so we can skip some distance calculations.
    std::ranges::sort(points, std::less<>{});

    // Calculate all pairwise distances, and keep track of the smallest N ones.
    std::vector<distance_t> distances;
    distances.reserve(connections_to_make);

    auto max_accepted_dist = std::numeric_limits<uint64_t>::max();
    uint16_t const num_points = points.size();  // Because this doesn't seem to get cached.
    [[maybe_unused]] uint32_t num_calc_skipped = 0;

    for (uint16_t idx_a = 0; idx_a < num_points - 1; ++idx_a) {
      for (uint16_t idx_b = idx_a + 1; idx_b < num_points; ++idx_b) {
        if (auto const dx =
                static_cast<int64_t>(points[idx_a].x) - static_cast<int64_t>(points[idx_b].x);
            static_cast<uint64_t>(dx * dx) >= max_accepted_dist) {
          num_calc_skipped += num_points - idx_b;
          break;  // Since points are sorted, we can't obtain any closer distances.
        }

        auto const distance = dist(idx_a, idx_b, points[idx_a], points[idx_b]);
        auto const num_pushed = distances.size();

        if (num_pushed < connections_to_make) {
          distances.push_back(distance);
          if (num_pushed + 1 == connections_to_make) {
            std::make_heap(distances.begin(), distances.end());
            max_accepted_dist = distances.front().distance;
          }
        } else if (distance.distance < max_accepted_dist) {
          // Note: Using a combined pop + push (i.e. replace) doesn't really speed things up.
          std::pop_heap(distances.begin(), distances.end());
          distances.back() = distance;
          std::push_heap(distances.begin(), distances.end());
          max_accepted_dist = distances.front().distance;
        }
      }
    }

    SPDLOG_DEBUG("Skipped {} / {} distance calculations", num_calc_skipped,
                 num_points * num_points / 2);

    // Connect the nearest N pairs.
    auto circuit_sets = disjoint_set_t(points.size());

    for (auto const & distance : distances) {
      [[maybe_unused]] auto const new_size = circuit_sets.merge(distance.idx_a, distance.idx_b);
      SPDLOG_DEBUG("Connected {:3d} & {:3d}, group size: {}, distance: {}", distance.idx_a,
                   distance.idx_b, new_size, distance);
    }

    // No need to sort, just find the largest 3 circuits.
    auto circuit_sizes = std::move(circuit_sets).sizes();
    std::ranges::nth_element(circuit_sizes, circuit_sizes.begin() + 3, std::greater{});
    auto const largest_circuits = std::span(circuit_sizes).first(3);

    SPDLOG_DEBUG("Largest circuits: {}", largest_circuits);
    return std::accumulate(largest_circuits.begin(), largest_circuits.end(), 1ULL,
                           std::multiplies<uint64_t>());
  }

  uint64_t day_t<8>::solve(part_t<2>, version_t<0>, simd_string_view_t input) {
    std::vector<point_t> const points = parse_input(input);

    // Do a very dumb thing and calculate all pairwise distances.
    std::vector<distance_t> distances = calc_pairwise_distances(points);

    // Make it a heap, so we can pop off the smallest distances.
    std::ranges::make_heap(distances, std::greater{});

    // Connect until everything is connected.
    auto circuit_sets = disjoint_set_t(points.size());

    size_t num_connected = 0;
    auto distances_end = distances.end();

    while (num_connected != points.size()) {
      std::pop_heap(distances.begin(), distances_end--, std::greater{});
      auto const distance = *distances_end;

      num_connected = circuit_sets.merge(distance.idx_a, distance.idx_b).value_or(num_connected);
      SPDLOG_DEBUG("Connection # {}: {} & {}, group size: {}, distance: {}",
                   std::distance(distances_end, distances.end()), distance.idx_a, distance.idx_b,
                   num_connected, distance);
    }

    // Find X coordinates of the last two connected points.
    auto const & last_distance = *distances_end;
    auto const & point_a = points.at(last_distance.idx_a);
    auto const & point_b = points.at(last_distance.idx_b);
    SPDLOG_DEBUG("Last connected points: {} & {}", point_a, point_b);

    return point_a.x * point_b.x;
  }

  uint64_t day_t<8>::solve(part_t<2>, version_t<1>, simd_string_view_t input) {
    std::vector<point_t> const points = parse_input(input);

    // Calculate the 3D Delaunay triangulation of the points. The edges of this triangulation are
    // guaranteed to contain the minimum spanning tree. Since this is a tricky afair, so we use
    // the CGAL library.
    using cgal_kernel_t = CGAL::Exact_predicates_inexact_constructions_kernel;
    using cgal_vertex_t = CGAL::Triangulation_vertex_base_with_info_3<uint16_t, cgal_kernel_t>;
    using cgal_data_t = CGAL::Triangulation_data_structure_3<cgal_vertex_t>;
    using cgal_triangulation_t = CGAL::Delaunay_triangulation_3<cgal_kernel_t, cgal_data_t>;
    using cgal_point_t = cgal_triangulation_t::Point;
    using cgal_edge_t = cgal_triangulation_t::Edge;

    auto const cgal_points = points | std::views::transform([&](point_t const & p) {
                               uint16_t const idx = &p - points.data();
                               return std::make_pair(cgal_point_t(p.x, p.y, p.z), idx);
                             });

    auto const delaunay = cgal_triangulation_t(cgal_points.begin(), cgal_points.end());
    SPDLOG_DEBUG("Delaunay triangulation has {} edges", delaunay.number_of_edges());
    assert(delaunay.is_valid());

    // Extract edges from the triangulation and sort them by distance.
    auto delaunay_edges = delaunay.finite_edges() |
                          std::views::transform([&](cgal_edge_t const & edge) -> distance_t {
                            int const idx_a = edge.first->vertex(edge.second)->info();
                            int const idx_b = edge.first->vertex(edge.third)->info();
                            auto const & point_a = points.at(idx_a);
                            auto const & point_b = points.at(idx_b);
                            return dist(idx_a, idx_b, point_a, point_b);
                          }) |
                          std::ranges::to<std::vector>();
    std::ranges::sort(delaunay_edges, std::less{});
    SPDLOG_DEBUG("Extracted {} edges from Delaunay triangulation", delaunay_edges.size());

    // Connect edges from the Delaunay triangulation until we have a minimum spanning tree.
    auto circuit_sets = disjoint_set_t(points.size());
    [[maybe_unused]] size_t num_edges_processed = 0;

    for (auto const & dist : delaunay_edges) {
      num_edges_processed++;
      auto const tree_size = circuit_sets.merge(dist.idx_a, dist.idx_b);

      if (tree_size.has_value()) {
        SPDLOG_DEBUG("Connected {} & {}, group size: {}, distance: {}", dist.idx_a, dist.idx_b,
                     *tree_size, dist.distance);
      }

      if (tree_size == points.size()) {  // This connection completed the spanning tree.
        auto const & point_a = points.at(dist.idx_a);
        auto const & point_b = points.at(dist.idx_b);

        SPDLOG_DEBUG("Processed {} edges, last connected points: {} & {}", num_edges_processed,
                     point_a, point_b);
        return point_a.x * point_b.x;
      }
    }

    std::unreachable();
  }

}  // namespace aoc25

#endif  // HWY_ONCE
