#include "aoc25/day_08.hpp"

#include "aoc25/day.hpp"
#include "aoc25/simd.hpp"
#include "aoc25/string.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>

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
      uint64_t const dx = static_cast<int64_t>(lhs.x) - static_cast<int64_t>(rhs.x);
      uint64_t const dy = static_cast<int64_t>(lhs.y) - static_cast<int64_t>(rhs.y);
      uint64_t const dz = static_cast<int64_t>(lhs.z) - static_cast<int64_t>(rhs.z);

      return distance_t{
          .distance = dx * dx + dy * dy + dz * dz,
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
       * @return The size of the merged set.
       */
      uint16_t merge(uint16_t a, uint16_t b) {
        uint16_t const root_a = find(a);
        uint16_t const root_b = find(b);

        if (root_a == root_b) {
          return size_.at(root_a);
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

    // Do a very dumb thing to find the largest circuits.
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

    // TODO: Create an octree and then calculate distances for each point within 1 block away.
    // Then keep expanding the number of blocks to search until we have enough distances. Well,
    // maybe not an octree, because we want fixed size blocks. We can instead do something similar
    // to the Rabin nearest pair algorithm: sample some random pairs, use shortest distance between
    // them to define block size, then partition all points into blocks of that size. Then for we
    // can detect for any point in a certain block which its neighboring blocks are, and check if
    // they exist, and if so, calculate distances to all points in those blocks. And then repeat.
    // Note that we probably have to take into account that points in the very corners of some
    // blocks may be "far away", so make sure we include all necessary blocks to ensure there's no
    // shorter distances missed. I.e. we'll have to calculate more distances than we can use, but
    // then use those "potentially too far" distances in the next iteration.

    // Calculate all pairwise distances, and keep track of the smallest N ones.
    std::vector<distance_t> distances;
    distances.reserve(connections_to_make);

    uint16_t num_pushed = 0;
    uint16_t num_points = points.size();  // Because for some reason this call result is not cached.

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
      SPDLOG_DEBUG("Connected {} & {}, group size: {}, distance: {}", distance.idx_a,
                   distance.idx_b, new_size, distance);
    }

    // Do a very dumb thing to find the largest circuits.
    auto circuit_sizes = std::move(circuit_sets).sizes();
    std::ranges::nth_element(circuit_sizes, circuit_sizes.begin() + 3, std::greater{});
    auto const largest_circuits = std::span(circuit_sizes).first(3);

    SPDLOG_DEBUG("Largest circuits: {}", largest_circuits);
    return std::accumulate(largest_circuits.begin(), largest_circuits.end(), 1ULL,
                           std::multiplies<uint64_t>());
  }

  uint64_t day_t<8>::solve(part_t<2>, version_t<0>, simd_string_view_t input) {
    std::vector<point_t> const points = parse_input(input);

    // TODO: Same as in part 1, but dynamic? Not clear how that could work...
    // Instead use an oct-tree and keep track of best distances so far? E.g. partition everything
    // into an oct-tree, then for each point calculate distances to all neighbors within N blocks.
    // Sort those distances and connect all of them. If not yet fully connected, expand distance
    // calculation to next "layer" of blocks, and repeat until fully connected.

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

      num_connected = circuit_sets.merge(distance.idx_a, distance.idx_b);
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

}  // namespace aoc25

#endif  // HWY_ONCE
