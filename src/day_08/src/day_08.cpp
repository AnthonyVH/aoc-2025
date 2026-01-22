#include "aoc25/day_08.hpp"

#include "aoc25/day.hpp"
#include "aoc25/n_ary_tree.hpp"
#include "aoc25/simd.hpp"
#include "aoc25/string.hpp"

#include <fmt/std.h>
#include <hwy/base.h>
#include <spdlog/spdlog.h>

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <ranges>
#include <utility>

namespace aoc25 {
  namespace {

    /* Note: We use float instead of uint32_t here, because the result of a distance calculation
     * requires more than 32 bits. However, the result fits in a float, and the loss of precision
     * does not influence the result. Of course given different input this might fail...
     *
     * Also note that the squared distance does not fit into a float, i.e. we would need a double.
     * Which stresses the cache much more and is slower than the just compare the square roots, but
     * only store half as many bytes.
     */
    struct point_t {
      float x;
      float y;
      float z;

      [[maybe_unused]] friend auto operator<(point_t const & lhs, point_t const & rhs) {
        return std::tie(lhs.x, lhs.y, lhs.z) < std::tie(rhs.x, rhs.y, rhs.z);
      }
    };

    [[maybe_unused]] std::string format_as(point_t const & obj) {
      return fmt::format("<x: {}, y: {}, z: {}>", obj.x, obj.y, obj.z);
    }

    /* Structure of arrays to allow SIMD calculations on the points. */
    struct points_t {
      points_t()
          : points_t(0) {}

      explicit points_t(size_t reserve_size) {
        x_.reserve(reserve_size);
        y_.reserve(reserve_size);
        z_.reserve(reserve_size);
      }

      size_t size() const { return x_.size(); }
      bool empty() const { return x_.empty(); }

      void push_back(float x, float y, float z) {
        x_.push_back(x);
        y_.push_back(y);
        z_.push_back(z);
      }

      float & x(size_t index) {
        assert(index < x_.size());
        return x_[index];
      }

      float x(size_t index) const {
        assert(index < x_.size());
        return x_[index];
      }

      float & y(size_t index) {
        assert(index < y_.size());
        return y_[index];
      }
      float y(size_t index) const {
        assert(index < y_.size());
        return y_[index];
      }
      float & z(size_t index) {
        assert(index < z_.size());
        return z_[index];
      }
      float z(size_t index) const {
        assert(index < z_.size());
        return z_[index];
      }

      simd_span_t<float const> x() const { return x_; }
      simd_span_t<float const> y() const { return y_; }
      simd_span_t<float const> z() const { return z_; }

      void erase(size_t index) {
        assert(index < x_.size());

        using std::swap;
        swap(x_.back(), x_[index]);
        swap(y_.back(), y_[index]);
        swap(z_.back(), z_[index]);

        x_.pop_back();
        y_.pop_back();
        z_.pop_back();
      }

     private:
      simd_vector_t<float> x_;
      simd_vector_t<float> y_;
      simd_vector_t<float> z_;
    };

    [[maybe_unused]] std::string format_as(points_t const & obj) {
      return fmt::format(
          "{::s}",
          std::views::zip(obj.x(), obj.y(), obj.z()) | std::views::transform([](auto const & e) {
            return fmt::format("({}, {}, {})", std::get<0>(e), std::get<1>(e), std::get<2>(e));
          }));
    }

  }  // namespace
}  // namespace aoc25

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "src/day_08.cpp"

// clang-format off
#include <hwy/foreach_target.h>
// clang-format on

#include <hwy/cache_control.h>
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();

namespace aoc25 {
  namespace {
    namespace HWY_NAMESPACE {

      namespace hn = hwy::HWY_NAMESPACE;

      void calculate_inverse_distances(point_t const & first_point,
                                       simd_span_t<float const> points_x,
                                       simd_span_t<float const> points_y,
                                       simd_span_t<float const> points_z,
                                       simd_span_t<float> distances_out) {
        static constexpr auto tag = hn::ScalableTag<float>{};
        static constexpr size_t lanes = hn::Lanes(tag);

        assert(points_x.size() == points_y.size());
        assert(points_x.size() == points_z.size());
        assert(points_x.size() == distances_out.size());

        auto const x_first = hn::Set(tag, first_point.x);
        auto const y_first = hn::Set(tag, first_point.y);
        auto const z_first = hn::Set(tag, first_point.z);

        auto const * HWY_RESTRICT src_x = points_x.data();
        auto const * HWY_RESTRICT src_y = points_y.data();
        auto const * HWY_RESTRICT src_z = points_z.data();

        size_t const num_points = points_x.size();
        size_t pos = 0;

        auto const calculate_inv_dist = [&](auto const & x_chunk, auto const & y_chunk,
                                            auto const & z_chunk) {
          auto const dx = hn::Sub(x_chunk, x_first);
          auto const dy = hn::Sub(y_chunk, y_first);
          auto const dz = hn::Sub(z_chunk, z_first);

          auto sum = hn::Mul(dx, dx);
          sum = hn::MulAdd(dy, dy, sum);
          sum = hn::MulAdd(dz, dz, sum);

          // This is faster than calculating the actual root, and only differs a tiny bit, which
          // doesn't seem to affect finding the proper solution to the problems.
          return hn::ApproximateReciprocalSqrt(sum);
        };

        for (; pos + lanes <= num_points; pos += lanes) {
          auto const x_chunk = hn::LoadU(tag, src_x + pos);
          auto const y_chunk = hn::LoadU(tag, src_y + pos);
          auto const z_chunk = hn::LoadU(tag, src_z + pos);

          auto const sum = calculate_inv_dist(x_chunk, y_chunk, z_chunk);
          hn::StoreU(sum, tag, distances_out.data() + pos);
        }

        if (pos < num_points) {  // Handle last partial chunk.
          size_t const lanes_remaining = num_points - pos;
          assert(pos + lanes_remaining == num_points);

          auto const x_chunk = hn::LoadN(tag, src_x + pos, lanes_remaining);
          auto const y_chunk = hn::LoadN(tag, src_y + pos, lanes_remaining);
          auto const z_chunk = hn::LoadN(tag, src_z + pos, lanes_remaining);

          auto const sum = calculate_inv_dist(x_chunk, y_chunk, z_chunk);
          hn::StoreN(sum, tag, distances_out.data() + pos, lanes_remaining);
        }
      }

      size_t calculate_limited_distances(point_t const & first_point,
                                         simd_span_t<float const> points_x,
                                         simd_span_t<float const> points_y,
                                         simd_span_t<float const> points_z,
                                         simd_span_t<float> distances_out,
                                         uint16_t & idx_b_flags_out,
                                         float max_distance) {
        using flags_t = std::remove_cvref_t<decltype(idx_b_flags_out)>;

        static constexpr auto tag = hn::ScalableTag<float>{};
        static constexpr size_t lanes = hn::Lanes(tag);
        [[maybe_unused]] static constexpr size_t bits_per_flag_word = sizeof(flags_t) * 8;

        assert(points_x.size() == points_y.size());
        assert(points_x.size() == points_z.size());
        assert(points_x.size() == distances_out.size());

        auto const x_first = hn::Set(tag, first_point.x);
        auto const y_first = hn::Set(tag, first_point.y);
        auto const z_first = hn::Set(tag, first_point.z);

        auto const max_dist = hn::Set(tag, max_distance);

        auto const * const HWY_RESTRICT src_x = points_x.data();
        auto const * const HWY_RESTRICT src_y = points_y.data();
        auto const * const HWY_RESTRICT src_z = points_z.data();

        auto * const HWY_RESTRICT dist_ptr = distances_out.data();

        size_t const num_points = points_x.size();
        assert(num_points <= bits_per_flag_word);  // All flags fit in a single word.

        size_t in_pos = 0;
        size_t out_pos = 0;
        flags_t idx_b_flags = 0;

        auto const calculate_dist = [&](auto const & x_chunk, auto const & y_chunk,
                                        auto const & z_chunk) {
          auto const dx = hn::Sub(x_chunk, x_first);
          auto const dy = hn::Sub(y_chunk, y_first);
          auto const dz = hn::Sub(z_chunk, z_first);

          auto sum = hn::Mul(dx, dx);
          sum = hn::MulAdd(dy, dy, sum);
          sum = hn::MulAdd(dz, dz, sum);

          // This is faster than calculating the actual root, and only differs a tiny bit, which
          // doesn't seem to affect finding the proper solution to the problems.
          return hn::ApproximateReciprocalSqrt(sum);
        };

        auto const update_flags = [&](auto const & mask) {
          flags_t const new_flags =
              static_cast<flags_t>(hn::BitsFromMask(tag, mask)) & ((1 << lanes) - 1);
          idx_b_flags |= new_flags << in_pos;
        };

        for (; in_pos + lanes <= num_points; in_pos += lanes) {
          auto const x_chunk = hn::LoadU(tag, src_x + in_pos);
          auto const y_chunk = hn::LoadU(tag, src_y + in_pos);
          auto const z_chunk = hn::LoadU(tag, src_z + in_pos);

          auto const sum = calculate_dist(x_chunk, y_chunk, z_chunk);
          auto const below_max = hn::Ge(sum, max_dist);  // Distance is actually 1 / dist.

          out_pos += hn::CompressStore(sum, below_max, tag, dist_ptr + out_pos);
          update_flags(below_max);
        }

        if (in_pos < num_points) {  // Handle last partial chunk.
          size_t const lanes_remaining = num_points - in_pos;
          assert(in_pos + lanes_remaining == num_points);
          auto const mask = hn::FirstN(tag, lanes_remaining);

          auto const x_chunk = hn::MaskedLoad(mask, tag, src_x + in_pos);
          auto const y_chunk = hn::MaskedLoad(mask, tag, src_y + in_pos);
          auto const z_chunk = hn::MaskedLoad(mask, tag, src_z + in_pos);

          auto const sum = calculate_dist(x_chunk, y_chunk, z_chunk);
          auto const below_max = hn::MaskedGe(mask, sum, max_dist);

          out_pos += hn::CompressStore(sum, below_max, tag, dist_ptr + out_pos);
          update_flags(below_max);
        }

        // All flags fit into a single word.
        idx_b_flags_out = idx_b_flags;

        return out_pos;
      }

      void update_distances(simd_span_t<float> prev_dist,
                            simd_span_t<uint32_t> prev_target_idx,
                            simd_span_t<float const> new_dist,
                            uint32_t new_target_idx) {
        static constexpr auto dist_tag = hn::ScalableTag<float>{};
        static constexpr size_t lanes = hn::Lanes(dist_tag);
        static constexpr auto idx_tag = hn::FixedTag<uint32_t, lanes>{};

        // Update all entries in prev_dist and prev_target_idx if the new distance is smaller.
        assert(prev_target_idx.size() == prev_dist.size());
        assert(prev_target_idx.size() == new_dist.size());

        size_t const num_points = new_dist.size();
        auto const target_idx = hn::Set(idx_tag, new_target_idx);

        // Ensure inputs are aligned.
        assert(reinterpret_cast<uintptr_t>(new_dist.data()) % lanes == 0);
        assert(reinterpret_cast<uintptr_t>(prev_dist.data()) % lanes == 0);

        float * const HWY_RESTRICT prev_ptr = prev_dist.data();
        float const * const HWY_RESTRICT new_ptr = new_dist.data();
        uint32_t * const HWY_RESTRICT prev_target_ptr = prev_target_idx.data();

        size_t pos = 0;

        // NOTE: Using > instead of <, because the values are 1 / distance. So a larger value is
        // actually a smaller distance.

        for (; pos + lanes <= num_points; pos += lanes) {
          // Note: Loading 2 chunks at a time, combining the masks, and then processing a
          // double-width chunk of indices is not faster.
          auto const new_chunk = hn::Load(dist_tag, new_ptr + pos);
          auto const prev_chunk = hn::Load(dist_tag, prev_ptr + pos);
          auto const use_new = hn::Gt(new_chunk, prev_chunk);

          hn::BlendedStore(new_chunk, use_new, dist_tag, prev_ptr + pos);
          hn::BlendedStore(target_idx, hn::RebindMask(idx_tag, use_new), idx_tag,
                           prev_target_ptr + pos);
        }

        if (pos < num_points) {  // Handle last partial chunk.
          size_t const lanes_remaining = num_points - pos;
          assert(pos + lanes_remaining == num_points);

          auto const new_chunk = hn::LoadN(dist_tag, new_ptr + pos, lanes_remaining);
          auto const prev_chunk = hn::LoadN(dist_tag, prev_ptr + pos, lanes_remaining);
          auto const use_new = hn::Gt(new_chunk, prev_chunk);

          hn::BlendedStore(new_chunk, use_new, dist_tag, prev_ptr + pos);
          hn::BlendedStore(target_idx, hn::RebindMask(idx_tag, use_new), idx_tag,
                           prev_target_ptr + pos);
        }
      }

    }  // namespace HWY_NAMESPACE
  }  // namespace
}  // namespace aoc25

HWY_AFTER_NAMESPACE();

#ifdef HWY_ONCE

namespace aoc25 {
  namespace {

    void calculate_inverse_distances(point_t const & first_point,
                                     simd_span_t<float const> points_x,
                                     simd_span_t<float const> points_y,
                                     simd_span_t<float const> points_z,
                                     simd_span_t<float> distances_out) {
      HWY_EXPORT(calculate_inverse_distances);
      return HWY_DYNAMIC_DISPATCH(calculate_inverse_distances)(first_point, points_x, points_y,
                                                               points_z, distances_out);
    }

    size_t calculate_limited_distances(point_t const & first_point,
                                       simd_span_t<float const> points_x,
                                       simd_span_t<float const> points_y,
                                       simd_span_t<float const> points_z,
                                       simd_span_t<float> distances_out,
                                       uint16_t & idx_b_flags_out,
                                       float max_distance) {
      HWY_EXPORT(calculate_limited_distances);
      return HWY_DYNAMIC_DISPATCH(calculate_limited_distances)(
          first_point, points_x, points_y, points_z, distances_out, idx_b_flags_out, max_distance);
    }

    void update_distances(simd_span_t<float> prev_distances,
                          simd_span_t<uint32_t> prev_target_idx,
                          simd_span_t<float const> new_distances,
                          uint32_t new_target_idx) {
      HWY_EXPORT(update_distances);
      return HWY_DYNAMIC_DISPATCH(update_distances)(prev_distances, prev_target_idx, new_distances,
                                                    new_target_idx);
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

        auto [small_idx, large_idx] = std::minmax(
            root_a, root_b, [&](uint16_t lhs, uint16_t rhs) { return size_[lhs] < size_[rhs]; });

        parent_[small_idx] = large_idx;
        size_[large_idx] += size_[small_idx];
        size_[small_idx] = 0;  // Clear, so we can find max values later.
        return size_[large_idx];
      }

      uint16_t size(uint16_t item) const { return size_.at(find(item)); }

      std::vector<uint16_t> extract_sizes() const && { return std::move(size_); }

     private:
      mutable std::vector<uint16_t> parent_;
      std::vector<uint16_t> size_;
    };

    points_t parse_input(simd_string_view_t input) {
      size_t const num_points = aoc25::count(input.as_span(), '\n');
      auto result = points_t(num_points);

      split(input, [&](simd_string_view_t line) {
        std::array<uint32_t, 3> coords;

        for (size_t coord_idx = 0; coord_idx < coords.size(); ++coord_idx) {
          size_t const comma_pos = line.find(',');
          simd_string_view_t const token = line.substr(0, comma_pos);
          convert_single_int<uint64_t>(token, [&](uint64_t value) { coords[coord_idx] = value; });

          if (coord_idx + 1 < coords.size()) {  // Don't update line after parsing last number.
            assert(comma_pos != line.npos);
            line.remove_prefix(token.size() + (comma_pos == line.npos ? 0 : 1));
          }
        }

        result.push_back(coords[0], coords[1], coords[2]);
      });

      SPDLOG_TRACE("Parsed {} points: {}", result.size(), result);
      return result;
    }

    /* Structure of arrays object to improve runtime. */
    struct distances_t {
      distances_t()
          : distances_t(0) {}

      explicit distances_t(size_t reserve_size) {
        idx_a_.reserve(reserve_size);
        idx_b_.reserve(reserve_size);
        distances_.reserve(reserve_size);
      }

      size_t size() const { return distances_.size(); }
      bool empty() const { return distances_.empty(); }

      uint32_t & idx_a(size_t index) {
        assert(index < idx_a_.size());
        return idx_a_[index];
      }

      uint32_t idx_a(size_t index) const {
        assert(index < idx_a_.size());
        return idx_a_[index];
      }

      uint32_t & idx_b(size_t index) {
        assert(index < idx_b_.size());
        return idx_b_[index];
      }

      uint32_t idx_b(size_t index) const {
        assert(index < idx_b_.size());
        return idx_b_[index];
      }

      float & distance(size_t index) {
        assert(index < distances_.size());
        return distances_[index];
      }

      float distance(size_t index) const {
        assert(index < distances_.size());
        return distances_[index];
      }

      void resize(size_t new_size) {
        idx_a_.resize(new_size);
        idx_b_.resize(new_size);
        distances_.resize(new_size);
      }

      simd_span_t<uint32_t> idx_a() { return idx_a_; }
      simd_span_t<uint32_t const> idx_a() const { return idx_a_; }

      simd_span_t<uint32_t> idx_b() { return idx_b_; }
      simd_span_t<uint32_t const> idx_b() const { return idx_b_; }

      simd_span_t<float> distances() { return distances_; }
      simd_span_t<float const> distances() const { return distances_; }

      void push_back(uint32_t idx_a, uint32_t idx_b, float distance) {
        idx_a_.push_back(idx_a);
        idx_b_.push_back(idx_b);
        distances_.push_back(distance);
      }

      void erase(size_t index) {
        assert(index < distances_.size());

        using std::swap;
        swap(idx_a_.back(), idx_a_[index]);
        swap(idx_b_.back(), idx_b_[index]);
        swap(distances_.back(), distances_[index]);

        idx_a_.pop_back();
        idx_b_.pop_back();
        distances_.pop_back();
      }

     private:
      // Indices are 32 bit, because there's no proper 16-bit SIMD instructions.
      // Distances are float, because there's no proper 64-bit integer SIMD multiplications.
      simd_vector_t<uint32_t> idx_a_;
      simd_vector_t<uint32_t> idx_b_;
      simd_vector_t<float> distances_;
    };

    [[maybe_unused]] std::string format_as(distances_t const & obj) {
      return fmt::format("{::s}", std::views::zip(obj.idx_a(), obj.idx_b(), obj.distances()) |
                                      std::views::transform([](auto const & e) {
                                        return fmt::format("<idx_a: {}, idx_b: {}, distance: {}>",
                                                           std::get<0>(e), std::get<1>(e),
                                                           std::get<2>(e));
                                      }));
    }

    struct connection_idx_t {
      uint16_t idx_a;
      uint16_t idx_b;
    };

    [[maybe_unused]] std::string format_as(connection_idx_t const & obj) {
      return fmt::format("[{} - {}]", obj.idx_a, obj.idx_b);
    }

    using tree_t = aoc25::sized_n_ary_min_tree_t<float, connection_idx_t, 4>;

    void push_elements_to_tree(tree_t & tree,
                               std::span<float const> distances,
                               std::span<uint16_t const> idxes_a,
                               std::span<uint16_t const> idx_b_flags,
                               uint16_t const b_offset) {
      assert(idxes_a.size() == idx_b_flags.size());
      uint16_t const num_idx_a = idxes_a.size();
      uint16_t dist_idx = 0;

      for (uint16_t idx_pos = 0; idx_pos < num_idx_a; ++idx_pos) {
        uint16_t const idx_a = idxes_a[idx_pos];
        uint16_t const idx_b_begin = idx_a + b_offset;
        uint16_t idx_b_flag = idx_b_flags[idx_pos];
        SPDLOG_TRACE("  idx_pos: {}, idx_a: {}, idx_b_flag: {:016b}", idx_pos, idx_a, idx_b_flag);

        while (idx_b_flag != 0) {
          uint16_t const b_step = std::countr_zero(idx_b_flag);
          idx_b_flag &= idx_b_flag - 1;  // Clear lowest set bit.

          // Distances are already pre-filtered by SIMD, so they are all better than whatever the
          // worst distance in the tree at the start of this function was.
          using element_t = typename tree_t::element_t;
          tree.push_or_replace(element_t{
              .key = distances[dist_idx++],
              .value =
                  connection_idx_t{
                      .idx_a = idx_a,
                      .idx_b = static_cast<uint16_t>(idx_b_begin + b_step),
                  },
          });
        }
      }
    }

    std::vector<connection_idx_t> calculate_n_shortest_distances(points_t const & points,
                                                                 size_t num_distances) {
      auto result = tree_t(num_distances);

      [[maybe_unused]] uint16_t const num_points = points.size();  // Ensure this is cached.

      // The main slowdown is calculating way too many distances. If the point cloud is more or
      // less uniformly distributed in space, then most distances will be large. Hence we only
      // need a few distances around each point. So instead of calculating all distances from a
      // point A, and then advancing to A + 1, we calculate a handful of distances from each A.
      // Each time this handful of distances is calculated, we update an the maximum accepted
      // distance. This way we can skip almost all distance calculations, assuming a uniform
      // distribution.
      static uint16_t constexpr b_step = 16;
      size_t const chunk_size = num_points * b_step;

      float min_accepted_recip_dist = 0;

      std::vector<uint16_t> idxes_a =
          std::views::iota(static_cast<uint16_t>(0u), static_cast<uint16_t>(num_points - 1)) |
          std::ranges::to<std::vector>();

      [[maybe_unused]] size_t num_calc_stored = 0;

      auto distance_chunk = simd_vector_t<float>(chunk_size);
      auto idx_b_flags = std::vector<uint16_t>(num_points);

      for (uint16_t b_offset = 1; !idxes_a.empty() && (b_offset < num_points); b_offset += b_step) {
        size_t dist_pos = 0;
        size_t idx_a_pos = 0;

        // Calculate the reciprocal of the reciprocal here, so we don't have to do a 1 / dist
        // calculation inside the loop.
        auto const max_accepted_dist = 1 / min_accepted_recip_dist;

        for (uint16_t const idx_a : idxes_a) {
          auto const point_a = point_t{
              .x = points.x(idx_a),
              .y = points.y(idx_a),
              .z = points.z(idx_a),
          };

          int16_t const idx_b_begin = idx_a + b_offset;
          int16_t const num_b = std::min<int16_t>(b_step, num_points - idx_b_begin);

          if (num_b <= 0) {
            // Indices are sorted, so no more distances to calculate in next iterations.
            SPDLOG_TRACE("No more distances to calculate from point {} onwards", idx_a);
            break;
          }

          auto const x_b = points.x().subspan(idx_b_begin, num_b);
          auto const y_b = points.y().subspan(idx_b_begin, num_b);
          auto const z_b = points.z().subspan(idx_b_begin, num_b);

          if (auto const diff_x = std::abs(point_a.x - x_b[0]); diff_x >= max_accepted_dist) {
            SPDLOG_TRACE("Skipping all distances from point {}", idx_a);
            continue;  // Since points are sorted, we can't obtain any closer distances.
          }

          SPDLOG_TRACE("Calculating distances from point {} to ({}, {}] (#: {})", , idx_a,
                       idx_b_begin, idx_b_begin + num_b, num_b);

          auto local_dist_chunk = simd_span_t{distance_chunk}.subspan(dist_pos, num_b);
          dist_pos += calculate_limited_distances(point_a, x_b, y_b, z_b, local_dist_chunk,
                                                  idx_b_flags[idx_a_pos], min_accepted_recip_dist);

          // Even if num_new_dist, current point might still generate viable distances later.
          idxes_a[idx_a_pos++] = idx_a;
        }

        idxes_a.resize(idx_a_pos);  // Keep only points that need further distance calculations.
        idx_b_flags.resize(idx_a_pos);
        SPDLOG_TRACE("# idxes_a: {}, idxes_a: {}", idxes_a.size(), idxes_a);

#  ifndef NDEBUG
        num_calc_stored += dist_pos;
        SPDLOG_DEBUG("b_offset: {}, {} distances from {} points", b_offset, dist_pos, idx_a_pos);
#  endif

        if (dist_pos > 0) {  // Merge distances from this chunk into result.
          // Note: It's faster just to try and push everything, than to do an nth_element on the
          // distances first.
          auto const new_distances = std::span<float const>{distance_chunk}.first(dist_pos);

          push_elements_to_tree(result, new_distances, idxes_a, idx_b_flags, b_offset);
          SPDLOG_TRACE("Tree keys after push: {}", std::span<float const>{result.keys()});

          assert(result.size() == num_distances);
          min_accepted_recip_dist = result.top().key;
          SPDLOG_DEBUG("  Updated min_accepted_recip_dist: {}", min_accepted_recip_dist);
        }
      }

#  ifndef NDEBUG
      [[maybe_unused]] size_t const num_max_calculations = num_points * (num_points - 1) / 2;
      SPDLOG_DEBUG("Stored {:.0f}% ({} / {}) distance calculations",
                   (100.0 * num_calc_stored) / num_max_calculations, num_calc_stored,
                   num_max_calculations);
#  endif

      return std::move(result).extract_values();
    }

  }  // namespace

  uint64_t day_t<8>::solve(part_t<1>, version_t<0>, simd_string_view_t input) {
    points_t const points =
        [&] {  // Sort points first, so we can skip some distance calculations later on.
          auto orig_points = parse_input(input);
          auto indices = std::views::iota(0u, static_cast<uint16_t>(orig_points.size())) |
                         std::ranges::to<std::vector>();

          std::ranges::sort(indices, [&](uint16_t lhs, uint16_t rhs) {
            return orig_points.x(lhs) < orig_points.x(rhs);
          });

          auto result = points_t(orig_points.size());
          for (auto const idx : indices) {
            result.push_back(orig_points.x(idx), orig_points.y(idx), orig_points.z(idx));
          }
          return result;
        }();

    // NOTE: Sketchy code, because the parameters for the example differ from the "real" input.
    size_t const connections_to_make = (points.size() < 1000) ? 10 : 1000;

    // Get the N shortest distances.
    auto const distances_idxes = calculate_n_shortest_distances(points, connections_to_make);

    // Connect the nearest N pairs.
    auto circuit_sets = disjoint_set_t(points.size());

    for (auto const & idxes : distances_idxes) {
      [[maybe_unused]] auto const new_size = circuit_sets.merge(idxes.idx_a, idxes.idx_b);
      SPDLOG_TRACE("Connected {:3d} & {:3d}, group size: {}", idxes.idx_a, idxes.idx_b, new_size);
    }

    // No need to sort, just find the largest 3 circuits.
    auto circuit_sizes = std::move(circuit_sets).extract_sizes();
    std::ranges::nth_element(circuit_sizes, circuit_sizes.begin() + 3, std::greater{});
    auto const largest_circuits = std::span(circuit_sizes).first(3);

    SPDLOG_DEBUG("Largest circuits: {}", largest_circuits);
    return std::accumulate(largest_circuits.begin(), largest_circuits.end(), 1ULL,
                           std::multiplies<uint64_t>());
  }

  uint64_t day_t<8>::solve(part_t<2>, version_t<0>, simd_string_view_t input) {
    points_t const points = parse_input(input);

    /* Calculating the 3D Delaunay triangulation is the mathematical "optimal" solution to solve
     * this, since it will give us approximately 6N edges which are guaranteed to contain the
     * minimum spanning tree. However, it's costly to compute, and extremely complex (i.e. you'd
     * use e.g. the CGAL library for it). A solution using it runs in approximately 4 ms.
     *
     * Instead of doing that, we will go for a much more brute-force approach, which works faster
     * here because there's only 1000 points.
     */

    /* Brute-force: for each point keep track of the shortest distance to any connected point.
     * Start with the last point connected, and keep connecting to the point with the shortest
     * distance to any of the connected points.
     *
     * Note: We use the last point, because it makes it easier to keep the remaining_points and
     * shortest_distances in sync (since an erase() call will swap the last entry with the erased
     * one, so if we erase the last point from remaining_points at the start, the other points
     * remain in the same order).
     */
    auto shortest_distances = distances_t(points.size() - 1);
    auto remaining_points = points;

    {
      uint32_t const first_point_idx = points.size() - 1;
      auto const first_point = point_t{
          .x = points.x(first_point_idx),
          .y = points.y(first_point_idx),
          .z = points.z(first_point_idx),
      };

      // Set indices.
      shortest_distances.resize(points.size() - 1);
      std::ranges::copy(std::views::iota(0u, static_cast<uint32_t>(points.size() - 1)),
                        shortest_distances.idx_a().begin());
      std::ranges::fill(shortest_distances.idx_b(), first_point_idx);

      // Calculate all distances from the first point to all others.
      simd_span_t<float const> const points_x = points.x().first(points.size() - 1);
      simd_span_t<float const> const points_y = points.y().first(points.size() - 1);
      simd_span_t<float const> const points_z = points.z().first(points.size() - 1);

      calculate_inverse_distances(first_point, points_x, points_y, points_z,
                                  shortest_distances.distances());

      // Remove the first point from remaining points.
      remaining_points.erase(first_point_idx);
    }

    // Scratch area for newly calculated distances.
    simd_vector_t<float> new_distances;

    for (uint16_t num_connections = 0; num_connections < points.size() - 1; ++num_connections) {
      // Find the shortest distance in the list (note that we find the maximum, since the values
      // stored are actualy 1 / distance, since that's cheaper to compute).
      auto const min_pos =
          aoc25::find_maximum_pos(simd_span_t<float const>{shortest_distances.distances()});

      SPDLOG_TRACE("Connection # {}: {} & {}, distance: {}", num_connections + 1,
                   shortest_distances.idx_a(min_pos), shortest_distances.idx_b(min_pos),
                   shortest_distances.distance(min_pos));

      if (shortest_distances.size() == 1) {
        // Find X coordinates of the last two connected points.
        auto const & x_a = points.x(shortest_distances.idx_a(min_pos));
        auto const & x_b = points.x(shortest_distances.idx_b(min_pos));
        SPDLOG_DEBUG("Last connected points at x: {} & {}", x_a, x_b);

        return static_cast<uint64_t>(x_a) * static_cast<uint64_t>(x_b);
      }

      auto const new_connected_idx = shortest_distances.idx_a(min_pos);

      // Ensure distances and points stay "in sync".
      shortest_distances.erase(min_pos);
      remaining_points.erase(min_pos);

      {  // Update the remaining distances with the newly connected point.
        // First calculate all distances from the new point to all others.
        auto const new_point = point_t{
            .x = points.x(new_connected_idx),
            .y = points.y(new_connected_idx),
            .z = points.z(new_connected_idx),
        };

        new_distances.resize(remaining_points.size());
        calculate_inverse_distances(new_point, remaining_points.x(), remaining_points.y(),
                                    remaining_points.z(), new_distances);

        // Now update the shortest distances.
        update_distances(shortest_distances.distances(), shortest_distances.idx_b(), new_distances,
                         new_connected_idx);
      }
    }

    std::unreachable();
  }

}  // namespace aoc25

#endif  // HWY_ONCE
