#include "aoc25/day_08.hpp"

#include "aoc25/day.hpp"
#include "aoc25/simd.hpp"
#include "aoc25/string.hpp"

#include <fmt/std.h>
#include <hwy/base.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <ranges>
#include <utility>

namespace aoc25 {
  namespace {

    /* Note: We use doubles instead of uint32_t here, because the result of a distance calculation
     * requires more than 32 bits. Hence we need at minimum a 64-bit result. However, there's no
     * proper 64-bit integer SIMD multiplication, so this must be done in floating point. And
     * converting from integer to floating point is expensive. Plus if we start with 32-bit values
     * we also have to expand the lane size, which means going from SSE to AVX2, which also costs a
     * lot of time. So instead we just pay the extra memory cost and store coordinates as double.
     */
    struct point_t {
      double x;
      double y;
      double z;

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

      void push_back(double x, double y, double z) {
        x_.push_back(x);
        y_.push_back(y);
        z_.push_back(z);
      }

      double & x(size_t index) {
        assert(index < x_.size());
        return x_[index];
      }

      double x(size_t index) const {
        assert(index < x_.size());
        return x_[index];
      }

      double & y(size_t index) {
        assert(index < y_.size());
        return y_[index];
      }
      double y(size_t index) const {
        assert(index < y_.size());
        return y_[index];
      }
      double & z(size_t index) {
        assert(index < z_.size());
        return z_[index];
      }
      double z(size_t index) const {
        assert(index < z_.size());
        return z_[index];
      }

      simd_span_t<double const> x() const { return x_; }
      simd_span_t<double const> y() const { return y_; }
      simd_span_t<double const> z() const { return z_; }

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
      simd_vector_t<double> x_;
      simd_vector_t<double> y_;
      simd_vector_t<double> z_;
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

      void calculate_distances(point_t const & first_point,
                               simd_span_t<double const> points_x,
                               simd_span_t<double const> points_y,
                               simd_span_t<double const> points_z,
                               simd_span_t<double> distances_out) {
        static constexpr auto tag = hn::ScalableTag<double>{};
        static constexpr size_t lanes = hn::Lanes(tag);

        assert(points_x.size() == points_y.size());
        assert(points_x.size() == points_z.size());
        assert(points_x.size() == distances_out.size());

        auto const x_first = hn::Set(tag, first_point.x);
        auto const y_first = hn::Set(tag, first_point.y);
        auto const z_first = hn::Set(tag, first_point.z);

        auto const * HWY_RESTRICT src_x = reinterpret_cast<double const *>(points_x.data());
        auto const * HWY_RESTRICT src_y = reinterpret_cast<double const *>(points_y.data());
        auto const * HWY_RESTRICT src_z = reinterpret_cast<double const *>(points_z.data());

        size_t const num_points = points_x.size();
        size_t pos = 0;

        auto const calculate_sum = [&](auto const & x_chunk, auto const & y_chunk,
                                       auto const & z_chunk) {
          auto const dx = hn::Sub(x_chunk, x_first);
          auto const dy = hn::Sub(y_chunk, y_first);
          auto const dz = hn::Sub(z_chunk, z_first);

          auto sum = hn::Mul(dx, dx);
          sum = hn::MulAdd(dy, dy, sum);
          sum = hn::MulAdd(dz, dz, sum);

          return sum;
        };

        for (; pos + lanes <= num_points; pos += lanes) {
          auto const x_chunk = hn::LoadU(tag, src_x + pos);
          auto const y_chunk = hn::LoadU(tag, src_y + pos);
          auto const z_chunk = hn::LoadU(tag, src_z + pos);

          auto const sum = calculate_sum(x_chunk, y_chunk, z_chunk);
          hn::Store(sum, tag, distances_out.data() + pos);
        }

        if (pos < num_points) {  // Handle last partial chunk.
          size_t const lanes_remaining = num_points - pos;
          assert(pos + lanes_remaining == num_points);

          auto const x_chunk = hn::LoadN(tag, src_x + pos, lanes_remaining);
          auto const y_chunk = hn::LoadN(tag, src_y + pos, lanes_remaining);
          auto const z_chunk = hn::LoadN(tag, src_z + pos, lanes_remaining);

          auto const sum = calculate_sum(x_chunk, y_chunk, z_chunk);
          hn::StoreN(sum, tag, distances_out.data() + pos, lanes_remaining);
        }
      }

      void update_distances(simd_span_t<double> prev_dist,
                            simd_span_t<uint32_t> prev_target_idx,
                            simd_span_t<double const> new_dist,
                            uint32_t new_target_idx) {
        static constexpr auto dist_tag = hn::ScalableTag<double>{};
        static constexpr size_t lanes = hn::Lanes(dist_tag);
        static constexpr auto idx_tag = hn::FixedTag<uint32_t, lanes>{};

        // Update all entries in prev_dist and prev_target_idx if the new distance is smaller.
        assert(prev_target_idx.size() == prev_dist.size());
        assert(prev_target_idx.size() == new_dist.size());

        auto const target_idx = hn::Set(idx_tag, new_target_idx);

        size_t pos = 0;

        // Ensure inputs are aligned.
        assert(reinterpret_cast<uintptr_t>(new_dist.data()) % lanes == 0);
        assert(reinterpret_cast<uintptr_t>(prev_dist.data()) % lanes == 0);

        for (; pos + lanes <= new_dist.size(); pos += lanes) {
          // Note: Loading 2 chunks at a time, combining the masks, and then processing a
          // double-width chunk of indices is not faster.
          auto const new_chunk = hn::Load(dist_tag, new_dist.data() + pos);
          auto const prev_chunk = hn::Load(dist_tag, prev_dist.data() + pos);
          auto const use_new = hn::Lt(new_chunk, prev_chunk);

          hn::BlendedStore(new_chunk, use_new, dist_tag, prev_dist.data() + pos);
          hn::BlendedStore(target_idx, hn::DemoteMaskTo(idx_tag, dist_tag, use_new), idx_tag,
                           prev_target_idx.data() + pos);
        }

        if (pos < new_dist.size()) {  // Handle last partial chunk.
          size_t const lanes_remaining = new_dist.size() - pos;
          assert(pos + lanes_remaining == new_dist.size());

          auto const new_chunk = hn::LoadN(dist_tag, new_dist.data() + pos, lanes_remaining);
          auto const prev_chunk = hn::LoadN(dist_tag, prev_dist.data() + pos, lanes_remaining);
          auto const use_new = hn::Lt(new_chunk, prev_chunk);

          hn::BlendedStore(new_chunk, use_new, dist_tag, prev_dist.data() + pos);

          hn::BlendedStore(target_idx, hn::DemoteMaskTo(idx_tag, dist_tag, use_new), idx_tag,
                           prev_target_idx.data() + pos);
        }
      }

      // Note: See call-site for an explanation why this single template argument is used.
      template <class SiftDownArgs>
      void sift_down_with_aux(SiftDownArgs args) {
        using key_t = std::remove_cvref_t<decltype(args.keys.front())>;
        using value_t = std::remove_cvref_t<decltype(args.values.front())>;

        static constexpr size_t branching_factor = SiftDownArgs::branching_factor;
        static constexpr auto tag = hn::FixedTag<key_t, branching_factor>{};

        static_assert(std::is_arithmetic_v<key_t>);
        static_assert((branching_factor == 2) || (branching_factor == 4),
                      "Comparison code requires branching factor of 2 or 4");

        size_t const size = args.keys.size();
        key_t * HWY_RESTRICT key_ptr = args.keys.data();
        value_t * HWY_RESTRICT value_ptr = args.values.data();
        size_t pos = args.pos;

        while (true) {
          size_t const first_child = branching_factor * pos + 1;
          size_t const last_child = first_child + branching_factor - 1;

          if (first_child >= size) {
            break;  // No children exist.
          }

          size_t best_child = 0xDEAD'BEEF;

          if (last_child < size) {  // Handle all children using SIMD.
            auto const child_keys = hn::LoadU(tag, key_ptr + first_child);

            // Prefetch children of first current child.
            hwy::Prefetch(key_ptr + branching_factor * first_child + 1);

            // Calculate max such that each lane contains the max key among all children.
            auto const max = hn::MaxOfLanes(tag, child_keys);

            if (hn::GetLane(max) <= key_ptr[pos]) {
              break;
            }

            // Find which child has the max key.
            auto const is_max = hn::Eq(child_keys, max);
            best_child = first_child + std::countr_zero(hn::BitsFromMask(tag, is_max));
          } else {  // Fall back to scalar code for the last incomplete group.
            best_child = first_child;

            for (size_t child = first_child + 1; child < size; ++child) {
              best_child = (key_ptr[child] > key_ptr[best_child]) ? child : best_child;
            }

            if (key_ptr[best_child] <= key_ptr[pos]) {
              break;
            }
          }

          // Keep key and value position synchronized.
          using std::swap;
          swap(key_ptr[pos], key_ptr[best_child]);
          swap(value_ptr[pos], value_ptr[best_child]);

          pos = best_child;
        }
      }

    }  // namespace HWY_NAMESPACE
  }  // namespace
}  // namespace aoc25

HWY_AFTER_NAMESPACE();

#ifdef HWY_ONCE

namespace aoc25 {

  namespace {

    void calculate_distances(point_t const & first_point,
                             simd_span_t<double const> points_x,
                             simd_span_t<double const> points_y,
                             simd_span_t<double const> points_z,
                             simd_span_t<double> distances_out) {
      HWY_EXPORT(calculate_distances);
      return HWY_DYNAMIC_DISPATCH(calculate_distances)(first_point, points_x, points_y, points_z,
                                                       distances_out);
    }

    void update_distances(simd_span_t<double> prev_distances,
                          simd_span_t<uint32_t> prev_target_idx,
                          simd_span_t<double const> new_distances,
                          uint32_t new_target_idx) {
      HWY_EXPORT(update_distances);
      return HWY_DYNAMIC_DISPATCH(update_distances)(prev_distances, prev_target_idx, new_distances,
                                                    new_target_idx);
    }

    /* Google Highway does not support creating dispatch tables to functions with more than one
     * template argument. Hence we cheat by bundling all template arguments into a single struct,
     * and unpacking it again inside the function that gets dispatched to.
     */
    template <size_t BranchingFactor, class Key, class Value>
    struct sift_down_with_aux_args_t {
      static constexpr size_t branching_factor = BranchingFactor;

      simd_span_t<Key> keys;
      std::span<Value> values;
      size_t pos;
    };

    template <size_t BranchingFactor, class Key, class Value>
    [[maybe_unused]] void sift_down_with_aux(simd_span_t<Key> keys,
                                             std::span<Value> values,
                                             size_t pos) {
      using args_t = sift_down_with_aux_args_t<BranchingFactor, Key, Value>;
      HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(sift_down_with_aux<args_t>)(args_t{keys, values, pos});
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

      double & distance(size_t index) {
        assert(index < distances_.size());
        return distances_[index];
      }

      double distance(size_t index) const {
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

      simd_span_t<double> distances() { return distances_; }
      simd_span_t<double const> distances() const { return distances_; }

      void push_back(uint32_t idx_a, uint32_t idx_b, double distance) {
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
      // Distances are double, because there's no proper 64-bit integer SIMD multiplications.
      simd_vector_t<uint32_t> idx_a_;
      simd_vector_t<uint32_t> idx_b_;
      simd_vector_t<double> distances_;
    };

    [[maybe_unused]] std::string format_as(distances_t const & obj) {
      return fmt::format("{::s}", std::views::zip(obj.idx_a(), obj.idx_b(), obj.distances()) |
                                      std::views::transform([](auto const & e) {
                                        return fmt::format("<idx_a: {}, idx_b: {}, distance: {}>",
                                                           std::get<0>(e), std::get<1>(e),
                                                           std::get<2>(e));
                                      }));
    }

    /** @brief An N-ary max tree with a maximum size.
     *
     * The tree is builds a max-tree of up the given number of elements. New elements can be pushed
     * using push_or_replace(), which either adds the new element (if the tree is not full), or
     * replaces the current maximum (if the tree is full and the new element is smaller than the
     * current maximum). This allows keeping track of the "best N" elements seen so far.
     */
    template <class Key, class Value, size_t BranchingFactor>
      requires std::is_arithmetic_v<Key>
    class sized_n_ary_max_tree_t {
     public:
      struct element_t {
        Key key;
        Value value;
      };

      struct element_ref_t {
        Key const & key;
        Value const & value;
      };

      explicit sized_n_ary_max_tree_t(size_t max_size)
          : keys_{}, values_{}, current_size_{0}, max_size_{max_size} {
        keys_.reserve(max_size_);
        values_.reserve(max_size_);
      }

      void push(element_t elem) {
        assert(current_size_ < max_size_);
        keys_.push_back(elem.key);
        values_.push_back(std::move(elem.value));
        sift_up(current_size_++);
      }

      /**
       * @brief Inserts a value or replaces the max:
       *   - If heap is not full, push the new value normally.
       *   - If heap is full AND (new_value < current max), replace root and sift down.
       *   - Otherwise, do nothing (new value is too large to enter the "best X" set).
       */
      void push_or_replace(element_t elem) {
        if (current_size_ < max_size_) {
          push(std::move(elem));
        } else if (is_higher_in_heap(keys_[0], elem.key)) {
          keys_[0] = elem.key;
          values_[0] = std::move(elem.value);
          sift_down(0);
        }
      }

      element_t pop() {
        assert(current_size_ != 0);
        auto result = element_t{.key = keys_[0], .value = std::move(values_[0])};

        if (current_size_-- >= 2) {
          keys_[0] = keys_[current_size_];
          values_[0] = std::move(values_[current_size_]);
          sift_down(0);
        }

        keys_.pop_back();
        values_.pop_back();

        return result;
      }

      element_ref_t top() const {
        assert(current_size_ > 0);
        return element_ref_t{.key = keys_[0], .value = values_[0]};
      }

      size_t size() const { return current_size_; }

      // Allow iterating over all the elements without popping them off.
      auto keys_begin() const { return keys_.begin(); }
      auto keys_end() const { return keys_.begin() + current_size_; }

      auto values_begin() const { return values_.begin(); }
      auto values_end() const { return values_.begin() + current_size_; }

      std::span<Key const> keys() const { return keys_; }
      std::span<Value const> values() const { return values_; }

      std::vector<Value> extract_values() && { return std::move(values_); }

     private:
      static constexpr size_t branching_factor = BranchingFactor;

      simd_vector_t<Key> keys_;
      std::vector<Value> values_;
      size_t current_size_;
      size_t max_size_;

      bool is_higher_in_heap(Key const & lhs, Key const & rhs) const {
        // By inverting the order of rhs and lhs, std::less creates a max-heap, which is the same
        // behavior as STL.
        return rhs < lhs;
      }

      void sift_up(size_t pos) {
        while (pos > 0) {
          size_t parent = (pos - 1) / branching_factor;
          if (!is_higher_in_heap(keys_[pos], keys_[parent])) {
            break;
          }

          using std::swap;
          swap(keys_[pos], keys_[parent]);
          swap(values_[pos], values_[parent]);
          pos = parent;
        }
      }

      void sift_down(size_t pos) {
        if constexpr ((branching_factor == 2) || (branching_factor == 4)) {
          sift_down_with_aux<branching_factor>(simd_span_t{keys_}, std::span{values_}, pos);
        } else {
          sift_down_scalar(pos);
        }
      }

      [[maybe_unused]] void sift_down_scalar(size_t pos) {
        while (true) {
          size_t first_child = branching_factor * pos + 1;
          if (first_child >= current_size_) {
            break;
          }

          // Find the best among up to branching_factor children.
          size_t best_child = first_child;
          size_t end_child = std::min(first_child + branching_factor, current_size_);

          for (size_t child = first_child + 1; child < end_child; ++child) {
            best_child = (is_higher_in_heap(keys_[child], keys_[best_child])) ? child : best_child;
          }

          if (!is_higher_in_heap(keys_[best_child], keys_[pos])) {
            break;
          }

          using std::swap;
          swap(keys_[pos], keys_[best_child]);
          swap(values_[pos], values_[best_child]);
          pos = best_child;
        }
      }
    };

    struct connection_idx_t {
      uint16_t idx_a;
      uint16_t idx_b;
    };

    std::vector<connection_idx_t> calculate_n_shortest_distances(points_t const & points,
                                                                 size_t num_distances) {
      // Calculate all pairwise distances, and keep track of the shortest N ones.
      auto result = sized_n_ary_max_tree_t<double, connection_idx_t, 4>(num_distances);
      using push_t = typename decltype(result)::element_t;

      // Prime with 1 element of maximum distance, so we can always check against largest distance.
      result.push(push_t{
          .key = std::numeric_limits<double>::infinity(),
          .value = connection_idx_t{0, 0},
      });
      auto & max_accepted_dist = result.top().key;

      uint16_t const num_points = points.size();  // Because this doesn't seem to get cached.
      [[maybe_unused]] uint32_t num_calc_skipped = 0;

      // Calculate a chunk of distances at a time.
      static constexpr size_t max_chunk_size = 32;  // Found by trying out various sizes.
      auto distance_chunk = simd_vector_t<double>(max_chunk_size);

      for (uint16_t idx_a = 0; idx_a < num_points - 1; ++idx_a) {
        auto const point_a = point_t{
            .x = points.x(idx_a),
            .y = points.y(idx_a),
            .z = points.z(idx_a),
        };

        uint16_t idx_b = idx_a + 1;

        auto const process_distance_chunk = [&](size_t chunk_size) -> bool {
          auto const x_b = points.x().subspan(idx_b, chunk_size);
          auto const y_b = points.y().subspan(idx_b, chunk_size);
          auto const z_b = points.z().subspan(idx_b, chunk_size);

          // If X-distance is already too large, skip the entire chunk.
          if (auto const dx = point_a.x - x_b[0]; dx * dx >= max_accepted_dist) {
            return false;  // Since points are sorted, we can't obtain any closer distances.
          }

          auto dist_out = simd_span_t{distance_chunk}.subspan(0, chunk_size);
          calculate_distances(point_a, x_b, y_b, z_b, dist_out);

          return true;
        };

        for (; idx_b < num_points; idx_b += max_chunk_size) {
          auto const chunk_size = std::min<size_t>(num_points - idx_b, max_chunk_size);
          bool const continue_processing = process_distance_chunk(chunk_size);

          if (!continue_processing) {
            num_calc_skipped += num_points - idx_b;
            break;
          }

          // Push distances to heap.
          for (size_t dist_idx = 0; dist_idx < chunk_size; ++dist_idx) {
            auto const distance = distance_chunk[dist_idx];
            result.push_or_replace(push_t{
                .key = distance,
                .value = connection_idx_t{idx_a, static_cast<uint16_t>(idx_b + dist_idx)},
            });
          }
        }
      }

      [[maybe_unused]] size_t const num_max_calculations = num_points * (num_points - 1) / 2;
      SPDLOG_DEBUG("Skipped {:.0f}% ({} / {}) distance calculations",
                   (100.0 * num_calc_skipped) / num_max_calculations, num_calc_skipped,
                   num_max_calculations);
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
      SPDLOG_DEBUG("Connected {:3d} & {:3d}, group size: {}", idxes.idx_a, idxes.idx_b, new_size);
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
      simd_span_t<double const> const points_x = points.x().subspan(0, points.size() - 1);
      simd_span_t<double const> const points_y = points.y().subspan(0, points.size() - 1);
      simd_span_t<double const> const points_z = points.z().subspan(0, points.size() - 1);

      calculate_distances(first_point, points_x, points_y, points_z,
                          shortest_distances.distances());

      // Remove the first point from remaining points.
      remaining_points.erase(first_point_idx);
    }

    // Scratch area for newly calculated distances.
    simd_vector_t<double> new_distances;

    for (uint16_t num_connections = 0; num_connections < points.size() - 1; ++num_connections) {
      // Find the shortest distance in the list.
      auto const min_pos =
          aoc25::find_minimum_pos(simd_span_t<double const>{shortest_distances.distances()});

      SPDLOG_DEBUG("Connection # {}: {} & {}, distance: {}", num_connections + 1,
                   shortest_distances.idx_a(min_pos), shortest_distances.idx_b(min_pos),
                   shortest_distances.distance(min_pos));

      if (shortest_distances.size() == 1) {
        // Find X coordinates of the last two connected points.
        auto const & x_a = points.x(shortest_distances.idx_a(min_pos));
        auto const & x_b = points.x(shortest_distances.idx_b(min_pos));
        SPDLOG_DEBUG("Last connected points at x: {} & {}", x_a, x_b);

        return x_a * x_b;
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
        calculate_distances(new_point, remaining_points.x(), remaining_points.y(),
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
