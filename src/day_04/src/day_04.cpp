#include "aoc25/day_04.hpp"

#include "aoc25/simd.hpp"

#include <fmt/base.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <fmt/std.h>
#include <spdlog/common.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <ranges>
#include <span>
#include <string_view>
#include <type_traits>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "src/day_04.cpp"

#ifdef HWY_ONCE

namespace aoc25 {
  namespace {

    static constexpr uint8_t max_neighbors_for_removable_roll = 4;

    struct grid_t {
      simd_vector_t<uint8_t> data;
      size_t dimension;

      simd_span_t<uint8_t> row(size_t row_idx) {
        return simd_span_t(data).subspan(row_idx * (dimension + 2), dimension + 2);
      }

      simd_span_t<uint8_t const> row(size_t row_idx) const {
        assert(row_idx < dimension + 2);
        return simd_span_t(data).subspan(row_idx * (dimension + 2), dimension + 2);
      }
    };

    [[maybe_unused]] std::string format_as(grid_t const & obj) {
      return fmt::format(
          "{}", fmt::join(std::views::iota(size_t{0}, obj.dimension) |
                              std::views::transform([&obj](size_t row_idx) {
                                return fmt::format(
                                    "{}",
                                    fmt::join(obj.row(row_idx + 1).subspan(1, obj.dimension), ""));
                              }),
                          "\n"));
    }

    /** @brief Grid which stores which neighbor contains a paper roll, using a bitset.
     * Note that it doesn't keep track of whether a cell itself contains a roll of paper.
     */
    struct bitset_grid_t {
     public:
      explicit bitset_grid_t(grid_t const & rolls)
          : neighbors_(calculate_neighbors_impl(rolls))
          , mask_bit_idx_to_offset_{std::to_array<int16_t>({
                +1,                                          // 0: right
                -1,                                          // 1: left
                static_cast<int16_t>(+rolls.dimension + 0),  // 2: below
                static_cast<int16_t>(+rolls.dimension + 1),  // 3: below right
                static_cast<int16_t>(+rolls.dimension - 1),  // 4: below left
                static_cast<int16_t>(-rolls.dimension + 0),  // 5: above
                static_cast<int16_t>(-rolls.dimension + 1),  // 6: above right
                static_cast<int16_t>(-rolls.dimension - 1),  // 7: above left
            })}
          , dimension_(rolls.dimension) {}

      uint16_t dimension() const { return dimension_; }

      uint8_t & operator[](uint16_t idx) { return neighbors_.at(idx); }
      uint8_t operator[](uint16_t idx) const { return neighbors_.at(idx); }

      uint8_t & operator[](uint8_t row, uint8_t col) { return operator[](row * dimension_ + col); }

      uint8_t operator[](uint8_t row, uint8_t col) const {
        return operator[](row * dimension_ + col);
      }

      simd_span_t<uint8_t> row(uint8_t row_idx) {
        assert(row_idx < dimension_);
        return simd_span_t(neighbors_).subspan(row_idx * dimension_, dimension_);
      }

      simd_span_t<uint8_t const> row(uint8_t row_idx) const {
        assert(row_idx < dimension_);
        return simd_span_t(neighbors_).subspan(row_idx * dimension_, dimension_);
      }

      inline int16_t mask_bit_idx_to_offset(uint8_t bit_idx) const {
        // Faster than using a big lookup table of std::spans to offsets for each possible bitset.
        assert(bit_idx < mask_bit_idx_to_offset_.size());
        return mask_bit_idx_to_offset_[bit_idx];
      }

      /** @brief Given an bit index from a mask, return the mask that identifies that neighbor.
       * I.e. if a cell has a neighbor indicated at bit index 6 (above right), the returned mask
       * should have the bit set for the position <below left>. This mask can then be used to clear
       * that bit in the cell's neighbor's bitset.
       */
      uint8_t mask_bit_idx_to_neighbor_mask(uint8_t bit_idx) const {
        static constexpr std::array<uint8_t, 8> const lut = std::to_array<uint8_t>({
            0b000'000'10,  // right: 0 -> left
            0b000'000'01,  // left: 1 -> right

            0b001'000'00,  // below: 2 -> above
            0b100'000'00,  // below right: 3 -> above left
            0b010'000'00,  // below left: 4  -> above right

            0b000'001'00,  // above: 5 -> below
            0b000'100'00,  // above right: 6 -> below left
            0b000'010'00,  // above left: 7 -> below right
        });

        assert(bit_idx < 8);
        return lut[bit_idx];
      }

     private:
      // Bitsets for each position in the grid. The format of the bitset, from MSB to LSB, is:
      // <above left> <above right> <above> <below left> <below right> <below> <left> <right>
      simd_vector_t<uint8_t> neighbors_;
      std::array<int16_t, 8> mask_bit_idx_to_offset_;
      int16_t dimension_;

      static simd_vector_t<uint8_t> calculate_neighbors_impl(grid_t const & rolls);
    };

    [[maybe_unused]] std::string format_as(bitset_grid_t const & obj) {
      return fmt::format("{}", fmt::join(std::views::iota(size_t{0}, obj.dimension()) |
                                             std::views::transform([&obj](size_t row_idx) {
                                               return fmt::format("{::08b}", obj.row(row_idx));
                                             }),
                                         "\n"));
    }

  }  // namespace
}  // namespace aoc25

#endif  // HWY_ONCE

// clang-format off
#include <hwy/foreach_target.h>
// clang-format on

#include <hwy/highway.h>
#include <hwy/print-inl.h>

HWY_BEFORE_NAMESPACE();

namespace aoc25 {
  namespace {
    namespace HWY_NAMESPACE {

      namespace hn = hwy::HWY_NAMESPACE;

      uint32_t count_removables(grid_t const & grid) {
        // For the code below, note that since storage is aligned, and once row i is processed, we
        // only ever touch rows > i. Hence it's fine to write into row i + 1's storage.
        static constexpr hn::ScalableTag<uint8_t> tag{};
        static constexpr size_t lanes = hn::Lanes(tag);

        auto row_sum_a = simd_vector_t<uint8_t>(grid.dimension);
        auto row_sum_b = simd_vector_t<uint8_t>(grid.dimension);
        auto row_sum_c = simd_vector_t<uint8_t>(grid.dimension);

        auto * prev_row_sum = &row_sum_a;
        auto * cur_row_sum = &row_sum_b;
        auto * next_row_sum = &row_sum_c;

        // Accumulate sum across all rows and reduce at the end.
        assert(grid.dimension < std::numeric_limits<uint8_t>::max());
        auto removable_counts = hn::Zero(tag);

        // Convert mask to vector of 0 & 1 by subtracting from zero.
        auto const mask_to_ones = [](auto const & mask) {
          static constexpr hn::ScalableTag<int8_t> signed_tag{};
          return hn::BitCast(
              tag, hn::Sub(hn::Zero(signed_tag), hn::BitCast(signed_tag, hn::VecFromMask(mask))));
        };

        // Start at row 0, which is actually row -1. That way all "intermediate" sums are ready to
        // be used when the actual row 0 is processed.
        for (size_t row_idx = 0; row_idx <= grid.dimension; ++row_idx) {
          // Calculate sum of next row, and the result for the current row.
          auto const next_grid_row = grid.row(row_idx + 1);
          auto const * const HWY_RESTRICT next_row_src_ptr = next_grid_row.data();

          auto const cur_grid_row =
              grid.row(row_idx).subspan(1);  // Offset by one, because we skip left padding.

          size_t const num_chunks = (grid.dimension + lanes - 1) / lanes;

          for (size_t col_idx = 0; col_idx < num_chunks * lanes; col_idx += lanes) {
            // Note that grid rows are padded with an extra cell on each side. I.e. the column index
            // into grid_row is offset by -1 compared to the column index into the output row sums.
            auto const grid_left = hn::LoadU(tag, next_row_src_ptr + col_idx + 0);
            auto const grid_center = hn::LoadU(tag, next_row_src_ptr + col_idx + 1);
            auto const grid_right = hn::LoadU(tag, next_row_src_ptr + col_idx + 2);
            auto const sum_for_next_row = grid_center + grid_right + grid_left;
            hn::Store(sum_for_next_row, tag, next_row_sum->data() + col_idx);

            // Create final sum for current row by adding all row sums. Mask out cells which don't
            // contain a roll of paper themselves.
            auto const sum = sum_for_next_row + hn::Load(tag, prev_row_sum->data() + col_idx) +
                             hn::Load(tag, cur_row_sum->data() + col_idx);
            auto const is_reachable = sum <= hn::Set(tag, max_neighbors_for_removable_roll);

            // Roll in own cell is counted as well. We need to mask out lanes past the end, because
            // otherwise we might count cells from the next row.
            uint8_t const num_valid_lanes = std::min<uint8_t>(lanes, grid.dimension - col_idx);
            auto const has_roll =
                hn::MaskedEq(hn::FirstN(tag, num_valid_lanes),
                             hn::LoadU(tag, cur_grid_row.data() + col_idx), hn::Set(tag, 1));
            auto const is_removable = hn::And(has_roll, is_reachable);

            removable_counts += mask_to_ones(is_removable);
          }

          // Rotate row sums.
          std::tie(prev_row_sum, cur_row_sum, next_row_sum) =
              std::make_tuple(cur_row_sum, next_row_sum, prev_row_sum);
        }

        // Sum into wider lanes first, otherwise the result is an uint8_t, which overflows.
        auto const summed_lanes = hn::SumsOf4(removable_counts);
        return hn::ReduceSum(hn::DFromV<decltype(summed_lanes)>(), summed_lanes);
      }

      /** @brief Calculate neighbors similarly to how we calculate sums for part 1. The big
       * difference however is that we don't store a count, but rather a bitset indicating which of
       * the 8 neighbors contain a roll of paper.
       *
       * See bitset_grid_t below for more details.
       *
       * @note A cell which doesn't contain a roll of paper itself will always have a zero bitset.
       */
      static simd_vector_t<uint8_t> calculate_neighbors(grid_t const & rolls) {
        auto result = simd_vector_t<uint8_t>(rolls.dimension * rolls.dimension);

        auto const get_out_row = [&](uint8_t row) -> simd_span_t<uint8_t> {
          return simd_span_t(result).subspan(row * rolls.dimension, rolls.dimension);
        };

        auto row_mask_a = simd_vector_t<uint8_t>(rolls.dimension);
        auto row_mask_b = simd_vector_t<uint8_t>(rolls.dimension);
        auto row_mask_c = simd_vector_t<uint8_t>(rolls.dimension);

        auto * prev_row_mask = &row_mask_a;
        auto * cur_row_mask = &row_mask_b;
        auto * next_row_mask = &row_mask_c;

        // For the code below, note that since storage is aligned, and once row i is processed, we
        // only ever touch rows > i. Hence it's fine to write into row i + 1's storage.
        static constexpr hn::ScalableTag<uint8_t> tag{};
        static constexpr size_t lanes = hn::Lanes(tag);

        auto const calculate_row_masks = [&](uint8_t roll_row_idx, simd_span_t<uint8_t> dest,
                                             auto callback_fn) {
          // NOTE: The callback_fn is to allow using this code both for the first row calculation,
          // as well as for the "rolling" calculation in the main loop.
          auto const grid_row = rolls.row(roll_row_idx);
          auto const * const HWY_RESTRICT src_ptr = grid_row.data();
          auto * const HWY_RESTRICT dest_ptr = dest.data();

          // Calculate a row's bitset, where bit 0 = center, 1 = right neighbor, 2 = left neighbor.
          // Note the offset into the source data: the input grid has one extra column to the left,
          // so offset 0 into the source equals 1 position to the left in the destination.
          size_t const num_chunks = (rolls.dimension + lanes - 1) / lanes;

          for (uint8_t idx = 0; idx < num_chunks * lanes; idx += lanes) {
            auto const grid_chunk_left = hn::LoadU(tag, src_ptr + idx + 0);
            auto const grid_chunk_center = hn::LoadU(tag, src_ptr + idx + 1);
            auto const grid_chunk_right = hn::LoadU(tag, src_ptr + idx + 2);

            auto const combined_cells = grid_chunk_center | hn::ShiftLeft<1>(grid_chunk_right) |
                                        hn::ShiftLeft<2>(grid_chunk_left);

            hn::Store(combined_cells, tag, dest_ptr + idx);

            // "Forward" mask for further processing.
            callback_fn(idx, combined_cells);
          }
        };

        // Prepare row 0 masks (note that rows in rolls are offset by 1), to allow "rolling"
        // calculation in next loop.
        calculate_row_masks(1, *cur_row_mask, [](auto, auto) {});

        for (size_t row_idx = 0; row_idx < rolls.dimension; ++row_idx) {
          uint8_t const rolls_row_idx = row_idx + 1;  // Due to input grid padding by 1.

          // Create final mask by combining masks from previous, current and next row.
          auto cur_row = get_out_row(row_idx);

          // Calculate bitset for next row.
          calculate_row_masks(
              rolls_row_idx + 1, *next_row_mask, [&](uint8_t col_idx, auto mask_for_next_row) {
                // For each neighbor direction, set the corresponding bit if that neighbor has a
                // roll. Note that each row mask contains bits for three directions per cell.
                auto const has_roll =
                    hn::TestBit(hn::LoadU(tag, cur_row_mask->data() + col_idx), hn::Set(tag, 1));
                auto const bitmask =
                    hn::ShiftLeft<5>(hn::LoadU(tag, prev_row_mask->data() + col_idx)) |
                    hn::ShiftRight<1>(hn::LoadU(tag, cur_row_mask->data() + col_idx)) |
                    hn::ShiftLeft<2>(mask_for_next_row);
                auto const masked_bitmask = hn::And(bitmask, hn::VecFromMask(has_roll));

                hn::StoreU(masked_bitmask, tag, cur_row.data() + col_idx);
              });

          // Rotate row masks.
          std::tie(prev_row_mask, cur_row_mask, next_row_mask) =
              std::make_tuple(cur_row_mask, next_row_mask, prev_row_mask);
        }

        return result;
      }

      grid_t to_grid(simd_string_view_t input) {
        // Assume a square grid. Each line has one extra character (the '\n'), so the total size of
        // the input should equal N * (N + 1), where N is the dimension of the grid.
        size_t const grid_size = (std::sqrt(1 + 4 * input.size()) - 1) / 2;

        [[maybe_unused]] size_t line_idx = 0;
        auto result = grid_t{
            // Reserve space for the grid data, including one padding row on top and bottom, as well
            // as one space of padding left and right.
            .data = simd_vector_t<uint8_t>((grid_size + 2) * (grid_size + 2)),
            .dimension = grid_size,
        };

        assert((std::is_same_v<char, decltype(input)::value_type>));
        auto const * HWY_RESTRICT src_ptr = reinterpret_cast<uint8_t const *>(input.data());
        auto * HWY_RESTRICT dest_ptr = result.data.data();

        dest_ptr += grid_size + 3;  // Start after top padding row and left padding.

        for (size_t row = 0; row < grid_size; ++row) {
          static constexpr hn::ScalableTag<uint8_t> tag{};
          static constexpr size_t lanes = hn::Lanes(tag);
          size_t const num_chunks = (grid_size + lanes - 1) / lanes;

          // Input is either '.' or '@'. The ASCII value for '@' is higher, so we can do a
          // saturating subtraction to convert to either 0 or 1.
          auto const roll_mask = hn::Set(tag, static_cast<uint8_t>('@') - 1);

          for (size_t idx = 0; idx < num_chunks * lanes; idx += lanes) {
            // Need to mask out lanes past the end, to ensure those stay zero.
            uint8_t const num_valid_lanes = std::min<uint8_t>(lanes, grid_size - idx);

            auto const chunk = hn::LoadU(tag, src_ptr + idx);
            auto const has_roll =
                hn::MaskedSaturatedSub(hn::FirstN(tag, num_valid_lanes), chunk, roll_mask);

            hn::StoreU(has_roll, tag, dest_ptr + idx);
          }

          src_ptr += grid_size + 1;   // +1 to skip newline
          dest_ptr += grid_size + 2;  // Skip padding at end of line and at begin of next line.
        }

        return result;
      }

      template <class D>
        requires std::is_same_v<hn::TFromD<D>, uint8_t>
      hn::VFromD<D> popcount(D const & tag, hn::VFromD<D> const & v) {
        auto const bit_counts = hn::Dup128VecFromValues(tag,
                                                        /* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
                                                        /* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
                                                        /* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
                                                        /* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4);

        auto const lower_nibbles = hn::And(v, hn::Set(tag, 0x0f));
        auto const upper_nibbles = hn::And(hn::ShiftRight<4>(v), hn::Set(tag, 0x0f));

        auto const lower_counts = hn::TableLookupBytes(bit_counts, lower_nibbles);
        auto const upper_counts = hn::TableLookupBytes(bit_counts, upper_nibbles);

        return hn::Add(lower_counts, upper_counts);
      }

      size_t mark_initially_removable(grid_t const & roll_grid,
                                      bitset_grid_t const & neighbors_grid,
                                      simd_span_t<int16_t> removable_positions) {
        static constexpr auto in_tag = hn::ScalableTag<uint8_t>{};
        static constexpr auto out_tag = hn::ScalableTag<int16_t>{};
        static constexpr auto half_in_tag = hn::Half<decltype(in_tag)>{};

        static constexpr size_t in_lanes = hn::Lanes(in_tag);
        static constexpr size_t out_lanes = hn::Lanes(out_tag);
        static_assert(in_lanes == 2 * out_lanes);

        uint8_t const dimension = neighbors_grid.dimension();

        int16_t * HWY_RESTRICT dest_ptr = removable_positions.data();
        auto const num_neighbors_limit = hn::Set(in_tag, max_neighbors_for_removable_roll);
        auto idxes = hn::Iota(out_tag, 0);

        auto const write_idxes = [&](hn::Mask<decltype(out_tag)> const mask, size_t max_lanes) {
          size_t const num_written = hn::CompressStore(idxes, mask, out_tag, dest_ptr);
          dest_ptr += num_written;
          idxes = hn::Add(idxes, hn::Set(out_tag, max_lanes));
        };

        // Process each row separately. This way we don't have to worry about padding in roll grid.
        for (uint8_t row = 0; row < dimension; ++row) {
          simd_span_t const roll_row = roll_grid.row(row + 1).subspan(1, dimension);
          simd_span_t const neighbors_row = neighbors_grid.row(row);

          size_t col_pos = 0;
          for (; col_pos + in_lanes <= dimension; col_pos += in_lanes) {
            auto const roll_chunk = hn::LoadU(in_tag, roll_row.data() + col_pos);
            auto const neighbors_chunk = hn::LoadU(in_tag, neighbors_row.data() + col_pos);

            auto const num_neighbors = popcount(in_tag, neighbors_chunk);
            auto is_removable = hn::Lt(num_neighbors, num_neighbors_limit);
            is_removable = hn::And(is_removable, hn::Eq(roll_chunk, hn::Set(in_tag, 1)));

            write_idxes(hn::PromoteMaskTo(out_tag, half_in_tag,
                                          hn::LowerHalfOfMask(half_in_tag, is_removable)),
                        out_lanes);
            write_idxes(hn::PromoteMaskTo(out_tag, half_in_tag,
                                          hn::UpperHalfOfMask(half_in_tag, is_removable)),
                        out_lanes);
          }

          if (col_pos < dimension) {
            uint8_t const lanes_remaining = dimension - col_pos;

            auto const roll_chunk = hn::LoadU(in_tag, roll_row.data() + col_pos);
            auto const neighbors_chunk = hn::LoadU(in_tag, neighbors_row.data() + col_pos);

            auto const num_neighbors = popcount(in_tag, neighbors_chunk);
            auto is_removable = hn::MaskedLt(hn::FirstN(in_tag, lanes_remaining), num_neighbors,
                                             num_neighbors_limit);
            is_removable = hn::And(is_removable, hn::Eq(roll_chunk, hn::Set(in_tag, 1)));

            write_idxes(hn::PromoteMaskTo(out_tag, half_in_tag,
                                          hn::LowerHalfOfMask(half_in_tag, is_removable)),
                        std::min<size_t>(out_lanes / 2, lanes_remaining));

            if (lanes_remaining >= out_lanes / 2) {
              write_idxes(hn::PromoteMaskTo(out_tag, half_in_tag,
                                            hn::UpperHalfOfMask(half_in_tag, is_removable)),
                          lanes_remaining - out_lanes / 2);
            }
          }
        }

        return std::ranges::distance(removable_positions.data(), dest_ptr);
      }

    }  // namespace HWY_NAMESPACE
  }  // namespace
}  // namespace aoc25

HWY_AFTER_NAMESPACE();

#ifdef HWY_ONCE

namespace aoc25 {
  namespace {

    simd_vector_t<uint8_t> calculate_neighbors(grid_t const & rolls) {
      HWY_EXPORT(calculate_neighbors);
      return HWY_DYNAMIC_DISPATCH(calculate_neighbors)(rolls);
    }

    uint32_t count_removables(grid_t const & rolls) {
      HWY_EXPORT(count_removables);
      return HWY_DYNAMIC_DISPATCH(count_removables)(rolls);
    }

    grid_t to_grid(simd_string_view_t input) {
      HWY_EXPORT(to_grid);
      return HWY_DYNAMIC_DISPATCH(to_grid)(input);
    }

    size_t mark_initially_removable(grid_t const & roll_grid,
                                    bitset_grid_t const & neighbors_grid,
                                    simd_span_t<int16_t> removable_positions) {
      HWY_EXPORT(mark_initially_removable);
      return HWY_DYNAMIC_DISPATCH(mark_initially_removable)(roll_grid, neighbors_grid,
                                                            removable_positions);
    }

    simd_vector_t<uint8_t> bitset_grid_t::calculate_neighbors_impl(grid_t const & rolls) {
      return calculate_neighbors(rolls);
    }

    uint16_t remove_rolls(bitset_grid_t & neighbors_grid,
                          uint16_t num_removable,
                          std::span<int16_t> removable_positions) {
      // Remove rolls until no more are accessible. Note that in this case, there's no need to check
      // whether the cell itself contains a roll of paper, since if it doesn't, it's not marked as a
      // neighbor in any other cell.
      uint32_t num_removed = 0;

      while (num_removable > 0) {
        int16_t const cell_idx = removable_positions[--num_removable];
        SPDLOG_DEBUG("Processing cell {} (neighbors: {:08b})", cell_idx, neighbors_grid[cell_idx]);

        num_removed += 1;

        // Update neighbors, and enqueue any newly removable positions.
        uint8_t neighbors_mask = neighbors_grid[cell_idx];
        assert(std::popcount(neighbors_mask) < max_neighbors_for_removable_roll);

        // Note: It's faster iterating over the bits this way than using a LUT.
        while (neighbors_mask != 0) {
          uint8_t const bit_idx = std::countr_zero(neighbors_mask);
          neighbors_mask &= neighbors_mask - 1;  // Clear lowest set bit.

          int16_t const offset = neighbors_grid.mask_bit_idx_to_offset(bit_idx);
          int16_t const neighbor_idx = cell_idx + offset;

          uint8_t & neighbor_bits = neighbors_grid[neighbor_idx];
          uint8_t const num_neighbors = std::popcount(neighbor_bits);

          // Branching really messes up the execution speed here, because it's so unpredictable
          // (around 45% mispredictions). So just do idempotent operations. If the cell is not ready
          // to be visited, then this has no effect on the outcome, since we ignore the "push" by
          // not increasing num_removable.
          assert(num_removable < removable_positions.size());
          removable_positions[num_removable] = neighbor_idx;
          num_removable += num_neighbors == max_neighbors_for_removable_roll;

          SPDLOG_DEBUG(
              " Updating neighbor {:4d} (offset {:+4d}), bits: {:08b} (#: {}), mask: {:08b}",
              neighbor_idx, offset, neighbor_bits, num_neighbors,
              neighbors_grid.mask_bit_idx_to_neighbor_mask(bit_idx));

          // Update neighbor's neighbor bitset to remove reference to this cell.
          neighbor_bits &= ~neighbors_grid.mask_bit_idx_to_neighbor_mask(bit_idx);
        }
      }

      return num_removed;
    }

  }  // namespace

  uint32_t day_t<4>::solve(part_t<1>, version_t<0>, simd_string_view_t input) {
    auto const grid = to_grid(input);
    return count_removables(grid);
  }

  uint32_t day_t<4>::solve(part_t<2>, version_t<0>, simd_string_view_t input) {
    auto const grid = to_grid(input);
    SPDLOG_DEBUG("Grid:\n{}", grid);

    auto neighbors_grid = bitset_grid_t(grid);
    SPDLOG_DEBUG("Neighbors grid:\n{}", neighbors_grid);

    // Populate initial removable positions.
    // Because push_back() is costing time, we avoid it by creating a maximum size vector.
    auto removable_positions =
        simd_vector_t<int16_t>(aoc25::count(simd_span_t{grid.data}, static_cast<uint8_t>(1)));
    SPDLOG_DEBUG("# removable positions buffer size: {}", removable_positions.size());

    size_t num_removable = mark_initially_removable(grid, neighbors_grid, removable_positions);
    SPDLOG_DEBUG("# initial removable positions: {}", num_removable);

    // Remove rolls until no more are accessible.
    return remove_rolls(neighbors_grid, num_removable, removable_positions);
  }

}  // namespace aoc25

#endif  // HWY_ONCE
