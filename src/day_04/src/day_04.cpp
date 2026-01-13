#include "aoc25/day_04.hpp"

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

    struct grid_t {
      std::vector<int8_t> data;
      size_t dimension;

      std::span<int8_t> row(size_t row_idx) {
        return std::span(const_cast<int8_t *>(std::as_const(*this).row(row_idx).data()),
                         dimension + 2);
      }

      std::span<int8_t const> row(size_t row_idx) const {
        assert(row_idx < dimension + 2);
        return std::span(data.data() + row_idx * (dimension + 2), dimension + 2);
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

    grid_t to_grid(std::string_view input) {
      // TODO: Implement this using SIMD.
      [[maybe_unused]] size_t line_idx = 0;
      grid_t result;

      // Assume a square grid. Each line has one extra character (the '\n'), so the total size of
      // the input should equal N * (N + 1), where N is the dimension of the grid.
      result.dimension = (std::sqrt(1 + 4 * input.size()) - 1) / 2;

      // Reserve space for the grid data, including one padding row on top and bottom, as well as
      // one space of padding left and right.
      result.data.resize((result.dimension + 2) * (result.dimension + 2));

      size_t pos = result.dimension + 3;  // Start after top padding row and left padding.

      while (!input.empty()) {
        // Process whole line at a time.
        auto const line = input.substr(0, result.dimension);
        std::ranges::transform(line, result.data.begin() + pos,
                               [](char c) { return (c == '@') ? 1 : 0; });
        pos += result.dimension + 2;  // Skip padding at end of line and at begin of next line.
        input.remove_prefix(result.dimension + 1);  // +1 to skip newline
      }

      return result;
    }

    uint32_t count_removables(grid_t const & grid) {
      uint32_t num_removable = 0;

      auto const get_in_row = [&](uint8_t row, int8_t offset) -> std::span<int8_t const> {
        return grid.row(row).subspan(1 + offset, grid.dimension);
      };

      auto row_sum_a = std::vector<int8_t>(grid.dimension);
      auto row_sum_b = std::vector<int8_t>(grid.dimension);
      auto row_sum_c = std::vector<int8_t>(grid.dimension);

      auto * prev_row_sum = &row_sum_a;
      auto * cur_row_sum = &row_sum_b;
      auto * next_row_sum = &row_sum_c;

      // Start at row 0, which is actually row -1. That way all "intermediate" sums are ready to be
      // used when the actual row 0 is processed.
      for (size_t row_idx = 0; row_idx <= grid.dimension; ++row_idx) {
        // Calculate sum of next row.
        auto const grid_row = grid.row(row_idx + 1);
        for (size_t col_idx = 0; col_idx < grid.dimension; ++col_idx) {
          // Note that grid rows are padded with one extra cell on each side.
          (*next_row_sum)[col_idx] =
              grid_row[col_idx + 0] + grid_row[col_idx + 1] + grid_row[col_idx + 2];
        }

        // Create final sum for current row by adding all row sums. Mask out cells which don't
        // contain a roll of paper themselves.
        static constexpr uint8_t max_neighboring_rolls_for_removable = 4;
        auto const cur_row_has_roll = get_in_row(row_idx, 0);

        for (size_t col_idx = 0; col_idx < grid.dimension; ++col_idx) {
          auto const sum =
              (*prev_row_sum)[col_idx] + (*cur_row_sum)[col_idx] + (*next_row_sum)[col_idx];

          // Roll in own cell is counted as well.
          bool const is_removable =
              cur_row_has_roll[col_idx] & (sum <= max_neighboring_rolls_for_removable);
          num_removable += is_removable;
        }

        // Swap row sums.
        std::tie(prev_row_sum, cur_row_sum, next_row_sum) =
            std::make_tuple(cur_row_sum, next_row_sum, prev_row_sum);
      }

      return num_removable;
    }

    /** @brief Grid which stores which neighbor contains a paper roll, using a bitset.
     * Note that it doesn't keep track of whether a cell itself contains a roll of paper.
     */
    struct bitset_grid_t {
     public:
      explicit bitset_grid_t(grid_t const & rolls)
          : neighbors_(calculate_neighbors(rolls))
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

      std::span<uint8_t> row(uint8_t row_idx) {
        return std::span(const_cast<uint8_t *>(std::as_const(*this).row(row_idx).data()),
                         dimension_);
      }

      std::span<uint8_t const> row(uint8_t row_idx) const {
        assert(row_idx < dimension_);
        return std::span(neighbors_.data() + row_idx * dimension_, dimension_);
      }

      int16_t mask_bit_idx_to_offset(uint8_t bit_idx) const {
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
      std::vector<uint8_t> neighbors_;
      std::array<int16_t, 8> mask_bit_idx_to_offset_;
      int16_t dimension_;

      static std::vector<uint8_t> calculate_neighbors(grid_t const & rolls) {
        auto result = std::vector<uint8_t>(rolls.dimension * rolls.dimension);

        // Calculate neighbors similarly to how we calculate sums for part 1. The big difference
        // however, is that we don't store a count, but rather a bitset indicating which of the 8
        // neighbors contain a roll of paper.
        // Note: A cell which doesn't contain a roll of paper itself will always have a zero bitset.
        [[maybe_unused]] auto const get_in_row = [&](uint8_t row,
                                                     int8_t offset) -> std::span<int8_t const> {
          return rolls.row(row).subspan(1 + offset, rolls.dimension);
        };

        auto const get_out_row = [&](uint8_t row) -> std::span<uint8_t> {
          return std::span(result.data() + row * rolls.dimension, rolls.dimension);
        };

        auto row_sum_a = std::vector<uint8_t>(rolls.dimension);
        auto row_sum_b = std::vector<uint8_t>(rolls.dimension);
        auto row_sum_c = std::vector<uint8_t>(rolls.dimension);

        auto * prev_row_sum = &row_sum_a;
        auto * cur_row_sum = &row_sum_b;
        auto * next_row_sum = &row_sum_c;

        auto const calculate_row_masks = [&](uint8_t roll_row_idx, std::span<uint8_t> dest) {
          auto const grid_row = rolls.row(roll_row_idx);
          for (size_t col_idx = 0; col_idx < rolls.dimension; ++col_idx) {
            dest[col_idx] = (grid_row[col_idx + 0] << 2) | (grid_row[col_idx + 1] << 0) |
                            (grid_row[col_idx + 2] << 1);
          }
        };

        // Prepare row 0 sums (note that rows in rolls are offset by 1), to allow "rolling"
        // calculation in next loop.
        calculate_row_masks(1, *cur_row_sum);

        for (size_t row_idx = 0; row_idx < rolls.dimension; ++row_idx) {
          // Calculate bitset for next row.
          uint8_t const rolls_row_idx = row_idx + 1;
          calculate_row_masks(rolls_row_idx + 1, *next_row_sum);

          // Create final sum for current row by adding all row sums.
          auto cur_row = get_out_row(row_idx);

          for (size_t col_idx = 0; col_idx < rolls.dimension; ++col_idx) {
            // For each neighbor direction, set the corresponding bit if that neighbor has a roll.
            // Note that each row sum contains bits for three directions per cell.
            uint8_t const has_roll = (*cur_row_sum)[col_idx] & 1;
            uint8_t const mask = ((*prev_row_sum)[col_idx] << 5) | ((*cur_row_sum)[col_idx] >> 1) |
                                 ((*next_row_sum)[col_idx] << 2);
            cur_row[col_idx] = has_roll * mask;
          }

          // Swap row sums.
          std::tie(prev_row_sum, cur_row_sum, next_row_sum) =
              std::make_tuple(cur_row_sum, next_row_sum, prev_row_sum);
        }

        return result;
      }
    };

    [[maybe_unused]] std::string format_as(bitset_grid_t const & obj) {
      return fmt::format("{}", fmt::join(std::views::iota(size_t{0}, obj.dimension()) |
                                             std::views::transform([&obj](size_t row_idx) {
                                               return fmt::format("{::08b}", obj.row(row_idx));
                                             }),
                                         "\n"));
    }

  }  // namespace

  uint32_t day_t<4>::solve(part_t<1>, version_t<0>, std::string_view input) {
    auto const grid = to_grid(input);
    return count_removables(grid);
  }

  uint32_t day_t<4>::solve(part_t<2>, version_t<0>, std::string_view input) {
    auto const grid = to_grid(input);
    SPDLOG_DEBUG("Grid:\n{}", grid);

    auto neighbors_grid = bitset_grid_t(grid);
    SPDLOG_DEBUG("Neighbors grid:\n{}", neighbors_grid);

    // Populate initial removable positions.
    uint16_t const dimension = neighbors_grid.dimension();

    std::vector<uint16_t> removable_positions;
    removable_positions.reserve(dimension * dimension);

    for (uint8_t row = 0; row < dimension; ++row) {
      auto const grid_row = grid.row(row + 1).subspan(1, dimension);
      auto const neighbors_row = neighbors_grid.row(row);

      for (uint8_t col = 0; col < dimension; ++col) {
        if (bool const has_roll = (grid_row[col] == 1); has_roll) {
          uint8_t const num_neighbors = std::popcount(neighbors_row[col]);
          if (num_neighbors < 4) {
            uint16_t const cell_idx = row * dimension + col;
            removable_positions.push_back(cell_idx);
          }
        }
      }
    }

    // Remove rolls until no more are accessible. Note that in this case, there's no need to check
    // whether the cell itself contains a roll of paper, since if it doesn't, it's not marked as a
    // neighbor in any other cell.
    uint32_t num_removed = 0;

    while (!removable_positions.empty()) {
      uint16_t const cell_idx = removable_positions.back();
      removable_positions.pop_back();
      SPDLOG_DEBUG("Processing cell {} (neighbors: {:08b})", cell_idx, neighbors_grid[cell_idx]);

      num_removed += 1;

      // Update neighbors, and enqueue any newly removable positions.
      uint8_t neighbors_mask = neighbors_grid[cell_idx];
      while (neighbors_mask != 0) {
        uint8_t const bit_idx = std::countr_zero(neighbors_mask);
        neighbors_mask &= neighbors_mask - 1;  // Clear lowest set bit.

        int16_t const offset = neighbors_grid.mask_bit_idx_to_offset(bit_idx);
        uint16_t const neighbor_idx = cell_idx + offset;

        uint8_t & neighbor_bits = neighbors_grid[neighbor_idx];
        uint8_t const num_neighbors = std::popcount(neighbor_bits);

        if (num_neighbors <= 3) {  // Cell was already enqueued.
          continue;
        }

        SPDLOG_DEBUG(" Updating neighbor {:4d} (offset {:+4d}), bits: {:08b} (#: {}), mask: {:08b}",
                     neighbor_idx, offset, neighbor_bits, num_neighbors,
                     neighbors_grid.mask_bit_idx_to_neighbor_mask(bit_idx));

        // Update neighbor's neighbor bitset to remove reference to this cell.
        neighbor_bits &= ~neighbors_grid.mask_bit_idx_to_neighbor_mask(bit_idx);

        // Update of neighbor_bits unsets one bit, so if the number of set bits was 4 before,
        // then the cell is now accessible.
        if (num_neighbors == 4) {
          removable_positions.push_back(neighbor_idx);
        }
      }
    }

    return num_removed;
  }

}  // namespace aoc25

#endif  // HWY_ONCE
