#include "aoc25/day_04.hpp"

#include <fmt/base.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <mdspan/mdspan.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <span>
#include <string_view>
#include <type_traits>

namespace fmt {

  namespace {

    template <class T>
    using derived_formatter_t =
        std::conditional_t<std::is_same_v<std::remove_cvref_t<T>, uint8_t>,
                           unsigned char,
                           std::conditional_t<std::is_same_v<std::remove_cvref_t<T>, int8_t>,
                                              signed char,
                                              std::remove_cvref_t<T>>>;

  }  // namespace

  template <typename Char, class T, class Extents, class Layout, class Accessor>
  struct range_format_kind<Kokkos::mdspan<T, Extents, Layout, Accessor>, Char>
      : std::integral_constant<range_format, range_format::disabled> {};

  template <class T, class Extents, class Layout, class Accessor>
    requires(Extents::rank() == 2)
  struct formatter<Kokkos::mdspan<T, Extents, Layout, Accessor>>
      : formatter<derived_formatter_t<T>> {
    using obj_t = Kokkos::mdspan<T, Extents, Layout, Accessor>;
    using subtype_t = derived_formatter_t<T>;
    using base_t = formatter<subtype_t>;

    constexpr auto parse(format_parse_context & ctx) -> format_parse_context::iterator {
      return base_t::parse(ctx);
    }

    auto format(obj_t const & obj, format_context & ctx) const -> format_context::iterator {
      auto out = ctx.out();

      *out++ = '[';
      for (size_t i = 0; i < obj.extent(0); ++i) {
        if (i > 0) {
          *out++ = '\n';
          *out++ = ' ';
        }

        *out++ = '[';

        for (size_t j = 0; j < obj.extent(1); ++j) {
          if (j > 0) {
            *out++ = ' ';
          }
          formatter<subtype_t>::format(obj[i, j], ctx);
        }

        *out++ = ']';
      }
      *out++ = ']';

      return out;
    }
  };

}  // namespace fmt

namespace aoc25 {
  namespace {

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

      auto as_mdspan() { return Kokkos::mdspan(data.data(), dimension + 2, dimension + 2); }

      auto as_mdspan() const { return Kokkos::mdspan(data.data(), dimension + 2, dimension + 2); }
    };

    grid_t to_grid(std::string_view input) {
      [[maybe_unused]] size_t line_idx = 0;
      grid_t result;

      // Assume a square grid. Each line has one extra character (the '\n'), so the total size of
      // the input should equal N * (N + 1), where N is the dimension of the grid.
      result.dimension = (std::sqrt(1 + 4 * input.size()) - 1) / 2;

      // Reserve space for the grid data, including one padding row on top and bottom, as well as
      // one space of padding left and right.
      result.data.resize((result.dimension + 2) * (result.dimension + 2));

      size_t pos = result.dimension + 3;  // Start after top padding row and left padding.
      [[maybe_unused]] size_t line_start = pos;

      for (char c : input) {
        assert(pos < result.data.size());

        if (c == '\n') {
          assert(pos - line_start == result.dimension);
          ++line_idx;
          pos += 2;  // Skip right padding and move to past next line's left padding.
          line_start = pos;
        } else {
          result.data[pos] = (c == '@') ? 1 : 0;
          ++pos;
        }
      }

      assert(line_idx == result.dimension);

      return result;
    }

    grid_t add_neighbors(grid_t const & grid) {
      grid_t result = grid;

      // For each cell, we add the value of the eight surrounding cells. So we take each row, and
      // add the previous and next rows to it, as well as left and right shifted versions of each of
      // them, and of itself.
      for (size_t row_idx = 1; row_idx <= grid.dimension; ++row_idx) {
        auto row = result.row(row_idx);

        auto const prev_row = grid.row(row_idx - 1);
        auto const cur_row = grid.row(row_idx + 0);
        auto const next_row = grid.row(row_idx + 1);

        for (size_t col_idx = 1; col_idx <= grid.dimension; ++col_idx) {
          if (row[col_idx] == 0) {
            continue;
          }

          row[col_idx] += prev_row[col_idx - 1];
          row[col_idx] += prev_row[col_idx + 0];
          row[col_idx] += prev_row[col_idx + 1];

          row[col_idx] += next_row[col_idx - 1];
          row[col_idx] += next_row[col_idx + 0];
          row[col_idx] += next_row[col_idx + 1];

          row[col_idx] += cur_row[col_idx - 1];
          row[col_idx] += cur_row[col_idx + 1];
        }
      }

      return result;
    }

    bool is_removable(int8_t const e) {
      return 0 < e && e <= 4;
    };

  }  // namespace

  uint32_t day_t<4>::solve(part_t<1>, version_t<0>, std::string_view input) {
    auto const grid = to_grid(input);

    auto grid_sums = add_neighbors(grid);
    SPDLOG_DEBUG("Grid sums:\n{}", grid_sums.as_mdspan());

    // Count how many cells have a value less than 4.
    return std::ranges::count_if(grid_sums.data, is_removable);
  }

  uint32_t day_t<4>::solve(part_t<2>, version_t<0>, std::string_view input) {
    auto const grid = to_grid(input);

    auto grid_sums = add_neighbors(grid);
    SPDLOG_DEBUG("# removable: {}, grid sums:\n{}",
                 std::ranges::count_if(grid_sums.data, is_removable), grid_sums.as_mdspan());

    // Remove rolls until no more are accessible (and thus can't be removed).
    auto next_grid_sums = grid_sums;
    uint32_t num_removed = 0;
    bool continue_removal = true;

    auto const update_row = [](std::span<int8_t> row, size_t modifier_col_idx) {
      row[modifier_col_idx - 1] = row[modifier_col_idx - 1] ? row[modifier_col_idx - 1] - 1 : 0;
      row[modifier_col_idx + 0] = row[modifier_col_idx + 0] ? row[modifier_col_idx + 0] - 1 : 0;
      row[modifier_col_idx + 1] = row[modifier_col_idx + 1] ? row[modifier_col_idx + 1] - 1 : 0;
    };

    while (continue_removal) {
      continue_removal = false;

      for (size_t row_idx = 1; row_idx <= grid_sums.dimension; ++row_idx) {
        auto row = grid_sums.row(row_idx);

        for (size_t col_idx = 1; col_idx <= grid.dimension; ++col_idx) {
          if (!is_removable(row[col_idx])) {
            continue;
          }

          continue_removal = true;  // At least one roll got removed, so keep trying.
          num_removed += 1;

          // Update neighbor counts. Note that counts might go negative, but that doesn't
          // matter, since is_removable will ignore those.
          update_row(next_grid_sums.row(row_idx - 1), col_idx);
          update_row(next_grid_sums.row(row_idx + 0), col_idx);
          update_row(next_grid_sums.row(row_idx + 1), col_idx);
          next_grid_sums.row(row_idx)[col_idx] = 0;
        }
      }

      // Update grid.
      grid_sums = next_grid_sums;
      SPDLOG_DEBUG("# removed: {}, grid:\n{}", num_removed, grid_sums.as_mdspan());
    }

    return num_removed;
  }

}  // namespace aoc25
