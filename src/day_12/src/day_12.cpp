#include "aoc25/day_12.hpp"

#include "aoc25/day.hpp"
#include "aoc25/simd.hpp"
#include "aoc25/string.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <mdspan/mdspan.hpp>
#include <spdlog/spdlog.h>
#include <sys/types.h>

#include <array>
#include <bit>
#include <cstdint>
#include <numeric>
#include <ranges>
#include <type_traits>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "src/day_12.cpp"

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

    struct present_t {
      // NOTE: Might be even faster if all 9 bits are packed in a single uint16_t.
      // But then we need a bunch more shifts.
      static constexpr uint8_t side_length = 3;

      present_t()
          : data_{} {}

      explicit present_t(std::span<uint8_t const, side_length> data) {
        std::ranges::copy(data, data_.begin());

        for ([[maybe_unused]] auto const & row : data_) {
          assert((row >> side_length) == 0);  // Ensure only side_length bits are used.
        }
      }

      uint8_t const & row(uint8_t row) const { return data_[row]; }

      auto begin() const { return data_.begin(); }
      auto end() const { return data_.end(); }

      auto const & raw() const { return data_; }

     private:
      std::array<uint8_t, side_length> data_;
    };

    [[maybe_unused]] auto format_as(present_t const & obj) {
      auto const formatted_rows =
          std::views::iota(0u, present_t::side_length) | std::views::transform([&](uint8_t row) {
            return fmt::format("{:c}{:0{}b}{:c}", (row == 0 ? '[' : ' '), obj.row(row),
                               present_t::side_length,
                               (row == (present_t::side_length - 1) ? ']' : '\n'));
          });
      return fmt::format("{:s}", fmt::join(formatted_rows, ""));
    }

  }  // namespace
}  // namespace aoc25

namespace fmt {

  template <typename Char>
  struct range_format_kind<aoc25::present_t, Char>
      : std::integral_constant<range_format, range_format::disabled> {};

}  // namespace fmt

namespace aoc25 {
  namespace {

    struct region_t {
      uint8_t width;
      uint8_t height;
      std::vector<uint8_t> required_presents;
    };

    [[maybe_unused]] std::string format_as(region_t const & obj) {
      return fmt::format("{}x{}: {}", obj.width, obj.height, obj.required_presents);
    }

    struct problem_t {
      std::vector<present_t> presents;
      std::vector<region_t> regions;
    };

    [[maybe_unused]] std::string format_as(problem_t const & obj) {
      return fmt::format("{} presents:\n{}\n\n{} regions:\n{:s}", obj.presents.size(),
                         fmt::join(obj.presents, "\n\n"), obj.regions.size(),
                         fmt::join(obj.regions, "\n"));
    }

    problem_t parse_input(simd_string_view_t input) {
      problem_t result;

      /* TODO: This can be sped up:
       * - For the presents, just find the last # in the file. From that value, we can calculate the
       *   number of presents.
       * - No need to properly parse the presents, just count number of #.
       * - The "solution" doesn't work for the example anyway, so might as well tune the parsing to
       *   the actual input. I.e. each region has 2 digits for both width and height, as well as for
       *   the count per present type. I.e. the exact position of each value on a line is known.
       * - No no need to actually parse and store the input. Just solve as we go through it.
       */

      // Each present starts with an index, followed by a colon, then present_t::side_length lines,
      // and finally a blank line. Assume there's less than 10 presents. So we know the present's
      // index is at most one digit. Keep reading until there's no more colon at the expected
      // position.
      while (input.at(1) == ':') {
        // Skip the line (index, colon, and newline).
        input.remove_prefix(3);

        auto data = std::array<uint8_t, present_t::side_length>{};
        for (size_t row = 0; row < present_t::side_length; ++row) {
          for (size_t col = 0; col < present_t::side_length; ++col) {
            // Skip over past lines + newlines.
            char const c = input[col + (present_t::side_length + 1) * row];
            data.at(row) |= static_cast<uint8_t>(c == '#') << col;
          }
        }
        result.presents.emplace_back(data);

        // Jump to the next present (3 lines of 3 characters + newline, plus one extra newline).
        input.remove_prefix(present_t::side_length * (present_t::side_length + 1) + 1);
      }

      // All remaining lines contain info on regions and their required presents.
      uint8_t const num_presents = result.presents.size();

      while (input.size() > 1) {
        // Line format: "WxH: p1 p2 p3 ... pN". Numbers can be up to two digits long.
        auto & region = result.regions.emplace_back();

        auto const x_pos = input.find('x');
        region.width = (x_pos == 1) ? (input[0] - '0') : (10 * (input[0] - '0') + (input[1] - '0'));
        input.remove_prefix(x_pos + 1);  // Skip past 'x'.

        auto const colon_pos = input.find(':');
        region.height =
            (colon_pos == 1) ? (input[0] - '0') : (10 * (input[0] - '0') + (input[1] - '0'));
        input.remove_prefix(colon_pos + 2);  // Skip past colon and space.

        // Now parse required amounts for each present.
        auto const parse_required = [&](char separator) {
          auto const sep_pos = input.find(separator);
          uint8_t const num_required =
              (sep_pos == 1) ? (input[0] - '0') : (10 * (input[0] - '0') + (input[1] - '0'));
          region.required_presents.push_back(num_required);
          input.remove_prefix(sep_pos + 1);  // Skip past separator.
        };

        for (uint8_t req_remaining = num_presents; req_remaining-- > 0;) {
          assert(input.size() > 1);
          parse_required(req_remaining != 0 ? ' ' : '\n');
        }
      }

      SPDLOG_DEBUG("Parsed: {}", result);
      return result;
    }

    using present_set_t = std::vector<present_t>;

  }  // namespace

  uint64_t day_t<12>::solve(part_t<1>, version_t<0>, simd_string_view_t input) {
    /* NOTE: This problem is just dumb. Apparently all regions into which the requested number of
     * presents could possibly fit (i.e. total area of presents <= area of region) do fit. I.e.
     * there is no need to solve the (NP-complete) problem at all (which isn't feasibly anyway).
     *
     * There used to be a heuristic-based solver in this file to attempt actually solving the
     * problem, which attempt to find a best-first fit across all present rotations/mirrorings on as
     * low- and right-most as possible position. How course it didn't work, because exactly solving
     * the 2D bin-packing problem is NP-complete.
     */
    auto const problem = parse_input(input);

    // Calculate area per present.
    auto const present_areas =
        problem.presents | std::views::transform([](auto const & present) -> uint16_t {
          return std::accumulate(present.raw().begin(), present.raw().end(), 0u,
                                 [](uint16_t sum, uint8_t row) {
                                   return sum + static_cast<uint16_t>(std::popcount(row));
                                 });
        }) |
        std::ranges::to<std::vector>();

    // For each region, test if it definitely doesn't fit.
    uint16_t const num_definitely_no_fit_regions = std::accumulate(
        problem.regions.begin(), problem.regions.end(), 0u, [&](uint16_t sum, auto const & region) {
          uint32_t const region_area = static_cast<uint32_t>(region.width) * region.height;
          uint32_t const min_required_area =
              std::inner_product(region.required_presents.begin(), region.required_presents.end(),
                                 present_areas.begin(), 0u);
          return sum + (min_required_area > region_area ? 1u : 0u);
        });

    return problem.regions.size() - num_definitely_no_fit_regions;
  }

}  // namespace aoc25

#endif  // HWY_ONCE
