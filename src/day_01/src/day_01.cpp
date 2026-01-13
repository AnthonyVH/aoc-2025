#include "aoc25/day_01.hpp"

#include "aoc25/math.hpp"
#include "aoc25/simd.hpp"
#include "aoc25/string.hpp"

#include <fmt/ranges.h>
#include <spdlog/spdlog.h>

#include <array>
#include <cassert>
#include <cstdint>

namespace aoc25 {
  namespace {

    // Somehow this is measurably faster than parse_input(version<2>).
    std::vector<int8_t> parse_input(version_t<1>, simd_string_view_t input) {
      auto rotations = std::vector<int8_t>(count(simd_span_t{input}, '\n'));
      size_t idx = 0;

      split(input, [&](simd_string_view_t line) {
        assert((line.size() >= 2) && (line.size() <= 4));
        int8_t const sign = line[0] == 'L' ? -1 : 1;
        uint8_t const num_digits = line.size() - 1;
        int8_t const partial_turn =
            (num_digits == 1) ? (line[1] - '0')
                              : ((num_digits == 2) ? ((line[1] - '0') * 10 + (line[2] - '0'))
                                                   : ((line[2] - '0') * 10 + (line[3] - '0')));

        assert(idx < rotations.size());
        rotations[idx++] = sign * partial_turn;
      });

      return rotations;
    }

    std::pair<uint16_t, std::vector<int8_t>> parse_input(version_t<2>, simd_string_view_t input) {
      auto partial_rotations = std::vector<int8_t>(count(simd_span_t{input}, '\n'));
      uint16_t num_full_rotations = 0;
      size_t idx = 0;

      split(input, [&](simd_string_view_t line) {
        assert((line.size() >= 2) && (line.size() <= 4));
        int8_t const sign = line[0] == 'L' ? -1 : 1;
        uint8_t const num_digits = line.size() - 1;
        uint8_t const full_turns = (num_digits <= 2) ? 0 : (line[1] - '0');
        int8_t const partial_turn =
            (num_digits == 1) ? (line[1] - '0')
                              : ((num_digits == 2) ? ((line[1] - '0') * 10 + (line[2] - '0'))
                                                   : ((line[2] - '0') * 10 + (line[3] - '0')));

        assert(idx < partial_rotations.size());
        partial_rotations[idx++] = sign * partial_turn;
        num_full_rotations += full_turns;
      });

      return std::make_pair(num_full_rotations, std::move(partial_rotations));
    }

    static constexpr int16_t dial_size = 100;
    static constexpr int16_t start_pos = 50;

    struct lut_t {
      static constexpr int lower_bound = -dial_size + 1;
      static constexpr int upper_bound = 2 * (dial_size - 1);

      static constexpr uint16_t zero_idx = -lower_bound;  // Index in table where value equals zero.
      static constexpr uint16_t one_rotation_idx = zero_idx + dial_size;

      static constexpr uint16_t value_to_idx(int16_t value) {
        assert(value >= lower_bound && value <= upper_bound);
        return value + zero_idx;
      }

      static constexpr uint16_t modulo(uint16_t index) {
        assert(index < lut.size());
        return lut[index];
      }

      static constexpr auto lut = [] {
        // Create an offset table, which points back into itself (incl. the offset).
        // That way we don't have to explicitly calculate the proper offset each lookup.
        static constexpr int num_entries = upper_bound - lower_bound + 1;

        auto table = std::array<uint16_t, num_entries>{};

        for (size_t i = 0; i < num_entries; ++i) {
          int16_t const value = i + lower_bound;
          int16_t const result = aoc25::mod(value, dial_size);
          uint16_t const table_idx = result - lower_bound;
          table[i] = table_idx;
        }
        return table;
      }();
    };

  }  // namespace

  int day_t<1>::solve(part_t<1>, version_t<0>, simd_string_view_t input) {
    auto const rotations = parse_input(version<1>, input);

    uint16_t pos = lut_t::value_to_idx(start_pos);
    uint16_t at_zero = 0;

    // TODO: Something very weird is going on here. Replacing lut.at() with modulo() causes a 25-50%
    // slowndown, consistently... Same for part 2. Release is also much slower than RelWithDebInfo.
    for (auto const & rotation : rotations) {
      pos = lut_t::lut.at(pos + rotation);
      at_zero += (pos == lut_t::zero_idx);
    }

    return at_zero;
  }

  int day_t<1>::solve(part_t<2>, version_t<0>, simd_string_view_t input) {
    auto const [full_rotations, partial_rotations] = parse_input(version<2>, input);

    uint16_t pos = lut_t::value_to_idx(start_pos);
    uint16_t passed_zero = full_rotations;

    for (auto const & rotation : partial_rotations) {
      uint16_t const lookup_pos = pos + rotation;

      // Don't count going negative when starting at zero as crossing zero.
      bool const crossed_zero = (pos != lut_t::zero_idx) & ((lookup_pos < lut_t::zero_idx) |
                                                            (lookup_pos > lut_t::one_rotation_idx));

      pos = lut_t::lut.at(lookup_pos);
      passed_zero += (pos == lut_t::zero_idx) + crossed_zero;
    }

    return passed_zero;
  }

}  // namespace aoc25
