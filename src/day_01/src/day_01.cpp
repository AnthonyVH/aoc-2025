#include "aoc25/day_01.hpp"

#include "aoc25/math.hpp"
#include "aoc25/memory.hpp"

#include <fmt/ranges.h>
#include <hwy/base.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <numeric>

namespace aoc25 {

  namespace {

    static constexpr uint8_t dial_size = 100;
    static constexpr uint8_t start_pos = 50;

    std::vector<int16_t> parse_input(version_t<0>, std::string_view input) {
      std::vector<int16_t> numbers;

      // Assume one number per line.
      auto const num_lines = std::ranges::count(input, '\n');
      numbers.reserve(num_lines);

      int16_t value = 0;
      bool invert = false;

      for (char c : input) {
        if (c == '\n') {
          numbers.push_back(invert ? -value : value);
          value = 0;
        } else if (c == 'L') {
          invert = true;
        } else if (c == 'R') {
          invert = false;
        } else {
          assert(c >= '0' && c <= '9');
          value = value * 10 + static_cast<int16_t>(c - '0');
        }
      }

      return numbers;
    }

    /** @brief Calculate lookup table for modulo operation on input between given ranges. */
    template <auto Lower, auto Upper, auto Modulo>
      requires std::integral<decltype(Lower)> && std::integral<decltype(Upper)> &&
               std::same_as<decltype(Lower), decltype(Upper)> && std::integral<decltype(Modulo)> &&
               (Lower < Upper) && (Modulo > 0)
    constexpr auto calculate_modulo_lut() {
      static constexpr size_t lut_size = Upper - Lower + 1;

      using T = decltype(Lower);

      std::array<T, lut_size> lut{};
      size_t idx = 0;

      for (T value = Lower; value <= Upper; ++value, ++idx) {
        lut[idx] = aoc25::mod(value, Modulo);
      }

      return lut;
    }

    template <class T>
    auto lut_mod(T value) -> T {
      static constexpr int16_t lut_lower = -1100;
      static constexpr int16_t lut_upper = 1100;
      static constexpr auto lut = calculate_modulo_lut<lut_lower, lut_upper, dial_size>();

      assert(value >= lut_lower && value <= lut_upper);
      [[assume(value >= lut_lower && value <= lut_upper)]];
      return lut.at(static_cast<size_t>(value - lut_lower));
    }

  }  // namespace

  int day_t<1>::solve(part_t<1>, version_t<0>, std::string_view input) {
    auto const numbers = parse_input(version<0>, input);

    int32_t pos = start_pos;
    uint16_t at_zero = 0;

    for (auto const number : numbers) {
      pos = aoc25::mod(pos + number, dial_size);
      at_zero += (pos == 0);
    }

    return at_zero;
  }

  int day_t<1>::solve(part_t<1>, version_t<1>, std::string_view input) {
    auto const numbers = parse_input(version<0>, input);

    int32_t pos = start_pos;
    uint16_t at_zero = 0;

    for (auto const number : numbers) {
      pos = lut_mod(pos + number);
      at_zero += (pos == 0);
    }

    return at_zero;
  }

  int day_t<1>::solve(part_t<1>, version_t<2>, std::string_view input) {
    auto numbers = parse_input(version<0>, input);
    numbers.at(0) += start_pos;

    using simd_allocator = aoc25::aligned_allocator<int16_t, 512>;
    auto prefix_sum = std::vector<int16_t, simd_allocator>(numbers.size());
    std::inclusive_scan(numbers.begin(), numbers.end(), prefix_sum.begin());

    mod_dial(prefix_sum);
    return count_zeros(prefix_sum);
  }

  int day_t<1>::solve(part_t<2>, version_t<0>, std::string_view input) {
    auto const numbers = parse_input(version<0>, input);

    int32_t pos = start_pos;
    uint16_t passed_zero = 0;

    for (auto const number : numbers) {
      int32_t after_move;
      if (number < 0) {  // Mirror the dial.
        auto const mirrored_pos = aoc25::mod(dial_size - pos, dial_size);
        after_move = mirrored_pos - number;
      } else {
        after_move = pos + number;
      }

      passed_zero += std::abs(after_move / 100);
      pos = aoc25::mod(pos + number, dial_size);
    }

    return passed_zero;
  }

  int day_t<1>::solve(part_t<2>, version_t<1>, std::string_view input) {
    auto const numbers = parse_input(version<0>, input);

    int32_t pos = start_pos;
    uint16_t passed_zero = 0;

    for (auto const number : numbers) {
      int32_t after_move;
      if (number < 0) {  // Mirror the dial.
        auto const mirrored_pos = lut_mod(dial_size - pos);
        after_move = mirrored_pos - number;
      } else {
        after_move = pos + number;
      }

      passed_zero += std::abs(after_move / 100);
      pos = lut_mod(pos + number);
    }

    return passed_zero;
  }

}  // namespace aoc25
