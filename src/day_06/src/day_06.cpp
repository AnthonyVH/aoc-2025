#include "aoc25/day_06.hpp"

#include "aoc25/day.hpp"
#include "aoc25/simd.hpp"
#include "aoc25/string.hpp"

#include <fmt/ranges.h>
#include <spdlog/spdlog.h>

#include <cstdint>
#include <numeric>

namespace aoc25 {

  namespace {}  // namespace

  /* TODO: Ideas:
   * - Just create a LUT for the numbers, they're at most 4 digits. Try to load groups of 4
   *   characters, if a digit is unused, default to e.g. '0'. Then subtract '0' from all, and
   *   compress the 4 * 8 = 32 bits into a 4 * 4 = 16 bit number (each byte is ['0','9'] or a
   *   tombstone marker). Use these 4 values as an index for a gather from a LUT of precomputed
   *   values. This LUT should easily fit into L2 cache (2^16 * 16 bit = 128 KiB).
   *
   * - Once we detect that we'er on the last line (with the operators), stop parsing. Then execute
   *   an SIMD loop where we match against non-spaces and Compress the result into an array.
   *
   * - We can also not parse the numbers initially. Instead search for the operators row. Then from
   *   there, we know how many "number" lines there are. This allows us to calculate offsets for
   *   each digit given that we know the operator's offset.
   */
  uint64_t day_t<6>::solve(part_t<1>, version_t<0>, simd_string_view_t input) {
    // To avoid having to search in each line where numbers begin/end, first split into lines.
    // We can then use the location of each operator to determine where numbers are.
    std::vector<simd_string_view_t> lines;
    lines.reserve(5);
    split(input, [&](simd_string_view_t const & line) { lines.push_back(line); }, '\n');

    size_t const max_line_length = std::accumulate(
        lines.begin(), lines.end(), 0U,
        [](size_t max, simd_string_view_t line) { return std::max(max, line.size()); });

    // Find position of each operator.
    std::vector<uint16_t> operator_offsets;
    operator_offsets.reserve(2'000);

    auto const & operators_line = lines.back();
    split(
        operators_line,
        [&, line_begin = operators_line.data()](simd_string_view_t const & line) {
          if (!line.empty()) {
            operator_offsets.push_back(std::distance(line_begin, line.data()));
          }
        },
        ' ');
    operator_offsets.push_back(max_line_length + 1);  // Sentinel value, including "virtual" ' '.
    SPDLOG_DEBUG("operator offsets: {}", operator_offsets);

    uint32_t const num_problems = operator_offsets.size() - 1;
    size_t const num_numbers = lines.size() - 1;
    SPDLOG_DEBUG("num problems: {}, num numbers: {}", num_problems, num_numbers);

    uint64_t total = 0;

    for (size_t problem_idx = 0; problem_idx < num_problems; ++problem_idx) {
      size_t const operator_offset = operator_offsets[problem_idx];
      size_t const next_operator_offset = operator_offsets[problem_idx + 1];
      size_t const num_digits = next_operator_offset - operator_offset - 1;
      char const op = operators_line[operator_offset];

      SPDLOG_DEBUG("problem: {}, num digits: {}", problem_idx, num_digits);
      uint64_t result = (op == '*') ? 1 : 0;

      // TODO: Seems inefficient to not iterate over a whole line first.
      // Although this way we don't have to find the operator offsets each time.
      // But we could precompute them once...
      for (size_t number_idx = 0; number_idx < num_numbers; ++number_idx) {
        auto line = lines.at(number_idx);
        size_t const number_begin = operator_offset;
        line = line.substr(number_begin, num_digits);  // Requesting a substr past the end is fine.

        // Eliminate spaces from the number. They can be either at the start or the end, not both.
        // NOTE: We don't care about space at the beginning. When converting the number, we do a
        // saturating subtraction by '0', so spaces become 0 (since their ASCII value is 32, and '0'
        // is 48).
        while (line.back() == ' ') {  // NOTE: Slightly faster than find_last_not_of.
          line.remove_suffix(1);      // Skip spaces at the end.
        };

        // Benchmarked to be faster than iterating over the digits one by one.
        uint16_t value = 0;
        convert_single_int<uint16_t>(line, [&](uint16_t e) { value = e; });

        result = (op == '*') ? (result * value) : (result + value);
      }

      total += result;
    }

    return total;
  }

  uint64_t day_t<6>::solve(part_t<1>, version_t<1>, simd_string_view_t input) {
    // To avoid having to search in each line where numbers begin/end, first split into lines.
    // We can then use the location of each operator to determine where numbers are.
    std::vector<simd_string_view_t> lines;
    lines.reserve(5);
    split(input, [&](simd_string_view_t const & line) { lines.push_back(line); }, '\n');

    size_t const max_line_length = std::accumulate(
        lines.begin(), lines.end(), 0U,
        [](size_t max, simd_string_view_t line) { return std::max(max, line.size()); });
    auto operators_line = lines.back();
    size_t operator_offset = 0;

    uint32_t const num_spaces = std::ranges::count(operators_line, ' ');
    uint32_t const num_problems = operators_line.size() - num_spaces;
    size_t const num_numbers = lines.size() - 1;
    SPDLOG_DEBUG("num problems: {}", num_problems);

    uint64_t total = 0;

    // Benchmarked to be faster than splitting the operators first, and then directly accessing each
    // number based on the pre-computed offsets.
    for (size_t problem_idx = 0; problem_idx < num_problems; ++problem_idx) {
      // Find next operator, and if there's none, place its offset past the line end.
      auto const next_operator_it =
          std::find_if(operators_line.begin() + (operator_offset + 1), operators_line.end(),
                       [](char ch) { return ch != ' '; });
      size_t const next_operator_offset =
          (next_operator_it != operators_line.end())
              ? std::distance(operators_line.begin(), next_operator_it)
              : max_line_length + 1;

      size_t const num_digits = next_operator_offset - operator_offset - 1;
      char const op = operators_line[operator_offset];

      SPDLOG_DEBUG("problem: {}, num digits: {}", problem_idx, num_digits);
      uint64_t result = (op == '*') ? 1 : 0;

      for (size_t number_idx = 0; number_idx < num_numbers; ++number_idx) {
        auto line = lines.at(number_idx);
        size_t const number_begin = operator_offset;
        line = line.substr(number_begin, num_digits);  // Requesting a substr past the end is fine.

        // Eliminate spaces from the number. They can be either at the start or the end, not both.
        // NOTE: We don't care about space at the beginning. When converting the number, we do a
        // saturating subtraction by '0', so spaces become 0 (since their ASCII value is 32, and '0'
        // is 48).
        while (line.back() == ' ') {  // NOTE: Slightly faster than find_last_not_of.
          line.remove_suffix(1);      // Skip spaces at the end.
        };

        // Benchmarked to be faster than iterating over the digits one by one.
        uint16_t value = 0;
        convert_single_int<uint16_t>(line, [&](uint16_t e) { value = e; });
        result = (op == '*') ? (result * value) : (result + value);
      }

      total += result;
      operator_offset = next_operator_offset;
    }

    return total;
  }

  uint64_t day_t<6>::solve(part_t<2>, version_t<0>, simd_string_view_t input) {
    // We need to know the maximum length of the lines to be able to extract the numbers
    // correctly. The easiest way is to just find all lines.
    std::vector<simd_string_view_t> lines;
    lines.reserve(5);
    split(input, [&](simd_string_view_t const & line) { lines.push_back(line); }, '\n');

    size_t const max_line_length = std::accumulate(
        lines.begin(), lines.end(), 0U,
        [](size_t max, simd_string_view_t line) { return std::max(max, line.size()); });
    auto operators_line = lines.back();
    size_t operator_offset = 0;

    uint32_t const num_spaces = std::ranges::count(operators_line, ' ');
    uint32_t const num_problems = operators_line.size() - num_spaces;
    size_t const num_digits = lines.size() - 1;
    SPDLOG_DEBUG("num problems: {}", num_problems);

    uint64_t total = 0;

    for (size_t problem_idx = 0; problem_idx < num_problems; ++problem_idx) {
      // Find next operator, and if there's none, place its offset past the line end.
      auto const next_operator_it =
          std::find_if(operators_line.begin() + (operator_offset + 1), operators_line.end(),
                       [](char ch) { return ch != ' '; });
      size_t const next_operator_offset =
          (next_operator_it != operators_line.end())
              ? std::distance(operators_line.begin(), next_operator_it)
              : max_line_length + 1;

      size_t const num_numbers = next_operator_offset - operator_offset - 1;
      char const op = operators_line[operator_offset];

      uint64_t result = (op == '*') ? 1 : 0;

      // Grab one digit per line, starting from the top, ignoring spaces.
      // Note that not all lines might be the same length, so check bounds.
      for (size_t number_idx = 0; number_idx < num_numbers; ++number_idx) {
        size_t const char_offset = operator_offset + number_idx;
        uint16_t value = 0;

        for (size_t digit_idx = 0; digit_idx < num_digits; ++digit_idx) {
          auto const & line = lines.at(digit_idx);
          if ((char_offset < line.size()) && (line[char_offset] != ' ')) {
            value = 10 * value + (line[char_offset] - '0');
          }
        }

        result = (op == '*') ? (result * value) : (result + value);
      }

      SPDLOG_DEBUG("problem: {}, result: {}", problem_idx, result);

      total += result;
      operator_offset = next_operator_offset;
    }

    return total;
  }

}  // namespace aoc25
