#include "aoc25/day_06.hpp"

#include "aoc25/day.hpp"
#include "aoc25/simd.hpp"
#include "aoc25/string.hpp"

#include <fmt/ranges.h>
#include <spdlog/spdlog.h>

#include <cstdint>
#include <numeric>

namespace aoc25 {
  namespace {

    std::vector<simd_string_view_t> split_into_lines(simd_string_view_t input) {
      std::vector<simd_string_view_t> result;
      result.reserve(5);
      split(input, [&](auto const & line) { result.push_back(line); }, '\n');
      return result;
    };

    std::vector<size_t> calculate_operator_offsets(simd_string_view_t operators_line,
                                                   size_t max_line_length,
                                                   size_t num_problems) {
      std::vector<size_t> result;
      result.reserve(num_problems);

      size_t operator_offset = 0;

      for (size_t problem_idx = 0; problem_idx < num_problems; ++problem_idx) {
        // Find next operator, and if there's none, place its offset past the line end.
        auto const next_operator_it =
            std::find_if(operators_line.begin() + (operator_offset + 1), operators_line.end(),
                         [](char ch) { return ch != ' '; });
        size_t const next_operator_offset =
            (next_operator_it != operators_line.end())
                ? std::distance(operators_line.begin(), next_operator_it)
                : max_line_length + 1;
        result.push_back(operator_offset);
        operator_offset = next_operator_offset;
      }
      result.push_back(operator_offset);  // Sentinel value.

      return result;
    }

  }  // namespace

  /* TODO: Ideas:
   * - Just create a LUT for the numbers, they're at most 4 digits. Try to load groups of 4
   *   characters, if a digit is unused, default to e.g. '0'. Then subtract '0' from all, and
   *   compress the 4 * 8 = 32 bits into a 4 * 4 = 16 bit number (each byte is ['0','9'] or a
   *   tombstone marker). Use these 4 values as an index for a gather from a LUT of precomputed
   *   values. This LUT should easily fit into L2 cache (2^16 * 16 bit = 128 KiB).
   *
   * - Once we detect that we're on the last line (with the operators), stop parsing. Then execute
   *   an SIMD loop where we match against non-spaces and Compress the result into an array.
   *
   * - We can also not parse the numbers initially. Instead search for the operators row. Then from
   *   there, we know how many "number" lines there are. This allows us to calculate offsets for
   *   each digit given that we know the operator's offset.
   */
  uint64_t day_t<6>::solve(part_t<1>, version_t<0>, simd_string_view_t input) {
    // To avoid having to search in each line where numbers begin/end, first split into lines.
    // We can then use the location of each operator to determine where numbers are.
    auto const lines = split_into_lines(input);
    size_t const max_line_length =
        std::ranges::max_element(lines, {}, [](auto const & e) { return e.size(); })->size();
    auto const operators_line = lines.back();

    uint32_t const num_spaces = aoc25::count(operators_line.as_span(), ' ');
    uint32_t const num_problems = operators_line.size() - num_spaces;
    size_t const num_numbers = lines.size() - 1;
    SPDLOG_DEBUG("num problems: {}", num_problems);

    // Detect location of all operators first to allow processing multiple problems in parallel.
    auto const operator_offsets =
        calculate_operator_offsets(operators_line, max_line_length, num_problems);

    uint64_t total = 0;

#pragma omp parallel for reduction(+ : total) schedule(static) num_threads(4)
    for (size_t problem_idx = 0; problem_idx < num_problems; ++problem_idx) {
      size_t const operator_offset = operator_offsets[problem_idx];
      size_t const num_digits = operator_offsets[problem_idx + 1] - operator_offset - 1;
      char const op = operators_line[operator_offset];

      SPDLOG_DEBUG("problem: {}, num digits: {}", problem_idx, num_digits);
      uint64_t result = (op == '*') ? 1 : 0;

      for (size_t number_idx = 0; number_idx < num_numbers; ++number_idx) {
        auto line = lines.at(number_idx);
        line =
            line.substr(operator_offset, num_digits);  // Requesting a substr past the end is fine.

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

  uint64_t day_t<6>::solve(part_t<2>, version_t<0>, simd_string_view_t input) {
    // We need to know the maximum length of the lines to be able to extract the numbers
    // correctly. The easiest way is to just find all lines.
    auto const lines = split_into_lines(input);
    size_t const max_line_length =
        std::ranges::max_element(lines, {}, [](auto const & e) { return e.size(); })->size();
    auto const operators_line = lines.back();

    uint32_t const num_spaces = aoc25::count(operators_line.as_span(), ' ');
    uint32_t const num_problems = operators_line.size() - num_spaces;
    size_t const num_digits = lines.size() - 1;
    SPDLOG_DEBUG("num problems: {}", num_problems);

    auto const operator_offsets =
        calculate_operator_offsets(operators_line, max_line_length, num_problems);
    uint64_t total = 0;

#pragma omp parallel for reduction(+ : total) schedule(static) num_threads(4)
    for (size_t problem_idx = 0; problem_idx < num_problems; ++problem_idx) {
      size_t const operator_offset = operator_offsets[problem_idx];
      size_t const num_numbers = operator_offsets[problem_idx + 1] - operator_offset - 1;
      char const op = operators_line[operator_offset];

      uint64_t result = (op == '*') ? 1 : 0;

      // Grab one digit per line, starting from the top, ignoring spaces.
      // Note that not all lines might be the same length, so check bounds.
      for (size_t number_idx = 0; number_idx < num_numbers; ++number_idx) {
        size_t const char_offset = operator_offset + number_idx;
        uint16_t value = 0;

        for (size_t digit_idx = 0; digit_idx < num_digits; ++digit_idx) {
          auto const & line = lines.at(digit_idx);
          char const ascii = (char_offset < line.size()) ? line[char_offset] : ' ';
          bool const is_digit = (ascii != ' ');

          value = is_digit ? (10 * value + (ascii - '0')) : value;
        }

        result = (op == '*') ? (result * value) : (result + value);
      }

      SPDLOG_DEBUG("problem: {}, result: {}", problem_idx, result);

      total += result;
    }

    return total;
  }

}  // namespace aoc25
