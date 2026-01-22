#include "aoc25/day_03.hpp"

#include "aoc25/algorithm.hpp"
#include "aoc25/simd.hpp"
#include "aoc25/string.hpp"

#include <fmt/ranges.h>
#include <spdlog/spdlog.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>

namespace aoc25 {

  uint32_t day_t<3>::solve(part_t<1>, version_t<0>, simd_string_view_t input) {
    size_t const line_length = input.find('\n');                // All lines have the same length.
    size_t const num_lines = input.size() / (line_length + 1);  // +1 for newlines.

    uint32_t sum = 0;

#pragma omp parallel for reduction(+ : sum) schedule(static)
    for (size_t line_idx = 0; line_idx < num_lines; ++line_idx) {
      auto const line = input.substr(line_idx * (line_length + 1), line_length);  // Skip newlines.

      // The first digit is the first largest in the range [0, N - 1). The second one is the
      // largest in the remaining range. This could also be done in a single pass, but it's faster
      // to do two passes with SIMD.
      size_t const first_digit_pos = find_maximum_pos(line.substr(0, line.size() - 1).as_span());
      char const second_digit = find_maximum(line.substr(first_digit_pos + 1).as_span());

      std::array<char, 2> const max_digits = {line[first_digit_pos], second_digit};
      SPDLOG_DEBUG("line: {:s}, max_digits: {}", line, max_digits);
      sum += 10 * (max_digits[0] - '0') + (max_digits[1] - '0');
    }

    return sum;
  }

  uint64_t day_t<3>::solve(part_t<2>, version_t<0>, simd_string_view_t input) {
    size_t const line_length = input.find('\n');                // All lines have the same length.
    size_t const num_lines = input.size() / (line_length + 1);  // +1 for newlines.

    static constexpr uint8_t digits_to_use = 12;

    uint64_t sum = 0;

#pragma omp parallel
    {
      // Note: Faster than an uninitialized array (i.e. std::make_unique_for_overwrite).
      auto stack = std::vector<char>(line_length);
      stack[0] = '0' - 1;  // Sentinel value.

#pragma omp for reduction(+ : sum) schedule(static)
      for (size_t line_idx = 0; line_idx < num_lines; ++line_idx) {
        auto const line =
            input.substr(line_idx * (line_length + 1), line_length);  // Skip newlines.

        // If digit i in the final joltage uses the value at position x, then digit i + 1
        // should use the maximum value in the range [x + 1, N - (D - (i + 1)) ), where N is the
        // length of the input and D is the total number of digits to use. So we walk backwards over
        // the input, keeping a stack of maximum values, leaving D digits at the end to process
        // afterwards. This gives us a list of candidate values for the first Y digits, where Y is
        // the length of the stack (the length of which will depend on the input).
        int input_pos = line_length - digits_to_use - 1;
        int stack_size = 2;

        stack[1] = line[input_pos];  // Last element is always larger than nothing.

        for (; input_pos-- > 0;) {
          auto const value = line[input_pos];
          // Much faster than always writing and using a ternary operator for stack_size increment.
          // Probably because this branch can be pretty well predicted.
          if (value >= stack[stack_size - 1]) {
            stack[stack_size++] = value;
          }
        }

        // For each digit use either the largest value on the stack (which is the one at the top),
        // or the largest available value in the (so far) unused remainder of the input. To keep
        // track of this unused remainer, we create queue. If at any point, the value used for the
        // current digit comes from this queue, then the original stack is discarded, because using
        // it would be equivalent to using values from earlier positions in the input, which is not
        // allowed.
        std::array<char, digits_to_use + 1> remainder_queue{};
        remainder_queue[0] = '9' + 1;  // Sentinel value.

        int remainder_queue_size = 1;
        int remainder_queue_pos = 1;  // Don't actually use the sentinel.

        input_pos = line_length - digits_to_use;
        SPDLOG_DEBUG("  remaining line: {}", line.substr(input_pos));

        uint64_t jolts = 0;

        for (uint8_t digit_idx = 0; digit_idx < digits_to_use; ++digit_idx) {
          // Push next input value onto remainder queue. Before pushing it on, pop off all smaller
          // values, since they will never be used. Note that the remainder_queue_pos always points
          // to the element to be used next, so we shouldn't pop off elements in front of it.
          assert(static_cast<size_t>(input_pos) < line_length);
          char const next_input_value = line[input_pos + digit_idx];
          while ((remainder_queue_size > remainder_queue_pos) &&
                 (remainder_queue[remainder_queue_size - 1] < next_input_value)) {
            --remainder_queue_size;
          }
          remainder_queue[remainder_queue_size++] = next_input_value;

          char const stack_value = stack[stack_size - 1];
          bool const use_remainder_value = remainder_queue[remainder_queue_pos] > stack_value;
          char const digit =
              use_remainder_value ? remainder_queue[remainder_queue_pos] : stack_value;

          SPDLOG_DEBUG(
              "  # digit: {}, next input: {}, stack: {}, remainder queue: {}, use_stack: {}, jolt "
              "digit: {}",
              digit_idx, next_input_value,
              std::span{stack.data() + 1, static_cast<size_t>(stack_size - 1)},
              std::span{remainder_queue.data() + remainder_queue_pos,
                        static_cast<size_t>(remainder_queue_size - remainder_queue_pos)},
              !use_remainder_value, digit);

          // If a value from the remainder queue is unused, the stack size goes to 1. That position
          // holds a sentinel value lower than any input, so the stack will never be used again.
          remainder_queue_pos += use_remainder_value;
          stack_size = use_remainder_value ? 1 : (stack_size - 1);

          jolts *= 10;
          jolts += digit - '0';
        }

        SPDLOG_DEBUG("line: {:s}, jolts: {}", line, jolts);
        sum += jolts;
      }
    }

    return sum;
  }

}  // namespace aoc25
