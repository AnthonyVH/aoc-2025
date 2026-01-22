#include "aoc25/day_12.hpp"

#include "aoc25/day.hpp"
#include "aoc25/simd.hpp"
#include "aoc25/string.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <spdlog/spdlog.h>
#include <sys/types.h>

#include <array>
#include <cstdint>
#include <stdexcept>

namespace aoc25 {
  namespace {

    // Each line in the problem has the format: "WWxHH: AA BB CC ... XX", where WW is width, HH is
    // height, and (AA, BB, ..., XX) are the number of presents of a given index to be placed inside
    // the region. Note that there's a newline at the end.
    constexpr size_t problem_line_length(size_t num_presents) {
      return 2 + 1 + 2 + 2 + (num_presents * (2 + 1));
    }

  }  // namespace
}  // namespace aoc25

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

      uint8_t num_hash_occurences(simd_string_view_t input) {
        static constexpr auto tag = hn::Full128<uint8_t>{};

        uint8_t const * data = reinterpret_cast<uint8_t const *>(input.data());
        auto const bytes = hn::LoadU(tag, data);
        auto const equals = hn::Eq(bytes, hn::Set(tag, static_cast<uint8_t>('#')));

        return hn::CountTrue(tag, equals);
      }

      uint16_t parse_and_decide_if_fit(simd_string_view_t input,
                                       simd_span_t<uint16_t const> present_volumes) {
        assert(present_volumes.size() <= 6);
        size_t const line_length = problem_line_length(present_volumes.size());
        size_t const num_lines = input.size() / line_length;
        assert(num_lines * line_length == input.size());

        static constexpr auto in_tag = hn::Full256<uint8_t>{};
        static constexpr auto calc_tag = hn::Full128<int16_t>{};

        [[maybe_unused]] static constexpr size_t in_lanes = hn::Lanes(in_tag);
        [[maybe_unused]] static constexpr size_t calc_lanes = hn::Lanes(calc_tag);

        // We need 3 bytes per present count (2 digits + space), minus 1 space for the last present.
        size_t const num_presents = present_volumes.size();
        size_t const num_unpacked_digit_lanes = num_presents * 3 - 1;
        assert(in_lanes >= num_unpacked_digit_lanes);
        assert(calc_lanes >= num_presents);

        uint8_t const * HWY_RESTRICT data = reinterpret_cast<uint8_t const *>(input.data());
        auto const present_volume = hn::LoadN(
            calc_tag, reinterpret_cast<int16_t const *>(present_volumes.data()), num_presents);

        // The mask of where digits characters are, as well as the multipliers to convert from
        // characters to integers are the same for each line, and can be precomputed.

        // The loaded "present counts" string looks like: "AA AA AA AA AA ...". So we have two
        // digits followed by a space, i.e. 0b110 for each digit group. We need to repeat this at
        // most (in_lanes / 3) times. Note that Highway requires an array of at least 8 elements to
        // load a mask from.
        static constexpr auto bytes_for_mask = std::max<size_t>(8, (in_lanes + 8 - 1) / 8);
        static constexpr uint32_t digits_bit_mask_multiplier =
            0b001'001'001'001'001'001'001'001'001'001;
        static constexpr uint32_t digits_bit_mask_word = 0b011 * digits_bit_mask_multiplier;
        HWY_ALIGN static constexpr std::array<uint8_t, bytes_for_mask> digit_char_pos_mask =
            std::to_array<uint8_t>({
                (digits_bit_mask_word >> 0 * 8) & 0xFF,
                (digits_bit_mask_word >> 1 * 8) & 0xFF,
                (digits_bit_mask_word >> 2 * 8) & 0xFF,
                (digits_bit_mask_word >> 3 * 8) & 0xFF,
                0,
                0,
                0,
                0,
            });
        auto const is_digit_pos = hn::And(hn::FirstN(in_tag, num_unpacked_digit_lanes),
                                          hn::LoadMaskBits(in_tag, digit_char_pos_mask.data()));

        HWY_ALIGN static constexpr std::array<uint8_t, hn::Lanes(in_tag)> digit_value_per_lane =
            std::to_array<uint8_t>({10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1,
                                    10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1});
        auto const digit_multipliers = hn::Load(
            hn::Full256<int8_t>{}, reinterpret_cast<int8_t const *>(digit_value_per_lane.data()));

        uint16_t presents_fit = 0;

        for (size_t line_idx = 0; line_idx < num_lines; ++line_idx) {
          uint8_t const * const line_ptr = data + line_idx * line_length;

          // First parse width and height.
          uint16_t const width = (line_ptr[0] - '0') * 10 + (line_ptr[1] - '0');
          uint16_t const height = (line_ptr[3] - '0') * 10 + (line_ptr[4] - '0');
          uint16_t const area = width * height;

          // Load characters representing all present counts. Then detect where the actual digits
          // are, and compress them to the front of the vector.
          auto const unpacked_chars = hn::LoadU(in_tag, line_ptr + 7);
          auto const packed_chars = hn::Compress(unpacked_chars, is_digit_pos);

          // Convert to 16-bit integers. Note that these are signed, because there's no proper
          // unsigned 16-bit support in AVX2.
          auto const zero_char = hn::Set(in_tag, '0');
          auto const packed_digits = hn::Sub(packed_chars, zero_char);

          hn::Vec<decltype(calc_tag)> const present_counts = hn::LowerHalf(
              hn::SatWidenMulPairwiseAdd(hn::Full256<int16_t>{}, packed_digits, digit_multipliers));

          // Now calculate the total volume required for these presents.
          int16_t const required_volume =
              hn::ReduceSum(calc_tag, hn::Mul(present_counts, present_volume));
          presents_fit += (required_volume <= area) ? 1 : 0;
        }

        return presents_fit;
      }

    }  // namespace HWY_NAMESPACE
  }  // namespace
}  // namespace aoc25

HWY_AFTER_NAMESPACE();

#ifdef HWY_ONCE

namespace aoc25 {
  namespace {

    uint8_t num_hash_occurences(simd_string_view_t input) {
      HWY_EXPORT(num_hash_occurences);
      return HWY_DYNAMIC_DISPATCH(num_hash_occurences)(input);
    }

    uint16_t parse_and_decide_if_fit(simd_string_view_t input,
                                     simd_span_t<uint16_t const> present_volumes) {
      HWY_EXPORT(parse_and_decide_if_fit);
      return HWY_DYNAMIC_DISPATCH(parse_and_decide_if_fit)(input, present_volumes);
    }

  }  // namespace

  uint64_t day_t<12>::solve(part_t<1>, version_t<0>, simd_string_view_t input) {
    // Count number of '#' for each input. We know each present is 3x3, so it takes 5 lines to
    // describe it.
    auto present_volumes = simd_vector_t<uint16_t>{};
    present_volumes.reserve(6);

    /* Total number of symbols per present description is:
     * - 1 digit for index, 1 colon, 1 newline
     * - 3 lines of 3 symbols + newline
     * - 1 extra newline
     */
    static constexpr size_t present_desc_length = 1 + 1 + 1 + (3 * (3 + 1)) + 1;

    auto are_next_lines_a_present = [&]() -> bool {
      // If the next lines describe a present, the second character must be a colon.
      static constexpr size_t next_present_check_offset = 1;
      return input.at(next_present_check_offset) == ':';
    };

    while (are_next_lines_a_present()) {
      size_t const present_volume = num_hash_occurences(input);
      present_volumes.push_back(present_volume);
      input.remove_prefix(present_desc_length);  // Jump to the next present.
    }
    SPDLOG_DEBUG("Parsed present volumes: {}", present_volumes);

    // Now parse regions and decide whether requested presents fit the given area.
    size_t const num_lines = aoc25::count(input.as_span(), '\n');
    size_t const line_length = problem_line_length(present_volumes.size());
    size_t const expected_input_length = num_lines * line_length;

    SPDLOG_DEBUG("# lines: {}, chars per line: {}, # expected chars: {}, input size: {}", num_lines,
                 line_length, expected_input_length, input.size());

    if (expected_input_length != input.size()) {
      // Exit on mismatch, i.e. for example. No big deal, since the code can't solve it anyway.
      throw std::invalid_argument(fmt::format(
          "Unsupported input format: expected {} lines of {} chars, got {} (!= {}) total chars",
          num_lines, line_length, input.size(), expected_input_length));
    }

    // Process in chunks of lines in parallel,
    static constexpr size_t num_threads = 8;
    size_t const lines_per_chunk = (num_lines + num_threads - 1) / num_threads;
    [[maybe_unused]] size_t const num_chunks = (num_lines + lines_per_chunk - 1) / lines_per_chunk;
    assert(num_chunks == num_threads);

    uint64_t result = 0;

#  pragma omp parallel for reduction(+ : result) num_threads(num_threads)
    for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
      size_t const line_start = chunk_idx * lines_per_chunk;
      size_t const line_end = std::min<size_t>(line_start + lines_per_chunk, num_lines);
      size_t const chunk_size = line_end - line_start;

      simd_string_view_t const chunk_input =
          input.substr(line_start * line_length, chunk_size * line_length);
      auto const num_fits_in_chunk = parse_and_decide_if_fit(chunk_input, present_volumes);
      SPDLOG_DEBUG("Chunk {:2d} (lines {}-{}): {} regions fit", chunk_idx, line_start, line_end - 1,
                   num_fits_in_chunk);

      result += num_fits_in_chunk;
    }

    return result;
  }

}  // namespace aoc25

#endif  // HWY_ONCE
