#include "aoc25/day_07.hpp"

#include "aoc25/day.hpp"
#include "aoc25/simd.hpp"

#include <fmt/ranges.h>
#include <fmt/std.h>
#include <spdlog/spdlog.h>

#include <cstddef>
#include <cstdint>
#include <numeric>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "src/day_07.cpp"

// clang-format off
#include <hwy/foreach_target.h>
// clang-format on

#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();

namespace aoc25 {
  namespace HWY_NAMESPACE {

    namespace hn = hwy::HWY_NAMESPACE;

    using char_tag_t = hn::ScalableTag<uint8_t>;
    using bool_tag_t = hn::ScalableTag<uint64_t>;

    using char_vec_t = hn::VFromD<char_tag_t>;
    using bool_vec_t = hn::VFromD<bool_tag_t>;

    bool_vec_t convert_line_to_splitter_word(simd_string_view_t line) {
      static constexpr size_t max_line_length = 160;
      static constexpr size_t bits_in_vec =
          8 * sizeof(hn::TFromV<bool_vec_t>) * hn::Lanes(bool_tag_t{});
      static_assert(bits_in_vec >= max_line_length);
      assert(line.size() <= bits_in_vec);  // Ensure line fits into a single vector.

      auto const * data = reinterpret_cast<uint8_t const *>(line.data());
      auto const splitters = hn::Set(char_tag_t{}, '^');

      // Setup storage for bit-packed masks. Note that for problem 1, the array does not need to be
      // intialized, since we don't care about data in unused parts, even if it ends up in the
      // splitter mask. This is not an issue, since the split beam can never reach these positions.
      // However, for part 2, we want to avoid "false detection" of splitters in unused parts. So we
      // do initialize the array to zero.
      static constexpr size_t chars_per_vec = hn::Lanes(char_tag_t{});
      static_assert(chars_per_vec >= 16 && chars_per_vec <= 64,
                    "Implementation supports 16, 32, or 64 character vectors");
      using mask_storage_t =
          std::conditional_t<chars_per_vec == 16, uint16_t,
                             std::conditional_t<chars_per_vec == 32, uint32_t, uint64_t>>;
      using mask_tag_t = hn::ScalableTag<mask_storage_t>;
      static constexpr size_t num_mask_storage_entries =
          std::max(hn::Lanes(hn::ScalableTag<mask_storage_t>{}),
                   (max_line_length + chars_per_vec - 1) / chars_per_vec);

      HWY_ALIGN auto mask_data = std::array<mask_storage_t, num_mask_storage_entries>{};
      size_t mask_data_idx = 0;

      // Load all characters in the line, detect where there are splitters, and store
      // the resulting bit-packed masks into an array.
      for (size_t col = 0; col < line.size(); col += hn::Lanes(char_tag_t{})) {
        auto const entries = hn::LoadU(char_tag_t{}, data + col);
        auto const splitter_mask = hn::Eq(entries, splitters);
        mask_data[mask_data_idx++] = hn::BitsFromMask(char_tag_t{}, splitter_mask);
      }

      return hn::BitCast(bool_tag_t{}, hn::Load(mask_tag_t{}, mask_data.data()));
    }

    /** @brief Bitshifts a whole vector to the left as a single element. */
    template <size_t Shift, class D>
    hn::VFromD<D> unified_shift_left(D tag, hn::VFromD<D> const & vec) {
      static_assert(Shift < 64, "Shift value must be less than 64");
      static_assert(std::is_integral_v<hn::TFromD<D>>,
                    "Template parameter D must be of integral type");

      static constexpr hn::ScalableTag<uint64_t> shift_tag;
      static constexpr size_t input_size = sizeof(hn::TFromD<D>) * hn::Lanes(tag);
      static constexpr size_t shift_size =
          sizeof(hn::TFromD<decltype(shift_tag)>) * hn::Lanes(shift_tag);
      static_assert(input_size == shift_size);

      // Work with 64-bit elements.
      auto const vec_u64 = hn::BitCast(shift_tag, vec);

      // Shift everything to the left.
      auto const lanes_left = hn::ShiftLeft<Shift>(vec_u64);

      // Isolate the "carry" bits and move them 1 word up.
      auto const carry_shift = hn::ShiftRight<64 - Shift>(vec_u64);
      auto const carries_word_shifted = hn::Slide1Up(shift_tag, carry_shift);

      // Combine the shift elements with the carries.
      auto const combined_shift = hn::Or(lanes_left, carries_word_shifted);

      // Cast back to the original type.
      return hn::BitCast(tag, combined_shift);
    }

    /** @brief Bitshifts a whole vector to the right as a single element. */
    template <size_t Shift, class D>
    hn::VFromD<D> unified_shift_right(D tag, hn::VFromD<D> const & vec) {
      static_assert(Shift < 64, "Shift value must be less than 64");
      static_assert(std::is_integral_v<hn::TFromD<D>>,
                    "Template parameter D must be of integral type");

      static constexpr hn::ScalableTag<uint64_t> shift_tag;
      static constexpr size_t input_size = sizeof(hn::TFromD<D>) * hn::Lanes(tag);
      static constexpr size_t shift_size =
          sizeof(hn::TFromD<decltype(shift_tag)>) * hn::Lanes(shift_tag);
      static_assert(input_size == shift_size);

      // Work with 64-bit elements.
      auto const vec_u64 = hn::BitCast(shift_tag, vec);

      // Shift everything to the right.
      auto const lanes_right = hn::ShiftRight<Shift>(vec_u64);

      // Isolate the "carry" bits and move them 1 word down.
      auto const carry_shift = hn::ShiftLeft<64 - Shift>(vec_u64);
      auto const carries_word_shifted = hn::Slide1Down(shift_tag, carry_shift);

      // Combine the shift elements with the carries.
      auto const combined_shift = hn::Or(lanes_right, carries_word_shifted);

      // Cast back to the original type.
      return hn::BitCast(tag, combined_shift);
    }

    uint16_t count_beam_splits(simd_string_view_t lines, size_t line_width, size_t start_pos) {
      // Prepare initial beam position.
      static constexpr size_t bits_per_lane = 8 * sizeof(hn::TFromD<bool_tag_t>);
      HWY_ALIGN std::array<hn::TFromD<bool_tag_t>, hn::Lanes(bool_tag_t{})> start_beam = {};
      start_beam[start_pos / bits_per_lane] = static_cast<hn::TFromD<bool_tag_t>>(1)
                                           << (start_pos % bits_per_lane);
      auto beam_positions = hn::Load(bool_tag_t{}, start_beam.data());

      // Loop over every odd line in the input (since even lines are empty).
      size_t const height = lines.size() / (line_width + 1);  // +1 for newline character.
      uint16_t num_splits = 0;

      for (size_t row = 2; row < height; row += 2) {
        auto line = lines.substr(row * (line_width + 1), line_width);  // +1 for newline character.
        assert(line[0] == '.' && line[line_width - 1] == '.');         // Borders are always empty.

        // Convert line to beam mask.
        SPDLOG_DEBUG("Processing line {:3d}: {:s}", row, line);
        auto const splitters = convert_line_to_splitter_word(line);

        // Determine beam splits.
        auto const hit_splitters = hn::And(splitters, beam_positions);
        auto const passing_beams = hn::AndNot(splitters, beam_positions);

        auto const left_split_beams = unified_shift_left<1>(bool_tag_t{}, hit_splitters);
        auto const right_split_beams = unified_shift_right<1>(bool_tag_t{}, hit_splitters);

        num_splits += hn::ReduceSum(bool_tag_t{}, hn::PopulationCount(hit_splitters));

        auto const new_beam_positions =
            hn::Or(hn::Or(left_split_beams, right_split_beams), passing_beams);
        beam_positions = std::move(new_beam_positions);

        SPDLOG_DEBUG("# hits: {}, # passing: {}, # beams: {}, # splits: {}",
                     hn::ReduceSum(bool_tag_t{}, hn::PopulationCount(hit_splitters)),
                     hn::ReduceSum(bool_tag_t{}, hn::PopulationCount(passing_beams)),
                     hn::ReduceSum(bool_tag_t{}, hn::PopulationCount(new_beam_positions)),
                     num_splits);
      }

      return num_splits;
    }

  }  // namespace HWY_NAMESPACE
}  // namespace aoc25

HWY_AFTER_NAMESPACE();

#ifdef HWY_ONCE

namespace aoc25 {
  namespace {

    uint16_t count_beam_splits(simd_string_view_t line, size_t line_width, size_t start_pos) {
      HWY_EXPORT(count_beam_splits);
      return HWY_DYNAMIC_DISPATCH(count_beam_splits)(line, line_width, start_pos);
    }

  }  // namespace

  uint64_t day_t<7>::solve(part_t<1>, version_t<0>, simd_string_view_t input) {
    // Find beam start point (it's probably the middle, but let's be safe).
    uint8_t const start_offset = input.find('S');

    // Input is a matrix, so only detect width once, instead of searching for newlines.
    uint8_t const width = input.find('\n', start_offset);  // Start from the beam start point.
    [[maybe_unused]] uint8_t const height =
        input.size() / (width + 1);  // +1 for newline character.
    SPDLOG_DEBUG("start: {}, width: {}, height: {}", start_offset, width, height);

    return count_beam_splits(input, width, start_offset);
  }

  uint64_t day_t<7>::solve(part_t<2>, version_t<0>, simd_string_view_t input) {
    // Find beam start point (it's probably the middle, but let's be safe).
    uint8_t const start_offset = input.find('S');

    // Input is a matrix, so only detect width once, instead of searching for newlines.
    uint8_t const width = input.find('\n', start_offset);  // Start from the beam start point.
    uint8_t const height = input.size() / (width + 1);     // +1 for newline character.
    SPDLOG_DEBUG("start: {}, width: {}, height: {}", start_offset, width, height);

    // Keep track of how many ways a beam can arrive at a given position. To avoid needing to add
    // special logic for the edges, add one extra position on the left side. This means that all
    // indexing into this vector needs to be offset by 1. We work around it by accessing an offset
    // pointer into the data.
    auto beam_arrival_counts = std::vector<uint64_t>(width + 1);
    auto * const beam_arrival_counts_ptr = beam_arrival_counts.data() + 1;
    beam_arrival_counts_ptr[start_offset] = 1;

    // Keep track of addition for next position separately. This way, we don't overwrite data for
    // the next position before we have processed it. This then allows us to avoid using a second
    // array to keep track of updated counts.
    uint64_t beam_count_for_next_pos = 0;

    // Every even line is empty, so skip them.
    for (uint8_t row = 2; row < height; row += 2) {
      [[maybe_unused]] auto const prev_line = input.substr((row - 1) * (width + 1), width);
      assert(prev_line.find('^') == std::string_view::npos);  // Previous line has no splitters.

      auto line = input.substr(row * (width + 1), width);  // +1 for newline character.
      assert(line[0] == '.' && line[width - 1] == '.');    // Borders are always empty.
      SPDLOG_DEBUG("Processing line {:3d}: {:s}", row, line);

      // Check if any beams hit a splitter on this line.
      for (size_t pos = 0; pos < width; ++pos) {
        // No branches, it's faster to do no-op additions/multiplications instead.
        uint64_t const arrival_count = beam_arrival_counts_ptr[pos];
        bool const is_splitter = (line[pos] == '^');

        beam_arrival_counts_ptr[pos] += beam_count_for_next_pos;
        beam_count_for_next_pos = is_splitter ? arrival_count : 0;

        beam_arrival_counts_ptr[pos - 1] += is_splitter * arrival_count;
        beam_arrival_counts_ptr[pos] -= is_splitter * arrival_count;
      }

      SPDLOG_DEBUG("  Beam arrival counts: {::d}", std::span{beam_arrival_counts}.subspan(1));
    }

    // Don't count padding column on the left.
    return std::accumulate(beam_arrival_counts.begin() + 1, beam_arrival_counts.end(), 0ULL);
  }

}  // namespace aoc25

#endif  // HWY_ONCE
