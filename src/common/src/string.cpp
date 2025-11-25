#include "aoc25/string.hpp"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "src/string.cpp"

// clang-format off
#include <hwy/foreach_target.h>
// clang-format on

#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();

namespace aoc25 {
  namespace HWY_NAMESPACE {

    namespace hn = hwy::HWY_NAMESPACE;

    std::span<simd_aligned_span_t<char const>> split_lines(
        simd_aligned_span_t<char const> input,
        std::span<simd_aligned_span_t<char const>> output,
        char splitter) {
      static constexpr hn::ScalableTag<uint8_t> tag{};
      static constexpr size_t lane_size = hn::Lanes(tag);

      uint8_t const* HWY_RESTRICT data = reinterpret_cast<uint8_t const*>(input.data());
      auto const splitters = hn::Set(tag, splitter);
      size_t line_start = 0;
      size_t line_idx = 0;

      for (size_t idx = 0; idx < input.size(); idx += lane_size) {
        auto vec = hn::Load(tag, &data[idx]);
        auto const eq = hn::Eq(vec, splitters);
        uint64_t bits = hn::BitsFromMask(tag, eq);

        // NOTE: This is faster than Iota/CompressStore to get all wanted line offsets.
        // TODO: Store a few of these, and process them in chunks of multiple uint64_t.
        for (; bits != 0; bits &= (bits - 1)) {
          size_t const bit_pos = std::countr_zero(bits);
          size_t const line_length = idx + bit_pos - line_start;

          assert(line_start + line_length <= input.size());
          assert(line_idx < output.size());

          output[line_idx++] = simd_aligned_span_t<char const>(
              reinterpret_cast<char const*>(&data[line_start]), line_length);
          line_start += line_length + 1;  // +1 to skip the splitter.
        }
      }

      if (line_start < input.size()) {  // Handle last line if not ending with a splitter.
        size_t const line_length = input.size() - line_start;
        assert(line_idx < output.size());
        output[line_idx++] = simd_aligned_span_t<char const>(
            reinterpret_cast<char const*>(&data[line_start]), line_length);
      }

      return output.first(line_idx);
    }

  }  // namespace HWY_NAMESPACE
}  // namespace aoc25

HWY_AFTER_NAMESPACE();

#ifdef HWY_ONCE

namespace aoc25 {
  HWY_EXPORT(split_lines);

  HWY_DLLEXPORT std::span<simd_aligned_span_t<char const>> split_lines(
      simd_aligned_span_t<char const> input,
      std::span<simd_aligned_span_t<char const>> output,
      char splitter) {
    return HWY_DYNAMIC_DISPATCH(split_lines)(input, output, splitter);
  }

}  // namespace aoc25

#endif