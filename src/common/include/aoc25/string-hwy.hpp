#include "aoc25/simd.hpp"

#if defined(AOC25_STRING_HWY_H_TARGET) == defined(HWY_TARGET_TOGGLE)
#  ifdef AOC25_STRING_HWY_H_TARGET
#    undef AOC25_STRING_HWY_H_TARGET
#  else
#    define AOC25_STRING_HWY_H_TARGET
#  endif

#  include "aoc25/string.hpp"

#  include <fmt/ranges.h>
#  include <spdlog/spdlog.h>

#  include <cassert>
#  include <concepts>
#  include <cstdint>
#  include <string_view>

#  undef HWY_TARGET_INCLUDE
#  define HWY_TARGET_INCLUDE "src/string-hwy.hpp"

// clang-format off
#include <hwy/foreach_target.h>
// clang-format on

#  include <hwy/base.h>
#  include <hwy/highway.h>
#  include <hwy/highway_export.h>

HWY_BEFORE_NAMESPACE();

namespace aoc25 {

  namespace HWY_NAMESPACE {

    namespace hn = hwy::HWY_NAMESPACE;

    template <class Ignore, class Mask>
    concept can_create_mask_from_uint = requires(uint32_t value) {
      { Mask::FromBits(value) } -> std::same_as<Mask>;
    } && std::same_as<void, std::void_t<Ignore>>;

    // These shenanigans are neccessary because without them calling Mask::FromBits() inside an
    // if-constexpr doesn't depend on the template parameter and will cause the compiler to
    // instantiate it even if the branch is not taken. Which in case of AVX2 causes a compile error.
    template <class T, class Mask>
    struct call_mask_from_bits {
      static Mask call(uint32_t mask) { return Mask::FromBits(mask); }
    };

    // Based on: https://lemire.me/blog/2023/09/22/parsing-integers-quickly-with-avx-512/
    inline hn::VFromD<hn::Full128<uint32_t>> parse_8digit_integers_simd_reverse(
        hn::VFromD<hn::Full256<uint8_t>> base10_8bit) {
      HWY_ALIGN static constexpr std::array<int8_t, 32> digit_value_base10e1 =
          std::to_array<int8_t>({10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1,
                                 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1});

      HWY_ALIGN static constexpr std::array<int8_t, 16> digit_value_base10e2 =
          std::to_array<int8_t>({100, 1, 100, 1, 100, 1, 100, 1, 100, 1, 100, 1, 100, 1, 100, 1});

      HWY_ALIGN static constexpr std::array<uint16_t, 8> digit_value_base10e4 =
          std::to_array<uint16_t>({10000, 1, 10000, 1, 10000, 1, 10000, 1});

      // Multiply pairs of base-10 digits by [10,1] and add them to create 8 base-10^2 digits.
      auto const base10e2_8bit = hn::DemoteTo(
          hn::Full128<uint8_t>{},
          hn::SatWidenMulPairwiseAdd(hn::Full256<int16_t>{}, base10_8bit,
                                     hn::Load(hn::Full256<int8_t>{}, digit_value_base10e1.data())));

      [[maybe_unused]] uint8_t const last_digit_e2 =
          hn::ExtractLane(base10e2_8bit, hn::Lanes(hn::Full128<uint8_t>{}) - 1);

      // Multiply pairs of base-10^2 digits by [10^2,1] and add to create 8 base-10^4 digits.
      auto const base10e4_16bit = hn::BitCast(
          hn::Full128<uint16_t>{},
          hn::SatWidenMulPairwiseAdd(hn::Full128<int16_t>{}, base10e2_8bit,
                                     hn::Load(hn::Full128<int8_t>{}, digit_value_base10e2.data())));

      [[maybe_unused]] uint16_t const last_digit_e4 =
          hn::ExtractLane(base10e4_16bit, hn::Lanes(hn::Full128<uint16_t>{}) - 1);

      // Multiply pairs of base-10^4 digits by [10^4,1] and add them to create 4 base-10^8 digits.
      auto const base10e8_32bit =
          hn::WidenMulPairwiseAdd(hn::Full128<uint32_t>{}, base10e4_16bit,
                                  hn::Load(hn::Full128<uint16_t>{}, digit_value_base10e4.data()));

      [[maybe_unused]] uint32_t const last_digit_e8 =
          hn::ExtractLane(base10e8_32bit, hn::Lanes(hn::Full128<uint32_t>{}) - 1);

      return base10e8_32bit;
    }

    template <class CallbackFn>
    void convert_single_uint64(simd_string_view_t input, CallbackFn && callback) {
      // Based on: https://lemire.me/blog/2023/09/22/parsing-integers-quickly-with-avx-512/
      auto const digit_count = input.size();
      assert((digit_count > 0) && (digit_count <= 20));
      assert((digit_count == 1) || (input[0] != '0'));

      // The code requires fixed 128- and 256-bit vectors, so use Full... tags.
      static constexpr hn::Full256<uint8_t> tag_32x8;
      static constexpr hn::Full128<uint32_t> tag_4x32;

      // Load bytes with end of data aligned to the end of the SIMD register.
      auto const ascii_zero = hn::Set(tag_32x8, '0');
      auto const nine = hn::Set(tag_32x8, 9);

      // Set the last digit_count bits of the mask to 1.
      uint32_t const mask = 0xFFFFFFFFULL << (32 - digit_count);
      auto const simd_mask = [&] {
        using mask_t = hn::MFromD<decltype(tag_32x8)>;

        if constexpr (can_create_mask_from_uint<CallbackFn, mask_t>) {
          return call_mask_from_bits<CallbackFn, mask_t>::call(mask);
        } else {
          // No way to directly load the mask. Need to create a vector and convert it.
          // See e.g.: https://stackoverflow.com/a/24242696/255803.
          HWY_ALIGN static constexpr std::array<uint64_t, 4> shuffle_indices =
              std::to_array<uint64_t>(
                  {0x0000000000000000, 0x0101010101010101, 0x0202020202020202, 0x0303030303030303});

          // Broadcast the mask to all 32-bit words in the vector.
          auto const broadcast_mask =
              hn::BitCast(hn::Full256<uint8_t>{}, hn::Set(hn::Full256<uint32_t>{}, mask));

          // For each of the 32 bytes in the vector, select byte 0 from the input for the first 8,
          // byte 1 for the next 8, etc.
          auto const shuffled_bytes = hn::TableLookupBytes(
              broadcast_mask,
              hn::BitCast(hn::Full256<uint8_t>{},
                          hn::Load(hn::Full256<uint64_t>{}, shuffle_indices.data())));

          // Create a mask to AND each byte with its corresponding bit in the mask.
          auto const and_mask = hn::BitCast(
              hn::Full256<uint8_t>{}, hn::Set(hn::Full256<uint64_t>{}, 0x80'40'20'10'08'04'02'01));

          // Generate a non-zero byte for each bit that is set in the mask.
          auto const masked_bits = hn::And(shuffled_bytes, and_mask);

          // Create final mask, where each entry is set if the AND produced a non-zero byte.
          return hn::Eq(masked_bits, and_mask);
        }
      }();

      SPDLOG_TRACE("Converting input '{}' with mask {:#032b}", input, mask);

      // Load input, using the mask.
      // NOTE: This load is never out of bounds, because we take an simd_string_view_t as input,
      // which guarantees sufficient padding both before and after the actual string data.
      uint8_t const * HWY_RESTRICT end =
          reinterpret_cast<uint8_t const *>(input.data() + input.size());
      auto const digits = hn::MaskedLoad(simd_mask, tag_32x8, end - 32);
      auto const base10_8bit = hn::MaskedSaturatedSub(simd_mask, digits, ascii_zero);

      // Verify that all characters are digits.
      [[maybe_unused]] auto const non_digits = hn::MaskedGt(simd_mask, base10_8bit, nine);
      assert(hn::AllFalse(tag_32x8, non_digits));

      // Convert bytes to a 32-digit base-10 integer by subtracting '0'.
      auto base10_32bit = parse_8digit_integers_simd_reverse(base10_8bit);
      HWY_ALIGN std::array<uint32_t, hn::MaxLanes(tag_4x32)> base10_words;
      hn::Store(base10_32bit, tag_4x32, base10_words.data());

      SPDLOG_TRACE("Parsed base-10 words: {::#08X}", base10_words);

      // Maximum 64-bit unsigned integer is 1844 67440737 09551615 (20 digits).
      uint64_t result = base10_words[3];

      if ((mask & 0xFFFFFFU) != 0) {
        auto const middle_part = base10_words[2];
        result = result + 100000000ULL * middle_part;

        if ((mask & 0xFFFFU) != 0) {
          auto const result_32bit = result;
          auto const high_part = base10_words[1];
          result = result + 10000000000000000ULL * high_part;

          if ((high_part > 1844) || (result < result_32bit)) [[unlikely]] {  // Check for overflow.
            throw std::overflow_error("Parsed uint64_t value is too large");
          }
        }
      }

      callback(result);
    }

    template <detail::is_split_callback CallbackFn>
    simd_string_view_t split(simd_string_view_t input, CallbackFn && callback, char splitter) {
      static_assert(std::is_same_v<char, simd_string_view_t::value_type>,
                    "Implementation requires char strings");
      static constexpr hn::ScalableTag<uint8_t> tag{};
      static constexpr size_t lanes = hn::Lanes(tag);

      auto const splitters = hn::Set(tag, static_cast<uint8_t>(splitter));
      auto const offsets = hn::Iota(tag, 0);
      auto const * HWY_RESTRICT data = reinterpret_cast<uint8_t const *>(input.data());

      size_t substr_start = 0;
      size_t input_pos = 0;
      HWY_ALIGN std::array<uint8_t, lanes> match_offset;

      auto const process_matches = [&](size_t num_matches) -> bool {
        SPDLOG_DEBUG("Found {} matches at input position {}", num_matches, input_pos);

        for (size_t offset_idx = 0; offset_idx < num_matches; ++offset_idx) {
          size_t const substr_end = input_pos + match_offset[offset_idx];
          size_t const substr_length = substr_end - substr_start;
          auto const substr = input.substr(substr_start, substr_length);

          substr_start = substr_end + 1;
          if (!callback(substr)) {  // Exit if callback wants to stop processing.
            return false;
          }
        }

        return true;
      };

      SPDLOG_DEBUG("Searching for {:?} in {} bytes", splitter, input.size());

      // Process initial unaligned bytes.
      size_t const unaligned_bytes = reinterpret_cast<uintptr_t>(data) % lanes;

      if (unaligned_bytes != 0) {
        size_t const initial_bytes = std::min(lanes - unaligned_bytes, input.size());
        SPDLOG_DEBUG("Processing {} unaligned bytes", initial_bytes);

        auto const chunk = hn::LoadU(tag, data);
        auto const eq = hn::Eq(chunk, splitters);
        auto const masked_eq = hn::And(eq, hn::FirstN(tag, initial_bytes));

        if (!hn::AllFalse(tag, masked_eq)) {  // Process only if there is a match.
          hn::Store(hn::Compress(offsets, masked_eq), tag, match_offset.data());
          if (!process_matches(hn::CountTrue(tag, masked_eq))) {
            return input.substr(substr_start);
          }
        }

        input_pos += initial_bytes;  // Update input position for aligned loads afterwards.
        SPDLOG_DEBUG("Processed {} unaligned bytes", initial_bytes);
      }

      // Process inputs in chunks of 'lanes' bytes.
      [[maybe_unused]] size_t const num_chunks = (input.size() - input_pos) / lanes;
      if (num_chunks > 0) {
        assert((reinterpret_cast<uintptr_t>(data) + input_pos) % lanes == 0);
        SPDLOG_DEBUG("Processing {} bytes ({} chunks)", num_chunks * lanes, num_chunks);
      }

      for (; input_pos + lanes <= input.size(); input_pos += lanes) {
        auto const chunk = hn::Load(tag, data + input_pos);
        auto const eq = hn::Eq(chunk, splitters);

        if (!hn::AllFalse(tag, eq)) {  // Process only if there is a match.
          hn::Store(hn::Compress(offsets, eq), tag, match_offset.data());
          if (!process_matches(hn::CountTrue(tag, eq))) {
            return input.substr(substr_start);
          }
        }
      }

      // Process remaining bytes. This amount is smaller than a lane, so no need for a loop.
      // We also know that the input is padded, so no risk of reading out of bounds.
      if (input_pos < input.size()) {
        size_t const remaining_bytes = input.size() - input_pos;
        SPDLOG_DEBUG("Processing {} remaining bytes", remaining_bytes);
        assert((reinterpret_cast<uintptr_t>(data) + input_pos) % lanes == 0);

        auto const chunk = hn::Load(tag, data + input_pos);
        auto const eq = hn::Eq(chunk, splitters);
        auto const masked_eq = hn::And(eq, hn::FirstN(tag, remaining_bytes));

        if (!hn::AllFalse(tag, masked_eq)) {  // Process only if there is a match.
          hn::Store(hn::Compress(offsets, masked_eq), tag, match_offset.data());
          if (!process_matches(hn::CountTrue(tag, masked_eq))) {
            return input.substr(substr_start);
          }
        }
      }

      if (substr_start < input.size()) {  // Final substring after the last splitter.
        // No need to check return value, there's nothing more to process.
        callback(input.substr(substr_start, input.size() - substr_start));
      }

      // Everything was processed, return empty view.
      return {};
    }

  }  // namespace HWY_NAMESPACE

}  // namespace aoc25

HWY_AFTER_NAMESPACE();

namespace aoc25 {

  namespace detail {

    // Highway doesn't support dispatch to multi-argument templates, so work around it by partial
    // class specialization and calling a single-templated function inside of it.
    template <std::integral T, class CallbackFn>
      requires std::is_unsigned_v<T> && std::invocable<CallbackFn, T>
    struct convert_single_int_impl {
      static_assert(false, "convert_single_int is not implemented for this type");
    };

    template <class CallbackFn>
      requires std::invocable<CallbackFn, uint64_t>
    struct convert_single_int_impl<uint64_t, CallbackFn> {
      static void invoke(simd_string_view_t input, CallbackFn && callback) {
        HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(convert_single_uint64<CallbackFn>)
        (input, std::forward<CallbackFn>(callback));
      }
    };

  }  // namespace detail

  template <std::integral T, class CallbackFn>
    requires std::is_unsigned_v<T> && std::invocable<CallbackFn, T>
  void convert_single_int(simd_string_view_t input, CallbackFn && callback) {
    return detail::convert_single_int_impl<T, CallbackFn>::invoke(
        input, std::forward<CallbackFn>(callback));
  }

  template <detail::is_split_callback CallbackFn>
  simd_string_view_t split(simd_string_view_t input, CallbackFn && callback, char splitter) {
    HWY_EXPORT_T(table, split<CallbackFn>);
    return HWY_DYNAMIC_DISPATCH_T(table)(input, std::forward<CallbackFn>(callback), splitter);
  }

}  // namespace aoc25

#endif  // AOC25_STRING_HWY_H_TARGET
