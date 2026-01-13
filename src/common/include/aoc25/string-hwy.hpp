#include "aoc25/simd.hpp"

#include <bit>

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

  namespace detail {

    /** @brief Creates an array that can be loaded as a mask (i.e. each element is either all 0s or
     * all 1s). All entries are set to zero, except for the last `num_digits` entries, which are set
     * to all 1s.
     */
    template <std::integral T, size_t NumLanes>
      requires std::is_unsigned_v<T>
    constexpr std::array<T, NumLanes> mask_for_single_uint_conversion(size_t num_digits) {
      std::array<T, NumLanes> result = {};  // Initialize all elements to zero.
      for (size_t i = 0; i < num_digits; ++i) {
        result[result.size() - 1 - i] = static_cast<T>(-1);
      }
      return result;
    }

    /** @brief Creates an array of mask data for the convert_single_uint functions. Each entry I in
     * the array holds data for the mask for an I digit number. Note that the 0th entry is unused.
     * It is generated to allow loading the required array entry directly with the digit count as
     * index.
     */
    template <std::integral T, size_t NumLanes, size_t MaxDigits>
    constexpr std::array<std::array<T, NumLanes>, MaxDigits + 1>
    masks_array_for_single_uint_conversion() {
      std::array<std::array<T, NumLanes>, MaxDigits + 1> result = {};
      for (size_t digits = 1; digits <= MaxDigits; ++digits) {
        result[digits] = mask_for_single_uint_conversion<T, NumLanes>(digits);
      }
      return result;
    }

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

  }  // namespace detail

  namespace HWY_NAMESPACE {

    namespace hn = hwy::HWY_NAMESPACE;

    // Based on: https://lemire.me/blog/2023/09/22/parsing-integers-quickly-with-avx-512/
    [[maybe_unused]] inline hn::VFromD<hn::Full128<uint32_t>> parse_10e8_integer_simd_reverse(
        hn::VFromD<hn::Full256<uint8_t>> base10_8bit) {
      HWY_ALIGN static constexpr std::array<uint8_t, hn::Lanes(hn::Full256<uint8_t>{})>
          digit_value_base10e1 =
              std::to_array<uint8_t>({10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1,
                                      10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1});

      HWY_ALIGN static constexpr std::array<uint8_t, hn::Lanes(hn::Full128<uint8_t>{})>
          digit_value_base10e2 = std::to_array<uint8_t>(
              {100, 1, 100, 1, 100, 1, 100, 1, 100, 1, 100, 1, 100, 1, 100, 1});

      HWY_ALIGN static constexpr std::array<uint16_t, hn::Lanes(hn::Full128<uint16_t>{})>
          digit_value_base10e4 =
              std::to_array<uint16_t>({10'000, 1, 10'000, 1, 10'000, 1, 10'000, 1});

      // Multiply 32 pairs of base-10 digits by [10,1] and add them to create 16 base-10^2 digits.
      auto const base10e2_8bit = hn::DemoteTo(
          hn::Full128<uint8_t>{},
          hn::SatWidenMulPairwiseAdd(
              hn::Full256<int16_t>{}, base10_8bit,
              hn::Load(hn::Full256<int8_t>{},
                       reinterpret_cast<int8_t const *>(digit_value_base10e1.data()))));

      // Multiply 16 pairs of base-10^2 digits by [10^2,1] and add to create 8 base-10^4 digits.
      auto const base10e4_16bit =
          hn::BitCast(hn::Full128<uint16_t>{},
                      hn::SatWidenMulPairwiseAdd(
                          hn::Full128<int16_t>{}, base10e2_8bit,
                          hn::Load(hn::Full128<int8_t>{},
                                   reinterpret_cast<int8_t const *>(digit_value_base10e2.data()))));

      // Multiply 8 pairs of base-10^4 digits by [10^4,1] and add them to create 4 base-10^8 digits.
      auto const base10e8_32bit =
          hn::WidenMulPairwiseAdd(hn::Full128<uint32_t>{}, base10e4_16bit,
                                  hn::Load(hn::Full128<uint16_t>{}, digit_value_base10e4.data()));

      return base10e8_32bit;
    }

    /** @brief Converts a byte-packed decimal value into a 16-bit value. The packed bytes should
     * be stored with the least significant digit at the highest address.
     */
    [[maybe_unused]] inline hn::VFromD<hn::Full128<uint16_t>> parse_10e4_integer_simd_reverse(
        hn::VFromD<hn::Full128<uint8_t>> base10_8bit) {
      // Unused digits are ignored by setting the corresponding multiplier to zero.
      HWY_ALIGN static constexpr std::array<uint16_t, hn::Lanes(hn::Full128<uint16_t>{})>
          digit_value_base10e4 = std::to_array<uint16_t>({0, 0, 0, 1, 1'000, 100, 10, 1});

      // Widen the upper 8 of 16 bytes (i.e. digits) to 16-bit values.
      auto const base10_16bit = hn::PromoteUpperTo(hn::Full128<uint16_t>{}, base10_8bit);

      // Multiply 8 pairs of base-10 digits by [0, 0, 0, 1, 1000, 100, 10, 1] and add them pairwise.
      auto const base10e4_32bit = hn::WidenMulPairwiseAdd(
          hn::Full128<int32_t>{}, hn::BitCast(hn::Full128<int16_t>{}, base10_16bit),
          hn::BitCast(hn::Full128<int16_t>{},
                      hn::Load(hn::Full128<uint16_t>{}, digit_value_base10e4.data())));

      // Add adjacent pairs of 32-bit values. We are only interest in the first two summations,
      // which are generated from the first argument.
      auto const base10e4_sum_32bit =
          hn::PairwiseAdd128(hn::Full128<int32_t>{}, base10e4_32bit, base10e4_32bit);

      // Truncate the results back to 16 bit. We know they can't have overflowed.
      auto const base10e4_16bit = hn::BitCast(hn::Full128<uint16_t>{}, base10e4_sum_32bit);

      return base10e4_16bit;
    }

    template <class CallbackFn>
    void convert_single_uint64(simd_string_view_t input, CallbackFn && callback) {
      // Based on: https://lemire.me/blog/2023/09/22/parsing-integers-quickly-with-avx-512/
      // Maximum digits for uint64_t is 20 (18'446'744'073'709'551'615).
      [[maybe_unused]] static constexpr uint8_t max_digits = 20;

      auto const digit_count = input.size();
      assert((digit_count > 0) && (digit_count <= max_digits));
      assert((digit_count == 1) || (input[0] != '0'));

      // The code requires fixed 128- and 256-bit vectors, so use Full... tags.
      static constexpr hn::Full256<uint8_t> tag_32x8;
      static constexpr hn::Full128<uint32_t> tag_4x32;

      // Load bytes with end of data aligned to the end of the SIMD register.
      auto const ascii_zero = hn::Set(tag_32x8, '0');
      auto const nine = hn::Set(tag_32x8, 9);

      // Set the last digit_count bits of the mask to 1.
      uint32_t const mask = 0xFFFFFFFFULL << (hn::Lanes(tag_32x8) - digit_count);
      auto const simd_mask = [&] {
        using mask_t = hn::MFromD<decltype(tag_32x8)>;

        if constexpr (detail::can_create_mask_from_uint<CallbackFn, mask_t>) {
          return detail::call_mask_from_bits<CallbackFn, mask_t>::call(mask);
        } else {
          // No way to directly load the mask. Need to create a vector and convert it.
          // See e.g.: https://stackoverflow.com/a/24242696/255803.

          /* Two alternatives to the LUT below:
           *
           * - A very simple "direct" approach, benchmarks slower than the other alternative.
           *
           *   return hn::SlideMaskUpLanes(tag_32x8, hn::FirstN(tag_32x8, digit_count),
           *                               hn::Lanes(tag_32x8) - digit_count);
           *
           * - A more complex approach using table lookups. Based on SO answer linked above.
           *   Benchmarks more or less as fast as the LUT method used here.
           * HWY_ALIGN static constexpr std::array<uint64_t, 4> shuffle_indices =
           *     std::to_array<uint64_t>(
           *         {0x0000000000000000, 0x0101010101010101,
           *          0x0202020202020202, 0x0303030303030303});
           *
           * // Broadcast the mask to all 32-bit words in the vector.
           * auto const broadcast_mask =
           *     hn::BitCast(hn::Full256<uint8_t>{}, hn::Set(hn::Full256<uint32_t>{}, mask));
           *
           * // For each of the 32 bytes in the vector, select byte 0 from the input for the first
           * // 8 mask bits, byte 1 for the next 8 mask bits, etc.
           * auto const shuffled_bytes = hn::TableLookupBytes(
           *     broadcast_mask,
           *     hn::BitCast(hn::Full256<uint8_t>{},
           *                 hn::Load(hn::Full256<uint64_t>{}, shuffle_indices.data())));
           *
           * // Create a mask to AND each byte with its corresponding bit in the mask.
           * auto const and_mask = hn::BitCast(
           *     hn::Full256<uint8_t>{}, hn::Set(hn::Full256<uint64_t>{},
           *                                     0x80'40'20'10'08'04'02'01));
           *
           * // Generate a non-zero byte for each bit that is set in the mask.
           * auto const masked_bits = hn::And(shuffled_bytes, and_mask);
           *
           * // Create final mask, where each entry is set if the AND produced a non-zero byte.
           * return hn::Eq(masked_bits, and_mask);
           */

          // Since there's only a few possible number of digits, using a LUT is fastest.
          HWY_ALIGN static constexpr std::array masks =
              detail::masks_array_for_single_uint_conversion<uint8_t, hn::Lanes(tag_32x8),
                                                             max_digits>();
          return hn::MaskFromVec(hn::Load(tag_32x8, masks[digit_count].data()));
        }
      }();

      SPDLOG_TRACE("Converting input '{}' with mask {:#032b}", input, mask);

      // Load input, using the mask.
      // NOTE: This load is never out of bounds, because we take an simd_string_view_t as input,
      // which guarantees sufficient padding both before and after the actual string data.
      uint8_t const * HWY_RESTRICT end =
          reinterpret_cast<uint8_t const *>(input.data() + input.size());
      auto const digits = hn::LoadU(tag_32x8, end - hn::Lanes(tag_32x8));

      // Convert bytes to a 32-digit base-10 integer by subtracting '0'.
      auto const base10_8bit = hn::MaskedSaturatedSub(simd_mask, digits, ascii_zero);

      // Verify that all characters are digits.
      [[maybe_unused]] auto const non_digits = hn::MaskedGt(simd_mask, base10_8bit, nine);
      assert(hn::AllFalse(tag_32x8, non_digits));

      // Multiply-accumulate the digits into base-10^8 words.
      auto const base10_32bit = parse_10e8_integer_simd_reverse(base10_8bit);
      HWY_ALIGN std::array<uint32_t, hn::MaxLanes(tag_4x32)> base10_words;
      hn::Store(base10_32bit, tag_4x32, base10_words.data());

      SPDLOG_TRACE("Parsed base-10 words: {::d}", base10_words);

      // Maximum 64-bit unsigned integer is 1844 67440737 09551615 (20 digits).
      uint64_t result = base10_words[3];

      if (digit_count > 8) {
        auto const middle_part = base10_words[2];
        result = result + 100'000'000ULL * middle_part;

        if (digit_count > 16) {
          auto const result_32bit = result;
          auto const high_part = base10_words[1];
          result = result + 10'000'000'000'000'000ULL * high_part;

          if ((high_part > 1844) || (result < result_32bit)) [[unlikely]] {  // Check for overflow.
            throw std::overflow_error("Parsed uint64_t value is too large");
          }
        }
      }

      callback(result);
    }

    template <class CallbackFn>
    void convert_single_uint16(simd_string_view_t input, CallbackFn && callback) {
      // Adapted from the convert_single_uint64() function.
      // Maximum digits for uint16_t is 5 (65'535).
      [[maybe_unused]] static constexpr uint8_t max_digits = 5;

      auto const digit_count = input.size();
      assert((digit_count > 0) && (digit_count <= max_digits));
      assert((digit_count == 1) || (input[0] != '0'));

      // The code requires fixed length vectors, so use Full... tags.
      static constexpr hn::Full128<uint8_t> tag_16x8;
      static constexpr hn::Full128<uint16_t> tag_8x16;

      // Load bytes with end of data aligned to the end of the SIMD register.
      auto const ascii_zero = hn::Set(tag_16x8, '0');
      auto const nine = hn::Set(tag_16x8, 9);

      // Set the last digit_count bits of the mask to 1.
      [[maybe_unused]] uint16_t const mask = 0xFFFFULL << (hn::Lanes(tag_16x8) - digit_count);
      auto const simd_mask = [&] {
        using mask_t = hn::MFromD<decltype(tag_16x8)>;

        if constexpr (detail::can_create_mask_from_uint<CallbackFn, mask_t>) {
          return detail::call_mask_from_bits<CallbackFn, mask_t>::call(mask);
        } else {
          // No way to directly load the mask. Need to create a vector and convert it.
          // See e.g.: https://stackoverflow.com/a/24242696/255803.
          // An alterantive, slower than the LUT method used here, is:
          //   return hn::SlideMaskUpLanes(tag_16x8, hn::FirstN(tag_16x8, digit_count),
          //                               hn::Lanes(tag_16x8) - digit_count);

          // Since there's only a few possible number of digits, using a LUT is fastest.
          HWY_ALIGN static constexpr std::array masks =
              detail::masks_array_for_single_uint_conversion<uint8_t, hn::Lanes(tag_16x8),
                                                             max_digits>();
          return hn::MaskFromVec(hn::Load(tag_16x8, masks[digit_count].data()));
        }
      }();

      SPDLOG_DEBUG("Converting input '{}' with mask {:#016b}", input, mask);

      // Load input, using the mask.
      // NOTE: This load is never out of bounds, because we take an simd_string_view_t as input,
      // which guarantees sufficient padding both before and after the actual string data.
      uint8_t const * HWY_RESTRICT end =
          reinterpret_cast<uint8_t const *>(input.data() + input.size());
      auto const digits = hn::LoadU(tag_16x8, end - hn::Lanes(tag_16x8));

      // Convert bytes to a 16-digit base-10 integer by subtracting '0'.
      auto const base10_8bit = hn::MaskedSaturatedSub(simd_mask, digits, ascii_zero);

      // Verify that all characters are digits.
      [[maybe_unused]] auto const non_digits = hn::MaskedGt(simd_mask, base10_8bit, nine);
      assert(hn::AllFalse(tag_16x8, non_digits));

      // Multiply-accumulate the digits into base-10^8 words.
      auto base10_16bit = parse_10e4_integer_simd_reverse(base10_8bit);
      HWY_ALIGN std::array<uint16_t, hn::MaxLanes(tag_8x16)> base10_words;
      hn::Store(base10_16bit, tag_8x16, base10_words.data());

      SPDLOG_DEBUG("Parsed base-10 words: {::d}", base10_words);

      // Maximum 16-bit unsigned integer is 65 535 (5 digits)
      uint16_t result = base10_words[2];  // Due to bit casting, result is at index 2.

      if (digit_count > 4) {  // The first word holds the sum of the lower 4 digits.
        auto const prev_result = result;
        auto const high_part = base10_words[0];  // Again, result in odd place due to bit casting.
        result = result + 10'000U * high_part;

        SPDLOG_DEBUG("Inside the high part (high part: {}, prev result: {}, result: {})", high_part,
                     prev_result, result);

        if ((high_part > 6) || (result < prev_result)) [[unlikely]] {  // Check for overflow.
          throw std::overflow_error("Parsed uint16_t value is too large");
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
      static constexpr bool callback_returns_bool =
          std::is_same_v<bool, std::invoke_result_t<CallbackFn, simd_string_view_t>>;

      auto const splitters = hn::Set(tag, static_cast<uint8_t>(splitter));
      auto const * HWY_RESTRICT data = reinterpret_cast<uint8_t const *>(input.data());

      size_t substr_start = 0;
      size_t input_pos = 0;

      auto const process_matches = [&](uint32_t match_bits) -> bool {
        SPDLOG_DEBUG("Found {} matches at input position {}", std::popcount(match_bits), input_pos);

        while (match_bits != 0) {
          uint8_t const match_offset = std::countr_zero(match_bits);
          uint32_t const lsb_bit = match_bits & static_cast<uint32_t>(-match_bits);
          match_bits &= ~lsb_bit;  // Unset LSB.

          size_t const substr_end = input_pos + match_offset;
          size_t const substr_length = substr_end - substr_start;
          auto const substr = input.substr(substr_start, substr_length);
          substr_start = substr_end + 1;

          if constexpr (callback_returns_bool) {
            if (!callback(substr)) {
              return false;  // Early exit if callback wants to stop processing.
            }
          } else {
            callback(substr);
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
        uint32_t const match_bits = hn::BitsFromMask(tag, masked_eq);

        if (match_bits != 0) {  // Process only if there is a match.
          if (!process_matches(match_bits)) {
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
        uint32_t const match_bits = hn::BitsFromMask(tag, eq);

        if (match_bits != 0) {  // Process only if there is a match.
          if (!process_matches(match_bits)) {
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
        uint32_t const match_bits = hn::BitsFromMask(tag, masked_eq);

        if (match_bits) {  // Process only if there is a match.
          if (!process_matches(match_bits)) {
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

    template <class CallbackFn>
      requires std::invocable<CallbackFn, uint16_t>
    struct convert_single_int_impl<uint16_t, CallbackFn> {
      static void invoke(simd_string_view_t input, CallbackFn && callback) {
        HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(convert_single_uint16<CallbackFn>)
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
