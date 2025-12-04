#include "aoc25/day_01.hpp"

#include <cassert>
#include <span>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "src/day_01-hwy.cpp"

// clang-format off
#include <hwy/foreach_target.h>
// clang-format on

#include <hwy/highway.h>
#include <hwy/highway_export.h>

HWY_BEFORE_NAMESPACE();

namespace aoc25 {
  namespace HWY_NAMESPACE {
    namespace hn = hwy::HWY_NAMESPACE;

    void mod_dial(std::span<int16_t> values) {
      static constexpr hn::ScalableTag<int16_t> tag{};
      static constexpr uint8_t dial_size = 100;
      static constexpr size_t lane_size = hn::Lanes(tag);

      // Assuming that the span points to padded/aligned memory.
      size_t const count = hwy::RoundUpTo(values.size(), lane_size);

      for (size_t idx = 0; idx < count; idx += lane_size) {
        auto * const ptr = &values[idx];
        auto vec = hn::Load(tag, ptr);
        vec = hn::Add(vec, hn::Set(tag, 20 * dial_size));
        vec = hn::Mod(vec, hn::Set(tag, dial_size));
        hn::Store(vec, tag, ptr);
      }
    }

    uint16_t count_zeros(std::span<int16_t const> values) {
      static constexpr hn::ScalableTag<int16_t> tag{};
      static constexpr size_t lane_size = hn::Lanes(tag);

      // Assuming that the span points to padded/aligned memory.
      size_t const count = hwy::RoundUpTo(values.size(), lane_size);

      uint16_t total_zeros = 0;

      for (size_t idx = 0; idx < count; idx += lane_size) {
        auto * const ptr = &values[idx];
        auto vec = hn::Load(tag, ptr);
        auto const eq = hn::Eq(vec, hn::Zero(tag));
        total_zeros += std::popcount(hn::BitsFromMask(tag, eq));
      }

      return total_zeros;
    }

  }  // namespace HWY_NAMESPACE
}  // namespace aoc25

HWY_AFTER_NAMESPACE();

#ifdef HWY_ONCE

namespace aoc25 {

  HWY_EXPORT(mod_dial);
  HWY_EXPORT(count_zeros);

  HWY_DLLEXPORT void mod_dial(std::span<int16_t> in) {
    return HWY_DYNAMIC_DISPATCH(mod_dial)(in);
  }

  HWY_DLLEXPORT uint16_t count_zeros(std::span<int16_t const> in) {
    return HWY_DYNAMIC_DISPATCH(count_zeros)(in);
  }

}  // namespace aoc25

#endif
