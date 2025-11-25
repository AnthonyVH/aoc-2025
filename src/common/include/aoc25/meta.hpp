#pragma once

#include <cstddef>

namespace aoc25 {

  // Priority tag for overload resolution.
  template <size_t N>
  struct prio_tag_t : prio_tag_t<N - 1> {
    static constexpr size_t value = N;
  };

  template <>
  struct prio_tag_t<0> {
    static constexpr size_t value = 0;
  };

  template <size_t N>
  inline constexpr prio_tag_t<N> prio_tag{};

}  // namespace aoc25
