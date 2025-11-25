#pragma once

#include "aoc25/simd.hpp"

#include <concepts>
#include <cstddef>
#include <filesystem>
#include <string_view>
#include <type_traits>
#include <vector>

namespace aoc25 {

  /// @brief Template for a day's solutions. Must be specialized for each day.
  template <size_t N>
  struct day_t;

  /// @brief Returns path to input file for given day and part.
  std::filesystem::path input_file_path(std::filesystem::path dir, size_t day, size_t part);

  /// @brief Returns paths to all example files for given day and part.
  std::vector<std::filesystem::path> example_file_paths(std::filesystem::path const & dir,
                                                        size_t day,
                                                        size_t part);

  /// @brief Part tag for dispatching to solution part-specific implementations.
  template <size_t N>
  struct part_t : std::integral_constant<size_t, N> {};

  template <size_t N>
  inline constexpr part_t<N> part{};

  // Version tag for dispatching to different implementations of a part's solution.
  template <size_t N>
  struct version_t : std::integral_constant<size_t, N> {};

  template <size_t N>
  inline constexpr version_t<N> version{};

  // Concepts to check if a given day can solve a given part (using an optional
  // version).
  template <size_t Day, size_t Part, class Input>
  concept invocable_for_part = requires(day_t<Day> t, Input in) {
    { t.solve(part<Part>, in) };
  };

  template <size_t Day, size_t Part, size_t Version, class Input>
  concept invocable_for_part_version = requires(day_t<Day> t, Input in) {
    { t.solve(part<Part>, version<Version>, in) };
  };

  namespace detail {

    // Trait to check the highest available version a part can be solved with.
    template <size_t Day, size_t Part, class Input>
    struct highest_version_for_part_t {
     private:
      struct stop_recursion_t {
        static constexpr size_t num_versions_ge = -1;
      };

      template <size_t Version>
      struct impl_t {
        // Need std::condition_t to avoid instantiating an infinite recursion.
        using recurse_t = std::conditional_t<invocable_for_part_version<Day, Part, Version, Input>,
                                             impl_t<Version + 1>,
                                             stop_recursion_t>;

        static constexpr size_t num_versions_ge = recurse_t::num_versions_ge + 1;
      };

     public:
      static constexpr bool has_versions = invocable_for_part_version<Day, Part, 0, Input>;
      static constexpr size_t highest_version = impl_t<0>::num_versions_ge - 1;
    };

  }  // namespace detail

  template <size_t Day, size_t Part, class Input>
  inline constexpr auto highest_version_for_part =
      detail::highest_version_for_part_t<Day, Part, Input>{};

}  // namespace aoc25
