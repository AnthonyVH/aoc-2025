#if defined(AOC25_N_ARY_TREE_HWY_H_TARGET) == defined(HWY_TARGET_TOGGLE)
#  ifdef AOC25_N_ARY_TREE_HWY_H_TARGET
#    undef AOC25_N_ARY_TREE_HWY_H_TARGET
#  else
#    define AOC25_N_ARY_TREE_HWY_H_TARGET
#  endif

#  include "aoc25/n_ary_tree.hpp"

#  include <fmt/ranges.h>
#  include <spdlog/spdlog.h>

#  include <limits>
#  include <type_traits>

#  undef HWY_TARGET_INCLUDE
#  define HWY_TARGET_INCLUDE "src/n_ary_tree-hwy.hpp"

// clang-format off
#include <hwy/foreach_target.h>
// clang-format on

#  include <hwy/cache_control.h>
#  include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();

namespace aoc25 {
  namespace HWY_NAMESPACE {

    namespace hn = hwy::HWY_NAMESPACE;

    // Note: See call-site for an explanation why this single template argument is used.
    template <class SiftDownArgs>
    void sift_down_with_values(SiftDownArgs args) {
      using key_t = std::remove_cvref_t<decltype(args.keys.front())>;
      using value_t = std::remove_cvref_t<decltype(args.values.front())>;

      static constexpr detail::n_ary_tree_variant_t variant = SiftDownArgs::variant;
      static constexpr size_t branching_factor = SiftDownArgs::branching_factor;

      // If branching factor is higher than the number of SIMD lanes, fall back to scalar code.
      static constexpr size_t num_supported_lanes = HWY_LANES(key_t);
      static constexpr bool power_of_two_branching = std::has_single_bit(branching_factor);
      static constexpr bool can_use_simd =
          (branching_factor <= num_supported_lanes) && power_of_two_branching;

      if constexpr (!can_use_simd) {
        return std::invoke(args.sift_down_scalar, args.tree, args.pos);
      }

      // This weird construction is required because if we try to create a tag that has more lanes
      // than what Highway supports, we'll get a compile-time error.
      static constexpr auto tag =
          hn::FixedTag<key_t, can_use_simd ? branching_factor : num_supported_lanes>{};

      static_assert(std::is_arithmetic_v<key_t>);
      static_assert(branching_factor >= 2, "Branching factor must be at least 2");

      size_t const size = args.keys.size();
      key_t * const HWY_RESTRICT key_ptr = args.keys.data();
      value_t * const HWY_RESTRICT value_ptr = args.values.data();
      size_t pos = args.pos;

      // For a min-tree, the lowest value should be at the top, so the "worst" value is max(). For a
      // max-tree, it's the opposite.
      auto const worst_value = hn::Set(tag, (variant == detail::n_ary_tree_variant_t::min)
                                                ? std::numeric_limits<key_t>::max()
                                                : std::numeric_limits<key_t>::min());

      while (true) {
        size_t const first_child = branching_factor * pos + 1;

        if (first_child >= size) {
          break;  // No children exist.
        }

        size_t const num_children = std::min(branching_factor, size - first_child);
        auto const child_keys = hn::LoadNOr(worst_value, tag, key_ptr + first_child, num_children);

        // Prefetch children of first current child. The higher the branching factor, the more
        // useless this is.
        hwy::Prefetch(key_ptr + branching_factor * first_child + 1);

        // Calculate best such that each lane contains the best key among all children.
        auto const best = (variant == detail::n_ary_tree_variant_t::min)
                            ? hn::MinOfLanes(tag, child_keys)
                            : hn::MaxOfLanes(tag, child_keys);

        if (hn::GetLane(best) >= key_ptr[pos]) {
          break;
        }

        // Find which child has the best key.
        auto const is_best = hn::Eq(child_keys, best);
        size_t const best_child = first_child + std::countr_zero(hn::BitsFromMask(tag, is_best));

        // Keep key and value position synchronized.
        using std::swap;
        swap(key_ptr[pos], key_ptr[best_child]);
        swap(value_ptr[pos], value_ptr[best_child]);

        pos = best_child;
      }
    }

  }  // namespace HWY_NAMESPACE
}  // namespace aoc25

HWY_AFTER_NAMESPACE();

namespace aoc25 {
  namespace detail {

    /* Google Highway does not support creating dispatch tables to functions with more than one
     * template argument. Hence we cheat by bundling all template arguments into a single struct,
     * and unpacking it again inside the function that gets dispatched to.
     */
    template <class Tree>
    struct sift_down_with_values_args_t {
      using key_t = typename Tree::key_t;
      using value_t = typename Tree::value_t;

      static constexpr size_t branching_factor = Tree::branching_factor;
      static constexpr n_ary_tree_variant_t variant = Tree::variant;

      // Required info for SIMD sift-down to fall back to scalar code.
      using member_void_fn_ptr_t = void (Tree::*)(size_t);
      Tree & tree;
      member_void_fn_ptr_t sift_down_scalar;

      simd_span_t<key_t> keys;
      std::span<value_t> values;
      size_t pos;
    };

    template <class Tree>
    void simd_sift_down_with_values(Tree & tree, size_t pos) {
      // The Tree class should make this function a friend.
      using args_t = sift_down_with_values_args_t<Tree>;
      HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(sift_down_with_values<args_t>)(args_t{
          tree, &Tree::sift_down_scalar, simd_span_t{tree.keys_}, std::span{tree.values_}, pos});
    }

  }  // namespace detail
}  // namespace aoc25

#endif  // AOC25_N_ARY_TREE_H_TARGET
