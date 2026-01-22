#pragma once

#include <functional>
#include <ranges>

namespace aoc25 {

  /** A branchless implementation of a lower bound binary search. */
  template <std::ranges::forward_range R, class T, class Compare = std::less<>>
  std::ranges::borrowed_iterator_t<R> lower_bound(R && haystack,
                                                  T const & needle,
                                                  Compare compare = Compare{});

  template <std::forward_iterator I, std::sentinel_for<I> S, class T, class Compare = std::less<>>
  I lower_bound(I first, S last, T const & needle, Compare compare = Compare{});

  /** A branchless implementation of a upper bound binary search. */
  template <std::ranges::forward_range R, class T, class Compare = std::less<>>
  std::ranges::borrowed_iterator_t<R> upper_bound(R && haystack,
                                                  T const & needle,
                                                  Compare compare = Compare{});

  template <std::forward_iterator I, std::sentinel_for<I> S, class T, class Compare = std::less<>>
  I upper_bound(I first, S last, T const & needle, Compare compare = Compare{});

}  // namespace aoc25

#include "aoc25/algorithm.tpp"
