#include "aoc25/algorithm.hpp"

namespace aoc25 {

  template <std::forward_iterator I, std::sentinel_for<I> S, class T, class Compare>
  I lower_bound(I first, S last, T const & needle, Compare compare) {
    size_t length = std::ranges::distance(first, last);

    while (length > 0) {
      size_t const half = length / 2;
      auto const middle = std::next(first, half);
      bool const go_right = compare(*middle, needle);
      std::advance(first, go_right * (length - half));
      length = half;
    }

    return first;
  }

  template <std::ranges::forward_range R, class T, class Compare>
  std::ranges::borrowed_iterator_t<R> lower_bound(R && haystack,
                                                  T const & needle,
                                                  Compare compare) {
    return lower_bound(std::ranges::begin(haystack), std::ranges::end(haystack), needle,
                       std::move(compare));
  }

  template <std::forward_iterator I, std::sentinel_for<I> S, class T, class Compare>
  I upper_bound(I first, S last, T const & needle, Compare compare) {
    size_t length = std::ranges::distance(first, last);

    while (length > 0) {
      size_t const half = length / 2;
      auto const middle = std::next(first, half);
      bool const go_right = !compare(needle, *middle);
      std::advance(first, go_right * (length - half));
      length = half;
    }

    return first;
  }

  template <std::ranges::forward_range R, class T, class Compare>
  std::ranges::borrowed_iterator_t<R> upper_bound(R && haystack,
                                                  T const & needle,
                                                  Compare compare) {
    return upper_bound(std::ranges::begin(haystack), std::ranges::end(haystack), needle,
                       std::move(compare));
  }

}  // namespace aoc25
