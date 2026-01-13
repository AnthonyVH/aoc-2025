#pragma once

#include "aoc25/math.hpp"  // Only for IDE.

#include <cassert>

namespace aoc25 {

  /// @brief Calculate the non-negative mod(value, modulus).
  template <std::integral T, std::integral U>
  constexpr T mod(T value, U modulus) {
    assert(modulus > 0);
    [[assume(modulus > 0)]];
    auto const cast_mod = static_cast<T>(modulus);
    return ((value % cast_mod) + cast_mod) % cast_mod;
  }

  constexpr uint64_t num_combinations(int8_t n, int8_t k) {
    if ((k <= 0) || (k > n)) {
      return 0;
    }

    uint64_t result = 1;
    for (uint64_t d = 1; d <= static_cast<uint64_t>(k); ++d) {
      result *= n--;
      result /= d;
    }

    return result;
  }

  /** From: https://stackoverflow.com/a/27670035
   *
   * #d = number length, #c = number of comparisons
   *
   * #d | #c   #d | #c     #d | #c   #d | #c
   * ---+---   ---+---     ---+---   ---+---
   * 20 | 5    15 | 5      10 | 5     5 | 5
   * 19 | 5    14 | 5       9 | 5     4 | 5
   * 18 | 4    13 | 4       8 | 4     3 | 4
   * 17 | 4    12 | 4       7 | 4     2 | 4
   * 16 | 4    11 | 4       6 | 4     1 | 4
   */
  constexpr unsigned num_digits(uint64_t x) {
    return                                      // Num-># Digits->[0-9] 64->bits bs->Binary Search
        (x >= 10000000000ul                     // [11-20] [1-10]
             ? (x >= 1000000000000000ul         // [16-20] [11-15]
                    ?                           // [16-20]
                    (x >= 100000000000000000ul  // [18-20] [16-17]
                         ?                      // [18-20]
                         (x >= 1000000000000000000ul        // [19-20] [18]
                              ?                             // [19-20]
                              (x >= 10000000000000000000ul  // [20] [19]
                                   ? 20
                                   : 19)
                              : 18)
                         :                          // [16-17]
                         (x >= 10000000000000000ul  // [17] [16]
                              ? 17
                              : 16))
                    :                                  // [11-15]
                    (x >= 1000000000000ul              // [13-15] [11-12]
                         ?                             // [13-15]
                         (x >= 10000000000000ul        // [14-15] [13]
                              ?                        // [14-15]
                              (x >= 100000000000000ul  // [15] [14]
                                   ? 15
                                   : 14)
                              : 13)
                         :                     // [11-12]
                         (x >= 100000000000ul  // [12] [11]
                              ? 12
                              : 11)))
             :                                  // [1-10]
             (x >= 100000ul                     // [6-10] [1-5]
                  ?                             // [6-10]
                  (x >= 10000000ul              // [8-10] [6-7]
                       ?                        // [8-10]
                       (x >= 100000000ul        // [9-10] [8]
                            ?                   // [9-10]
                            (x >= 1000000000ul  // [10] [9]
                                 ? 10
                                 : 9)
                            : 8)
                       :                // [6-7]
                       (x >= 1000000ul  // [7] [6]
                            ? 7
                            : 6))
                  :                        // [1-5]
                  (x >= 100ul              // [3-5] [1-2]
                       ?                   // [3-5]
                       (x >= 1000ul        // [4-5] [3]
                            ?              // [4-5]
                            (x >= 10000ul  // [5] [4]
                                 ? 5
                                 : 4)
                            : 3)
                       :           // [1-2]
                       (x >= 10ul  // [2] [1]
                            ? 2
                            : 1))));
  }

}  // namespace aoc25
