#pragma once

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <span>
#include <type_traits>

namespace aoc25 {

  /** @brief STL allocator that aligns allocations to the specified alignment. The alignment equals
   * the largest realistic SIMD alignment. The allocated size is always rounded up to a multiple of
   * the alignment. Both beginning and end of the allocation are padded, such that any kind of SIMD
   * load within the requested address range is guaranteed to be within allocation bounds.
   */
  template <class T, size_t Alignment>
  struct aligned_allocator {
   public:
    static_assert((Alignment & (Alignment - 1)) == 0, "Alignment must be power of 2");
    static_assert(Alignment >= alignof(T), "Alignment must be at least alignof(T)");

    using value_type = T;
    using size_type = size_t;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;

    static constexpr size_t alignment = Alignment;

    template <typename U>
    struct rebind {
      using other = aligned_allocator<U, Alignment>;
    };

    aligned_allocator() = default;

    template <class U, class Enable = std::enable_if_t<!std::is_same_v<T, U>>>
    aligned_allocator(aligned_allocator<U, Alignment> const &) noexcept;

    T * allocate(size_type n);
    void deallocate(T * ptr, size_type) noexcept;
  };

  template <class T, class U, size_t AlignmentLhs, size_t AlignmentRhs>
  bool operator==(aligned_allocator<T, AlignmentLhs> const & lhs,
                  aligned_allocator<U, AlignmentRhs> const & rhs) noexcept;

}  // namespace aoc25

#include "aoc25/memory.tpp"
