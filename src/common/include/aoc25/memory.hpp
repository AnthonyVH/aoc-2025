#pragma once

#include "aoc25/simd.hpp"

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <span>
#include <type_traits>

namespace aoc25 {

  /** @brief STL allocator that aligns allocations to the specified alignment. The alignment equals
   * the largest realistic SIMD alignment. The allocated size is always rounded up to a multiple of
   * the alignment and extended such that if the address of the last element were to be used to load
   * an SIMD register, all the addresses after it will also be part of the allocation.
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
    aligned_allocator(aligned_allocator<U, Alignment> const&) noexcept;

    T* allocate(size_type n);
    void deallocate(T* ptr, size_type) noexcept;
  };

  template <class T, class U, size_t AlignmentLhs, size_t AlignmentRhs>
  bool operator==(aligned_allocator<T, AlignmentLhs> const& lhs,
                  aligned_allocator<U, AlignmentRhs> const& rhs) noexcept;

  namespace detail {
    template <class T, class Elem, size_t Alignment>
    struct uses_aligned_allocator
        : std::uses_allocator<std::remove_cvref_t<T>, aligned_allocator<Elem, Alignment>> {};

    template <class T, class Elem, size_t Alignment>
    static constexpr auto uses_aligned_allocator_v =
        uses_aligned_allocator<T, Elem, Alignment>::value;

    template <class Rng, size_t Alignment>
    concept aligned_range =
        std::ranges::range<Rng> &&
        detail::uses_aligned_allocator_v<Rng,
                                         std::remove_cvref_t<std::ranges::range_value_t<Rng>>,
                                         Alignment>;
  }  // namespace detail

  template <size_t Alignment, class T>
  class aligned_span : public std::span<T> {
   private:
    using base_t = std::span<T>;

   public:
    aligned_span() = default;

    aligned_span(T* data, size_t size);

    template <class Rng>
      requires detail::aligned_range<Rng, Alignment> && (!std::is_rvalue_reference_v<Rng>)
    aligned_span(Rng&& rng);

    size_t aligned_size() const noexcept;
  };

}  // namespace aoc25

#include "aoc25/memory.tpp"
