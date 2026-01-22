#pragma once

#include "aoc25/memory.hpp"  // Enable IDE syntax highlighting.

#include <cassert>
#include <new>
#include <stdexcept>

namespace aoc25 {

  template <class T, size_t Alignment>
  template <class U, class Enable>
  aligned_allocator<T, Alignment>::aligned_allocator(
      aligned_allocator<U, Alignment> const &) noexcept {};

  template <class T, size_t Alignment>
  T * aligned_allocator<T, Alignment>::allocate(size_type n) {
    size_t size = n * sizeof(T);

    size = (size + (Alignment - 1)) / Alignment * Alignment;
    size += 2 * Alignment;  // Add padding before and after to allow for SIMD loads.

    // GCC keeps printing errors about too large allocations during PGO + LTO-enabled builds. So
    // let's make sure we see this if it actually happens.
    static constexpr size_t max_alloc_size = PTRDIFF_MAX;
    if (size > max_alloc_size) [[unlikely]] {
      throw std::invalid_argument("aligned_allocator: allocation size too large");
    }

    void * ptr = std::aligned_alloc(Alignment, size);
    if (!ptr) {
      throw std::bad_alloc();
    }

    // Can't do arithmetic on void *, so cast to char * first.
    void * const shifted_ptr = static_cast<char *>(ptr) + Alignment;  // Adjust for front padding.
    return static_cast<T *>(shifted_ptr);
  }

  template <class T, size_t Alignment>
  void aligned_allocator<T, Alignment>::deallocate(T * ptr, size_type) noexcept {
    void * const original_ptr =
        static_cast<char *>(static_cast<void *>(ptr)) - Alignment;  // Adjust for front padding.
    std::free(original_ptr);
  }

  template <class T, class U, size_t AlignmentLhs, size_t AlignmentRhs>
  bool operator==(aligned_allocator<T, AlignmentLhs> const & /* lhs */,
                  aligned_allocator<U, AlignmentRhs> const & /* rhs */) noexcept {
    return AlignmentLhs == AlignmentRhs;
  }

}  // namespace aoc25
