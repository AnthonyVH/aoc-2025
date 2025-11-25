#pragma once

#include "aoc25/memory.hpp"  // Enable IDE syntax highlighting.

#include <spdlog/pattern_formatter.h>
#include <spdlog/spdlog.h>

#include <new>

namespace aoc25 {

  template <class T, size_t Alignment>
  template <class U, class Enable>
  aligned_allocator<T, Alignment>::aligned_allocator(
      aligned_allocator<U, Alignment> const&) noexcept {};

  template <class T, size_t Alignment>
  T* aligned_allocator<T, Alignment>::allocate(size_type n) {
    size_t size = n * sizeof(T);

    size = (size + (Alignment - 1)) / Alignment * Alignment;
    size += Alignment;  // Add extra space to allow for SIMD loads right at the end.
    assert(size % Alignment == 0);

    void* ptr = std::aligned_alloc(Alignment, size);
    if (!ptr) {
      throw std::bad_alloc();
    }
    return static_cast<T*>(ptr);
  }

  template <class T, size_t Alignment>
  void aligned_allocator<T, Alignment>::deallocate(T* ptr, size_type) noexcept {
    std::free(ptr);
  }

  template <class T, class U, size_t AlignmentLhs, size_t AlignmentRhs>
  bool operator==(aligned_allocator<T, AlignmentLhs> const& /* lhs */,
                  aligned_allocator<U, AlignmentRhs> const& /* rhs */) noexcept {
    return AlignmentLhs == AlignmentRhs;
  }

  template <size_t Alignment, class T>
  aligned_span<Alignment, T>::aligned_span(T* data, size_t size) : base_t(data, size) {}

  template <size_t Alignment, class T>
  template <class Rng>
    requires detail::aligned_range<Rng, Alignment> && (!std::is_rvalue_reference_v<Rng>)
  aligned_span<Alignment, T>::aligned_span(Rng&& rng) : base_t{rng} {}

  template <size_t Alignment, class T>
  size_t aligned_span<Alignment, T>::aligned_size() const noexcept {
    size_t size = this->size_bytes();
    return (size + (Alignment - 1)) / Alignment * Alignment;
  }

}  // namespace aoc25
