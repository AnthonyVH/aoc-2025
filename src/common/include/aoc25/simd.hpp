#pragma once

#include "aoc25/memory.hpp"

#include <cstddef>
#include <iterator>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace aoc25 {

  static constexpr size_t simd_alignment_bytes = 512 / 8;

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

  // The types below define objects that are safe to use with SIMD loads/stores.
  template <class CharT, class Traits = std::char_traits<CharT>>
  using basic_simd_string_t =
      std::basic_string<CharT, Traits, aligned_allocator<CharT, simd_alignment_bytes>>;

  using simd_string_t = basic_simd_string_t<char>;

  template <class T>
  using simd_vector_t = std::vector<T, aligned_allocator<T, simd_alignment_bytes>>;

  template <class T, size_t Extent = std::dynamic_extent>
  class simd_span_t {
   public:
    using span_t = std::span<T, Extent>;

    using element_type = typename span_t::element_type;
    using value_type = typename span_t::value_type;
    using size_type = typename span_t::size_type;
    using difference_type = typename span_t::difference_type;
    using pointer = typename span_t::pointer;
    using const_pointer = typename span_t::const_pointer;
    using reference = typename span_t::reference;
    using const_reference = typename span_t::const_reference;
    using iterator = typename span_t::iterator;
    using reverse_iterator = typename span_t::reverse_iterator;

#if __cpp_lib_ranges_as_const >= 202311L
    using const_iterator = typename span_t::const_iterator;
    using const_reverse_iterator = typename span_t::const_reverse_iterator;
#endif  // __cpp_lib_ranges_as_const >= 202311L

    template <size_t Offset, size_t Count>
    static constexpr size_t FinalExtent = (Count != std::dynamic_extent)  ? Count
                                        : (Extent != std::dynamic_extent) ? Extent - Offset
                                                                          : std::dynamic_extent;

    static constexpr size_t extent = Extent;

    constexpr simd_span_t() noexcept = default;

    template <class Rng>
      requires detail::aligned_range<Rng, simd_alignment_bytes> &&
               (!std::is_rvalue_reference_v<Rng>)
    explicit(extent != std::dynamic_extent) constexpr simd_span_t(Rng && rng);

    // Allow implicit conversion from simd_span_t<T> to simd_span_t<T const>.
    template <class U>
      requires std::is_const_v<T> && (!std::is_const_v<U>) &&
               std::is_same_v<std::remove_cv_t<T>, std::remove_cv_t<U>>
    constexpr simd_span_t(simd_span_t<U, Extent> const & other) noexcept;

    // Explicit conversions to std::span<T, Extent>.
    explicit operator span_t() const noexcept;
    span_t as_span() const noexcept;

    // Iterators
    constexpr auto begin() const noexcept;
    constexpr auto cbegin() const noexcept;

    constexpr auto end() const noexcept;
    constexpr auto cend() const noexcept;

    constexpr auto rbegin() const noexcept;
    constexpr auto crbegin() const noexcept;

    constexpr auto rend() const noexcept;
    constexpr auto crend() const noexcept;

    // Element access
    constexpr reference operator[](size_type idx) const noexcept;

    constexpr reference front() const;
    constexpr reference back() const;

    constexpr pointer data() const noexcept;

    // Observers
    constexpr size_type size() const noexcept;
    constexpr size_type size_bytes() const noexcept;

    constexpr bool empty() const noexcept;

    // Subviews
    template <size_t Count>
    constexpr simd_span_t<element_type, Count> first() const;

    constexpr simd_span_t<element_type, std::dynamic_extent> first(size_type count) const;

    template <size_t Count>
    constexpr simd_span_t<element_type, Count> last() const;

    constexpr simd_span_t<element_type, std::dynamic_extent> last(size_type count) const;

    template <size_t Offset, size_t Count = std::dynamic_extent>
    constexpr simd_span_t<element_type, FinalExtent<Offset, Count>> subspan() const;

    constexpr simd_span_t<element_type, std::dynamic_extent> subspan(
        size_type offset,
        size_type count = std::dynamic_extent) const;

   private:
    span_t span_;

    // Allow all other simd_span_t instantiations to call the private constructor.
    template <class U, size_t OtherExtent>
    friend class simd_span_t;

    constexpr explicit simd_span_t(span_t span) noexcept;
  };

  // Deduction guides for simd_span_t.
  template <class It, class EndOrSize>
  simd_span_t(It, EndOrSize) -> simd_span_t<std::remove_reference_t<std::iter_reference_t<It>>>;

  template <class T, std::size_t N>
  simd_span_t(T (&)[N]) -> simd_span_t<T, N>;

  template <class T, std::size_t N>
  simd_span_t(std::array<T, N> &) -> simd_span_t<T, N>;

  template <class T, std::size_t N>
  simd_span_t(std::array<T, N> const &) -> simd_span_t<T const, N>;

  template <class Rng>
  simd_span_t(Rng &&) -> simd_span_t<std::remove_reference_t<std::ranges::range_reference_t<Rng>>>;

  template <class CharT, class Traits = std::char_traits<CharT>>
  struct basic_simd_string_view_t {
   public:
    using basic_string_view_t = std::basic_string_view<CharT, Traits>;

    using traits_type = typename basic_string_view_t::traits_type;
    using value_type = typename basic_string_view_t::value_type;
    using pointer = typename basic_string_view_t::pointer;
    using const_pointer = typename basic_string_view_t::const_pointer;
    using reference = typename basic_string_view_t::reference;
    using const_reference = typename basic_string_view_t::const_reference;
    using const_iterator = typename basic_string_view_t::const_iterator;
    using iterator = const_iterator;
    using const_reverse_iterator = typename basic_string_view_t::const_reverse_iterator;
    using reverse_iterator = const_reverse_iterator;
    using size_type = typename basic_string_view_t::size_type;
    using difference_type = typename basic_string_view_t::difference_type;

    using allocator_type = typename basic_simd_string_t<CharT, Traits>::allocator_type;

    static constexpr size_type npos = basic_string_view_t::npos;

    constexpr basic_simd_string_view_t() noexcept = default;

    // Implicit conversion from basic_simd_string_t.
    constexpr basic_simd_string_view_t(basic_simd_string_t<CharT, Traits> const &) noexcept;

    // Implicit conversion to std::string_view.
    [[nodiscard]] constexpr operator basic_string_view_t() const noexcept;
    [[nodiscard]] constexpr basic_string_view_t as_view() const noexcept;

    [[nodiscard]] constexpr simd_span_t<CharT const> as_span() const noexcept;

    // Iterators
    [[nodiscard]] constexpr auto begin() const noexcept;
    [[nodiscard]] constexpr auto cbegin() const noexcept;

    [[nodiscard]] constexpr auto end() const noexcept;
    [[nodiscard]] constexpr auto cend() const noexcept;

    [[nodiscard]] constexpr auto rbegin() const noexcept;
    [[nodiscard]] constexpr auto crbegin() const noexcept;

    [[nodiscard]] constexpr auto rend() const noexcept;
    [[nodiscard]] constexpr auto crend() const noexcept;

    // Element access
    [[nodiscard]] constexpr const_reference operator[](size_type idx) const noexcept;
    [[nodiscard]] constexpr const_reference at(size_type idx) const;

    [[nodiscard]] constexpr const_reference front() const;
    [[nodiscard]] constexpr const_reference back() const;

    [[nodiscard]] constexpr const_pointer data() const noexcept;

    // Capacity
    [[nodiscard]] constexpr size_type size() const noexcept;
    [[nodiscard]] constexpr size_type length() const noexcept;

    [[nodiscard]] constexpr bool empty() const noexcept;

    // Modifiers
    constexpr void remove_prefix(size_type n) noexcept;
    constexpr void remove_suffix(size_type n) noexcept;

    constexpr void swap(basic_simd_string_view_t & other) noexcept;

    // Operations
    constexpr size_type copy(char * dest, size_type n, size_type pos = 0) const;

    [[nodiscard]] constexpr basic_simd_string_view_t substr(
        size_type pos = 0,
        size_type n = basic_simd_string_view_t::npos) const;

    // TODO: Implement search-related functions with SIMD acceleration.
    [[nodiscard]] constexpr bool starts_with(basic_simd_string_view_t sv) const noexcept;
    [[nodiscard]] constexpr bool starts_with(basic_string_view_t sv) const noexcept;
    [[nodiscard]] constexpr bool starts_with(char ch) const noexcept;
    [[nodiscard]] constexpr bool starts_with(char const * s) const noexcept;

    [[nodiscard]] constexpr bool ends_with(basic_simd_string_view_t sv) const noexcept;
    [[nodiscard]] constexpr bool ends_with(basic_string_view_t sv) const noexcept;
    [[nodiscard]] constexpr bool ends_with(char ch) const noexcept;
    [[nodiscard]] constexpr bool ends_with(char const * s) const noexcept;

    [[nodiscard]] constexpr size_type find(basic_simd_string_view_t v,
                                           size_type pos = 0) const noexcept;
    [[nodiscard]] constexpr size_type find(basic_string_view_t v, size_type pos = 0) const noexcept;
    [[nodiscard]] constexpr size_type find(CharT ch, size_type pos = 0) const noexcept;
    [[nodiscard]] constexpr size_type find(CharT const * s, size_type pos, size_type count) const;
    [[nodiscard]] constexpr size_type find(CharT const * s, size_type pos = 0) const;

    [[nodiscard]] constexpr size_type rfind(basic_simd_string_view_t v,
                                            size_type pos = npos) const noexcept;
    [[nodiscard]] constexpr size_type rfind(basic_string_view_t v,
                                            size_type pos = npos) const noexcept;
    [[nodiscard]] constexpr size_type rfind(CharT ch, size_type pos = npos) const noexcept;
    [[nodiscard]] constexpr size_type rfind(CharT const * s, size_type pos, size_type count) const;
    [[nodiscard]] constexpr size_type rfind(CharT const * s, size_type pos = npos) const;

    [[nodiscard]] constexpr size_type find_last_not_of(basic_string_view_t v,
                                                       size_type pos = npos) const noexcept;
    [[nodiscard]] constexpr size_type find_last_not_of(CharT ch,
                                                       size_type pos = npos) const noexcept;
    [[nodiscard]] constexpr size_type find_last_not_of(CharT const * s,
                                                       size_type pos,
                                                       size_type count) const;
    [[nodiscard]] constexpr size_type find_last_not_of(CharT const * s, size_type pos = npos) const;

    // Non-member functions
    [[nodiscard]] friend auto operator<=>(basic_simd_string_view_t const & lhs,
                                          basic_simd_string_view_t const & rhs) noexcept = default;

   private:
    basic_string_view_t view_;

    constexpr explicit basic_simd_string_view_t(basic_string_view_t view) noexcept;
  };

  using simd_string_view_t = basic_simd_string_view_t<char>;

  template <std::integral T>
  uint64_t count(simd_span_t<T const> span, T const & value);

  template <class T>
    requires std::is_arithmetic_v<T>
  size_t find_minimum_pos(simd_span_t<T const> input);

}  // namespace aoc25

#include "aoc25/simd-hwy.hpp"
#include "aoc25/simd.tpp"
