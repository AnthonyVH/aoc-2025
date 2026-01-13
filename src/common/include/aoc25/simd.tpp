#pragma once

#include "aoc25/simd.hpp"

namespace aoc25 {

  template <class T, size_t Extent>
  template <class Rng>
    requires detail::aligned_range<Rng, simd_alignment_bytes> && (!std::is_rvalue_reference_v<Rng>)
  constexpr simd_span_t<T, Extent>::simd_span_t(Rng && rng) : span_(rng) {}

  template <class T, size_t Extent>
  constexpr simd_span_t<T, Extent>::simd_span_t(span_t span) noexcept : span_(span) {}

  template <class T, size_t Extent>
  simd_span_t<T, Extent>::operator span_t() const noexcept {
    return as_span();
  }

  template <class T, size_t Extent>
  auto simd_span_t<T, Extent>::as_span() const noexcept -> span_t {
    return span_;
  }

  template <class T, size_t Extent>
  constexpr auto simd_span_t<T, Extent>::begin() const noexcept {
    return span_.begin();
  }

  template <class T, size_t Extent>
  constexpr auto simd_span_t<T, Extent>::cbegin() const noexcept {
    return span_.cbegin();
  }

  template <class T, size_t Extent>
  constexpr auto simd_span_t<T, Extent>::end() const noexcept {
    return span_.end();
  }

  template <class T, size_t Extent>
  constexpr auto simd_span_t<T, Extent>::cend() const noexcept {
    return span_.cend();
  }

  template <class T, size_t Extent>
  constexpr auto simd_span_t<T, Extent>::rbegin() const noexcept {
    return span_.rbegin();
  }

  template <class T, size_t Extent>
  constexpr auto simd_span_t<T, Extent>::crbegin() const noexcept {
    return span_.crbegin();
  }

  template <class T, size_t Extent>
  constexpr auto simd_span_t<T, Extent>::rend() const noexcept {
    return span_.rend();
  }

  template <class T, size_t Extent>
  constexpr auto simd_span_t<T, Extent>::crend() const noexcept {
    return span_.crend();
  }

  template <class T, size_t Extent>
  constexpr auto simd_span_t<T, Extent>::operator[](size_type idx) const noexcept -> reference {
    return span_[idx];
  }

  template <class T, size_t Extent>
  constexpr auto simd_span_t<T, Extent>::front() const -> reference {
    return span_.front();
  }

  template <class T, size_t Extent>
  constexpr auto simd_span_t<T, Extent>::back() const -> reference {
    return span_.back();
  }

  template <class T, size_t Extent>
  constexpr auto simd_span_t<T, Extent>::data() const noexcept -> pointer {
    return span_.data();
  }

  template <class T, size_t Extent>
  constexpr auto simd_span_t<T, Extent>::size() const noexcept -> size_type {
    return span_.size();
  }

  template <class T, size_t Extent>
  constexpr auto simd_span_t<T, Extent>::size_bytes() const noexcept -> size_type {
    return span_.size_bytes();
  }

  template <class T, size_t Extent>
  constexpr bool simd_span_t<T, Extent>::empty() const noexcept {
    return span_.empty();
  }

  template <class T, size_t Extent>
  template <size_t Count>
  constexpr auto simd_span_t<T, Extent>::first() const -> simd_span_t<element_type, Count> {
    return simd_span_t<element_type, Count>(span_.template first<Count>());
  }

  template <class T, size_t Extent>
  constexpr auto simd_span_t<T, Extent>::first(size_type count) const
      -> simd_span_t<element_type, std::dynamic_extent> {
    return simd_span_t<element_type, std::dynamic_extent>(span_.first(count));
  }

  template <class T, size_t Extent>
  template <size_t Count>
  constexpr auto simd_span_t<T, Extent>::last() const -> simd_span_t<element_type, Count> {
    return simd_span_t<element_type, Count>(span_.template last<Count>());
  }

  template <class T, size_t Extent>
  constexpr auto simd_span_t<T, Extent>::last(size_type count) const
      -> simd_span_t<element_type, std::dynamic_extent> {
    return simd_span_t<element_type, std::dynamic_extent>(span_.last(count));
  }

  template <class T, size_t Extent>
  template <size_t Offset, size_t Count>
  constexpr auto simd_span_t<T, Extent>::subspan() const
      -> simd_span_t<element_type, FinalExtent<Offset, Count>> {
    return simd_span_t<element_type, Count>(span_.template subspan<Offset, Count>());
  }

  template <class T, size_t Extent>
  constexpr auto simd_span_t<T, Extent>::subspan(size_type offset, size_type count) const
      -> simd_span_t<element_type, std::dynamic_extent> {
    return simd_span_t<element_type, std::dynamic_extent>(span_.subspan(offset, count));
  }

  template <class CharT, class Traits>
  constexpr basic_simd_string_view_t<CharT, Traits>::basic_simd_string_view_t(
      basic_simd_string_t<CharT, Traits> const & view) noexcept
      : view_(view) {}

  template <class CharT, class Traits>
  constexpr basic_simd_string_view_t<CharT, Traits>::basic_simd_string_view_t(
      basic_string_view_t view) noexcept
      : view_(view) {}

  template <class CharT, class Traits>
  constexpr basic_simd_string_view_t<CharT, Traits>::operator basic_string_view_t() const noexcept {
    return view_;
  }

  template <class CharT, class Traits>
  constexpr auto basic_simd_string_view_t<CharT, Traits>::as_view() const noexcept
      -> basic_string_view_t {
    return view_;
  }

  template <class CharT, class Traits>
  constexpr auto basic_simd_string_view_t<CharT, Traits>::begin() const noexcept {
    return view_.begin();
  }

  template <class CharT, class Traits>
  constexpr auto basic_simd_string_view_t<CharT, Traits>::cbegin() const noexcept {
    return view_.cbegin();
  }

  template <class CharT, class Traits>
  constexpr auto basic_simd_string_view_t<CharT, Traits>::end() const noexcept {
    return view_.end();
  }

  template <class CharT, class Traits>
  constexpr auto basic_simd_string_view_t<CharT, Traits>::cend() const noexcept {
    return view_.cend();
  }

  template <class CharT, class Traits>
  constexpr auto basic_simd_string_view_t<CharT, Traits>::rbegin() const noexcept {
    return view_.rbegin();
  }

  template <class CharT, class Traits>
  constexpr auto basic_simd_string_view_t<CharT, Traits>::crbegin() const noexcept {
    return view_.crbegin();
  }

  template <class CharT, class Traits>
  constexpr auto basic_simd_string_view_t<CharT, Traits>::rend() const noexcept {
    return view_.rend();
  }

  template <class CharT, class Traits>
  constexpr auto basic_simd_string_view_t<CharT, Traits>::crend() const noexcept {
    return view_.crend();
  }

  template <class CharT, class Traits>
  constexpr auto basic_simd_string_view_t<CharT, Traits>::operator[](size_type idx) const noexcept
      -> const_reference {
    return view_[idx];
  }

  template <class CharT, class Traits>
  constexpr auto basic_simd_string_view_t<CharT, Traits>::at(size_type idx) const
      -> const_reference {
    return view_.at(idx);
  }

  template <class CharT, class Traits>
  constexpr auto basic_simd_string_view_t<CharT, Traits>::front() const -> const_reference {
    return view_.front();
  }

  template <class CharT, class Traits>
  constexpr auto basic_simd_string_view_t<CharT, Traits>::back() const -> const_reference {
    return view_.back();
  }

  template <class CharT, class Traits>
  constexpr auto basic_simd_string_view_t<CharT, Traits>::data() const noexcept -> const_pointer {
    return view_.data();
  }

  template <class CharT, class Traits>
  constexpr auto basic_simd_string_view_t<CharT, Traits>::size() const noexcept -> size_type {
    return view_.size();
  }

  template <class CharT, class Traits>
  constexpr auto basic_simd_string_view_t<CharT, Traits>::length() const noexcept -> size_type {
    return view_.length();
  }

  template <class CharT, class Traits>
  constexpr bool basic_simd_string_view_t<CharT, Traits>::empty() const noexcept {
    return view_.empty();
  }

  template <class CharT, class Traits>
  constexpr void basic_simd_string_view_t<CharT, Traits>::remove_prefix(size_type n) noexcept {
    view_.remove_prefix(n);
  }

  template <class CharT, class Traits>
  constexpr void basic_simd_string_view_t<CharT, Traits>::remove_suffix(size_type n) noexcept {
    view_.remove_suffix(n);
  }

  template <class CharT, class Traits>
  constexpr void basic_simd_string_view_t<CharT, Traits>::swap(
      basic_simd_string_view_t & other) noexcept {
    view_.swap(other.view_);
  }

  template <class CharT, class Traits>
  constexpr auto basic_simd_string_view_t<CharT, Traits>::copy(char * dest,
                                                               size_type n,
                                                               size_type pos) const -> size_type {
    return view_.copy(dest, n, pos);
  }

  template <class CharT, class Traits>
  constexpr auto basic_simd_string_view_t<CharT, Traits>::substr(size_type pos, size_type n) const
      -> basic_simd_string_view_t {
    return basic_simd_string_view_t(view_.substr(pos, n));
  }

  template <class CharT, class Traits>
  constexpr bool basic_simd_string_view_t<CharT, Traits>::starts_with(
      basic_simd_string_view_t sv) const noexcept {
    return view_.starts_with(sv.view_);
  }

  template <class CharT, class Traits>
  constexpr bool basic_simd_string_view_t<CharT, Traits>::starts_with(
      basic_string_view_t sv) const noexcept {
    return view_.starts_with(sv);
  }

  template <class CharT, class Traits>
  constexpr bool basic_simd_string_view_t<CharT, Traits>::starts_with(char ch) const noexcept {
    return view_.starts_with(ch);
  }

  template <class CharT, class Traits>
  constexpr bool basic_simd_string_view_t<CharT, Traits>::starts_with(
      char const * s) const noexcept {
    return view_.starts_with(s);
  }

  template <class CharT, class Traits>
  constexpr bool basic_simd_string_view_t<CharT, Traits>::ends_with(
      basic_simd_string_view_t sv) const noexcept {
    return view_.ends_with(sv.view_);
  }

  template <class CharT, class Traits>
  constexpr bool basic_simd_string_view_t<CharT, Traits>::ends_with(
      basic_string_view_t sv) const noexcept {
    return view_.ends_with(sv);
  }

  template <class CharT, class Traits>
  constexpr bool basic_simd_string_view_t<CharT, Traits>::ends_with(char ch) const noexcept {
    return view_.ends_with(ch);
  }

  template <class CharT, class Traits>
  constexpr bool basic_simd_string_view_t<CharT, Traits>::ends_with(char const * s) const noexcept {
    return view_.ends_with(s);
  }

  template <class CharT, class Traits>
  constexpr auto basic_simd_string_view_t<CharT, Traits>::find(basic_simd_string_view_t v,
                                                               size_type pos) const noexcept
      -> size_type {
    return view_.find(v.view_, pos);
  }

  template <class CharT, class Traits>
  constexpr auto basic_simd_string_view_t<CharT, Traits>::find(basic_string_view_t v,
                                                               size_type pos) const noexcept
      -> size_type {
    return view_.find(v, pos);
  }

  template <class CharT, class Traits>
  constexpr auto basic_simd_string_view_t<CharT, Traits>::find(CharT ch,
                                                               size_type pos) const noexcept
      -> size_type {
    return view_.find(ch, pos);
  }

  template <class CharT, class Traits>
  constexpr auto basic_simd_string_view_t<CharT, Traits>::find(CharT const * s,
                                                               size_type pos,
                                                               size_type count) const -> size_type {
    return view_.find(s, pos, count);
  }

  template <class CharT, class Traits>
  constexpr auto basic_simd_string_view_t<CharT, Traits>::find(CharT const * s, size_type pos) const
      -> size_type {
    return view_.find(s, pos);
  }

  template <class CharT, class Traits>
  constexpr auto basic_simd_string_view_t<CharT, Traits>::rfind(basic_simd_string_view_t v,
                                                                size_type pos) const noexcept
      -> size_type {
    return view_.rfind(v.view_, pos);
  }

  template <class CharT, class Traits>
  constexpr auto basic_simd_string_view_t<CharT, Traits>::rfind(basic_string_view_t v,
                                                                size_type pos) const noexcept
      -> size_type {
    return view_.rfind(v, pos);
  }

  template <class CharT, class Traits>
  constexpr auto basic_simd_string_view_t<CharT, Traits>::rfind(CharT ch,
                                                                size_type pos) const noexcept
      -> size_type {
    return view_.rfind(ch, pos);
  }

  template <class CharT, class Traits>
  constexpr auto basic_simd_string_view_t<CharT, Traits>::rfind(CharT const * s,
                                                                size_type pos,
                                                                size_type count) const
      -> size_type {
    return view_.rfind(s, pos, count);
  }

  template <class CharT, class Traits>
  constexpr auto basic_simd_string_view_t<CharT, Traits>::rfind(CharT const * s,
                                                                size_type pos) const -> size_type {
    return view_.rfind(s, pos);
  }

}  // namespace aoc25
