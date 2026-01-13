#include "aoc25/string.hpp"

#include <fmt/base.h>

#include <algorithm>
#include <charconv>
#include <stdexcept>

namespace aoc25 {

  template <class StringLike>
    requires requires(StringLike s) {
      { s.substr(0, 1) } -> std::same_as<StringLike>;
    }
  StringLike trim(StringLike s) {
    auto const is_nonspace = [](unsigned char c) { return !std::isspace(c); };

    auto const new_begin = std::ranges::find_if(s, is_nonspace);

    if (new_begin == s.end()) {  // If the string is all whitespace, stop.
      return {};
    }

    // We know there's at least one non-whitespace character. So the result
    // from find_last_if can't be s.end().
    auto const new_end =
#if __cpp_lib_ranges_find_last >= 202207L
        std::next(std::ranges::find_last_if(s, is_nonspace).begin());
#else
        std::find_if(s.rbegin(), s.rend(), is_nonspace).base();
#endif

    size_t const start_index = std::distance(s.begin(), new_begin);
    size_t const new_length = std::distance(new_begin, new_end);

    return std::move(s).substr(start_index, new_length);
  }

  template <class T>
  T to_int(std::string_view str) {
    T value;
    [[maybe_unused]] auto const [_, ec] =
        std::from_chars(str.data(), str.data() + str.size(), value);
    if (ec != std::errc{}) [[unlikely]] {
      throw std::invalid_argument(fmt::format("Failed to convert '{}' to integer", str));
    }
    return value;
  }

}  // namespace aoc25
