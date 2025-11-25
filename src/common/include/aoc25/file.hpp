#pragma once

#include "aoc25/simd.hpp"

#include <filesystem>

namespace aoc25 {

  /** @brief Returns trimmed file contents.
   *
   * @returns Trimmed contents of the file at `file`. The last character is always a newline. This
   * is to simplify parsing when there's multiple lines. The string's buffer is guaranteed to be a
   * multiple of aoc25::simd_alignment bytes, with entries past the string's "actual" end equal to
   * the NULL character.
   *
   * @note The string's size will reflect the number of actual characters in the input. I.e. it does
   * not include the NULL characters that are present in the buffer.
   *
   * @throws file_read_error If the file could not be opened or read.
   */
  simd_string_t read_file(std::filesystem::path const & file);

  struct file_read_error : public std::runtime_error {
    using std::runtime_error::runtime_error;
  };

  /// @brief Resolves path to its target. If path is not a symlink, returns the path itself.
  std::filesystem::path resolve_symlink(std::filesystem::path path);

}  // namespace aoc25
