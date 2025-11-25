#include "aoc25/file.hpp"

#include "aoc25/memory.hpp"
#include "aoc25/simd.hpp"
#include "aoc25/string.hpp"

#include <fmt/std.h>

#include <cstdlib>
#include <fstream>
#include <iterator>

namespace aoc25 {

  simd_string_t read_file(std::filesystem::path const & file) {
    auto ifile = std::ifstream{file};
    if (!ifile.is_open()) {
      throw file_read_error(fmt::format("Failed to open file ({})", file));
    }

    auto result = trim(
        simd_string_t(std::istreambuf_iterator<char>{ifile}, std::istreambuf_iterator<char>{}));

    // Ensure the last character is a newline. This makes parsing lines easier (i.e. no need to
    // check for either '\n' or EOF).
    result.push_back('\n');

    return result;
  }

  std::filesystem::path resolve_symlink(std::filesystem::path path) {
    if (!std::filesystem::is_symlink(path)) {
      return path;
    }

    auto const target = std::filesystem::read_symlink(path);
    if (target.is_absolute()) {
      path = std::move(target);
    } else {
      path = std::move(path).parent_path() / target;
      path = std::move(path).lexically_normal();
    }

    return resolve_symlink(path);
  }

}  // namespace aoc25
