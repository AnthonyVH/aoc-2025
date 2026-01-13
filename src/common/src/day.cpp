#include "aoc25/day.hpp"

#include "aoc25/file.hpp"

#include <fmt/format.h>
#include <fmt/std.h>

#include <ranges>
#include <regex>

namespace aoc25 {

  std::filesystem::path input_file_path(std::filesystem::path input_dir, size_t day, size_t part) {
    return std::move(input_dir) / fmt::format("day_{:02d}-part_{}.txt", day, part);
  }

  std::vector<std::filesystem::path> verify_file_paths(std::filesystem::path const & dir,
                                                       size_t day,
                                                       size_t part) {
    // Find all files matching the pattern "{prefix}(?:-example_{number})?.txt"
    auto const pattern =
        std::regex(fmt::format(R"(^day_{:02d}-part_{:d}(?:-example_\d+)?\.txt$)", day, part));

    // Compiler error if entries is not a separate variable.
    auto entries = std::filesystem::directory_iterator(dir);
    auto matches = entries | std::views::filter([](auto const & e) {
                     return std::filesystem::is_regular_file(resolve_symlink(e));
                   }) |
                   std::views::filter([&](auto const & e) {
                     auto const filename = e.path().filename().native();
                     return std::regex_match(filename, pattern);
                   }) |
                   std::views::filter([&](auto const & e) {
                     // Check that solution file exists.
                     auto const & path = e.path();
                     auto const solution_file =
                         path.parent_path() /
                         path.stem().concat("-solution").concat(path.extension().native());
                     return std::filesystem::exists(solution_file);
                   }) |
                   std::views::transform([](auto const & e) { return e.path(); });

    // For some reason std::ranges::to doesn't compile here.
    std::vector<std::filesystem::path> result;
    std::ranges::copy(matches, std::back_inserter(result));

    // And sort...
    std::ranges::sort(result);

    return result;
  }

}  // namespace aoc25
