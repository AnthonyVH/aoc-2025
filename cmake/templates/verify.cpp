// clang-format off
#include GENERATED_HEADER_FILE
// clang-format on

#include "aoc25/day.hpp"
#include "aoc25/file.hpp"
#include "aoc25/preprocessor.hpp"
#include "aoc25/string.hpp"

#include <fmt/color.h>
#include <fmt/ranges.h>
#include <fmt/std.h>
#include <spdlog/cfg/argv.h>
#include <spdlog/cfg/env.h>
#include <spdlog/spdlog.h>

#include <cstddef>
#include <filesystem>
#include <utility>
#include <vector>

namespace {

  using namespace aoc25;

  struct test_count_t {
    test_count_t()
        : successful(0), total(0) {}

    test_count_t & operator+=(test_count_t const & other) {
      successful += other.successful;
      total += other.total;
      return *this;
    }

    [[maybe_unused]] friend test_count_t operator+(test_count_t lhs, test_count_t const & rhs) {
      lhs += rhs;
      return lhs;
    }

    unsigned successful;
    unsigned total;
  };

  template <size_t Day, size_t Part, size_t Version, class StringLike>
  bool verify_day_part_version(std::string_view msg_prefix,
                               StringLike const & input,
                               size_t example_number,
                               std::string_view expected) {
    static constexpr bool run_version = Version != static_cast<size_t>(-1);

    auto const actual = [&] {
      // Internal state might change when calling solve, so always recreate the day_t object.
      day_t<Day> day{};
      static constexpr auto tag = part<Part>;

      if constexpr (run_version) {
        return day.solve(tag, version<Version>, input);
      } else {
        return day.solve(tag, input);
      }
    }();
    auto const actual_str = fmt::format("{}", actual);

    spdlog::info("[{}{} - example {}] {} (actual: {}, expected: {})", msg_prefix,
                 run_version ? fmt::format(" v{:d}", Version) : std::string_view{}, example_number,
                 (actual_str == expected) ? fmt::styled("PASS", fmt::fg(fmt::terminal_color::green))
                                          : fmt::styled("FAIL", fmt::fg(fmt::terminal_color::red)),
                 actual_str, expected);
    return actual_str == expected;
  }

  template <size_t Day, size_t Part>
  test_count_t verify_day_part() {
    using input_t = decltype(read_file(std::declval<std::filesystem::path>()));
    static constexpr auto & version_info = highest_version_for_part<Day, Part, input_t>;
    static constexpr bool can_run =
        version_info.has_versions || invocable_for_part<Day, Part, input_t>;

    test_count_t test_count;

    if constexpr (can_run) {
      auto const msg_prefix = fmt::format("day {:02} - part {}", Day, Part);

      auto const example_files = example_file_paths(STRINGIFY(INPUT_DIR), Day, Part);

      if (example_files.empty()) {
        spdlog::warn("[{}] No example files found", msg_prefix);
        return test_count;
      }

      // Validate each example file for each version. Iterate over example files first, since
      // this makes it easier to compare the output of different versions against each other.
      for (auto const & example_file : example_files) {
        auto const & solution_file =
            example_file.parent_path() /
            example_file.stem().concat("-solution").concat(example_file.extension().native());

        if (!std::filesystem::exists(solution_file)) {  // Skip example if file doesn't exist.
          spdlog::error("[{}] Solution file does not exist ({})", msg_prefix, solution_file);
          continue;
        }

        auto const expected = trim(read_file(solution_file));
        size_t const example_number = std::stoul(example_file.stem().native().substr(
            example_file.stem().native().find_last_of('_') + 1));

        auto input = read_file(example_file);

        if constexpr (version_info.has_versions) {
          static constexpr auto versions =
              std::make_index_sequence<version_info.highest_version + 1>{};

          auto const invoker = [&]<size_t... Version>(std::index_sequence<Version...>) {
            return (... + verify_day_part_version<Day, Part, Version>(msg_prefix, input,
                                                                      example_number, expected));
          };

          test_count.total += versions.size();
          test_count.successful += invoker(versions);
        } else {
          test_count.total += 1;
          test_count.successful += verify_day_part_version<Day, Part, static_cast<size_t>(-1)>(
              msg_prefix, input, example_number, expected);
        }
      }
    }

    return test_count;
  }

  template <size_t Day>
  test_count_t verify_day() {
    static constexpr size_t max_parts = 2;
    static constexpr auto parts = std::make_index_sequence<max_parts>{};

    auto const invoker = []<size_t... Is>(std::index_sequence<Is...>) {
      // Need fold expression to ensure proper evaluation order. Hence this mess.
      test_count_t result{};
      auto const add = [&](test_count_t tc) { result += tc; };
      (add(verify_day_part<Day, Is + 1>()), ...);
      return result;
    };
    return invoker(parts);
  }

  template <size_t... Days>
  test_count_t verify_days() {
    return (... + verify_day<Days>());
  }

}  // namespace

int main(int argc, char ** argv) {
  // Setup default logging
  spdlog::set_level(static_cast<spdlog::level::level_enum>(SPDLOG_ACTIVE_LEVEL));

  // Try and set logging from env and command line args (in that order of preference).
  spdlog::cfg::load_env_levels();
  spdlog::cfg::load_argv_levels(argc, argv);

  auto const test_counts = verify_days<DAY_NUMBERS>();
  bool const success = test_counts.total == test_counts.successful;
  spdlog::info("[summary] {} ({} passed, {} failed, {} total)",
               success ? fmt::styled("PASS", fmt::fg(fmt::terminal_color::green))
                       : fmt::styled("FAIL", fmt::fg(fmt::terminal_color::red)),
               test_counts.successful, test_counts.total - test_counts.successful,
               test_counts.total);
  return success ? 0 : 1;
}
