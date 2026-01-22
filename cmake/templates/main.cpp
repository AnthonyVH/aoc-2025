// clang-format off
#include GENERATED_HEADER_FILE
// clang-format on

#include "aoc25/day.hpp"
#include "aoc25/file.hpp"
#include "aoc25/logging.hpp"
#include "aoc25/preprocessor.hpp"

#include <fmt/ranges.h>
#include <fmt/std.h>
#include <spdlog/cfg/argv.h>
#include <spdlog/cfg/env.h>
#include <spdlog/common.h>
#include <spdlog/spdlog.h>

#include <cstddef>
#include <filesystem>
#include <utility>

namespace {

  using namespace aoc25;

  template <size_t Day, size_t Part>
  void run_day_part() {
    using input_t = decltype(read_file(std::declval<std::filesystem::path>()));
    static constexpr auto & version_info = highest_version_for_part<Day, Part, input_t>;
    static constexpr bool can_run =
        version_info.has_versions || invocable_for_part<Day, Part, input_t>;

    if constexpr (can_run) {
      auto const msg_prefix = fmt::format("[day {:02} - part {}]", Day, Part);

      std::filesystem::path const input_path = input_file_path(STRINGIFY(INPUT_DIR), Day, Part);
      if (!std::filesystem::exists(input_path)) {  // Skip part if file doesn't exist.
        spdlog::error("{} Input file does not exist ({})", msg_prefix, input_path);
        return;
      }

      day_t<Day> day{};
      static constexpr auto tag = part<Part>;

      auto input = read_file(std::move(input_path));
      auto const result = [&] {  // Only run a single version.
        if constexpr (version_info.has_versions) {
          return day.solve(tag, version<version_info.highest_version>, std::move(input));
        } else {
          return day.solve(tag, std::move(input));
        }
      }();
      spdlog::info("{} {}", msg_prefix, result);
    }
  }

  template <size_t Day>
  void run_day() {
    static constexpr size_t max_parts = 2;
    static constexpr auto parts = std::make_index_sequence<max_parts>{};

    auto const invoker = []<size_t... Part>(std::index_sequence<Part...>) {
      (..., run_day_part<Day, Part + 1>());
    };
    invoker(parts);
  }

  template <size_t... Days>
  void run_days() {
    (..., run_day<Days>());
  }

}  // namespace

int main(int argc, char ** argv) {
  // Setup default logging.
  aoc25::setup_logging(argc, argv);

  run_days<DAY_NUMBERS>();
}
