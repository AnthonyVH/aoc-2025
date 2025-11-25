// clang-format off
#include GENERATED_HEADER_FILE
// clang-format on

#include "aoc25/day.hpp"
#include "aoc25/file.hpp"
#include "aoc25/preprocessor.hpp"
#include "aoc25/system.hpp"

#include <benchmark/benchmark.h>
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

  template <size_t Day, size_t Part, size_t Version = static_cast<size_t>(-1)>
  void benchmark_day_part(benchmark::State & state) {
    auto input_path = input_file_path(STRINGIFY(INPUT_DIR), Day, Part);

    // Skip benchmarks for which no input files exist.
    if (!std::filesystem::exists(input_path)) {
      auto const msg = fmt::format("Input file does not exist ({})", input_path);
      state.SkipWithError(msg);
      return;
    }

    auto const input = read_file(std::move(input_path));

    // Suppress all non-critical logging inside solvers. Note that we can't completely
    // disable logging, since the source code for each day is already compiled into a separate
    // library, with SPDLOG_ACTIVE_LEVEL set to a specific value.
    auto const prev_log_level = spdlog::get_level();
    spdlog::set_level(spdlog::level::warn);

    try {
      for (auto _ : state) {
        // Internal state might change when calling solve, so always recreate the day_t object.
        day_t<Day> day{};
        static constexpr auto tag = part<Part>;

        if constexpr (Version == static_cast<size_t>(-1)) {
          benchmark::DoNotOptimize(day.solve(tag, input));
        } else {
          benchmark::DoNotOptimize(day.solve(tag, version<Version>, input));
        }
      }
    } catch (std::exception const & ex) {
      auto const msg = fmt::format("Exception: {}", ex.what());
      state.SkipWithError(msg);
    } catch (...) {
      auto const msg = fmt::format("Unknown exception");
      state.SkipWithError(msg);
    }

    // Restore previous logging level.
    spdlog::set_level(prev_log_level);
  }

  template <size_t Day, size_t Part, bool IsMultiDayBenchmark>
  void register_day_part() {
    auto const msg_prefix = fmt::format("day {:02} - part {}", Day, Part);

    // If this is a multi-day benchmark, only run version 0 or the non-versioned solver. Otherwise
    // run all available versions.
    using input_t = decltype(read_file(std::declval<std::filesystem::path>()));
    static constexpr auto & version_info = highest_version_for_part<Day, Part, input_t>;
    static constexpr bool runMultipleVersions = !IsMultiDayBenchmark && version_info.has_versions;

    if constexpr (runMultipleVersions) {
      static constexpr auto versions = std::make_index_sequence<version_info.highest_version + 1>{};

      auto const invoker = [&]<size_t... Version>(std::index_sequence<Version...>) {
        (..., benchmark::RegisterBenchmark(fmt::format("{} v{:d}", msg_prefix, Version),
                                           &benchmark_day_part<Day, Part, Version>));
      };
      invoker(versions);
    } else {
      if constexpr (version_info.has_versions) {
        benchmark::RegisterBenchmark(msg_prefix,
                                     &benchmark_day_part<Day, Part, version_info.highest_version>);
      } else if constexpr (invocable_for_part<Day, Part, input_t>) {
        benchmark::RegisterBenchmark(msg_prefix, &benchmark_day_part<Day, Part>);
      }
    }
  }

  template <size_t Day, bool IsMultiDayBenchmark>
  void register_day() {
    static constexpr size_t max_parts = 2;
    static constexpr auto parts = std::make_index_sequence<max_parts>{};

    auto const invoker = []<size_t... Part>(std::index_sequence<Part...>) {
      (..., register_day_part<Day, Part + 1, IsMultiDayBenchmark>());
    };
    invoker(parts);
  }

  template <size_t... Days>
  void register_days() {
    bool constexpr static isMultiDayBenchmark = sizeof...(Days) > 1;
    (..., register_day<Days, isMultiDayBenchmark>());
  }

}  // namespace

int main(int argc, char ** argv) {
  // Disable ASLR, if possible, for more consistent benchmarking results.
  benchmark::MaybeReenterWithoutASLR(argc, argv);

  // Setup default logging
  spdlog::set_level(static_cast<spdlog::level::level_enum>(SPDLOG_ACTIVE_LEVEL));

  // Try and set logging from env and command line args (in that order of preference).
  spdlog::cfg::load_env_levels();
  spdlog::cfg::load_argv_levels(argc, argv);

  // Set CPU affinity to P-cores only, no weak E-cores.
  aoc25::set_affinity_to_p_cores();

  benchmark::Initialize(&argc, argv);

  // Register benchmarks for each day.
  register_days<DAY_NUMBERS>();

  benchmark::RunSpecifiedBenchmarks();
}
