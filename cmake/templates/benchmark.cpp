// clang-format off
#include <fmt/color.h>
#include <stdexcept>
#include <string_view>
#include GENERATED_HEADER_FILE
// clang-format on

#include "aoc25/day.hpp"
#include "aoc25/file.hpp"
#include "aoc25/logging.hpp"
#include "aoc25/preprocessor.hpp"
#include "aoc25/system.hpp"

#include <benchmark/benchmark.h>
#include <fmt/ranges.h>
#include <fmt/std.h>
#include <magic_enum/magic_enum.hpp>
#include <spdlog/cfg/argv.h>
#include <spdlog/cfg/env.h>
#include <spdlog/common.h>
#include <spdlog/spdlog.h>

#include <cstddef>
#include <filesystem>
#include <ranges>
#include <utility>

namespace {

  using namespace aoc25;

  class UnicodeReporter : public benchmark::ConsoleReporter {
   public:
    UnicodeReporter(OutputOptions opts = ConsoleReporter::OO_Defaults)
        : ConsoleReporter(opts) {
      if ((output_options_ & OO_Tabular) == 0) {
        throw std::invalid_argument("UnicodeReporter only supports tabular output");
      }
    }

   private:
    using base_t = benchmark::ConsoleReporter;

    enum class FixedColumn { Time, CPU, Iterations };

    struct stats_t {
      benchmark::TimeUnit time_unit;
      double time;
      double cpu;
    };

    static constexpr std::array<size_t, magic_enum::enum_count<FixedColumn>()> fixed_col_width =
        std::to_array<size_t>({13, 15, 12});
    static constexpr std::array<std::string_view, magic_enum::enum_count<FixedColumn>()>
        fixed_col_name = std::to_array<std::string_view>({"Time", "CPU", "Iterations"});

    static constexpr size_t misc_field_width = 10;

    size_t num_numeric_columns_;
    size_t total_width_;
    std::vector<stats_t> run_stats_;

    // Unicode character might not fit in a single character, so get creative.
    std::string repeat_unicode(std::string_view unicode_char, size_t count) {
      return fmt::format("{:s}", fmt::join(std::views::repeat(unicode_char, count), ""));
    };

    void clear_state() {
      num_numeric_columns_ = 0;
      total_width_ = 0;
      run_stats_.clear();
    }

    void verify_stats() {
      if (run_stats_.empty()) {
        return;
      }

      bool const time_units_match = std::ranges::all_of(run_stats_, [&](stats_t const & stats) {
        return stats.time_unit == run_stats_.at(0).time_unit;
      });
      if (!time_units_match) {
        throw std::runtime_error("Mismatched time units in benchmark runs");
      }
    }

    static std::string format_time(double time) {
      // Matches benchmark::ConsoleReporter::FormatTime() logic.

      // Assuming the time is at max 9.9999e+99 and we have 10 digits for the
      // number, we get 10-1(.)-1(e)-1(sign)-2(exponent) = 5 digits to print.
      if (time > 9999999999 /*max 10 digit number*/) {
        return fmt::format("{:1.4e}", time);
      }

      // For the time columns of the console printer 13 digits are reserved. One of
      // them is a space and max two of them are the time unit (e.g ns). That puts
      // us at 10 digits usable for the number.
      // Align decimal places...
      int width = 10;
      int precision = 0;

      if (time < 1.0) {
        precision = 3;
      } else if (time < 10.0) {
        precision = 2;
      } else if (time < 100.0) {
        precision = 1;
      }

      return fmt::format("{:>{}.{}f}", time, width, precision);
    }

   protected:
    void PrintHeader(Run const & run) override {
      // Left border + space + name field + # fields * (space + field width) + space + right border.
      bool const has_counters = !run.counters.empty();
      bool const is_tabular = (output_options_ & OO_Tabular) != 0;
      size_t const num_extra_columns = has_counters ? (is_tabular ? run.counters.size() : 1) : 0;
      size_t const width_from_fixed_cols = std::ranges::fold_left(
          fixed_col_width, magic_enum::enum_count<FixedColumn>(), std::plus<>());

      total_width_ = 1 + 1 + name_field_width_ + width_from_fixed_cols +
                     num_extra_columns * (1 + misc_field_width) + 1 + 1;

      // Border above header row.
      std::string str = fmt::format("╭{}╮\n", repeat_unicode("─", total_width_ - 2));

      // Header row.
      str += fmt::format("│ {:<{}}", "Benchmark", name_field_width_);

      for (auto const & e : magic_enum::enum_values<FixedColumn>()) {
        size_t const idx = std::to_underlying(e);
        str += fmt::format(" {:>{}}", fixed_col_name.at(idx), fixed_col_width.at(idx));
      }

      if (has_counters) {
        if (is_tabular) {
          for (auto const & e : run.counters) {
            str += fmt::format(" {:>{}}", e.first, misc_field_width);
          }
        } else {
          str += fmt::format(" {:>{}}", "User Ctrs", misc_field_width);
        }
      }

      str += " │\n";

      // Line underneath header row.
      str += fmt::format("╰{}╯\n", repeat_unicode("─", total_width_ - 2));

      fmt::print(GetOutputStream(), "{}", str);
    }

    void PrintRunData(Run const & result) override {
      fmt::print(GetOutputStream(), "  ");
      base_t::PrintRunData(result);

      // Store stats for later printing in Finalize().
      if (result.run_type != Run::RT_Aggregate ||
          result.aggregate_unit == benchmark::StatisticUnit::kTime) {
        run_stats_.emplace_back(stats_t{
            .time_unit = result.time_unit,
            .time = result.GetAdjustedRealTime(),
            .cpu = result.GetAdjustedCPUTime(),
        });
      }
    }

    void Finalize() override {
      bool const use_color = (output_options_ & OO_Color) != 0;
      auto const add_style = [&](std::string str, fmt::text_style style) {
        return fmt::styled(std::move(str), use_color ? style : fmt::text_style{});
      };

      std::string str = fmt::format("  {}\n", repeat_unicode("─", total_width_ - 4));

      str += fmt::format("  {:<{}}", add_style("Total", fmt::fg(fmt::terminal_color::green)),
                         name_field_width_);

      verify_stats();

      if (!run_stats_.empty()) {
        std::string const time_unit_str = benchmark::GetTimeUnitString(run_stats_.at(0).time_unit);

        auto const total_time = std::ranges::fold_left(
            run_stats_, 0.0, [](double acc, stats_t const & stats) { return acc + stats.time; });
        auto const total_cpu = std::ranges::fold_left(
            run_stats_, 0.0, [](double acc, stats_t const & stats) { return acc + stats.cpu; });

        size_t const time_value_width =
            fixed_col_width.at(std::to_underlying(FixedColumn::Time)) - time_unit_str.size() - 1;
        str += fmt::format(
            " {:>{}s} {}", add_style(format_time(total_time), fmt::fg(fmt::terminal_color::yellow)),
            time_value_width, add_style(time_unit_str, fmt::fg(fmt::terminal_color::yellow)));

        size_t const cpu_value_width =
            fixed_col_width.at(std::to_underlying(FixedColumn::CPU)) - time_unit_str.size() - 1;
        str += fmt::format(
            " {:>{}s} {}", add_style(format_time(total_cpu), fmt::fg(fmt::terminal_color::yellow)),
            cpu_value_width, add_style(time_unit_str, fmt::fg(fmt::terminal_color::yellow)));

        str += '\n';
      }

      fmt::print(GetOutputStream(), "{}", str);

      clear_state();
    }
  };

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
  // If OpenMP is used, ensure that it keeps threads spinning and pinned to the same cores.
  setenv("OMP_WAIT_POLICY", "active", 1);
  setenv("OMP_PROC_BIND", "spread", 1);  // Seems to be fastest overall.
  setenv("OMP_PLACES", "cores", 1);      // Pin to cores, not hyper-threads.

  // Disable ASLR, if possible, for more consistent benchmarking results.
  benchmark::MaybeReenterWithoutASLR(argc, argv);

  // Setup default logging
  aoc25::setup_logging(argc, argv);

  // Set CPU affinity to P-cores only, no weak E-cores.
  aoc25::set_affinity_to_p_cores();

  benchmark::Initialize(&argc, argv);

  // Register benchmarks for each day.
  register_days<DAY_NUMBERS>();

  UnicodeReporter reporter;
  benchmark::RunSpecifiedBenchmarks(&reporter);
}
