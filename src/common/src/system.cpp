#include "aoc25/system.hpp"

#include "aoc25/file.hpp"
#include "aoc25/string.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <spdlog/spdlog.h>

#include <sched.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <ranges>
#include <regex>
#include <vector>

namespace aoc25 {

  namespace {

    std::vector<uint8_t> get_all_core_indices() {
      std::vector<uint8_t> result;

      auto const matcher = std::regex(R"(cpu([0-9]+))");
      auto const cpu_info_dir = std::filesystem::path("/sys/devices/system/cpu/");

      for (auto const & entry : std::filesystem::directory_iterator{cpu_info_dir}) {
        if (!entry.is_directory()) {
          continue;
        }

        auto const dirname = entry.path().filename();
        std::smatch match;
        if (std::regex_match(dirname.native(), match, matcher)) {
          result.push_back(to_int<uint8_t>(match.str(1)));
        }
      }

      std::ranges::sort(result);

      return result;
    }

    /** @brief Returns a list of performance core indices. If the current CPU has only a single type
     * of core, then all of them are threated as performance cores.
     */
    std::vector<uint8_t> get_p_core_indices() {
      auto const cores_info_path = std::filesystem::path("/sys/devices/cpu_core/cpus");

      // Parse CPUs string, which consists of ranges (e.g. "0-3"), potentially separated by commas.
      simd_string_t cpus;

      try {
        cpus = read_file(cores_info_path);
      } catch (file_read_error const &) {
        return get_all_core_indices();
      }

      std::vector<uint8_t> result;
      auto cpus_view = std::string_view{cpus};

      while (!cpus_view.empty()) {
        auto const comma_pos = cpus_view.find(',');
        auto const range = cpus_view.substr(0, comma_pos);
        auto const dash_pos = range.find('-');

        uint8_t start_core;
        uint8_t end_core;

        if (dash_pos != std::string_view::npos) {  // Range of cores.
          start_core = to_int<uint8_t>(range.substr(0, dash_pos));
          end_core = to_int<uint8_t>(range.substr(dash_pos + 1));
        } else {  // Single core.
          start_core = to_int<uint8_t>(range);
          end_core = start_core;
        }

        // Add all cores to result.
        auto cores_range = std::views::iota(start_core, static_cast<uint8_t>(end_core + 1));
        result.insert(result.end(), cores_range.begin(), cores_range.end());

        // Remove processed part of input string.
        cpus_view.remove_prefix(comma_pos == std::string_view::npos ? cpus_view.size()
                                                                    : comma_pos + 1);
      }

      return result;
    }

    void set_cpu_affinity(std::span<uint8_t const> cpu_indices) {
      cpu_set_t set;
      CPU_ZERO(&set);

      for (uint8_t const cpu_index : cpu_indices) {
        CPU_SET(cpu_index, &set);
      }

      spdlog::info("Setting CPU affinity to cores {}", cpu_indices);
      auto const result = sched_setaffinity(0, sizeof(set), &set);
      if (result != 0) {
        throw std::runtime_error(fmt::format("Failed to set CPU affinity on cores {} (error: {})",
                                             cpu_indices, std::strerror(errno)));
      }
    }

  }  // namespace

  void set_affinity_to_p_cores() {
    auto const cpu_cores = get_p_core_indices();
    set_cpu_affinity(cpu_cores);
  }

}  // namespace aoc25
