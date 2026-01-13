#include "aoc25/day_10.hpp"

#include "aoc25/day.hpp"
#include "aoc25/math.hpp"
#include "aoc25/simd.hpp"
#include "aoc25/string.hpp"

#include <fmt/ranges.h>
#include <fmt/std.h>
#include <magic_enum/magic_enum.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <ranges>
#include <type_traits>
#include <utility>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "src/day_10.cpp"

// clang-format off
#include <hwy/foreach_target.h>
// clang-format on

#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();

namespace aoc25 {
  namespace {
    namespace HWY_NAMESPACE {

      namespace hn = hwy::HWY_NAMESPACE;

      [[maybe_unused]] void compiler_stop_complaining() {}

    }  // namespace HWY_NAMESPACE
  }  // namespace
}  // namespace aoc25

HWY_AFTER_NAMESPACE();

#ifdef HWY_ONCE

namespace aoc25 {

  namespace {

    HWY_EXPORT(compiler_stop_complaining);

    [[maybe_unused]] void compiler_stop_complaining() {
      return HWY_DYNAMIC_DISPATCH(compiler_stop_complaining)();
    }

    struct problem_t {
      std::vector<uint16_t> switches;
      std::vector<uint16_t> jolts;
      uint16_t target = 0;
      uint8_t num_bits = 0;
    };

    std::string format_as(problem_t const & obj) {
      return fmt::format("<target: {:#0{}b}, switches: {::#0{}b}, jolts: {::d}>", obj.target,
                         obj.num_bits + 2, obj.switches, obj.num_bits + 2, obj.jolts);
    }

    problem_t parse(simd_string_view_t input) {
      problem_t result;

      assert(!input.empty());
      assert(input[0] == '[');
      auto const end_bracket_pos = input.find(']');
      assert(end_bracket_pos != simd_string_view_t::npos);

      {
        // Input is basically given MSB first.
        result.num_bits = end_bracket_pos - 1;

        for (size_t idx = 0; idx < result.num_bits; ++idx) {
          char c = input[1 + idx];
          assert(c == '.' || c == '#');
          if (c == '#') {
            result.target |= (1 << idx);
          }
        }
      }
      input.remove_prefix(end_bracket_pos + 2);

      while (input[0] == '(') {
        auto const end_paren_pos = input.find(')');
        assert(end_paren_pos != simd_string_view_t::npos);
        simd_string_view_t token = input.substr(1, end_paren_pos - 1);

        uint16_t switch_value = 0;

        while (!token.empty()) {
          auto const comma_pos = token.find(',');
          simd_string_view_t bit_token = token.substr(0, comma_pos);
          assert(bit_token.size() == 1);
          uint8_t const bit_pos = bit_token[0] - '0';
          switch_value |= (1 << bit_pos);
          token.remove_prefix(bit_token.size() + (comma_pos == simd_string_view_t::npos ? 0 : 1));
        }

        result.switches.push_back(switch_value);
        input.remove_prefix(end_paren_pos + 2);
      }

      assert(!input.empty());
      assert(input.front() == '{');
      assert(input.back() == '}');
      input = input.substr(1, input.size() - 2);  // Remove braces.

      while (!input.empty()) {
        auto const comma_pos = input.find(',');
        simd_string_view_t bit_token = input.substr(0, comma_pos);
        result.jolts.push_back(to_int<uint16_t>(bit_token));
        input.remove_prefix((comma_pos == simd_string_view_t::npos ? input.size() : comma_pos + 1));
      }

      SPDLOG_DEBUG("Parsed {}", result);
      return result;
    }

    std::vector<problem_t> parse_input(simd_string_view_t input) {
      std::vector<problem_t> result;
      result.reserve(200);
      split(input, [&](simd_string_view_t line) { result.push_back(parse(line)); });
      return result;
    }

    uint16_t apply_diff_switches(std::span<uint16_t const> switches,
                                 uint16_t combo_diff,
                                 uint16_t state) {
      while (combo_diff != 0) {
        size_t const switch_idx = std::countr_zero(combo_diff);
        state ^= switches[switch_idx];
        combo_diff &= ~(1ULL << switch_idx);  // Unset LSB.
      }
      return state;
    }

    void push_switch([[maybe_unused]] uint8_t switch_idx,
                     std::span<uint16_t> remaining_jolts,
                     uint16_t switch_bitmask,
                     int16_t number_of_pushes) {
      SPDLOG_TRACE("  [push_switch] switch {} ({:#b}) on state {} with change {}", switch_idx,
                   switch_bitmask, remaining_jolts, number_of_pushes);
      while (switch_bitmask != 0) {
        size_t const idx = std::countr_zero(switch_bitmask);
        assert(idx < remaining_jolts.size());
        remaining_jolts[idx] -= number_of_pushes;
        switch_bitmask &= ~(1ULL << idx);  // Unset LSB.
      }
    }

    enum class solve_state_t {
      no_solution,
      found_solution,
      skip_switch,
    };

    struct solve_result_t {
      solve_state_t state = solve_state_t::no_solution;
      uint16_t num_pushes = 0;
    };

    [[maybe_unused]] std::string format_as(solve_result_t const & obj) {
      return fmt::format("<state: {}, num_pushes: {}>", magic_enum::enum_name(obj.state),
                         obj.num_pushes);
    }

    template <class Caches>
    [[maybe_unused]] solve_result_t solve_num_button_pushes_impl(
        uint64_t & call_count,
        Caches & caches,
        problem_t const & problem,
        uint16_t num_switch_pushes,
        std::span<uint16_t> remaining_jolts,
        uint8_t switch_idx,
        std::optional<uint16_t> best_so_far) {
      ++call_count;

      if (auto const it = caches.at(switch_idx).find(remaining_jolts);
          it != caches.at(switch_idx).end()) {
        SPDLOG_TRACE("  [cache hit] switch_idx {}, state {} => {}", switch_idx, remaining_jolts,
                     it->second);
        return it->second;
      } else if (std::ranges::all_of(remaining_jolts, [](auto const & e) { return e == 0; })) {
        // TODO: Move this to the caller.
        SPDLOG_INFO("Found solution for {} with {} switch pushes", problem, num_switch_pushes);
        return {solve_state_t::found_solution, 0};
      } else if (num_switch_pushes >= best_so_far.value_or(INT16_MAX)) {
        // assert(false);  // TODO: Don't do this call if this is going to happen. And stop
        // iterating.
        return {.state = solve_state_t::no_solution, .num_pushes = 0};
      } else if (switch_idx >= problem.switches.size()) {
        // TODO: Catch this earlier and don't recurse?
        return {.state = solve_state_t::no_solution, .num_pushes = 0};
      }
      if (best_so_far.has_value() &&
          std::ranges::any_of(remaining_jolts, [&](auto const & e) { return e > *best_so_far; })) {
        // Need more pushes than required to beat best.
        return {.state = solve_state_t::no_solution, .num_pushes = 0};
      }

      // Push a switch.
      auto result = solve_result_t{.state = solve_state_t::no_solution, .num_pushes = UINT16_MAX};
      uint8_t const num_switches = problem.switches.size();
      int16_t const max_remaining_pushes = best_so_far.value_or(INT16_MAX) - num_switch_pushes;

      assert(switch_idx < num_switches);
      auto const & cur_switch = problem.switches.at(switch_idx);

      // Determine number of pushes needed to reach target for this switch.
      SPDLOG_TRACE("  Calculating max pushes for switch {:d} ({:#b}), remaining: {}",
                   cur_switch_idx, cur_switch, remaining_jolts);
      int16_t max_pushes = max_remaining_pushes;
      bool single_modify_found = false;

      for (uint8_t target_idx = 0; target_idx < remaining_jolts.size(); ++target_idx) {
        bool const switch_increments_jolt = (cur_switch & (1 << target_idx)) != 0;
        if (switch_increments_jolt) {
          // Check if any remaining switches can also modify this jolt.
          bool const other_switches_modify =
              std::ranges::any_of(problem.switches.begin() + switch_idx + 1, problem.switches.end(),
                                  [target_idx](auto const & other_switch) {
                                    return (other_switch & (1 << target_idx)) != 0;
                                  });

          if (!other_switches_modify) {  // This is the only switch modifying this jolt.
            if (max_pushes < remaining_jolts[target_idx]) {
              // Impossible to reach target for this jolt.
              SPDLOG_DEBUG(
                  "    Switch {:d} is only one modifying jolt {:d}, can't reach push target ({}) "
                  "with {} max pushes",
                  switch_idx, target_idx, remaining_jolts[target_idx], max_pushes);
              max_pushes = 0;  // TODO: How do we skip this switch properly?
              result = {.state = solve_state_t::skip_switch,
                        .num_pushes = UINT16_MAX};  // Can't never find any solutions for
                                                    // this switch given input state.
              break;
            } else if (single_modify_found && (max_pushes != remaining_jolts[target_idx])) {
              // Two targets can only be modified by this switch, impossible.
              SPDLOG_DEBUG(
                  "    Switch {:d} is only one modifying jolt {:d}, but already had another "
                  "single-modify target with different needed pushes ({} vs {})",
                  switch_idx, target_idx, max_pushes, remaining_jolts[target_idx]);
              max_pushes = 0;  // TODO: How do we skip this switch properly?
              return {.state = solve_state_t::skip_switch,
                      .num_pushes = UINT16_MAX};  // Can't never find any solutions for
                                                  // this switch given input state.
              break;
            } else {
              SPDLOG_TRACE(
                  "    Switch {:d} is only one modifying jolt {:d}, setting max pushes to {}",
                  cur_switch_idx, target_idx, remaining_jolts[target_idx]);
              single_modify_found = true;
              max_pushes = remaining_jolts[target_idx];
            }
          } else {
            max_pushes = std::min<int16_t>(max_pushes, remaining_jolts[target_idx]);
          }
        }
      }

      if (result.state != solve_state_t::skip_switch) {
        // If it might be possible to find a solution, try.

        // If this is an "single modifying" switch or the last switch, only try the needed pushes.
        bool const push_all = single_modify_found || ((switch_idx + 1) == num_switches);
        uint16_t push_count = push_all ? max_pushes : 0;

        SPDLOG_DEBUG("  Pushing switch {:d} ({:#b}) with state {}, max pushes: {}, push all: {}",
                     switch_idx, cur_switch, remaining_jolts, max_pushes, push_all);

        // Try all possible number of pushes for this switch.
        uint16_t num_pushed = 0;
        for (; push_count <= max_pushes; ++push_count) {
          if (push_count > 0) {
            num_pushed += (push_all ? max_pushes : 1);
            push_switch(switch_idx, remaining_jolts, cur_switch, push_all ? max_pushes : 1);
          }

          SPDLOG_TRACE("    Calling next with state {}, switch pushes: {}", remaining_jolts,
                       num_switch_pushes + push_count);
          auto next_result = solve_num_button_pushes_impl(
              call_count, caches, problem, num_switch_pushes + push_count, remaining_jolts,
              switch_idx + 1,
              (result.state == solve_state_t::found_solution) ? result.num_pushes : best_so_far);

          /*
      if (next_result.state == solve_state_t::skip_switch) {
        SPDLOG_DEBUG("    Must skip switch {}", cur_switch_idx);
        break;  // No solution possible with current starting state, return up.
      } else
      */
          if (next_result.state == solve_state_t::found_solution) {
            assert(next_result.num_pushes != UINT16_MAX);
            result = {
                .state = solve_state_t::found_solution,
                .num_pushes =
                    std::min<uint16_t>(result.num_pushes, next_result.num_pushes + push_count),
            };
            SPDLOG_DEBUG(
                "    Found solution pushing switch {:d} a total of {} times: {} (total # pushes: "
                "{})",
                switch_idx, push_count, result, result.num_pushes + num_switch_pushes);
          }
        }

        if (num_pushed > 0) {
          push_switch(switch_idx, remaining_jolts, cur_switch, -num_pushed);
        }
      }

      SPDLOG_DEBUG("  Returning result {} for switch_idx {}", result, switch_idx);
      caches.at(switch_idx)
          .emplace(std::vector(remaining_jolts.begin(), remaining_jolts.end()), result);
      return result;
    }

    template <std::integral T>
    struct span_hash_t {
      using is_transparent = void;

      std::size_t operator()(std::span<T const> const & span) const {
        using data_t = std::remove_cvref_t<T>;

        std::size_t seed = 0;
        for (auto const & e : span) {
          // Re-using the hash_combine logic
          seed ^= std::hash<data_t>{}(e) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
      }
    };

    template <class T>
    struct span_compare_t {
      using is_transparent = void;

      bool operator()(std::span<T const> const & lhs, std::span<T const> const & rhs) const {
        return std::ranges::equal(lhs, rhs);
      }
    };

    [[maybe_unused]] uint16_t solve_num_button_pushes(problem_t const & problem) {
      SPDLOG_DEBUG("Solving {}", problem);
      auto state = problem.jolts;

      using cache_state_t [[maybe_unused]] = std::vector<uint16_t>;
      using cache_t = std::unordered_map<cache_state_t, solve_result_t, span_hash_t<uint16_t>,
                                         span_compare_t<uint16_t>>;
      auto caches = std::vector<cache_t>(problem.switches.size() + 1);
      uint64_t call_count = 0;

      auto const result =
          solve_num_button_pushes_impl(call_count, caches, problem, 0, state, 0, std::nullopt);
      assert(result.state == solve_state_t::found_solution);
      SPDLOG_INFO("{}, # button pushes: {}, total calls: {}", problem, result.num_pushes,
                  call_count);
      return result.num_pushes;
    }

    [[maybe_unused]] bool can_push_switch(std::span<uint16_t> jolts_remaining,
                                          uint16_t switch_bits,
                                          uint16_t num_pushes) {
      while (switch_bits != 0) {
        size_t const jolt_idx = std::countr_zero(switch_bits);
        switch_bits &= ~(1ULL << jolt_idx);  // Unset LSB.
        if (jolts_remaining[jolt_idx] < num_pushes) {
          return false;
        }
      }
      return true;
    }

    [[maybe_unused]] uint16_t max_switch_pushes(std::span<uint16_t> jolts_remaining,
                                                uint16_t switch_bits) {
      uint16_t max_pushes = UINT16_MAX;

      while ((max_pushes > 0) && (switch_bits != 0)) {
        size_t const jolt_idx = std::countr_zero(switch_bits);
        switch_bits &= ~(1ULL << jolt_idx);  // Unset LSB.
        max_pushes = std::min<uint16_t>(max_pushes, jolts_remaining[jolt_idx]);
      }

      return max_pushes;
    }

    std::optional<uint8_t> find_next_jolt(std::span<uint16_t const> jolt_to_switches,
                                          std::span<uint16_t> jolts_remaining,
                                          uint16_t active_switches) {
      std::optional<uint8_t> best_jolt_idx = std::nullopt;
      bool has_single_switch_jolt = false;

      // Keep track of minimal remaining jolts.
      uint16_t minimal_jolt_value = UINT16_MAX;

      for (uint8_t jolt_idx = 0; !has_single_switch_jolt && (jolt_idx < jolts_remaining.size());
           ++jolt_idx) {
        uint16_t const jolt_value = jolts_remaining[jolt_idx];
        if (jolt_value == 0) {
          continue;
        }

        // Check if jolt can only be modified by a single switch.
        uint16_t const modifying_switches = jolt_to_switches[jolt_idx] & active_switches;
        uint8_t const num_modifying_switches = std::popcount(modifying_switches);
        has_single_switch_jolt = (num_modifying_switches == 1);

        if (num_modifying_switches == 0) {  // Required amount of jolts can't be reached.
          return std::nullopt;
        } else if (num_modifying_switches == 1) {
          best_jolt_idx = jolt_idx;
        } else if (jolt_value < minimal_jolt_value) {
          // Select jolt if it has the minimal remaining jolts so far.
          best_jolt_idx = jolt_idx;
          minimal_jolt_value = jolt_value;
        }
      }

      SPDLOG_TRACE("[find_next_jolt] candidate: {}", best_jolt_idx);
      return best_jolt_idx;
    }

    uint16_t calc_min_remaining_pushes(std::span<uint16_t const> jolts_remaining,
                                       std::span<uint16_t const> switches,
                                       std::span<uint16_t const> jolt_to_switches,
                                       [[maybe_unused]] uint16_t active_switches) {
      uint16_t active_jolts = 0;
      for (uint8_t jolt_idx = 0; jolt_idx < jolts_remaining.size(); ++jolt_idx) {
        if (jolts_remaining[jolt_idx] != 0) {
          active_jolts |= (1 << jolt_idx);
        }
      }

      uint16_t result = 0;

      while (active_jolts != 0) {
        // Find largest remaining jolt.
        auto const jolt_idx = [&] {
          uint8_t max_jolt_idx = 0;
          uint16_t max_jolts = 0;

          auto remaining_jolts = active_jolts;
          while (remaining_jolts != 0) {
            uint8_t const idx = std::countr_zero(remaining_jolts);
            remaining_jolts &= ~(1ULL << idx);  // Unset LSB.
            if (jolts_remaining[idx] > max_jolts) {
              max_jolt_idx = idx;
              max_jolts = jolts_remaining[idx];
            }
          }
          return max_jolt_idx;
        }();

        uint16_t const jolt_value = jolts_remaining[jolt_idx];
        assert(jolt_value > 0);

        // Need at least as many pushes as there are remaining jolts.
        result += jolt_value;

        // Create mask of all jolt positions modified by switches that modify this jolt.
        // TODO: Take active switches into account? This must also be taken into account when
        // selecting jolt though, because otherwise we might not be able to set it to 0.
        uint16_t combined_switch_bits = 0;
        uint16_t switches_for_jolt = jolt_to_switches[jolt_idx] & active_switches;
        while (switches_for_jolt != 0) {
          uint8_t const switch_idx = std::countr_zero(switches_for_jolt);
          switches_for_jolt &= ~(1ULL << switch_idx);  // Unset LSB.
          combined_switch_bits |= switches[switch_idx];
        }

        // Mask all jolts that are modified by these switches. This will likely make the estimated
        // number of pushes too low, but it avoids having to deal with dependencies between
        // switches.
        active_jolts &= ~combined_switch_bits;
        SPDLOG_TRACE(
            "[calc_min_remaining_pushes] jolts remaining: {}, jolt_idx: {}, jolt_value: {}, "
            "combined_switch_bits: {:#b}, active_jolts: {:#b},  active switches: {:#b}, result: {}",
            jolts_remaining, jolt_idx, jolt_value, combined_switch_bits, active_jolts,
            active_switches, result);

        if (active_jolts & (1 << jolt_idx)) {  // Shouldn't happen.
          // Some bug somewhere, we end up in an unreachable state.
          return UINT16_MAX / 2;
        }
      }

      return result;
    }

    [[maybe_unused]] bool has_unreachable_jolts(std::span<uint16_t const> jolts_remaining,
                                                std::span<uint16_t const> jolt_to_switches,
                                                uint16_t active_switches) {
      for (uint8_t jolt_idx = 0; jolt_idx < jolts_remaining.size(); ++jolt_idx) {
        if (jolts_remaining[jolt_idx] == 0) {
          continue;
        } else if ((jolt_to_switches[jolt_idx] & active_switches) == 0) {
          SPDLOG_TRACE(
              "[has_unreachable_jolts] jolt {} with remaining {} is unreachable with active "
              "switches {:#b}",
              jolt_idx, jolts_remaining[jolt_idx], active_switches);
          return true;
        }
      }
      return false;
    }

    void solve_jolts_impl(uint64_t & call_count,
                          std::optional<uint16_t> & best_solution,
                          std::span<uint16_t const> switches,
                          std::span<uint16_t const> jolt_to_switches,
                          std::span<uint16_t> jolts_remaining,
                          uint16_t active_switches,
                          uint16_t pushes_by_parent) {
      [[maybe_unused]] uint8_t const log_indent = switches.size() - std::popcount(active_switches);

      if (std::ranges::all_of(jolts_remaining, [](auto const & e) { return e == 0; })) {
        SPDLOG_DEBUG("{:>{}s}Found solution, # pushes: {}", "", 2 * log_indent, pushes_by_parent);
        best_solution = std::min(best_solution.value_or(UINT16_MAX), pushes_by_parent);
        return;
      }

      ++call_count;  // Don't count "solution found" calls.

      auto const jolt_idx = find_next_jolt(jolt_to_switches, jolts_remaining, active_switches);
      if (!jolt_idx.has_value()) {
        return;
      }

      uint16_t possible_switches = jolt_to_switches[*jolt_idx] & active_switches;
      SPDLOG_TRACE("{:>{}}Checking switches {:#b} for jolt {}", "", 2 * log_indent,
                   possible_switches, *jolt_idx);

      while (possible_switches != 0) {
        uint8_t const num_modifying_switches = std::popcount(possible_switches);
        uint8_t const switch_idx = std::countr_zero(possible_switches);

        possible_switches &= ~(1ULL << switch_idx);  // Unset LSB.
        active_switches &= ~(1ULL << switch_idx);    // Disable switch for all future recursions.

        // Determine minimum and maximum number of pushes for this switch.
        uint16_t const jolt_value = jolts_remaining[*jolt_idx];
        uint16_t const min_pushes = (num_modifying_switches == 1) ? jolt_value : 1;

        uint16_t max_pushes = max_switch_pushes(jolts_remaining, switches[switch_idx]);
        if (best_solution.has_value()) {
          // Don't try pushes that can't beat the best result.
          // TODO: Subtract minimum required pushes for remaining jolts.
          auto const max_allowed_pushes = *best_solution - pushes_by_parent;
          max_pushes = std::min<uint16_t>(max_pushes, max_allowed_pushes);
        }

        // Try pushing this switch up to the maximum number of times possible.
        for (uint16_t num_pushes = max_pushes + 1; num_pushes-- > min_pushes;) {
          assert(can_push_switch(jolts_remaining, switches[switch_idx], num_pushes));

          // Push switch and recurse.
          push_switch(switch_idx, jolts_remaining, switches[switch_idx], num_pushes);

          // Disable any active switches which would modify jolts that are already zero.
          auto usable_switches = active_switches;
          for (uint8_t idx = 0; idx < jolts_remaining.size(); ++idx) {
            if (jolts_remaining[idx] == 0) {  // Disable all switches modifying this jolt.
              usable_switches &= ~jolt_to_switches[idx];
            }
          }

          /*
          bool const unreachable =
              has_unreachable_jolts(jolts_remaining, jolt_to_switches, usable_switches);
          if (unreachable) {
            SPDLOG_TRACE(
                "{:>{}s}Skipping switch {} ({:#b}) since it would lead to unreachable jolts", "",
                2 * log_indent, switch_idx, switches[switch_idx]);
          }
          */

          // Count miminum remaining pushes needed to reach target for remaining jolts.
          // TODO: Is there a way not to calculate this in this loop?
          uint16_t const min_remaining_child_pushes = calc_min_remaining_pushes(
              jolts_remaining, switches, jolt_to_switches, usable_switches);
          uint16_t const min_total_pushes =
              pushes_by_parent + num_pushes + min_remaining_child_pushes;
          SPDLOG_TRACE("{:>{}}min_remaining_child_pushes: {}, min_total_pushes: {}, best: {}", "",
                       2 * log_indent, min_remaining_child_pushes, min_total_pushes, best_solution);

          if (min_total_pushes < best_solution.value_or(UINT16_MAX)) {
            SPDLOG_DEBUG(
                "{:>{}s}Pushing switch {} ({:#b}) a total of {} times, jolts remaining: {}, "
                "switches active: {} ({:#b})",
                "", 2 * log_indent, switch_idx, switches[switch_idx], num_pushes, jolts_remaining,
                std::popcount(usable_switches), usable_switches);
            solve_jolts_impl(call_count, best_solution, switches, jolt_to_switches, jolts_remaining,
                             usable_switches, pushes_by_parent + num_pushes);
          }

          // Undo pushes before trying next number of pushes.
          push_switch(switch_idx, jolts_remaining, switches[switch_idx], -num_pushes);
        }
      }
    }

    uint16_t solve_jolts(problem_t const & problem) {
      SPDLOG_DEBUG("Solving {}", problem);

      // Prepare bookkeeping.
      assert(problem.switches.size() <= 16);

      std::optional<uint16_t> best_solution = std::nullopt;
      auto jolts_remaining = problem.jolts;
      uint16_t active_switches = (1ULL << problem.switches.size()) - 1;
      [[maybe_unused]] uint64_t call_count = 0;

      // For each jolt, keep track of which switches can modify it.
      auto jolt_to_switches = std::vector<uint16_t>(problem.jolts.size());
      for (uint8_t switch_idx = 0; switch_idx < problem.switches.size(); ++switch_idx) {
        auto switch_bits = problem.switches[switch_idx];
        while (switch_bits != 0) {
          size_t const jolt_idx = std::countr_zero(switch_bits);
          jolt_to_switches[jolt_idx] |= (1 << switch_idx);
          switch_bits &= ~(1ULL << jolt_idx);  // Unset LSB.
        }
      }

      solve_jolts_impl(call_count, best_solution, problem.switches, jolt_to_switches,
                       jolts_remaining, active_switches, 0);
      SPDLOG_INFO("{}, # button pushes: {}, total calls: {}", problem, best_solution.value(),
                  call_count);
      return best_solution.value();
    }

    // Structure holding precomputed tables of all values of 0-N bits, where each table lists (in
    // order) all possible combinations of least to most bits set.
    template <size_t N>
    class combination_tables_t {
     public:
      constexpr combination_tables_t() {
        // Generate all tables.
        uint16_t * table_begin = data_.data();
        std::span<uint16_t const> prev_table;

        for (uint8_t num_bits = 1; num_bits <= N; ++num_bits) {
          uint16_t * const table_end = generate_gray_table_for(table_begin, num_bits, prev_table);
          assert(static_cast<size_t>(table_end - table_begin) == table_size(num_bits));
          tables_[num_bits - 1] = std::span<uint16_t const>(table_begin, table_end);
          prev_table = tables_[num_bits - 1];
          table_begin = table_end;
        }
      }

      constexpr std::span<uint16_t const> table_for(size_t num_bits) const {
        assert(num_bits > 0);
        assert(num_bits <= N);
        return tables_[num_bits - 1];
      }

     private:
      static constexpr size_t table_size(size_t num_bits) {
        // Number of bits in a table for X bits is 2^X - 1, since we don't include the 0 value.
        return (1ull << num_bits) - 1;
      }

      static constexpr size_t total_table_size(size_t max_num_bits) {
        assert(max_num_bits > 0);
        auto const num_bits =
            std::views::iota(size_t{1}, max_num_bits + 1) | std::views::transform(table_size);
        return std::accumulate(num_bits.begin(), num_bits.end(), 0ull);
      }

      /// @brief Generate table for N bits and store result at dest.
      /// @return Pointer past the end of the generated table.
      static constexpr uint16_t * generate_gray_table_for(
          uint16_t * dest,
          uint8_t num_bits,
          std::span<uint16_t const> previous_table) {
        /* Generate table as a Gray code, where each element only differs from the previous one in
         * two places. The algorithm implemented here uses the Revolving Door method. An alternative
         * is Chase's sequence.
         * See e.g.:
         *  - https://encyclopediaofmath.org/wiki/Gray_code
         *  - https://www.baeldung.com/cs/generate-k-combinations
         */

        // List consists of the concatenation of:
        //  - result for k bits set for num_bits - 1,
        //  - reversed result for k - 1 bits set for num_bits - 1, with the bit at num_bits set.

        // First find the begin and end position of combinations with k bits in the previous table.
        auto const sum_of_num_combinations = [](int8_t n, int8_t k) -> size_t {
          if (k <= 0) {
            return 0;
          }

          auto const num_combos =
              std::views::iota(int8_t{1}, static_cast<int8_t>(k + 1)) |
              std::views::transform([n](int8_t bits_set) { return num_combinations(n, bits_set); });
          return std::accumulate(num_combos.begin(), num_combos.end(), 0ull);
        };

        for (uint8_t num_bits_set = 1; num_bits_set <= num_bits; ++num_bits_set) {
          {  // Copy entries from previous table with K bits set.
            size_t const prev_index_cur_k_begin =
                sum_of_num_combinations(num_bits - 1, num_bits_set - 1);
            size_t const prev_index_cur_k_size = num_combinations(num_bits - 1, num_bits_set);
            auto const prev_table_entries_cur_k =
                previous_table.subspan(prev_index_cur_k_begin, prev_index_cur_k_size);

            auto const copy_result = std::ranges::copy(prev_table_entries_cur_k, dest);
            dest = copy_result.out;
          }

          {  // Copy entries from previous table with K - 1 bits set.
            size_t const prev_index_prev_k_begin =
                sum_of_num_combinations(num_bits - 1, num_bits_set - 2);
            size_t const prev_index_prev_k_size = num_combinations(num_bits - 1, num_bits_set - 1);

            // If list is empty, treat it as a list with an unset bitstring.
            if (prev_index_prev_k_size == 0) {
              *dest++ = 1 << (num_bits - 1);
            } else {
              auto const prev_table_entries_prev_k =
                  previous_table.subspan(prev_index_prev_k_begin, prev_index_prev_k_size);

              auto const reversed_and_transformed =
                  prev_table_entries_prev_k | std::views::reverse |
                  std::views::transform(
                      [num_bits](uint16_t value) { return value | (1 << (num_bits - 1)); });
              auto const copy_result = std::ranges::copy(reversed_and_transformed, dest);
              dest = copy_result.out;
            }
          }
        }

        return dest;
      }

      static constexpr size_t max_bits = 15;
      static_assert(N <= max_bits, "Combination tables only supported up to 8 bits");

      std::array<uint16_t, total_table_size(N)> data_;
      std::array<std::span<uint16_t const>, N> tables_;
    };

  }  // namespace

  uint64_t day_t<10>::solve(part_t<1>, version_t<0>, simd_string_view_t input) {
    auto const problems = parse_input(input);

    // Note: the "problem.switches" is a spanning set, but not a minimum spanning set, i.e. not a
    // basis, so we can't just do Gaussian elimination to find the solution.
    uint64_t total = 0;

    // Iterate first over all values with 1 bit set, then 2 bits set, etc. This ensures that as soon
    // as we find a solution, it is the minimum number of toggles and we can stop the search.
    static constexpr size_t max_switches = 13;
    static constexpr combination_tables_t<max_switches> combination_tables;

    for (auto const & problem : problems) {
      SPDLOG_DEBUG("Solving {} ({} switches)", problem, problem.switches.size());
      assert(problem.switches.size() <= max_switches);

      auto const table = combination_tables.table_for(problem.switches.size());

      // Combinations are in Gray-code order, i.e. each entry differs from the previous one in
      // exactly two bits. So keep track of previous state and just apply the difference. Note that
      // when the number of set bits changes, the number of differences can be more than two.
      // However, the loop handles this just fine.
      uint16_t prev_combo = 0;
      uint16_t state = 0;

      for (uint16_t idx = 0; (idx < table.size()); ++idx) {
        uint16_t const combo = table[idx];
        uint16_t const combo_diff = prev_combo ^ combo;  // Determine which bits changed.
        prev_combo = combo;

        state = apply_diff_switches(problem.switches, combo_diff, state);

        if (state == problem.target) {
          uint16_t const num_used_switches = std::popcount(combo);
          SPDLOG_DEBUG(" Found solution for {}: combo {:#0{}b} (# toggles: {})", problem, combo,
                       problem.switches.size() + 2, num_used_switches);
          total += num_used_switches;
          break;  // Found solution is always the optimal one.
        }
      }
    }

    return total;
  }

  uint64_t day_t<10>::solve(part_t<2>, version_t<0>, simd_string_view_t input) {
    // Branch-and-bound:
    //  - Complete number of pushes for 1 switch first. I.e. never increase once checking
    //    the next switches.
    //  - The limit is maximum value for each switch.
    //  - Keep track of best solution with given state for given switch index. Then bail if
    //    we get to that same state with same switch again.
    //    => If we hash on the state & key index, we can't just return the # pushes, because
    //       we might have reached the same state with a different number of pushes. So we
    //       need to store the minimum number of pushes to get from that state to the target,
    //       instead of from the start to the target.
    //  - If a given switch is the only one remaining that modifies a certain target, then
    //    it has to be instantly pushed the maximum number of times to reach the target. And
    //    if not possible, then bail.
    //  - Count state down to 0, instead of checking against target.
    //  - What else can we prune on?
    auto problems = parse_input(input);

    // Try switches which toggle more jolts first.
    for (auto & problem : problems) {
      std::ranges::sort(problem.switches, std::greater<>{},
                        [](uint16_t switch_bits) { return std::popcount(switch_bits); });
    }
    return 0;

    // return std::accumulate(
    //     problems.begin(), problems.end(), 0ULL,
    //     [](uint64_t acc, auto const & problem) { return acc +
    //     solve_num_button_pushes(problem);
    //     });
  }

  uint64_t day_t<10>::solve(part_t<2>, version_t<1>, simd_string_view_t input) {
    auto problems = parse_input(input);

    return std::accumulate(
        problems.begin(), problems.end(), 0ULL,
        [](uint64_t acc, auto const & problem) { return acc + solve_jolts(problem); });
  }

}  // namespace aoc25

#endif  // HWY_ONCE
