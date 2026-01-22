#include "aoc25/day_11.hpp"

#include "aoc25/day.hpp"
#include "aoc25/simd.hpp"
#include "aoc25/string.hpp"

#include <fmt/ranges.h>
#include <fmt/std.h>
#include <magic_enum/magic_enum.hpp>
#include <spdlog/spdlog.h>

#include <cstdint>
#include <ranges>
#include <utility>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "src/day_11.cpp"

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

    enum class node_name_t : uint8_t {
      you,
      out,
      svr,
      dac,
      fft,
    };

    class problem_t {
     public:
      explicit problem_t(size_t num_nodes)
          : neighbors_(std::make_unique_for_overwrite<uint16_t[]>(num_nodes * (num_nodes - 1)))
          , neighbor_views_(num_nodes)
          , node_indices_{}
          , num_neighbors_pushed_(0)
          , last_neighbors_pushed_(0)
          , neighbors_size_(num_nodes * (num_nodes - 1)) {
        // Must allocate maximum possible amount of neighbors to avoid reallocations. Because
        // a reallocation would invalidate all existing spans.
      }

      problem_t(problem_t const &) = delete;
      problem_t & operator=(problem_t const &) = delete;

      problem_t(problem_t &&) = default;
      problem_t & operator=(problem_t &&) = default;

      size_t num_nodes() const { return neighbor_views_.size(); }

      uint16_t operator[](node_name_t name) const {
        return node_indices_[std::to_underlying(name)];
      }

      uint16_t & operator[](node_name_t name) { return node_indices_[std::to_underlying(name)]; }

      std::span<uint16_t const> neighbors(uint16_t node_idx) const {
        return neighbor_views_.at(node_idx);
      }

      void add_neighbor(uint16_t neighbor_idx) {
        assert(num_neighbors_pushed_ < neighbors_size_);
        neighbors_[num_neighbors_pushed_++] = neighbor_idx;
      }

      void finalize_neighbors_for(uint16_t node_idx) {
        neighbor_views_.at(node_idx) =
            std::span<uint16_t const>(neighbors_.get() + last_neighbors_pushed_,
                                      num_neighbors_pushed_ - last_neighbors_pushed_);
        last_neighbors_pushed_ = num_neighbors_pushed_;
      }

     private:
      // Single big storage to avoid having to create tons of small vectors. Note that we don't use
      // an std::vector, because the initialization takes a significant amount of time.
      std::unique_ptr<uint16_t[]> neighbors_;
      std::vector<std::span<uint16_t const>> neighbor_views_;
      std::array<uint16_t, magic_enum::enum_count<node_name_t>()> node_indices_;
      size_t num_neighbors_pushed_;
      size_t last_neighbors_pushed_;
      [[maybe_unused]] size_t neighbors_size_;

      friend std::string format_as(problem_t const & obj);
    };

    [[maybe_unused]] std::string format_as(problem_t const & obj) {
      auto const named_indices =
          magic_enum::enum_values<node_name_t>() | std::views::transform([&](node_name_t name) {
            return fmt::format("{}: {}", magic_enum::enum_name(name), obj[name]);
          });
      auto const neighbors = obj.neighbor_views_ | std::views::enumerate |
                             std::views::transform([](auto const & pair) {
                               auto const [idx, neigh] = pair;
                               return fmt::format("{}: {}", idx, neigh);
                             });
      return fmt::format("#nodes: {}, {::s},  <neighbors: {::s}>", obj.neighbor_views_.size(),
                         named_indices, neighbors);
    }

    problem_t parse_input(simd_string_view_t input) {
      // Note: Most of the solving time is spend parsing the input, hence the "funky" code.
      // Input pattern: "aaaa: bbbb cccc dddd" with part after the colon of variable length.
      // There's only 26^3 possible names, which is 17 576. So use big vector to avoid hashing.
      static constexpr size_t max_num_names = 26 * 26 * 26;
      auto const hash_name = [](std::string_view const & name) -> uint16_t {
        return (static_cast<uint16_t>(name[0] - 'a') * 26 * 26) +
               (static_cast<uint16_t>(name[1] - 'a') * 26) + (static_cast<uint16_t>(name[2] - 'a'));
      };

      enum class state_t { name, neighbors };

      auto name_to_idx = std::vector<uint16_t>(max_num_names, UINT16_MAX);
      uint16_t next_idx = 0;

      auto state = state_t::name;
      uint16_t parent_idx = 0;

      // Need to add 1 extra for the "out" node, which has no outgoing edges.
      size_t const num_nodes = count(input.as_span(), ':') + 1;

      auto result = problem_t(num_nodes);

      while (!input.empty()) {
        switch (state) {
          case state_t::name: {
            assert(input.at(3) == ':');
            auto const name_token = input.substr(0, 3);
            input.remove_prefix(5);  // Skip "aaa: "

            auto & idx = name_to_idx[hash_name(name_token)];
            if (idx == UINT16_MAX) {
              idx = next_idx++;
            }

            parent_idx = idx;
            state = state_t::neighbors;
            break;
          }

          case state_t::neighbors: {
            auto const name_token = input.substr(0, 3);
            auto & idx = name_to_idx[hash_name(name_token)];
            if (idx == UINT16_MAX) {
              idx = next_idx++;
            }

            result.add_neighbor(idx);

            assert(input.size() > 3);
            if (input[3] != ' ') {
              state = state_t::name;
              result.finalize_neighbors_for(parent_idx);
            }
            input = input.substr(4);

            break;
          }
        }
      }

      for (node_name_t const name : magic_enum::enum_values<node_name_t>()) {
        result[name] = name_to_idx[hash_name(magic_enum::enum_name(name))];
      }

      SPDLOG_DEBUG("Parsed: {}", result);
      return result;
    }

    struct solver_t {
      solver_t(problem_t const & problem)
          : problem_(problem) {}

      uint64_t solve(part_t<1>) {
        // Get all nodes reachable from start.
        // Note: Faster than combining reachable nodes and topological sort into a two DFSes.
        auto const reachable_nodes = find_all_reachable_nodes(problem_[node_name_t::you]);
        auto const sorted_nodes = topological_sort(reachable_nodes, problem_[node_name_t::you]);

        // Walk topological order, counting paths to each node.
        return count_paths(sorted_nodes, problem_[node_name_t::you], problem_[node_name_t::out], 1);
      }

      uint64_t solve(part_t<2>) {
        // Get all nodes reachable from start, these will also include nodes reachable from dac and
        // fft (otherwise it would mean there is no path going through them).
        auto const reachable_nodes = find_all_reachable_nodes(problem_[node_name_t::svr]);

        // The graph is a DAG, so there can't be a cycle between dac and fft. So on the way to the
        // end, one will always come before the other, and it will be in the order of the list
        // generated here.
        auto const sorted_nodes = topological_sort(reachable_nodes, problem_[node_name_t::svr]);

        // Find out whether dac or fft comes first.
        auto const dac_it = std::ranges::find(sorted_nodes, problem_[node_name_t::dac]);
        auto const fft_it =
            std::ranges::find(dac_it, sorted_nodes.end(), problem_[node_name_t::fft]);
        bool const dac_first = fft_it != sorted_nodes.end();

        uint16_t const first_idx =
            dac_first ? problem_[node_name_t::dac] : problem_[node_name_t::fft];
        uint16_t const second_idx =
            dac_first ? problem_[node_name_t::fft] : problem_[node_name_t::dac];

        // Walk topological order, counting paths to each node.
        uint64_t const count_to_first =
            count_paths(sorted_nodes, problem_[node_name_t::svr], first_idx, 1);
        uint64_t const count_to_second =
            count_paths(sorted_nodes, first_idx, second_idx, count_to_first);

        return count_paths(sorted_nodes, second_idx, problem_[node_name_t::out], count_to_second);
      }

     private:
      problem_t const & problem_;

      uint64_t count_paths(std::span<uint16_t const> sorted_nodes,
                           uint16_t start_idx,
                           uint16_t end_idx,
                           uint64_t initial_count) const {
        auto path_counts = std::vector<uint64_t>(problem_.num_nodes());
        path_counts.at(start_idx) = initial_count;

        for (auto const node_idx : sorted_nodes) {
          if (node_idx == end_idx) {
            break;
          }

          auto const cur_path_count = path_counts.at(node_idx);
          for (auto const neighbor_idx : problem_.neighbors(node_idx)) {
            path_counts.at(neighbor_idx) += cur_path_count;
          }
        }

        return path_counts.at(end_idx);
      }

      std::vector<uint16_t> topological_sort(std::vector<uint16_t> const & nodes,
                                             uint16_t start_idx) const {
        // Prepare number of neighbors for topological sort. Note that the count might be wrong for
        // nodes not on path, but we don't care, since those nodes won't be used.
        auto num_neighbors = std::vector<uint64_t>(problem_.num_nodes());
        for (auto const node_idx : nodes) {
          for (auto const neighbor_idx : problem_.neighbors(node_idx)) {
            num_neighbors.at(neighbor_idx) += 1;
          }
        }

        // Topological sort using Kahn's algorithm.
        std::vector<uint16_t> sorted_nodes;
        sorted_nodes.reserve(nodes.size());

        std::vector<uint16_t> nodes_to_process;
        nodes_to_process.push_back(start_idx);

        while (!nodes_to_process.empty()) {
          uint16_t const node_idx = nodes_to_process.back();
          nodes_to_process.pop_back();
          sorted_nodes.push_back(node_idx);

          if (node_idx == problem_[node_name_t::out]) {
            break;  // Don't bother continuing.
          }

          for (auto const neighbor_idx : problem_.neighbors(node_idx)) {
            num_neighbors.at(neighbor_idx) -= 1;
            if (num_neighbors.at(neighbor_idx) == 0) {
              nodes_to_process.push_back(neighbor_idx);
            }
          }
        }

        SPDLOG_DEBUG("Topological sort: {}", sorted_nodes);
        return sorted_nodes;
      }

      std::vector<uint16_t> find_all_reachable_nodes(uint16_t start_node) const {
        std::vector<uint16_t> result;
        result.reserve(problem_.num_nodes());

        auto visited = std::vector<bool>(problem_.num_nodes(), false);
        dfs(start_node, visited, result);
        SPDLOG_DEBUG("{} reachable nodes from {}: {}", result.size(), start_node, result);
        return result;
      }

      void dfs(uint16_t node_idx,
               std::vector<bool> & visited,
               std::vector<uint16_t> & result) const {
        visited.at(node_idx) = true;
        result.push_back(node_idx);

        for (auto const neighbor_idx : problem_.neighbors(node_idx)) {
          if (!visited.at(neighbor_idx)) {
            dfs(neighbor_idx, visited, result);
          }
        }
      }
    };

  }  // namespace

  uint64_t day_t<11>::solve(part_t<1>, version_t<0>, simd_string_view_t input) {
    auto const problem = parse_input(input);
    auto solver = solver_t(problem);
    return solver.solve(part<1>);
  }

  uint64_t day_t<11>::solve(part_t<2>, version_t<0>, simd_string_view_t input) {
    auto const problem = parse_input(input);
    auto solver = solver_t(problem);
    return solver.solve(part<2>);
  }

}  // namespace aoc25

#endif  // HWY_ONCE
