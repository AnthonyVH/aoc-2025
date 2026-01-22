#include "aoc25/day_10.hpp"

#include "aoc25/day.hpp"
#include "aoc25/math.hpp"
#include "aoc25/simd.hpp"
#include "aoc25/string.hpp"

#include <Eigen/Core>
#include <Eigen/src/Core/ArithmeticSequence.h>
#include <Eigen/src/Core/util/Constants.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <fmt/std.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <numeric>
#include <ranges>
#include <type_traits>
#include <utility>

// Enable formatting of Eigen matrices.
namespace aoc25 {
  namespace {

    template <class T, class Enable = void>
    struct is_eigen_type : std::false_type {};

    template <class T>
    struct is_eigen_type<Eigen::MatrixBase<T>, void> : std::true_type {};

    template <class T>
    struct is_eigen_type<T, std::enable_if_t<std::is_base_of_v<Eigen::PlainObjectBase<T>, T>>>
        : std::true_type {};

    template <class T>
    inline constexpr bool is_eigen_type_v = is_eigen_type<T>::value;

  }  // namespace
}  // namespace aoc25

namespace fmt {

  template <typename T>
    requires aoc25::is_eigen_type_v<T>
  struct formatter<T> : ostream_formatter {};

  template <typename T, typename Char>
    requires aoc25::is_eigen_type_v<T>
  struct range_format_kind<T, Char> : std::integral_constant<range_format, range_format::disabled> {
  };

};  // namespace fmt

namespace aoc25 {

  namespace {

    struct problem_t {
      std::vector<uint16_t> switches;
      std::vector<uint16_t> jolts;
      uint16_t target = 0;
      uint8_t num_bits = 0;
    };

    [[maybe_unused]] std::string format_as(problem_t const & obj) {
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

      SPDLOG_TRACE("Parsed {}", result);
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

    Eigen::MatrixX<int64_t> switches_to_matrix(std::span<uint16_t const> switches,
                                               size_t num_jolts) {
      Eigen::MatrixX<int64_t> matrix = Eigen::MatrixX<int64_t>::Zero(num_jolts, switches.size());

      // Switches are stored as bitmasks, so need to unpack.
      for (size_t col = 0; col < switches.size(); ++col) {
        uint16_t switch_bits = switches[col];

        while (switch_bits != 0) {
          size_t const row = std::countr_zero(switch_bits);
          switch_bits &= ~(1ULL << row);  // Unset LSB.
          matrix(row, col) = 1;
        }
      }

      return matrix;
    }

    template <class T>
    using DiagonalMatrixX = Eigen::DiagonalMatrix<T, Eigen::Dynamic>;

    /**
     * Computes Smith Normal Form, where D = PAQ.
     * This version ignores P and only returns D and Q.
     * @param input The input integer matrix.
     * @return Tuple of (D, P, Q).
     */
    template <class T>
    std::tuple<DiagonalMatrixX<T>, Eigen::MatrixX<T>, Eigen::MatrixX<T>>
    calculate_smith_normal_form(Eigen::MatrixX<T> input) {
      assert(input.rows() <= std::numeric_limits<uint8_t>::max());
      assert(input.cols() <= std::numeric_limits<uint8_t>::max());
      uint8_t const rows = input.rows();
      uint8_t const cols = input.cols();
      int const max_rank = std::min(rows, cols);

      auto P = Eigen::MatrixX<T>::Identity(rows, rows).eval();
      auto Q = Eigen::MatrixX<T>::Identity(cols, cols).eval();

      for (int k = 0; k < max_rank; ++k) {
        bool changed = true;

        while (changed) {
          changed = false;

          // 1. Pivot selection: Find smallest non-zero element in submatrix.
          int best_row = -1;
          int best_col = -1;
          auto const submatrix = input(Eigen::seq(k, rows - 1), Eigen::seq(k, cols - 1));
          T const min_val = (submatrix.array() == 0)
                                .select(std::numeric_limits<T>::max(), submatrix.cwiseAbs().array())
                                .minCoeff(&best_row, &best_col);
          best_row += k;
          best_col += k;
          bool const found_nonzero = min_val != std::numeric_limits<T>::max();

          if (!found_nonzero) {
            break;  // Entire submatrix is zero.
          }

          // Swap Rows (Track in P)
          if (best_row != k) {
            input.row(k).swap(input.row(best_row));
            P.row(k).swap(P.row(best_row));
          }

          // Swap Columns (Track in Q)
          if (best_col != k) {
            input.col(k).swap(input.col(best_col));
            Q.col(k).swap(Q.col(best_col));
          }

          // 2. Eliminate other entries in row k and column k
          // Column elimination (row operations, so mirror on P).
          for (int row = k + 1; row < rows; ++row) {
            if (input(row, k) != 0) {
              auto const quotient = input(row, k) / input(k, k);
              input.row(row) -= quotient * input.row(k);
              P.row(row) -= quotient * P.row(k);  // Mirror operation in P.
              if (input(row, k) != 0) {
                changed = true;
              }
            }
          }

          // Row elimination (column operations, so mirror on Q).
          for (int col = k + 1; col < cols; ++col) {
            if (input(k, col) != 0) {
              auto const quotient = input(k, col) / input(k, k);
              input.col(col) -= quotient * input.col(k);
              Q.col(col) -= quotient * Q.col(k);  // Mirror operation in Q.
              if (input(k, col) != 0) {
                changed = true;
              }
            }
          }
        }

        // 3. Ensure input diagonal is positive.
        if (input(k, k) < 0) {
          input.row(k) *= -1;
          P.row(k) *= -1;
        }
      }

      return std::make_tuple(DiagonalMatrixX<T>(input.diagonal()), std::move(P), std::move(Q));
    }

    /** @brief Calculate a linear system that generates all possible solutions to the given problem.
     */
    std::tuple<Eigen::VectorX<int16_t>, Eigen::MatrixX<int16_t>> generate_solution_system(
        problem_t const & problem) {
      // Solve system by first converting to Smith Normal Form.
      // See e.g. http://www.numbertheory.org/php/axbmodm.html
      auto const switch_matrix = switches_to_matrix(problem.switches, problem.jolts.size());
      auto const target =
          Eigen::Map<Eigen::VectorX<uint16_t> const>(problem.jolts.data(), problem.jolts.size())
              .cast<int64_t>()
              .eval();

      auto const [snf_d, snf_p, snf_q] = calculate_smith_normal_form(switch_matrix);
      auto b_alt = snf_p * target;
      SPDLOG_TRACE(" [SNF] D: {}\nP:\n{}\nQ:\n{}\nb_alt: {}", snf_d.diagonal().transpose(), snf_p,
                   snf_q, b_alt.transpose());

      // Prepare base solution.
      auto const [y, rank] = [&] {
        auto result = Eigen::VectorX<int64_t>::Zero(snf_q.rows()).eval();

        size_t const num_rows = std::min(b_alt.rows(), result.rows());
        uint8_t row = 0;

        for (; row < num_rows; ++row) {
          auto const d_entry = snf_d.diagonal()(row);

          if (d_entry == 0) {
            break;  // D is diagonal, once an entry is zero, all following are zero too.
          }

          auto const div_result = std::div(b_alt(row), d_entry);
          assert(div_result.rem == 0);  // Must be divisible.
          result(row) = div_result.quot;
        }

        SPDLOG_TRACE("  [SNF] rank: {}, y: {}", row, result.transpose());
        return std::make_tuple(std::move(result), row);
      }();

      auto const x_base = (snf_q * y).cast<int16_t>();
      auto null_space =
          snf_q(Eigen::placeholders::all, Eigen::placeholders::lastN(snf_q.cols() - rank))
              .cast<int16_t>();
      SPDLOG_TRACE("  [solution base] x_base: [{}], null space:{:s}{}", x_base.transpose().matrix(),
                   (null_space.cols() == 0) ? " []" : "\n", null_space.matrix());

      return std::make_tuple(x_base, null_space);
    }

    struct limits_t {
      int16_t lower = std::numeric_limits<int16_t>::min();
      int16_t upper = std::numeric_limits<int16_t>::max();
    };

    [[maybe_unused]] std::string format_as(limits_t const & obj) {
      return fmt::format("[{}, {}]", obj.lower, obj.upper);
    }

    /** @brief Find the limits between which the null space column can be added to the solution
     * while keeping all of the solution's entries positive.
     */
    limits_t calculate_limits(Eigen::VectorX<int16_t> const & solution,
                              std::span<int16_t const> null_space_column) {
      assert(static_cast<ssize_t>(null_space_column.size()) == solution.rows());

      limits_t result;

      for (int row = 0; row < solution.size(); ++row) {
        auto const ns_value = null_space_column[row];
        if (ns_value == 0) {  // Null space doesn't affect this row, so skip.
          continue;
        }

        auto const sol_value = solution(row);
        if (sol_value < 0) {
          if (ns_value > 0) {
            // Null space increases this row, which is currently negative. So calculate minimum
            // number of additions to make it non-negative.
            int16_t const min_additions = (-sol_value + (ns_value - 1)) / ns_value;
            result.lower = std::max(result.lower, min_additions);
          } else {
            assert(ns_value < 0);
            // Null space decreases this row, which is currently negative. So calculate minimum
            // number of subtractions to make it non-negative. Since this is the minimum number of
            // times we need to subtract a negative number, this is effectively a maximum number
            // of additions. E.g. if we calculate that at least -5 additions are required, then
            // adding only -4 would still leave the row negative.
            int16_t const pos_ns_value = -ns_value;
            int16_t const min_subtractions = (sol_value - (pos_ns_value - 1)) / pos_ns_value;
            result.upper = std::min(result.upper, min_subtractions);
          }
        } else {
          assert(sol_value >= 0);

          if (ns_value > 0) {
            // Null space increases this row, which is currently positive. So calculate maximum
            // number of subtractions to keep it non-negative.
            int16_t const max_subtractions = -sol_value / ns_value;
            result.lower = std::max(result.lower, max_subtractions);
          } else {
            assert(ns_value < 0);
            // Null space decreases this row, which is currently positive. So calculate maximum
            // number of additions to keep it non-negative.
            int16_t const max_additions = sol_value / -ns_value;
            result.upper = std::min(result.upper, max_additions);
          }
        }
      }

      return result;
    }

    // A very basic simplex implementation.
    // See e.g. https://www.cs.emory.edu/~cheung/Courses/253/Syllabus/ConstrainedOpt
    struct simplex_t {
      simplex_t(Eigen::RowVectorX<int16_t> const & objective,
                Eigen::MatrixX<int16_t> const & constraints,
                Eigen::VectorX<int16_t> const & bounds)
          : tableau_{}, offsets_{} {
        build_tableau(objective, constraints, bounds);
      }

      std::optional<std::pair<double, Eigen::RowVectorX<double>>> solve() {
        if (!has_solution_) {
          return std::nullopt;
        }

        solve_impl();

        auto const objective_value = tableau_(0, Eigen::placeholders::last);
        return std::make_pair(objective_value, extract_variables());
      }

     private:
      static constexpr double epsilon = 1e-10;

      Eigen::MatrixX<double> tableau_;
      Eigen::RowVectorX<double> offsets_;
      std::vector<bool> is_free_variable_;
      bool has_solution_ = true;

      static bool is_zero(double value) { return std::abs(value) < epsilon; }

      static Eigen::MatrixX<double> fix_zeros(Eigen::MatrixX<double> obj) {
        return obj.unaryExpr([](double value) { return (std::abs(value) < epsilon) ? 0. : value; });
      }

      template <class T>
        requires aoc25::is_eigen_type_v<T>
      static auto hide_negative_zero(T obj) {
        return (obj.array().cwiseAbs() < epsilon).select(0., obj).matrix().eval();
      }

      static void check_canonical(Eigen::MatrixX<double> const & tableau) {
        for (int col = 1; col < tableau.cols() - 1; ++col) {
          auto const column = tableau.col(col);
          auto const var_coeffs = column(Eigen::seq(1, Eigen::placeholders::last));
          uint8_t const num_nonzero = (var_coeffs.array() != 0).count();

          if (num_nonzero == 1) {  // Might be a canonical unit vector, i.e. basic variable.
            int row = -1;          // Find location of non-zero element.
            double const coeff = var_coeffs.maxCoeff(&row);

            bool const is_basic_variable = (coeff == 1);
            if (is_basic_variable) {
              assert(column(0) == 0);  // Objective row must be zero.
            }
          }
        }
      }

      Eigen::RowVectorX<double> extract_variables() const {
        Eigen::RowVectorX<double> result = Eigen::RowVectorX<double>::Zero(offsets_.size());

        // If variable is basic (i.e. canonical unit vector), get its value from the tableau.
        static constexpr int first_variable_col = 1;
        int var_col = first_variable_col;

        auto const get_value_from_col = [&](int col) -> double {
          auto const column = tableau_.col(col);
          uint8_t const num_nonzero = (column.array() != 0).count();

          if (num_nonzero == 1) {
            int row = -1;  // Find location of non-zero element.
            double const coeff = column.maxCoeff(&row);

            if (coeff == 1) {  // Canonical unit vector, so basic variable. Result is in RHS.
              return tableau_(row, Eigen::placeholders::last);
            }
          }

          return 0.;
        };

        for (int var_idx = 0; var_idx < offsets_.cols(); ++var_idx) {
          if (is_free_variable_.at(var_idx)) {
            // Free variable, so represented by two variables in tableau.
            double const pos_value = get_value_from_col(var_col);
            double const neg_value = get_value_from_col(var_col + 1);
            result(var_idx) = pos_value - neg_value;
            var_col += 2;
          } else {
            // Regular variable, represented by single variable in tableau.
            result(var_idx) = get_value_from_col(var_col);
            var_col += 1;
          }
        }

        result -= offsets_;

        SPDLOG_TRACE("  simplex result: {}", result);
        return result;
      }

      void build_tableau(Eigen::RowVectorX<int16_t> const & objective,
                         Eigen::MatrixX<int16_t> const & constraints,
                         Eigen::VectorX<int16_t> const & bounds) {
        assert(objective.cols() == constraints.cols());
        assert(bounds.rows() == constraints.rows());

        // Check for lower bounds on variables, i.e. keep track of their offset.
        // Also check which variables are maybe free, i.e. not explicitly bounded >= 0.
        enum class bound_type_t : uint8_t { ignore, lower, upper };

        is_free_variable_ = std::vector<bool>(constraints.cols(), true);
        auto bound_types = std::vector<bound_type_t>(constraints.rows(), bound_type_t::ignore);
        offsets_ = Eigen::RowVectorX<double>::Zero(constraints.cols());

        for (int row = 0; row < constraints.rows(); ++row) {
          // If all but one variable is zero, then that variable is non-free.
          uint8_t const num_nonzero = (constraints.row(row).array() != 0).count();
          int16_t const bound = bounds(row);

          // If bound is negative, then number of pushes must be at least -bound, i.e. lower bound.
          bound_types[row] = (bound < 0) ? bound_type_t::lower : bound_type_t::upper;

          if (num_nonzero == 1) {
            for (int col = 0; col < constraints.cols(); ++col) {
              if (int16_t const coef = constraints(row, col); coef != 0) {
                is_free_variable_.at(col) = coef < 0;

                // If the bound is negative, the constraint is Ax >= b, with b = -bound. Thus, this
                // is a lower bound on only this variable, which means we must replace it with a new
                // variable x' = x - offset, where offset = b / A.
                if (bound < 0) {
                  // NOTE: Stored offset is negative.
                  double lower_limit = static_cast<double>(bound) / coef;
                  lower_limit =
                      (lower_limit < 0) ? std::floor(lower_limit) : std::ceil(lower_limit);

                  offsets_[col] = std::min(offsets_[col], lower_limit);

                  // TODO: Using this info to remove the constraint breaks things...
                  // bound_types[row] = bound_type_t::ignore;
                }

                break;
              }
            }
          }
        }

        // After that we add slack variables for each constraint.
        // Hence, the dimension of the tableau is:
        //  - rows: # null vectors + 1 (for objective)
        //  - cols: # variables + # slack variables + 2 * # surplus variables
        //          (i.e. incl. artificial variables) + 1 (for RHS) + 1 for objective value.
        size_t const num_unknowns = objective.cols();
        size_t const num_free_variables = std::ranges::count(is_free_variable_, true);
        size_t const num_variable_columns = num_unknowns + num_free_variables;

        size_t const num_constraints = constraints.rows();
        size_t const num_lower_bounds = std::ranges::count(bound_types, bound_type_t::ignore);

        size_t const num_slack_variables = std::ranges::count(bound_types, bound_type_t::upper);
        size_t const num_surplus_variables = std::ranges::count(bound_types, bound_type_t::lower);

        size_t const num_rows = num_constraints - num_lower_bounds + 1;
        size_t const num_cols =
            num_variable_columns + num_slack_variables + 2 * num_surplus_variables + 1 + 1;

        /* The format of the tableau is:
         *
         *  1 -C 0 0
         *  0  A I b
         *
         * where:
         *  - C is the cost vector (objective function coefficients)
         *  - A is the constraint matrix
         *  - I is the identity matrix for slack variables
         *  - b is the bounds vector (RHS)
         */
        tableau_ = Eigen::MatrixX<double>::Zero(num_rows, num_cols);

        // Put all constraints per variable into the tableau. This makes it much easier to
        // deal with free variables for which we need to duplicate the columns. At this point we
        // also set the objective coefficients, since these also might need to be duplicated.
        //
        // The objective should be minimized, so keep the signs. The simplex algorithm will maximize
        // the objective, so we would negate the objective coefficients. However, to make the
        // algorithm maximize, those coefficients must then be negated again, hence we would do a
        // double negation.
        int var_col = 1;
        for (int var_idx = 0; var_idx < static_cast<int>(num_unknowns); ++var_idx) {
          if (is_free_variable_.at(var_idx)) {
            // Free variable, so represented by two variables in tableau.
            tableau_(0, var_col) = objective(var_idx);
            tableau_(0, var_col + 1) = -objective(var_idx);

            tableau_.col(var_col).tail(num_constraints) = constraints.col(var_idx).cast<double>();
            tableau_.col(var_col + 1).tail(num_constraints) =
                -constraints.col(var_idx).cast<double>();
            var_col += 2;
          } else {
            // Regular variable, represented by single variable in tableau.
            tableau_(0, var_col) = objective(var_idx);
            tableau_.col(var_col).tail(num_constraints) = constraints.col(var_idx).cast<double>();
            var_col += 1;
          }
        }

        // Set objective value entry.
        tableau_(0, 0) = 1;

        // Add constant due to offsets.
        tableau_(0, Eigen::placeholders::last) = offsets_.dot(objective.cast<double>());

        // Set the constraint coefficients, bounds, and slack/surplus/artificial variables.
        /* There are two options:
         *
         *  - negative bound: when the solved variables are substituted back into the constraint,
         *    the result must be >= -bound, to ensure the corresponding number of pushes becomes
         *    positive. This means the constraint is of the form Ax >= b, where b is positive. In
         *    this case we need to add a surplus variable, not a slack variable. We also have to add
         *    an artificial variable to be able to find an initial basic feasible solution. Note
         *    that neither a surplus, nor an artificial variable should be added if this constraint
         *    is a lower bound on a single variable.
         *
         *  - positive bound: when the solved variables are substituted back into the constraint,
         *    the result must be >= -bound, to ensure the corresponding number of pushes remains
         *    positive. Since bounds must always be positive, we must first negate the constraint.
         *    This reverses the inequality, i.e. the new constraints is of the form -Ax <= bound,
         *    with bound positive. Since the inequality is now of the <= type, we need to add a
         *    slack variable.
         */
        int slack_surplus_col = 1 + num_variable_columns;
        int artificial_col = slack_surplus_col + num_slack_variables + num_surplus_variables;
        int row_idx = 1;

        for (size_t constraint_idx = 0; constraint_idx < num_constraints; ++constraint_idx) {
          auto const bound_type = bound_types.at(constraint_idx);
          if (bound_type == bound_type_t::ignore) {
            continue;  // Skip constraints that are just variable bounds.
          }

          int16_t const bound = bounds(constraint_idx);
          auto constraint = tableau_.row(row_idx);
          bool negate_constraint = false;

          if (bound_type == bound_type_t::lower) {           // Ax >= -bound
            constraint(Eigen::placeholders::last) = -bound;  // Negate bound.

            constraint(slack_surplus_col) = -1;  // Subtract surplus variable.
            constraint(artificial_col) = 1;      // Add artificial variable.

            ++slack_surplus_col;
            ++artificial_col;
          } else if (bound_type == bound_type_t::upper) {  // -Ax <= bound
            negate_constraint = true;
            constraint = -constraint;                       // Negate the constraint.
            constraint(slack_surplus_col) = 1;              // Add slack variable.
            constraint(Eigen::placeholders::last) = bound;  // Positive bound.

            ++slack_surplus_col;
          }

          // Adjust for variable offsets. Need to use the original, not possibly duplicated due to
          // free variables, constraint here.
          constraint(Eigen::placeholders::last) +=
              (negate_constraint ? -1 : 1) *
              offsets_.dot(constraints.row(constraint_idx).cast<double>());

          ++row_idx;
        }

        check_canonical(tableau_);

        // If num_surplus_variables > 0, we need to perform Phase 1 of the two-phase simplex method
        // to find an initial basic feasible solution.
        if (num_surplus_variables > 0) {
          run_phase_one(1 + num_variable_columns + num_slack_variables + num_surplus_variables,
                        num_surplus_variables);
        }

        SPDLOG_TRACE("  has_solution: {}, offsets: [{}], tableau:\n{}", has_solution_, offsets_,
                     hide_negative_zero(tableau_));
      }

      void fix_basic_variable_objective_coeffs(Eigen::MatrixX<double> & tableau) {
        static constexpr int first_constraint_row = 1;

        for (int col = 1; col < tableau.cols() - 1; ++col) {
          auto const column = tableau.col(col);
          if (column(0) == 0) {
            continue;  // Objective coefficient is already zero.
          }

          auto const var_coeffs =
              column(Eigen::seq(first_constraint_row, Eigen::placeholders::last));

          if (uint8_t const num_nonzero = (var_coeffs.array() != 0).count(); num_nonzero == 1) {
            int row = -1;  // Find location of non-zero element.
            double const coeff = var_coeffs.maxCoeff(&row);

            if (coeff != 1) {  // Not a canonical unit vector, i.e. not a basic variable, so skip.
              continue;
            }

            assert(row != -1);
            row += first_constraint_row;  // Fix row offset.

            // Subtract multiple of this row from objective row to zero it out.
            tableau.row(0) -= tableau(0, col) * tableau.row(row);
          }
        }
      }

      void run_phase_one(int artificial_vars_begin, int num_artificial_vars) {
        // Keep track of the original objective function.
        Eigen::RowVectorX<double> const original_objective = tableau_.row(0);

        SPDLOG_TRACE("  [simplex] before Phase 1 tableau:\n{}", hide_negative_zero(tableau_));

        // Set Phase 1 objective function to minize the sum of artificial variables (i.e. make
        // them zero).
        tableau_.row(0).setZero();
        tableau_(0, 0) = 1;
        tableau_(0, Eigen::placeholders::lastN(num_artificial_vars + 1)).setOnes();
        tableau_(0, Eigen::placeholders::last) = 0;

        SPDLOG_TRACE("  [simplex] pre-canonical phase 1 tableau:\n{}",
                     hide_negative_zero(tableau_));

        // Make all artificial variable columns canonical unit vectors.
        static constexpr int first_constraint_row = 1;

        for (int offset = 0; offset < num_artificial_vars; ++offset) {
          int row = -1;

          int const artifical_var_col = artificial_vars_begin + offset;
          [[maybe_unused]] auto const value =
              tableau_
                  .col(artifical_var_col)(
                      Eigen::seq(first_constraint_row, Eigen::placeholders::last))
                  .maxCoeff(&row);
          assert(value == 1);

          // Fix row offset.
          row += first_constraint_row;

          // Subtract the constraint row from the objective row.
          tableau_.row(0) -= tableau_.row(row);
        }

        SPDLOG_TRACE("  [simplex] initial phase 1 tableau:\n{}", hide_negative_zero(tableau_));

        // Solve the system to find an initial basic feasible solution.
        [[maybe_unused]] bool const success = solve_impl();
        assert(success);

        // Check that the objective value is zero, i.e. all artificial variables are zero. If not,
        // there's no feasible solution.
        has_solution_ = is_zero(tableau_(0, Eigen::placeholders::last));
        if (!has_solution_) {
          SPDLOG_TRACE("  [simplex] phase 1 found no feasible solution, objective value: {}",
                       tableau_(0, Eigen::placeholders::last));
          return;
        }

        // Restore original objective function.
        tableau_.row(0) = original_objective;

        // Ensure objective coeffients in columns of basic variables are zero.
        fix_basic_variable_objective_coeffs(tableau_);

        // Remove artificial variable columns from tableau.
        tableau_.col(artificial_vars_begin) = tableau_.rightCols(1);
        tableau_.conservativeResize(Eigen::NoChange, tableau_.cols() - num_artificial_vars);

        SPDLOG_TRACE("  [simplex] phase 1 after objective restore:\n{}",
                     hide_negative_zero(tableau_));
      }

      bool solve_impl() {
        assert(has_solution_);

        static double constexpr limit = 0;
        bool success = false;

        check_canonical(tableau_);

        while (true) {
          // Find pivot column (most negative in objective row).
          static constexpr int first_pivot_col = 1;

          int pivot_col;
          double min_val =
              tableau_.row(0).middleCols(first_pivot_col, tableau_.cols() - 2).minCoeff(&pivot_col);

          if (min_val >= limit) {  // Optimal found.
            success = true;
            break;
          }

          pivot_col += first_pivot_col;  // Fix pivot column offset.

          // Find pivot row with lowest ratio.
          int pivot_row = -1;
          auto min_ratio = std::numeric_limits<double>::infinity();

          for (int row_idx = 1; row_idx < tableau_.rows(); ++row_idx) {
            auto row = tableau_.row(row_idx);

            if (row(pivot_col) > limit) {
              double const ratio = row(Eigen::placeholders::last) / row(pivot_col);

              if (ratio < min_ratio) {
                min_ratio = ratio;
                pivot_row = row_idx;
              }
            }
          }

          assert(pivot_row != -1);  // A solution should always exist.

          // Pivot operation.
          tableau_.row(pivot_row) /= tableau_(pivot_row, pivot_col);

          for (int row = 0; row < tableau_.rows(); ++row) {
            if (row != pivot_row) {
              tableau_.row(row) -= tableau_(row, pivot_col) * tableau_.row(pivot_row);
            }
          }

          // Prevent numerical instability by fixing very small values to zero.
          tableau_ = fix_zeros(std::move(tableau_));

          SPDLOG_TRACE("  [simplex] pivot: ({}, {}), min_val: {}, tableau:\n{}", pivot_row,
                       pivot_col, min_val, hide_negative_zero(tableau_));
        }

        SPDLOG_TRACE("  [simplex] has solution: {}, final tableau:\n{}", has_solution_,
                     hide_negative_zero(tableau_));
        return success;
      }
    };

    void find_num_pushes_impl(Eigen::VectorX<int16_t> const & base,
                              Eigen::MatrixX<int16_t> const & constraints,
                              Eigen::RowVectorX<int16_t> const & objective,
                              double & best_target_change,
                              uint64_t & branch_counter,
                              uint8_t log_indent) {
      branch_counter += 1;

      // Run simplex to find optimal (floating point) solution.
      auto simplex = simplex_t(objective, constraints, base);
      auto simplex_result = simplex.solve();

      if (!simplex_result.has_value()) {
        SPDLOG_DEBUG(" {:{}}simplex found no solution", "", log_indent);
        return;
      }

      auto const & [target_change, null_vector_changes] = simplex_result.value();

      // Results are floating point, so don't do exact comparison.
      auto const is_integer = [](double const val) {
        static constexpr double epsilon = 1e-10;
        return std::abs(std::round(val) - val) < epsilon;
      };

      // If the solution can't improve on the best found so far, stop searching.
      if (target_change <= best_target_change) {
        SPDLOG_DEBUG(" {:{}}pruning branch: target_change {} <= best_target_change {}", "",
                     log_indent, target_change, best_target_change);
        return;
      }

      // If the result is integer, the optimal solution for this branch is found.
      bool const all_integers = std::ranges::all_of(null_vector_changes, is_integer);

      if (all_integers) {
        // Only update best solution if it's integer. Otherwise we might not branch correctly.
        best_target_change = std::max(best_target_change, target_change);
        SPDLOG_DEBUG(" {:{}}found integer solution: {}, target_change {}", "", log_indent,
                     null_vector_changes, target_change);
        return;
      }

      // No integer solution: branch and bound on fractional variables.
      SPDLOG_DEBUG(
          " {:{}}branching on fractional solution: target_change {}, null_vector_changes: {}", "",
          log_indent, target_change, null_vector_changes);

      // First find which values are fractional.
      auto const is_fractional =
          null_vector_changes |
          std::views::transform([&](double const val) { return !is_integer(val); }) |
          std::ranges::to<std::vector>();
      uint8_t const num_fractional = std::ranges::count(is_fractional, true);
      assert(num_fractional < 8);

      // Use a counter as a bitmask to decide whether to create a lower or upper bound for each
      // fractional variable.
      uint8_t const num_branches = (1 << num_fractional);

      for (uint8_t branch_idx = 0; branch_idx < num_branches; ++branch_idx) {
        // Create extra constraints for this branch.
        Eigen::VectorX<int16_t> extended_base =
            Eigen::VectorX<int16_t>::Zero(base.rows() + num_fractional);
        Eigen::MatrixX<int16_t> extended_constraints =
            Eigen::MatrixX<int16_t>::Zero(constraints.rows() + num_fractional, constraints.cols());

        extended_base.topRows(base.rows()) = base;
        extended_constraints.topRows(constraints.rows()) = constraints;

        uint8_t const row_offset = base.rows();
        uint8_t fractional_idx = 0;

        for (size_t var_idx = 0; var_idx < is_fractional.size(); ++var_idx) {
          if (!is_fractional.at(var_idx)) {
            continue;
          }

          double const val = null_vector_changes(var_idx);
          bool const lower_not_upper = ((branch_idx >> fractional_idx) & 1) == 0;

          if (lower_not_upper) {  // A negative base value is a lower bound constraint.
            SPDLOG_DEBUG(" {:{}}branching on var {}: lower bound >= {}", "", log_indent, var_idx,
                         std::ceil(val));
            extended_base(row_offset + fractional_idx) = -std::ceil(val);
            extended_constraints(row_offset + fractional_idx, var_idx) = 1;
          } else {  // An positive base value is an upper bound constraint.
            SPDLOG_DEBUG(" {:{}}branching on var {}: upper bound <= {}", "", log_indent, var_idx,
                         std::floor(val));
            extended_base(row_offset + fractional_idx) = std::floor(val);
            extended_constraints(row_offset + fractional_idx, var_idx) = -1;
          }

          ++fractional_idx;
        }

        // Recurse into this branch.
        find_num_pushes_impl(extended_base, extended_constraints, objective, best_target_change,
                             branch_counter, log_indent + 2);
      }
    }

    uint64_t find_num_pushes(Eigen::VectorX<int16_t> & x_base,
                             Eigen::MatrixX<int16_t> const & null_space) {
      // Calculate how null space column influences the total number of pushes.
      Eigen::RowVectorX<int16_t> const push_influence = null_space.colwise().sum();
      SPDLOG_TRACE(" push influence: {}", push_influence);

      // If none of the columns influence the total number of pushes, just count the number of
      // pushes in the base solution. We know there's a solution with a positive number of pushes,
      // but we don't need to actually find it.
      if (push_influence.isZero()) {
        return x_base.sum();
      }

      // If there's only a single column in the null space, directly calculate optimal solution.
      if (null_space.cols() == 1) {
        // Calculate limits for each free variable in the null space.
        auto const & null_space_col = null_space.col(0);
        limits_t const limits = calculate_limits(x_base, std::span(null_space_col));
        SPDLOG_TRACE(" limits: {}", limits);

        // Determine optimal value within limits.
        int16_t optimal_value = (push_influence(0) > 0) ? limits.lower : limits.upper;
        return (x_base + optimal_value * null_space_col).sum();
      } else {
        double best_target_change = -std::numeric_limits<double>::infinity();
        uint64_t branch_counter = 0;

        // Branch and bound using simplex to find optimal integer solution.
        find_num_pushes_impl(x_base, null_space, push_influence, best_target_change, branch_counter,
                             0);
        SPDLOG_DEBUG(" [branch-and-bound] branch count: {}, best target change: {}", branch_counter,
                     best_target_change);

        return x_base.sum() - std::round(best_target_change);
      }
    }

  }  // namespace

  uint64_t day_t<10>::solve(part_t<1>, version_t<0>, simd_string_view_t input) {
    auto const problems = parse_input(input);

    // Note: the "problem.switches" is a spanning set, but not a minimum spanning set, i.e. not a
    // basis, so we can't just do Gaussian elimination to find the solution.
    uint64_t total = 0;

    // Iterate first over all values with 1 bit set, then 2 bits set, etc. This ensures that as
    // soon as we find a solution, it is the minimum number of toggles and we can stop the search.
    static constexpr size_t max_switches = 13;
    static constexpr combination_tables_t<max_switches> combination_tables;

#pragma omp parallel for reduction(+ : total) schedule(static) num_threads(8)
    for (auto const & problem : problems) {
      SPDLOG_DEBUG("Solving {} ({} switches)", problem, problem.switches.size());
      assert(problem.switches.size() <= max_switches);

      auto const table = combination_tables.table_for(problem.switches.size());

      // Combinations are in Gray-code order, i.e. each entry differs from the previous one in
      // exactly two bits. So keep track of previous state and just apply the difference. Note
      // that when the number of set bits changes, the number of differences can be more than two.
      // However, the loop handles this just fine.
      uint16_t prev_combo = 0;
      uint16_t state = 0;

      for (uint16_t idx = 0; (idx < table.size()); ++idx) {
        // TODO: An option here would be to precompute all possible combo_diff values. These can
        // then be used to compute a LUT for to the switch values. Since there's at most 13
        // switches, and we know that combo_diff has exactly two bits set, that means a LUT of
        // bin(13, 2) = 78 entries. In that case we do have to make sure that when transitioning
        // from N to N + 1 bits set, we apply all required differences, such that we start with all
        // the necessary switches toggled on.
        uint16_t const combo = table[idx];
        uint16_t const combo_diff = prev_combo ^ combo;  // Determine which bits changed.
        prev_combo = combo;

        state = apply_diff_switches(problem.switches, combo_diff, state);

        if (state == problem.target) {
          uint16_t const num_used_switches = std::popcount(combo);
          SPDLOG_DEBUG(" Found solution {:#0{}b} (# toggles: {})", combo,
                       problem.switches.size() + 2, num_used_switches);
          total += num_used_switches;
          break;  // Found solution is always the optimal one.
        }
      }
    }

    return total;
  }

  uint64_t day_t<10>::solve(part_t<2>, version_t<0>, simd_string_view_t input) {
    auto const problems = parse_input(input);
    uint64_t total = 0;

#pragma omp parallel for reduction(+ : total) schedule(guided) num_threads(4)
    for (auto const & problem : problems) {
      // Obtain generator for all possible integer solutions, with reduced number of free
      // variables.
      SPDLOG_DEBUG("Solving {}", problem);
      auto [x_base, null_space] = generate_solution_system(problem);

      // If the null space is empty, the only solution is already found. If not, we need to find
      // the best possible solution.
      uint64_t const num_pushes =
          (null_space.cols() == 0) ? x_base.sum() : find_num_pushes(x_base, null_space);
      total += num_pushes;
      SPDLOG_DEBUG("  minimal solution with # pushes: {}", num_pushes);
    }

    return total;
  }

}  // namespace aoc25
