#pragma once

#include "aoc25/simd.hpp"

#include <cstddef>
#include <iterator>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace aoc25 {

  namespace detail {

    template <class Key, class Value>
    struct n_ary_tree_element_t {
      Key key;
      Value value;
    };

    template <class Key, class Value>
    [[maybe_unused]] std::string format_as(n_ary_tree_element_t<Key, Value> const & obj) {
      return fmt::format("<{}: {}>", obj.key, obj.value);
    }

    template <class Key, class Value>
    struct n_ary_tree_element_ref_t {
      Key const & key;
      Value const & value;
    };

    template <class Key, class Value>
    [[maybe_unused]] std::string format_as(n_ary_tree_element_ref_t<Key, Value> const & obj) {
      return fmt::format("<{}: {}>", obj.key, obj.value);
    }

    enum class n_ary_tree_variant_t {
      min,
      max,
    };

    // Forward declare SIMD sift-down function.
    template <class Tree>
    void simd_sift_down_with_values(Tree &, size_t);

    /** @brief An N-ary min/max tree with a maximum size.
     *
     * The tree is builds a min/max-tree of up the given number of elements. New elements can be
     * pushed using push_or_replace(), which either adds the new element (if the tree is not full),
     * or replaces the first element  (if the tree is full and the new element is "better" than the
     * first one). This allows keeping track of the "best N" elements seen so far.
     */
    template <class Key, class Value, size_t BranchingFactor, n_ary_tree_variant_t Variant>
      requires std::is_arithmetic_v<Key>
    class sized_n_ary_tree_t {
     public:
      using key_t = Key;
      using value_t = Value;
      using element_t = detail::n_ary_tree_element_t<key_t, value_t>;
      using element_ref_t = detail::n_ary_tree_element_ref_t<key_t, value_t>;

      static constexpr size_t branching_factor = BranchingFactor;
      static constexpr n_ary_tree_variant_t variant = Variant;

      sized_n_ary_tree_t()
          : sized_n_ary_tree_t(0) {}

      explicit sized_n_ary_tree_t(size_t max_size)
          : keys_{}, values_{}, current_size_{0}, max_size_{0} {
        resize(max_size);
      }

      void resize(size_t new_size) {
        if (max_size_ != 0) {
          throw std::runtime_error(
              "Cannot resize a sized_n_ary_tree_t which already has a max size set");
        }
        assert(current_size_ == 0);

        max_size_ = new_size;
        keys_.reserve(max_size_);
        values_.reserve(max_size_);
      }

      void push(element_t elem) {
        // TODO: Throw instead.
        assert(current_size_ < max_size_);
        keys_.push_back(elem.key);
        values_.push_back(std::move(elem.value));
        sift_up(current_size_++);
      }

      /**
       * @brief Inserts a value or replaces the min:
       *   - If heap is not full, push the new value normally.
       *   - If heap is full AND (new_value > current min), replace root and sift down.
       *   - Otherwise, do nothing (new value is too large to enter the "best X" set).
       */
      bool push_or_replace(element_t elem) {
        assert(max_size_ > 0);

        if (current_size_ < max_size_) {
          push(std::move(elem));
          return true;
        } else if (is_higher_in_heap(keys_[0], elem.key)) {
          keys_[0] = elem.key;
          values_[0] = std::move(elem.value);
          sift_down(0);
          return true;
        }

        return false;
      }

      element_t pop() {
        assert(current_size_ != 0);
        auto result = element_t{.key = keys_[0], .value = std::move(values_[0])};

        if (--current_size_ > 0) {
          keys_[0] = keys_[current_size_];
          values_[0] = std::move(values_[current_size_]);
          sift_down(0);
        }

        keys_.pop_back();
        values_.pop_back();
        assert(keys_.size() == current_size_);
        assert(values_.size() == keys_.size());

        return result;
      }

      element_ref_t top() const {
        if (current_size_ == 0) {
          throw std::runtime_error("Cannot get top of empty sized_n_ary_tree_t");
        }
        return element_ref_t{.key = keys_[0], .value = values_[0]};
      }

      size_t size() const { return current_size_; }
      size_t max_size() const { return max_size_; }

      // Allow iterating over all the elements without popping them off.
      std::span<Key const> keys() const { return keys_; }
      std::span<value_t const> values() const { return values_; }

      /// @brief Extract the values (e.g. after building the tree).
      std::vector<value_t> extract_values() && { return std::move(values_); }

     private:
      using self_t = sized_n_ary_tree_t<key_t, value_t, branching_factor, variant>;

      simd_vector_t<Key> keys_;
      std::vector<value_t> values_;
      size_t current_size_;
      size_t max_size_;

      bool is_higher_in_heap(Key const & lhs, Key const & rhs) const {
        // By inverting the order of rhs and lhs, std::less creates a max-heap, which is the same
        // behavior as STL. So for a min-heap, use > instead of <.
        if constexpr (variant == n_ary_tree_variant_t::min) {
          return rhs > lhs;
        } else {
          static_assert(variant == n_ary_tree_variant_t::max);
          return rhs < lhs;
        }
      }

      void sift_up(size_t pos) {
        while (pos > 0) {
          size_t parent = (pos - 1) / branching_factor;
          if (!is_higher_in_heap(keys_[pos], keys_[parent])) {
            break;
          }

          using std::swap;
          swap(keys_[pos], keys_[parent]);
          swap(values_[pos], values_[parent]);
          pos = parent;
        }
      }

      // Allow the SIMD sift-down function to fall back to the scalar implementation.
      friend void simd_sift_down_with_values<self_t>(self_t &, size_t pos);

      void sift_down(size_t pos) {
        // The SIMD code will fall back to scalar code if the branching factor is too large.
        simd_sift_down_with_values(*this, pos);
      }

      [[maybe_unused]] void sift_down_scalar(size_t pos) {
        while (true) {
          size_t first_child = branching_factor * pos + 1;
          if (first_child >= current_size_) {
            break;
          }

          // Find the best among up to branching_factor children.
          size_t best_child = first_child;
          size_t end_child = std::min(first_child + branching_factor, current_size_);

          for (size_t child = first_child + 1; child < end_child; ++child) {
            best_child = (is_higher_in_heap(keys_[child], keys_[best_child])) ? child : best_child;
          }

          if (!is_higher_in_heap(keys_[best_child], keys_[pos])) {
            break;
          }

          using std::swap;
          swap(keys_[pos], keys_[best_child]);
          swap(values_[pos], values_[best_child]);
          pos = best_child;
        }
      }
    };

  }  // namespace detail

  template <class Key, class Value, size_t BranchingFactor>
  using sized_n_ary_min_tree_t =
      detail::sized_n_ary_tree_t<Key, Value, BranchingFactor, detail::n_ary_tree_variant_t::min>;

  template <class Key, class Value, size_t BranchingFactor>
  using sized_n_ary_max_tree_t =
      detail::sized_n_ary_tree_t<Key, Value, BranchingFactor, detail::n_ary_tree_variant_t::max>;

}  // namespace aoc25

#include "aoc25/n_ary_tree-hwy.hpp"
