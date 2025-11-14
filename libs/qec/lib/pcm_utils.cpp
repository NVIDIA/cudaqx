/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/pcm_utils.h"
#include <cassert>
#include <cstring>
#include <random>

namespace cudaq::qec {

/// @brief Return a vector of column indices that would sort the PCM columns
/// in topological order.
/// @param row_indices For each column, a vector of row indices that have a
/// non-zero value in that column.
/// @param num_syndromes_per_round The number of syndromes per round. (Defaults
/// to 0, which means that no secondary per-round sorting will occur.)
/// @details This function tries to make a matrix that is close to a block
/// diagonal matrix from its input. If \p num_syndromes_per_round is > 0, then
/// the columns are first sorted by rounds numbers in which the checks are
/// performed. Columns are then sorted by the index of the first non-zero entry
/// in the column, and if those match, then they are sorted by the index of the
/// last non-zero entry in the column. This ping pong continues for the indices
/// of the second non-zero element and the second-to-last non-zero element, and
/// so forth.
std::vector<std::uint32_t> get_sorted_pcm_column_indices(
    const std::vector<std::vector<std::uint32_t>> &row_indices,
    std::uint32_t num_syndromes_per_round) {
  std::vector<std::uint32_t> column_order(row_indices.size());
  std::iota(column_order.begin(), column_order.end(), 0);
  std::sort(column_order.begin(), column_order.end(),
            [&row_indices, num_syndromes_per_round](const std::uint32_t &a,
                                                    const std::uint32_t &b) {
              const auto &a_vec = row_indices[a];
              const auto &b_vec = row_indices[b];

              if (a_vec.size() == 0 && b_vec.size() != 0)
                return true;
              if (a_vec.size() != 0 && b_vec.size() == 0)
                return false;
              if (a_vec.size() == 0 && b_vec.size() == 0)
                return a < b; // stable sort.

              // Now we know both vectors have at least one element.

              // Have a and b iterators, both head and tail versions of both.
              auto a_it_head = a_vec.begin();
              auto a_it_tail = a_vec.end() - 1;
              auto b_it_head = b_vec.begin();
              auto b_it_tail = b_vec.end() - 1;

              // First sort by the span of rounds that the errors appear in. We
              // can only do this sorting if we know how many syndromes per
              // round.
              if (num_syndromes_per_round > 0) {
                auto a_first_round = *a_it_head / num_syndromes_per_round;
                auto a_last_round = *a_it_tail / num_syndromes_per_round;
                auto b_first_round = *b_it_head / num_syndromes_per_round;
                auto b_last_round = *b_it_tail / num_syndromes_per_round;
                if (a_first_round != b_first_round)
                  return a_first_round < b_first_round;
                if (a_last_round != b_last_round)
                  return a_last_round < b_last_round;
              }

              // Now we sort the columns corresponding to errors that occur in
              // the same rounds.
              do {
                // Compare the head elements.
                if (*a_it_head != *b_it_head)
                  return *a_it_head < *b_it_head;

                // Before checking the tail iterators, make sure they are not
                // aliased to the head elements that we just compared. If so,
                // we've exhausted one of the vectors and will return
                // accordingly.

                // Check if we ran out of "a" elements.
                if (a_it_head == a_it_tail && b_it_head != b_it_tail)
                  return true;
                // Check if we ran out of "b" elements.
                if (a_it_head != a_it_tail && b_it_head == b_it_tail)
                  return false;
                if (a_it_head == a_it_tail && b_it_head == b_it_tail)
                  return a < b; // stable sort.

                // Compare the tail elements.
                if (*a_it_tail != *b_it_tail)
                  return *a_it_tail < *b_it_tail;

                // Advance the head iterators.
                a_it_head++;
                b_it_head++;

                // Check to see if the new head iterators match the tail
                // iterators that we just compared. If so, we've exhausted one
                // of the vectors and will return accordingly.
                if (a_it_head == a_it_tail && b_it_head != b_it_tail)
                  return true;
                if (a_it_head != a_it_tail && b_it_head == b_it_tail)
                  return false;
                if (a_it_head == a_it_tail && b_it_head == b_it_tail)
                  return a < b; // stable sort.

                // Decrement the tail iterators.
                a_it_tail--;
                b_it_tail--;
              } while (true);

              // Unreachable.
              return a < b;
            });

  return column_order;
}

/// @brief Check if a PCM is sorted.
/// @param pcm The PCM to check.
/// @param num_syndromes_per_round The number of syndromes per round.
/// @return True if the PCM is sorted, false otherwise.
bool pcm_is_sorted(const cudaqx::tensor<uint8_t> &pcm,
                   std::uint32_t num_syndromes_per_round) {
  if (pcm.rank() != 2) {
    throw std::invalid_argument("pcm_is_sorted: PCM must be a 2D tensor");
  }

  auto column_indices =
      get_sorted_pcm_column_indices(pcm, num_syndromes_per_round);
  auto num_cols = pcm.shape()[1];
  for (std::size_t c = 0; c < num_cols; c++)
    if (column_indices[c] != c)
      return false;
  return true;
}

/// @brief Return a sparse representation of the PCM.
/// @return A vector of vectors that sparsely represents the PCM. The size of
/// the outer vector is the number of columns in the PCM, and the i-th element
/// contains an inner vector of the row indices of the non-zero elements in the
/// i-th column of the PCM.
std::vector<std::vector<std::uint32_t>>
get_sparse_pcm(const cudaqx::tensor<uint8_t> &pcm) {
  if (pcm.rank() != 2) {
    throw std::invalid_argument("get_sparse_pcm: PCM must be a 2D tensor");
  }

  auto num_rows = pcm.shape()[0];
  auto num_cols = pcm.shape()[1];

  // Form a sparse representation of the PCM.
  std::vector<std::vector<std::uint32_t>> row_indices(num_cols);
  for (std::size_t r = 0; r < num_rows; r++) {
    auto *row = &pcm.at({r, 0});
    for (std::size_t c = 0; c < num_cols; c++)
      if (row[c])
        row_indices[c].push_back(r);
  }

  return row_indices;
}

/// @brief Return a vector of column indices that would sort the pcm columns
/// in topological order.
std::vector<std::uint32_t>
get_sorted_pcm_column_indices(const cudaqx::tensor<uint8_t> &pcm,
                              std::uint32_t num_syndromes_per_round) {
  if (pcm.rank() != 2) {
    throw std::invalid_argument(
        "get_sorted_pcm_column_indices: PCM must be a 2D tensor");
  }

  auto row_indices = get_sparse_pcm(pcm);

  return get_sorted_pcm_column_indices(row_indices, num_syndromes_per_round);
}

/// @brief Reorder the columns of a PCM according to the given column order.
/// Note: this may return a subset of the columns in the original PCM if the
/// \p column_order does not contain all of the columns in the original PCM.
/// @param pcm The PCM to reorder.
/// @param column_order The column order to use for reordering.
/// @param row_begin The first row to include in the reordering. Leave at the
/// default value to include all rows.
/// @param row_end The last row to include in the reordering. Leave at the
/// default value to include all rows.
/// @return A new PCM with the columns reordered according to the given column
/// order.
cudaqx::tensor<uint8_t>
reorder_pcm_columns(const cudaqx::tensor<uint8_t> &pcm,
                    const std::vector<std::uint32_t> &column_order,
                    uint32_t row_begin, uint32_t row_end) {
  if (pcm.rank() != 2) {
    throw std::invalid_argument("reorder_pcm_columns: PCM must be a 2D tensor");
  }
  if (row_begin > row_end) {
    throw std::invalid_argument(
        "reorder_pcm_columns: row_begin must be less than or equal to row_end");
  }

  auto num_rows = pcm.shape()[0];
  auto num_cols = pcm.shape()[1];
  auto new_num_cols = column_order.size();

  // Clamp row_end to the last row in the PCM.
  row_end = std::min(row_end, static_cast<uint32_t>(num_rows - 1));
  auto num_rows_to_copy = row_end - row_begin + 1;

  for (auto c : column_order) {
    if (c >= num_cols) {
      throw std::invalid_argument(
          "reorder_pcm_columns: column_order contains a column index that is "
          "greater than the number of columns in PCM");
    }
  }

  auto transposed_pcm = pcm.transpose();
  cudaqx::tensor<uint8_t> new_pcm_t(
      std::vector<std::size_t>{new_num_cols, num_rows_to_copy});
  for (std::size_t c = 0; c < new_num_cols; c++) {
    auto *orig_col = &transposed_pcm.at({column_order[c], row_begin});
    auto *new_col = &new_pcm_t.at({c, 0});
    std::memcpy(new_col, orig_col, num_rows_to_copy * sizeof(uint8_t));
  }

  return new_pcm_t.transpose();
}

/// @brief Sort the columns of a PCM in topological order.
/// @param pcm The PCM to sort.
/// @return A new PCM with the columns sorted in topological order.
cudaqx::tensor<uint8_t>
sort_pcm_columns(const cudaqx::tensor<uint8_t> &pcm,
                 std::uint32_t num_syndromes_per_round) {
  auto column_order =
      get_sorted_pcm_column_indices(pcm, num_syndromes_per_round);
  return reorder_pcm_columns(pcm, column_order);
}

/// @brief Simplify a PCM by removing duplicate columns and 0-weight columns,
/// and combine the probability weight vectors accordingly.
/// @param pcm The PCM to simplify.
/// @param weights The probability weight vectors to combine. This assumes all
/// error mechanisms are independent of each other.
/// @return A new PCM with the columns sorted in topological order, and the
/// probability weight vectors combined accordingly.
std::pair<cudaqx::tensor<uint8_t>, std::vector<double>>
simplify_pcm(const cudaqx::tensor<uint8_t> &pcm,
             const std::vector<double> &weights,
             std::uint32_t num_syndromes_per_round) {
  auto row_indices = get_sparse_pcm(pcm);
  auto column_order =
      get_sorted_pcm_column_indices(pcm, num_syndromes_per_round);
  // March through the columns in topological order, and combine the probability
  // weight vectors if the columns have the same row indices.
  std::vector<std::vector<std::uint32_t>> new_row_indices;
  std::vector<double> new_weights;
  const auto num_cols = column_order.size();
  for (std::size_t c = 0; c < num_cols; c++) {
    auto column_index = column_order[c];
    auto &curr_row_indices = row_indices[column_index];
    // If the column has no non-zero elements, or a weight of 0, then we skip
    // it.
    if (curr_row_indices.size() == 0 || weights[column_index] == 0)
      continue;
    if (new_row_indices.empty()) {
      new_row_indices.push_back(curr_row_indices);
      new_weights.push_back(weights[column_index]);
    } else {
      auto &prev_row_indices = new_row_indices.back();
      if (prev_row_indices == curr_row_indices) {
        // The current column has the same row indices as the previous column,
        // so we update the weights and do NOT add the duplicate column.
        auto prev_weight = new_weights.back();
        auto curr_weight = weights[column_index];
        auto new_weight =
            prev_weight + curr_weight - 2.0 * prev_weight * curr_weight;
        new_weights.back() = new_weight;
      } else {
        // The current column has different row indices than the previous
        // column. So we add the current column to the new PCM, and update the
        // weights.
        new_row_indices.push_back(curr_row_indices);
        new_weights.push_back(weights[column_index]);
      }
    }
  }

  // The new PCM may have fewer columns than the original PCM.
  cudaqx::tensor<uint8_t> new_pcm(
      std::vector<std::size_t>{pcm.shape()[0], new_row_indices.size()});
  for (std::size_t c = 0; c < new_row_indices.size(); c++)
    for (auto r : new_row_indices[c])
      new_pcm.at({r, c}) = 1;

  return std::make_pair(new_pcm, new_weights);
}

std::tuple<cudaqx::tensor<uint8_t>, std::uint32_t, std::uint32_t>
get_pcm_for_rounds(const cudaqx::tensor<uint8_t> &pcm,
                   std::uint32_t num_syndromes_per_round,
                   std::uint32_t start_round, std::uint32_t end_round,
                   bool straddle_start_round, bool straddle_end_round) {
  if (num_syndromes_per_round == 0) {
    throw std::invalid_argument(
        "get_pcm_for_rounds: num_syndromes_per_round must be greater than 0");
  }
  if (num_syndromes_per_round > pcm.shape()[0]) {
    throw std::invalid_argument(
        "get_pcm_for_rounds: num_syndromes_per_round must be less than the "
        "number of rows in PCM");
  }

  // Trim down to the right rows
  auto first_row_to_keep = start_round * num_syndromes_per_round;
  auto last_row_to_keep = (end_round + 1) * num_syndromes_per_round - 1;

  if (first_row_to_keep >= pcm.shape()[0]) {
    throw std::invalid_argument(
        "get_pcm_for_rounds: first_row_to_keep is greater than the number of "
        "rows in PCM");
  }
  if (last_row_to_keep >= pcm.shape()[0]) {
    throw std::invalid_argument(
        "get_pcm_for_rounds: last_row_to_keep is greater than the number of "
        "rows in PCM");
  }

  // Get a sparse representation of the PCM.
  auto row_indices = get_sparse_pcm(pcm);

  // Get the columns that have any non-zero data in the range [start_round,
  // end_round]. Use straddle_start_round and straddle_end_round accordingly.
  std::vector<std::uint32_t> columns_in_range;
  for (std::size_t c = 0; c < row_indices.size(); c++) {
    auto &rows_for_this_column = row_indices[c];
    if (rows_for_this_column.size() == 0)
      continue;
    auto first_round = rows_for_this_column.front() / num_syndromes_per_round;
    auto last_round = rows_for_this_column.back() / num_syndromes_per_round;
    // If the first_round/last_round is fully within the range [start_round,
    // end_round], then we include this column.
    if (first_round >= start_round && last_round <= end_round)
      columns_in_range.push_back(c);
    // If it straddles the start_round, then we only include it if
    // straddle_start_round is true.
    else if (straddle_start_round && first_round <= start_round &&
             last_round >= start_round)
      columns_in_range.push_back(c);
    // If it straddles the end_round, then we only include it if
    // straddle_end_round is true.
    else if (straddle_end_round && first_round <= end_round &&
             last_round >= end_round)
      columns_in_range.push_back(c);
  }

  // Traverse columns_in_range to find the first and last columns that were
  // included.
  uint32_t first_column = std::numeric_limits<uint32_t>::max();
  uint32_t last_column = std::numeric_limits<uint32_t>::min();
  for (auto c : columns_in_range) {
    first_column = std::min(first_column, c);
    last_column = std::max(last_column, c);
  }

  return std::make_tuple(reorder_pcm_columns(pcm, columns_in_range,
                                             first_row_to_keep,
                                             last_row_to_keep),
                         first_column, last_column);
}

/// @brief Generate a random PCM with the given parameters.
/// @param n_rounds The number of rounds in the PCM.
/// @param n_errs_per_round The number of errors per round in the PCM.
/// @param n_syndromes_per_round The number of syndromes per round in the PCM.
/// @param weight The column weight of the PCM.
/// @param rng The random number generator to use (e.g.
/// std::mt19937_64(your_seed))
/// @return A random PCM with the given parameters.
cudaqx::tensor<uint8_t> generate_random_pcm(std::size_t n_rounds,
                                            std::size_t n_errs_per_round,
                                            std::size_t n_syndromes_per_round,
                                            int weight, std::mt19937_64 &&rng) {
  std::size_t n_cols = n_rounds * n_errs_per_round;
  std::size_t n_rows = n_rounds * n_syndromes_per_round;
  cudaqx::tensor<uint8_t> pcm(std::vector<std::size_t>{n_rows, n_cols});

  // Generate a random bit (either a 0 or 1) for each element of the PCM.
  std::uniform_int_distribution<> dis(0, 1);

  for (std::size_t r = 0; r < n_rounds; ++r) {
    for (std::size_t c = 0; c < n_errs_per_round; ++c) {
      auto c_ix = r * n_errs_per_round + c;
      // Randomly decide if this column has all errors appear within this round
      // or if they should also appear in the next round too.
      bool all_errors_in_this_round = dis(rng) ? true : false;
      if (r == n_rounds - 1)
        all_errors_in_this_round = true;
      std::size_t row_max = all_errors_in_this_round
                                ? n_syndromes_per_round
                                : 2 * n_syndromes_per_round;
      std::uniform_int_distribution<> row_dis(0, row_max - 1);
      for (std::size_t i = 0; i < weight; ++i) {
        auto row_ix = row_dis(rng);
        // Loop until we find a row that has not been set yet
        while (pcm.at({r * n_syndromes_per_round + row_ix, c_ix}) == 1)
          row_ix = row_dis(rng);
        pcm.at({r * n_syndromes_per_round + row_ix, c_ix}) = 1;
      }
    }
  }

  return pcm;
}

/// @brief Randomly permute the columns of a PCM.
/// @param pcm The PCM to permute.
/// @return A new PCM with the columns permuted randomly.
cudaqx::tensor<uint8_t> shuffle_pcm_columns(const cudaqx::tensor<uint8_t> &pcm,
                                            std::mt19937_64 &&rng) {
  std::vector<std::uint32_t> column_order(pcm.shape()[1]);
  std::iota(column_order.begin(), column_order.end(), 0);
  std::shuffle(column_order.begin(), column_order.end(), rng);
  return reorder_pcm_columns(pcm, column_order);
}

/// @brief Extend a PCM to the given number of rounds.
/// @param pcm The PCM to extend.
/// @param num_syndromes_per_round The number of syndromes per round.
/// @param n_rounds The number of rounds to extend the PCM to.
/// @return A pair of the new PCM and the list of column indices from the
/// original PCM that were used to form the new PCM.
std::pair<cudaqx::tensor<uint8_t>, std::vector<std::uint32_t>>
pcm_extend_to_n_rounds(const cudaqx::tensor<uint8_t> &pcm,
                       std::size_t num_syndromes_per_round,
                       std::uint32_t n_rounds) {
  // Current number of rounds
  auto orig_num_rounds = pcm.shape()[0] / num_syndromes_per_round;

  if (orig_num_rounds > n_rounds) {
    throw std::invalid_argument(
        "extend_pcm_to_n_rounds: original number of "
        "rounds must be less than or equal to n_rounds");
  }

  // Verify the PCM is already sorted.
  if (!pcm_is_sorted(pcm, num_syndromes_per_round)) {
    throw std::invalid_argument(
        "extend_pcm_to_n_rounds: input PCM is not sorted");
  }

  // Find out the number of non-zero columns in the PCM.
  auto num_orig_non_zero_cols = [&]() -> std::uint32_t {
    auto [sub_pcm, first_column, last_column] = get_pcm_for_rounds(
        pcm, num_syndromes_per_round, 0, orig_num_rounds - 1, true, true);
    return static_cast<std::uint32_t>(sub_pcm.shape()[1]);
  }();

  // Traverse the PCM, fetching one round at a time. Save the first and
  // last time the PCM matches the previous round. Once it mismatches after it
  // had started matching, then exit the loop.
  auto [prior_sub_pcm, prior_first_column, prior_last_column] =
      get_pcm_for_rounds(pcm, num_syndromes_per_round, 0, 0, true, true);
  std::uint32_t first_match = std::numeric_limits<std::uint32_t>::max();
  std::uint32_t last_match = std::numeric_limits<std::uint32_t>::min();
  std::vector<std::uint32_t> first_columns;
  std::vector<std::uint32_t> last_columns;
  first_columns.push_back(prior_first_column);
  last_columns.push_back(prior_last_column);
  for (std::uint32_t w = 1; w < orig_num_rounds; w++) {
    auto [sub_pcm, first_column, last_column] =
        get_pcm_for_rounds(pcm, num_syndromes_per_round, w, w, true, true);
    first_columns.push_back(first_column);
    last_columns.push_back(last_column);
    if (sub_pcm.shape()[0] == prior_sub_pcm.shape()[0] &&
        sub_pcm.shape()[1] == prior_sub_pcm.shape()[1] &&
        memcmp(sub_pcm.data(), prior_sub_pcm.data(),
               sub_pcm.size() * sizeof(uint8_t)) == 0) {
      // The PCM matches the previous round, so we start (or continue) tracking
      // first_match and last_match.
      first_match = std::min(first_match, w);
      last_match = std::max(last_match, w);
    } else if (first_match != std::numeric_limits<std::uint32_t>::max()) {
      // We've found a mismatch after we started matching, so we can stop here.
      break;
    }
    prior_sub_pcm = std::move(sub_pcm);
  }

  if (first_match == std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument(
        "extend_pcm_to_n_rounds: PCM round analysis determined that "
        "consecutive rounds never matched each other, so we don't know how to "
        "insert more rounds");
  }

  // prior_sub_pcm contains the sub-PCM that we are going to insert
  // (potentially multiple times).
  std::uint32_t num_overlap_cols_per_round =
      last_columns[first_match - 1] + 1 - first_columns[first_match];
  std::uint32_t num_cols_per_inserted_round =
      prior_sub_pcm.shape()[1] - num_overlap_cols_per_round;

  // Calculate the size of the new PCM.
  std::uint32_t num_added_rounds = n_rounds - orig_num_rounds;
  std::uint32_t num_added_cols = num_cols_per_inserted_round * num_added_rounds;
  std::uint32_t num_added_rows = num_syndromes_per_round * num_added_rounds;

  // Now we insert the new rounds into the PCM.

  auto copy_sub_pcm_into_big_pcm = [](const cudaqx::tensor<uint8_t> &sub_pcm,
                                      cudaqx::tensor<uint8_t> &big_pcm,
                                      std::uint32_t starting_row,
                                      std::uint32_t starting_col) {
    assert(starting_row + sub_pcm.shape()[0] <= big_pcm.shape()[0]);
    assert(starting_col + sub_pcm.shape()[1] <= big_pcm.shape()[1]);
    for (std::uint32_t r = 0; r < sub_pcm.shape()[0]; r++) {
      auto *sub_pcm_row = &sub_pcm.at({r, 0});
      auto *big_pcm_row = &big_pcm.at({starting_row + r, starting_col});
      std::memcpy(big_pcm_row, sub_pcm_row,
                  sub_pcm.shape()[1] * sizeof(uint8_t));
    }
  };

  auto new_pcm_size = std::vector<std::size_t>{
      pcm.shape()[0] + num_added_rows, num_orig_non_zero_cols + num_added_cols};
  cudaqx::tensor<uint8_t> new_pcm(new_pcm_size);
  std::vector<std::uint32_t> column_list;
  column_list.reserve(new_pcm_size[1]);
  std::uint32_t num_rows_populated = 0;
  std::uint32_t num_cols_populated = 0;
  // Copy the beginning.
  {
    auto [sub_pcm, first_column, last_column] = get_pcm_for_rounds(
        pcm, num_syndromes_per_round, 0, last_match - 1, true, true);
    copy_sub_pcm_into_big_pcm(sub_pcm, new_pcm, 0, 0);
    for (std::uint32_t c = 0; c <= last_column; c++)
      column_list.push_back(c);
    num_rows_populated = sub_pcm.shape()[0];
    num_cols_populated = sub_pcm.shape()[1];
  }
  // Do the extension.
  {
    auto [sub_pcm, first_column, last_column] = get_pcm_for_rounds(
        pcm, num_syndromes_per_round, first_match, first_match, true, true);
    for (std::uint32_t w = 0; w < num_added_rounds; w++) {
      copy_sub_pcm_into_big_pcm(sub_pcm, new_pcm, num_rows_populated,
                                num_cols_populated -
                                    num_overlap_cols_per_round);
      for (std::uint32_t c = first_column + num_overlap_cols_per_round;
           c <= last_column; c++)
        column_list.push_back(c);
      num_rows_populated += num_syndromes_per_round;
      num_cols_populated += num_cols_per_inserted_round;
    }
  }
  // Copy the end.
  {
    auto [sub_pcm, first_column, last_column] =
        get_pcm_for_rounds(pcm, num_syndromes_per_round, last_match,
                           orig_num_rounds - 1, true, true);
    std::uint32_t col_insert_start =
        new_pcm.shape()[1] - (pcm.shape()[1] - first_column);
    std::uint32_t first_orig_col = first_column + num_overlap_cols_per_round;
    copy_sub_pcm_into_big_pcm(sub_pcm, new_pcm, num_rows_populated,
                              col_insert_start);
    for (std::uint32_t c = first_orig_col; c <= last_column; c++)
      column_list.push_back(c);
    num_cols_populated += last_column - first_orig_col + 1;
    num_rows_populated += num_syndromes_per_round;
  }

  if (column_list.size() != new_pcm.shape()[1]) {
    throw std::runtime_error(
        "extend_pcm_to_n_rounds: column_list.size() [value: " +
        std::to_string(column_list.size()) +
        "] != new_pcm.shape()[1] [value: " +
        std::to_string(new_pcm.shape()[1]) + "]");
  }

  return std::make_pair(new_pcm, column_list);
}

std::string pcm_to_sparse_string(const cudaqx::tensor<uint8_t> &pcm) {
  // Output the string like:
  // 2,5,7,-1,2,5,32,-1,...
  std::stringstream ss;
  for (std::size_t r = 0; r < pcm.shape()[0]; r++) {
    const uint8_t *row = &pcm.at({r, 0});
    for (std::size_t c = 0; c < pcm.shape()[1]; c++) {
      if (row[c] == 1) {
        ss << c << ",";
      }
    }
    ss << "-1,";
  }
  std::string result = ss.str();
  result.pop_back(); // trim the final comma
  return result;
}

std::vector<std::int64_t>
pcm_to_sparse_vec(const cudaqx::tensor<uint8_t> &pcm) {
  std::vector<std::int64_t> sparse_vec;
  for (std::size_t r = 0; r < pcm.shape()[0]; r++) {
    for (std::size_t c = 0; c < pcm.shape()[1]; c++) {
      if (pcm.at({r, c}) == 1) {
        sparse_vec.push_back(static_cast<std::int64_t>(c));
      }
    }
    sparse_vec.push_back(-1);
  }
  return sparse_vec;
}

cudaqx::tensor<uint8_t> pcm_from_sparse_string(const std::string &sparse_str,
                                               std::size_t num_rows,
                                               std::size_t num_cols) {
  cudaqx::tensor<uint8_t> pcm(std::vector<std::size_t>{num_rows, num_cols});
  std::stringstream ss(sparse_str);
  std::string item;
  std::uint32_t row = 0;
  while (std::getline(ss, item, ',')) {
    if (item == "-1") {
      row++;
      continue;
    }
    std::uint32_t col = std::stoul(item);
    pcm.at({row, col}) = 1;
  }
  return pcm;
}

cudaqx::tensor<uint8_t>
pcm_from_sparse_vec(const std::vector<std::int64_t> &sparse_vec,
                    std::size_t num_rows, std::size_t num_cols) {
  cudaqx::tensor<uint8_t> pcm(std::vector<std::size_t>{num_rows, num_cols});
  std::uint64_t row = 0;
  for (std::int64_t col : sparse_vec) {
    if (col < 0) {
      row++;
      continue;
    }
    pcm.at({row, static_cast<uint64_t>(col)}) = 1;
  }
  return pcm;
}

std::vector<std::int64_t>
generate_timelike_sparse_detector_matrix(std::uint32_t num_syndromes_per_round,
                                         std::uint32_t num_rounds,
                                         bool include_first_round) {
  std::vector<std::int64_t> detector_matrix;
  if (include_first_round) {
    for (std::uint32_t i = 0; i < num_syndromes_per_round; i++) {
      detector_matrix.push_back(i);
      detector_matrix.push_back(-1);
    }
  }
  // Every round after this is a XOR of the prior round's syndrome with the
  // current round's syndrome.
  for (std::uint32_t i = 1; i < num_rounds; i++) {
    for (std::uint32_t j = 0; j < num_syndromes_per_round; j++) {
      detector_matrix.push_back((i - 1) * num_syndromes_per_round + j);
      detector_matrix.push_back(i * num_syndromes_per_round + j);
      detector_matrix.push_back(-1);
    }
  }

  return detector_matrix;
}

std::vector<std::int64_t> generate_timelike_sparse_detector_matrix(
    std::uint32_t num_syndromes_per_round, std::uint32_t num_rounds,
    std::vector<std::int64_t> first_round_matrix) {
  if (first_round_matrix.size() == 0) {
    throw std::invalid_argument("generate_timelike_sparse_detector_matrix: "
                                "first_round_matrix must be non-empty");
  }

  for (std::uint32_t i = 0; i < first_round_matrix.size(); i++) {
    bool index_parity = (i % 2 == 0);
    // even elements should be >= 0
    if (index_parity && (first_round_matrix[i] < 0)) {
      throw std::invalid_argument(
          "generate_timelike_sparse_detector_matrix: first_round_matrix should "
          "have one index per row (row end indicated by -1)");
    }
    // odd elements should be -1
    if (!index_parity && (first_round_matrix[i] != -1)) {
      throw std::invalid_argument(
          "generate_timelike_sparse_detector_matrix: first_round_matrix should "
          "have one index per row (row end indicated by -1)");
    }
  }

  std::vector<std::int64_t> detector_matrix(first_round_matrix);

  // Every round after this is a XOR of the prior round's syndrome with the
  // current round's syndrome.
  for (std::uint32_t i = 1; i < num_rounds; i++) {
    for (std::uint32_t j = 0; j < num_syndromes_per_round; j++) {
      detector_matrix.push_back((i - 1) * num_syndromes_per_round + j);
      detector_matrix.push_back(i * num_syndromes_per_round + j);
      detector_matrix.push_back(-1);
    }
  }

  return detector_matrix;
}
} // namespace cudaq::qec
