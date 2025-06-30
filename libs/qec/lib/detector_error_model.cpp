/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/detector_error_model.h"
#include "common/Logger.h"
#include "cudaq/qec/pcm_utils.h"

namespace cudaq::qec {

std::size_t detector_error_model::num_detectors() const {
  auto shape = detector_error_matrix.shape();
  if (shape.size() == 2)
    return shape[0];
  return 0;
}

std::size_t detector_error_model::num_error_mechanisms() const {
  auto shape = detector_error_matrix.shape();
  if (shape.size() == 2)
    return shape[1];
  return 0;
}

std::size_t detector_error_model::num_observables() const {
  auto shape = observables_flips_matrix.shape();
  if (shape.size() == 2)
    return shape[0];
  return 0;
}

/// @brief Return a sparse representation of the PCM.
/// @param pcm The PCM to convert to a sparse representation.
/// @return A vector of vectors that sparsely represents the PCM. The size of
/// the outer vector is the number of columns in the PCM, and the i-th element
/// contains an inner vector of the row indices of the non-zero elements in the
/// i-th column of the PCM.
std::vector<std::vector<std::uint32_t>>
dense_to_sparse(const cudaqx::tensor<uint8_t> &pcm) {
  if (pcm.rank() != 2) {
    throw std::invalid_argument("dense_to_sparse: PCM must be a 2D tensor");
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

void detector_error_model::canonicalize_for_rounds(
    uint32_t num_syndromes_per_round) {
  auto row_indices = dense_to_sparse(detector_error_matrix);
  auto column_order =
      get_sorted_pcm_column_indices(row_indices, num_syndromes_per_round);
  std::vector<std::uint32_t> final_column_order;
  // March through the columns in topological order, and combine the probability
  // weight vectors if the columns have the same row indices.
  std::vector<std::vector<std::uint32_t>> new_row_indices;
  std::vector<double> new_weights;
  std::vector<std::size_t> new_error_ids;
  const std::size_t num_obs = this->num_observables();
  const auto num_cols = column_order.size();
  bool has_error_ids =
      error_ids.has_value() && error_ids->size() == error_rates.size();
  for (std::size_t c = 0; c < num_cols; c++) {
    auto column_index = column_order[c];
    auto &curr_row_indices = row_indices[column_index];
    // If the column has no non-zero elements, or a weight of 0, then we skip
    // it.
    if (curr_row_indices.size() == 0 || error_rates[column_index] == 0)
      continue;
    if (new_row_indices.empty()) {
      new_row_indices.push_back(curr_row_indices);
      new_weights.push_back(error_rates[column_index]);
      final_column_order.push_back(column_index);
      if (has_error_ids)
        new_error_ids.push_back(error_ids->at(column_index));
    } else {
      auto &prev_row_indices = new_row_indices.back();
      if (prev_row_indices == curr_row_indices) {
        // The current column has the same row indices as the previous column
        // (i.e. has the same syndrome signature) so we update the error_rates
        // and do NOT add the duplicate column.
        auto prev_weight = new_weights.back();
        auto prev_error_id = has_error_ids
                                 ? new_error_ids.back()
                                 : std::numeric_limits<std::size_t>::max();
        auto curr_weight = error_rates[column_index];
        bool same_error_id =
            has_error_ids && prev_error_id == error_ids->at(column_index);
        double scale_factor = same_error_id ? 0.0 : 1.0;
        // The new weight is the probability that exactly ONE of the two errors
        // occurs. This is given by the formula: P(A xor B) = P(A) + P(B) - 2 *
        // P(A and B). If the errors originate from the same error mechanism,
        // then P(A and B) = 0.
        auto new_weight = prev_weight + curr_weight -
                          scale_factor * 2.0 * prev_weight * curr_weight;
        new_weights.back() = new_weight;
        // Arbitrarily choose to keep the smaller error ID.
        if (has_error_ids)
          new_error_ids.back() =
              std::min(prev_error_id, error_ids->at(column_index));
        // Verify that the observables are the same for the duplicate column.
        auto previous_column = column_order[c - 1];
        bool match = true;
        for (std::size_t r = 0; r < num_obs; r++) {
          if (this->observables_flips_matrix.at({r, previous_column}) !=
              this->observables_flips_matrix.at({r, column_index})) {
            match = false;
            break;
          }
        }
        if (!match) {
          // printf("detector_error_matrix:\n");
          // this->detector_error_matrix.dump_bits();
          // printf("observables_flips_matrix:\n");
          // this->observables_flips_matrix.dump_bits();
          // fflush(stdout);
          cudaq::warn(
              "detector_error_model::canonicalize_for_rounds: identical "
              "syndromes exist in detector_error_matrix but have different "
              "observables in observables_flips_matrix (columns {} and {})",
              previous_column, column_index);
        }
      } else {
        // The current column has different row indices than the previous
        // column. So we add the current column to the new PCM, and update the
        // error_rates.
        new_row_indices.push_back(curr_row_indices);
        new_weights.push_back(error_rates[column_index]);
        final_column_order.push_back(column_index);
        if (has_error_ids)
          new_error_ids.push_back(error_ids->at(column_index));
      }
    }
  }

  std::swap(this->error_rates, new_weights);
  if (has_error_ids)
    std::swap(*this->error_ids, new_error_ids);

  // Create the reordered, reduced Detector Error Matrix.
  this->detector_error_matrix = cudaq::qec::reorder_pcm_columns(
      this->detector_error_matrix, final_column_order);

  // Create the reordered, reduced Observables Flips Matrix.
  this->observables_flips_matrix = cudaq::qec::reorder_pcm_columns(
      this->observables_flips_matrix, final_column_order);
}

} // namespace cudaq::qec
