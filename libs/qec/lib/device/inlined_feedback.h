/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq.h"

namespace cudaq::qec {

/// @brief Number of nonzero entries in row @p row of a row-major 0/1 matrix
/// that has @p num_cols columns (i.e. the weight of one feedback/observable/
/// stabilizer row).
inline __qpu__ std::size_t
row_support_weight(const std::vector<std::size_t> &matrix_flat, std::size_t row,
                   std::size_t num_cols) {
  std::size_t weight = 0;
  for (std::size_t k = 0; k < num_cols; ++k)
    if (matrix_flat[row * num_cols + k] != 0)
      weight++;
  return weight;
}

/// @brief Records for the cross-round detector of syndrome record @p j: the
/// earlier-vs-current comparison `{prev[j], curr[j]}` followed by the
/// earlier-round herald records selected by the CSR row for j.
inline __qpu__ std::vector<cudaq::measure_result>
cross_round_detector_records(const std::vector<cudaq::measure_result> &prev,
                             const std::vector<cudaq::measure_result> &curr,
                             std::size_t j,
                             const std::vector<std::size_t> &feedback_indices,
                             const std::vector<std::size_t> &feedback_offsets) {
  std::size_t weight = feedback_offsets[j + 1] - feedback_offsets[j];
  std::vector<cudaq::measure_result> det(2 + weight);
  det[0] = prev[j];
  det[1] = curr[j];
  std::size_t idx = 2;
  for (std::size_t i = feedback_offsets[j]; i < feedback_offsets[j + 1]; ++i)
    det[idx++] = prev[feedback_indices[i]];
  return det;
}

/// @brief Records for the boundary detector of syndrome record @p record_row:
/// `{last_syndrome[record_row]}`, then the data-qubit readouts
/// `data_results[q]` for each data qubit q (ascending) in the stabilizer
/// support `stabilizers[row_base + q]`, then the last-round herald records
/// selected by the CSR row for @p record_row. Empty feedback offsets mean the
/// herald part contributes nothing, so one helper serves both feedback and
/// legacy branches.
inline __qpu__ std::vector<cudaq::measure_result> boundary_detector_records(
    const std::vector<cudaq::measure_result> &last_syndrome,
    std::size_t record_row,
    const std::vector<cudaq::measure_result> &data_results,
    const std::vector<std::size_t> &stabilizers, std::size_t row_base,
    std::size_t num_data, const std::vector<std::size_t> &feedback_indices,
    const std::vector<std::size_t> &feedback_offsets) {
  std::size_t support_weight = 0;
  for (std::size_t q = 0; q < num_data; ++q)
    if (stabilizers[row_base + q] != 0)
      support_weight++;
  std::size_t fb_weight =
      feedback_offsets.size() > 0
          ? feedback_offsets[record_row + 1] - feedback_offsets[record_row]
          : 0;
  std::vector<cudaq::measure_result> support(1 + support_weight + fb_weight);
  support[0] = last_syndrome[record_row];
  std::size_t idx = 1;
  for (std::size_t q = 0; q < num_data; ++q)
    if (stabilizers[row_base + q] != 0)
      support[idx++] = data_results[q];
  if (feedback_offsets.size() > 0)
    for (std::size_t i = feedback_offsets[record_row];
         i < feedback_offsets[record_row + 1]; ++i)
      support[idx++] = last_syndrome[feedback_indices[i]];
  return support;
}

/// @brief Round-major slice offsets into the observable-feedback buffer. Entry
/// m is the start index of observable m's slice; each observable occupies
/// `num_rounds * (obs_feedback_offsets[m+1] - obs_feedback_offsets[m])`
/// records. The returned vector has `num_observables + 1` entries; the last
/// entry is the total buffer size. Returns an empty vector when the input
/// offsets are empty (no observable feedback declared).
inline __qpu__ std::vector<std::size_t> observable_feedback_record_offsets(
    const std::vector<std::size_t> &obs_feedback_offsets,
    std::size_t num_observables, std::size_t num_rounds) {
  // Sized (not default) construction: the __qpu__ dialect forbids the default
  // std::vector constructor. Empty CSR offsets yield a size-0 vector.
  std::size_t num_offsets =
      obs_feedback_offsets.size() > 0 ? num_observables + 1 : 0;
  std::vector<std::size_t> offsets(num_offsets);
  if (obs_feedback_offsets.size() == 0)
    return offsets;
  std::size_t total = 0;
  for (std::size_t m = 0; m < num_observables; ++m) {
    offsets[m] = total;
    total +=
        num_rounds * (obs_feedback_offsets[m + 1] - obs_feedback_offsets[m]);
  }
  offsets[num_observables] = total;
  return offsets;
}

/// @brief Write the records that observable feedback collects during @p round
/// into the round-major @p obs_fb_records buffer. For each observable m, the
/// round-r block starts at `record_offsets[m] + round * row_weight(m)` and
/// holds the syndrome records selected by CSR row m. Call with round 0 for the
/// first round's syndrome and with the running round index for each subsequent
/// round.
inline __qpu__ void collect_observable_feedback_round(
    std::vector<cudaq::measure_result> &obs_fb_records,
    const std::vector<std::size_t> &record_offsets,
    const std::vector<cudaq::measure_result> &syndrome, std::size_t round,
    const std::vector<std::size_t> &obs_feedback_indices,
    const std::vector<std::size_t> &obs_feedback_offsets,
    std::size_t num_observables) {
  for (std::size_t m = 0; m < num_observables; ++m) {
    std::size_t weight = obs_feedback_offsets[m + 1] - obs_feedback_offsets[m];
    std::size_t idx = record_offsets[m] + round * weight;
    for (std::size_t i = obs_feedback_offsets[m];
         i < obs_feedback_offsets[m + 1]; ++i)
      obs_fb_records[idx++] = syndrome[obs_feedback_indices[i]];
  }
}

/// @brief Support records for logical observable @p obs: the feedback records
/// collected in @p obs_fb_records (round-major, occupying
/// `[offsets[obs], offsets[obs + 1])`), followed by the data-qubit readouts
/// `data_results[q]` for each data qubit q (ascending) with
/// `obs_matrix_flat(obs, q) != 0`. When @p offsets is empty (no observable
/// feedback declared) the feedback prefix is empty and only the data support
/// is returned.
inline __qpu__ std::vector<cudaq::measure_result> observable_support_records(
    std::size_t obs, const std::vector<std::size_t> &obs_matrix_flat,
    std::size_t num_data,
    const std::vector<cudaq::measure_result> &data_results,
    const std::vector<cudaq::measure_result> &obs_fb_records,
    const std::vector<std::size_t> &offsets) {
  std::size_t support_weight =
      row_support_weight(obs_matrix_flat, obs, num_data);
  std::size_t fb_count = 0;
  std::size_t base = 0;
  if (offsets.size() > 0) {
    base = offsets[obs];
    fb_count = offsets[obs + 1] - offsets[obs];
  }
  std::vector<cudaq::measure_result> obs_support(fb_count + support_weight);
  for (std::size_t i = 0; i < fb_count; ++i)
    obs_support[i] = obs_fb_records[base + i];
  std::size_t idx = fb_count;
  for (std::size_t q = 0; q < num_data; ++q)
    if (obs_matrix_flat[obs * num_data + q] != 0)
      obs_support[idx++] = data_results[q];
  return obs_support;
}

} // namespace cudaq::qec
