/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "memory_circuit.h"
#include <numeric>

namespace cudaq::qec {

__qpu__ void memory_circuit(const code::stabilizer_round &stabilizer_round,
                            const code::one_qubit_encoding &statePrep,
                            std::size_t num_data, std::size_t numAncx,
                            std::size_t numAncz, std::size_t num_rounds,
                            const std::vector<std::size_t> &x_stabilizers,
                            const std::vector<std::size_t> &z_stabilizers,
                            const std::vector<std::size_t> &obs_matrix_flat,
                            std::size_t num_observables,
                            bool measure_in_x_basis,
                            const std::vector<std::size_t> &fb_indices,
                            const std::vector<std::size_t> &fb_offsets,
                            const std::vector<std::size_t> &obs_fb_indices,
                            const std::vector<std::size_t> &obs_fb_offsets) {
  // Allocate the data and ancilla qubits
  cudaq::qvector data(num_data), xstab_anc(numAncx), zstab_anc(numAncz);

  // Create the logical patch
  patch logical(data, xstab_anc, zstab_anc);

  // Prepare the initial state
  statePrep({data, xstab_anc, zstab_anc});

  // The "off-basis" detectors will be non-deterministic after the first
  // stabilizer round.
  auto final_syndrome = stabilizer_round(logical, x_stabilizers, z_stabilizers);
  std::size_t num_fixed_measurements =
      measure_in_x_basis ? xstab_anc.size() : zstab_anc.size();
  std::size_t fixed_offset =
      measure_in_x_basis ? final_syndrome.size() - num_fixed_measurements : 0;
  for (std::size_t i = 0; i < num_fixed_measurements; ++i) {
    cudaq::detector(final_syndrome[fixed_offset + i]);
  }

  std::size_t numCols = numAncx + numAncz;

  // Observable inlined-feedback accumulation. Nested std::vector is not
  // supported in __qpu__ code, so observable m's per-round feedback records
  // occupy the round-major slice starting at num_rounds * obs_fb_offsets[m]
  // of a single flat buffer; the round-r block of observable m starts at
  // num_rounds * obs_fb_offsets[m] + r * row_weight(m) with
  // row_weight(m) = obs_fb_offsets[m + 1] - obs_fb_offsets[m].
  bool has_obs_fb = obs_fb_offsets.size() > 0;
  std::size_t obs_fb_total =
      has_obs_fb ? num_rounds * obs_fb_offsets[num_observables] : 0;
  std::vector<cudaq::measure_result> obs_fb_records(obs_fb_total);

  // Collect the first-round feedback records for each observable.
  if (has_obs_fb) {
    for (std::size_t m = 0; m < num_observables; ++m) {
      std::size_t idx = num_rounds * obs_fb_offsets[m];
      for (std::size_t i = obs_fb_offsets[m]; i < obs_fb_offsets[m + 1]; ++i)
        obs_fb_records[idx++] = final_syndrome[obs_fb_indices[i]];
    }
  }

  // Generate syndrome data
  for (std::size_t round = 1; round < num_rounds; ++round) {
    auto syndrome = stabilizer_round(logical, x_stabilizers, z_stabilizers);
    if (has_obs_fb) {
      for (std::size_t m = 0; m < num_observables; ++m) {
        std::size_t row_weight = obs_fb_offsets[m + 1] - obs_fb_offsets[m];
        std::size_t idx = num_rounds * obs_fb_offsets[m] + round * row_weight;
        for (std::size_t i = obs_fb_offsets[m]; i < obs_fb_offsets[m + 1]; ++i)
          obs_fb_records[idx++] = syndrome[obs_fb_indices[i]];
      }
    }
    if (fb_offsets.size() == 0) {
      cudaq::detectors(final_syndrome, syndrome);
    } else {
      // Cross-round detector for record j: earlier vs current round record,
      // augmented with the earlier-round herald records in row j of the
      // feedback layout.
      for (std::size_t j = 0; j < numCols; ++j) {
        std::size_t begin = fb_offsets[j];
        std::size_t end = fb_offsets[j + 1];
        std::vector<cudaq::measure_result> det(2 + end - begin);
        det[0] = final_syndrome[j];
        det[1] = syndrome[j];
        for (std::size_t i = begin; i < end; ++i)
          det[2 + (i - begin)] = final_syndrome[fb_indices[i]];
        cudaq::detector(det);
      }
    }
    final_syndrome = syndrome;
  }

  if (measure_in_x_basis) {
    h(data);
  }
  auto data_results = mz(data);

  // Emit one logical_observable per row of the observable matrix.
  for (std::size_t obs = 0; obs < num_observables; ++obs) {
    std::size_t support_weight = 0;
    for (std::size_t q = 0; q < num_data; ++q) {
      if (obs_matrix_flat[obs * num_data + q] != 0)
        support_weight++;
    }
    if (!has_obs_fb) {
      std::vector<cudaq::measure_result> obs_support(support_weight);
      std::size_t idx = 0;
      for (std::size_t q = 0; q < num_data; ++q) {
        if (obs_matrix_flat[obs * num_data + q] != 0)
          obs_support[idx++] = data_results[q];
      }
      cudaq::logical_observable(obs_support);
    } else {
      // Feedback records from every round, then the data-qubit support.
      std::size_t row_weight = obs_fb_offsets[obs + 1] - obs_fb_offsets[obs];
      std::size_t fb_count = num_rounds * row_weight;
      std::size_t base = num_rounds * obs_fb_offsets[obs];
      std::vector<cudaq::measure_result> obs_support(fb_count + support_weight);
      for (std::size_t i = 0; i < fb_count; ++i)
        obs_support[i] = obs_fb_records[base + i];
      std::size_t idx = fb_count;
      for (std::size_t q = 0; q < num_data; ++q) {
        if (obs_matrix_flat[obs * num_data + q] != 0)
          obs_support[idx++] = data_results[q];
      }
      cudaq::logical_observable(obs_support);
    }
  }

  // For each stabilizer, form detectors from data qubit readout connected with
  // final stabilizer round.
  const std::vector<size_t> &stabilizers =
      measure_in_x_basis ? x_stabilizers : z_stabilizers;

  for (std::size_t x = 0; x < num_fixed_measurements; ++x) {
    std::size_t row_base = x * num_data;

    std::size_t support_weight = 0;
    for (std::size_t q = 0; q < num_data; ++q) {
      if (stabilizers[row_base + q] != 0) {
        support_weight++;
      }
    }

    // With inlined feedback, the boundary detector for record
    // (fixed_offset + x) is extended with its declared last-round records.
    std::size_t fb_begin = 0;
    std::size_t fb_end = 0;
    if (fb_offsets.size() > 0) {
      fb_begin = fb_offsets[fixed_offset + x];
      fb_end = fb_offsets[fixed_offset + x + 1];
    }

    std::vector<cudaq::measure_result> support(support_weight + 1 +
                                               (fb_end - fb_begin));
    support[0] = final_syndrome[fixed_offset + x];
    std::size_t support_idx = 1;
    for (std::size_t q = 0; q < num_data; ++q) {
      if (stabilizers[row_base + q] != 0) {
        support[support_idx++] = data_results[q];
      }
    }
    for (std::size_t i = fb_begin; i < fb_end; ++i)
      support[support_idx++] = final_syndrome[fb_indices[i]];

    cudaq::detector(support);
  }
}

} // namespace cudaq::qec
