/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "memory_circuit.h"
#include "inlined_feedback.h"

namespace cudaq::qec {

__qpu__ void
memory_circuit(const code::stabilizer_round &stabilizer_round,
               const code::one_qubit_encoding &statePrep, std::size_t num_data,
               std::size_t numAncx, std::size_t numAncz, std::size_t num_rounds,
               const std::vector<std::size_t> &x_stabilizers,
               const std::vector<std::size_t> &z_stabilizers,
               const std::vector<std::size_t> &obs_matrix_flat,
               std::size_t num_observables, bool measure_in_x_basis,
               const std::vector<std::size_t> &feedback_indices,
               const std::vector<std::size_t> &feedback_offsets,
               const std::vector<std::size_t> &obs_feedback_indices,
               const std::vector<std::size_t> &obs_feedback_offsets) {
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

  // Observable inlined feedback accumulation. Nested std::vector is not
  // supported in __qpu__ code, so instead of one record vector per observable
  // we use a single flat buffer with per-observable round-major slices; see
  // observable_feedback_record_offsets / collect_observable_feedback_round in
  // inlined_feedback.h.
  std::vector<std::size_t> obs_fb_record_offsets =
      observable_feedback_record_offsets(obs_feedback_offsets, num_observables,
                                         num_rounds);
  std::size_t obs_fb_total = obs_feedback_offsets.size() > 0
                                 ? obs_fb_record_offsets[num_observables]
                                 : 0;
  std::vector<cudaq::measure_result> obs_fb_records(obs_fb_total);

  // Collect the first-round feedback records for each observable.
  if (obs_feedback_offsets.size() > 0)
    collect_observable_feedback_round(obs_fb_records, obs_fb_record_offsets,
                                      final_syndrome, 0, obs_feedback_indices,
                                      obs_feedback_offsets, num_observables);

  // Generate syndrome data
  for (std::size_t round = 1; round < num_rounds; ++round) {
    auto syndrome = stabilizer_round(logical, x_stabilizers, z_stabilizers);
    if (obs_feedback_offsets.size() > 0)
      collect_observable_feedback_round(obs_fb_records, obs_fb_record_offsets,
                                        syndrome, round, obs_feedback_indices,
                                        obs_feedback_offsets, num_observables);
    if (feedback_offsets.size() == 0) {
      cudaq::detectors(final_syndrome, syndrome);
    } else {
      // Cross-round detector for record j: earlier vs current round record,
      // augmented with the earlier-round records declared in row j of the
      // feedback matrix.
      for (std::size_t j = 0; j < numCols; ++j)
        cudaq::detector(cross_round_detector_records(
            final_syndrome, syndrome, j, feedback_indices, feedback_offsets));
    }
    final_syndrome = syndrome;
  }

  if (measure_in_x_basis) {
    h(data);
  }
  auto data_results = mz(data);

  // Emit one logical_observable per row of the observable matrix: the per-round
  // feedback records followed by the data-qubit support.
  for (std::size_t obs = 0; obs < num_observables; ++obs)
    cudaq::logical_observable(
        observable_support_records(obs, obs_matrix_flat, num_data, data_results,
                                   obs_fb_records, obs_fb_record_offsets));

  // For each stabilizer, form detectors from data qubit readout connected with
  // final stabilizer round. With inlined feedback, the boundary detector for
  // record (fixed_offset + x) is extended with the declared last-round records.
  const std::vector<size_t> &stabilizers =
      measure_in_x_basis ? x_stabilizers : z_stabilizers;

  for (std::size_t x = 0; x < num_fixed_measurements; ++x)
    cudaq::detector(boundary_detector_records(
        final_syndrome, fixed_offset + x, data_results, stabilizers,
        x * num_data, num_data, feedback_indices, feedback_offsets));
}

} // namespace cudaq::qec
