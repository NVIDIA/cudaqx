/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Shared memory_circuit_hw kernel body. Do NOT include this file directly:
// it carries no include guard and is textually included inside a namespace by
// cudaq/qec/device/memory_circuit.h (plain family, no-op decoding helpers) and
// cudaq/qec/device/memory_circuit_realtime.h (realtime family, forwarding
// decoding helpers). The including namespace must provide
// detail::enqueue_syndromes_if and detail::reset_decoder_if; everything else
// resolves to cudaq::qec via enclosing-namespace lookup. The thin
// memory_circuit / memory_circuit_multi forwarding wrappers are defined
// per-family in the including headers (with namespace-qualified forwarding
// calls, since unqualified calls are ambiguous under ADL when both families
// are visible).

__qpu__ void memory_circuit_hw_impl(
    cudaq::qview<> data, cudaq::qview<> xstab_anc, cudaq::qview<> zstab_anc,
    const code::stabilizer_round &stabilizer_round,
    const code::one_qubit_encoding &statePrep, std::size_t num_rounds,
    const std::vector<std::size_t> &x_stabilizers,
    const std::vector<std::size_t> &z_stabilizers,
    const std::vector<std::size_t> &obs_matrix_flat,
    std::size_t num_observables, bool measure_in_x_basis,
    std::int64_t enqueue_decoder_id) {
  std::size_t num_data = data.size();

  // Create the logical patch on the caller-allocated registers
  patch logical(data, xstab_anc, zstab_anc);

  // Prepare the initial state
  statePrep({data, xstab_anc, zstab_anc});

  // The "off-basis" detectors will be non-deterministic after the first
  // stabilizer round.
  auto final_syndrome = stabilizer_round(logical, x_stabilizers, z_stabilizers);
  {
    // Rebuild the syndrome in local scope before enqueueing: hardware QIR
    // profiles cannot lower a heap-materialized measurement vector, and the
    // optimizer only folds the enqueue's bit-packing into scalar ops when the
    // vector is constructed in the enqueueing scope.
    std::vector<cudaq::measure_result> syn(final_syndrome.size());
    for (std::size_t i = 0; i < final_syndrome.size(); ++i)
      syn[i] = final_syndrome[i];
    detail::enqueue_syndromes_if(enqueue_decoder_id, syn);
  }
  std::size_t num_fixed_measurements =
      measure_in_x_basis ? xstab_anc.size() : zstab_anc.size();
  std::size_t fixed_offset =
      measure_in_x_basis ? final_syndrome.size() - num_fixed_measurements : 0;
  for (std::size_t i = 0; i < num_fixed_measurements; ++i) {
    cudaq::detector(final_syndrome[fixed_offset + i]);
  }

  // Generate syndrome data
  for (std::size_t round = 1; round < num_rounds; ++round) {
    auto syndrome = stabilizer_round(logical, x_stabilizers, z_stabilizers);
    {
      std::vector<cudaq::measure_result> syn(syndrome.size());
      for (std::size_t i = 0; i < syndrome.size(); ++i)
        syn[i] = syndrome[i];
      detail::enqueue_syndromes_if(enqueue_decoder_id, syn);
    }
    cudaq::detectors(final_syndrome, syndrome);
    final_syndrome = syndrome;
  }

  if (measure_in_x_basis) {
    h(data);
  }
  auto data_results = mz(data);

  // Stream the raw data readout as the final "round" so the realtime decoder
  // can form the data-vs-stabilizer boundary detectors from it, matching the
  // detector layout emitted below (and by dem_from_memory_circuit).
  detail::enqueue_syndromes_if(enqueue_decoder_id, data_results);

  // Emit one logical_observable per row of the observable matrix.
  for (std::size_t obs = 0; obs < num_observables; ++obs) {
    std::size_t support_weight = 0;
    for (std::size_t q = 0; q < num_data; ++q) {
      if (obs_matrix_flat[obs * num_data + q] != 0)
        support_weight++;
    }
    std::vector<cudaq::measure_result> obs_support(support_weight);
    std::size_t idx = 0;
    for (std::size_t q = 0; q < num_data; ++q) {
      if (obs_matrix_flat[obs * num_data + q] != 0)
        obs_support[idx++] = data_results[q];
    }
    cudaq::logical_observable(obs_support);
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

    std::vector<cudaq::measure_result> support(support_weight + 1);
    support[0] = final_syndrome[fixed_offset + x];
    std::size_t support_idx = 1;
    for (std::size_t q = 0; q < num_data; ++q) {
      if (stabilizers[row_base + q] != 0) {
        support[support_idx++] = data_results[q];
      }
    }

    cudaq::detector(support);
  }
}
