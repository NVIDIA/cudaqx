/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Shared memory_circuit_hw_impl kernel body. No include guard: textually
// included inside each family's namespace by memory_circuit.h (plain, no-op
// helpers) and memory_circuit_realtime.h (realtime, forwarding helpers). The
// including namespace must define detail::enqueue_syndromes_if and
// detail::reset_decoder_if. All other public wrappers use namespace-qualified
// calls to memory_circuit_hw_impl because ADL on code::stabilizer_round
// arguments would otherwise make unqualified calls ambiguous when both
// families are visible in the same TU.

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
  patch logical(data, xstab_anc, zstab_anc);
  statePrep({data, xstab_anc, zstab_anc});

  // First round: off-basis stabilizers are non-deterministic, so only emit
  // detectors for the in-basis half.
  auto final_syndrome = stabilizer_round(logical, x_stabilizers, z_stabilizers);
  detail::enqueue_syndromes_if(enqueue_decoder_id, final_syndrome);

  std::size_t num_fixed_measurements =
      measure_in_x_basis ? xstab_anc.size() : zstab_anc.size();
  std::size_t fixed_offset =
      measure_in_x_basis ? final_syndrome.size() - num_fixed_measurements : 0;
  for (std::size_t i = 0; i < num_fixed_measurements; ++i)
    cudaq::detector(final_syndrome[fixed_offset + i]);

  for (std::size_t round = 1; round < num_rounds; ++round) {
    auto syndrome = stabilizer_round(logical, x_stabilizers, z_stabilizers);
    detail::enqueue_syndromes_if(enqueue_decoder_id, syndrome);

    cudaq::detectors(final_syndrome, syndrome);
    final_syndrome = syndrome;
  }

  if (measure_in_x_basis)
    h(data);
  auto data_results = mz(data);

  // Stream data readout as the final syndrome round so the decoder can form
  // boundary detectors matching the layout emitted below.
  detail::enqueue_syndromes_if(enqueue_decoder_id, data_results);

  for (std::size_t obs = 0; obs < num_observables; ++obs) {
    std::size_t support_weight = 0;
    for (std::size_t q = 0; q < num_data; ++q)
      if (obs_matrix_flat[obs * num_data + q] != 0)
        support_weight++;
    std::vector<cudaq::measure_result> obs_support(support_weight);
    std::size_t idx = 0;
    for (std::size_t q = 0; q < num_data; ++q)
      if (obs_matrix_flat[obs * num_data + q] != 0)
        obs_support[idx++] = data_results[q];
    cudaq::logical_observable(obs_support);
  }

  const std::vector<size_t> &stabilizers =
      measure_in_x_basis ? x_stabilizers : z_stabilizers;
  for (std::size_t x = 0; x < num_fixed_measurements; ++x) {
    std::size_t row_base = x * num_data;
    std::size_t support_weight = 0;
    for (std::size_t q = 0; q < num_data; ++q)
      if (stabilizers[row_base + q] != 0)
        support_weight++;
    std::vector<cudaq::measure_result> support(support_weight + 1);
    support[0] = final_syndrome[fixed_offset + x];
    std::size_t support_idx = 1;
    for (std::size_t q = 0; q < num_data; ++q)
      if (stabilizers[row_base + q] != 0)
        support[support_idx++] = data_results[q];
    cudaq::detector(support);
  }
}
