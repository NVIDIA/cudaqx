/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq.h"
#include "cudaq/qec/code.h"

namespace cudaq::qec {

/// \pure_device_kernel
/// @brief QEC memory circuit on caller-allocated qubit views.
///
/// For realtime decoding see
/// cudaq::qec::realtime in cudaq/qec/device/memory_circuit_realtime.h.
__qpu__ void memory_circuit_hw(cudaq::qview<> data, cudaq::qview<> xstab_anc,
                               cudaq::qview<> zstab_anc,
                               const code::stabilizer_round &stabilizer_round,
                               const code::one_qubit_encoding &statePrep,
                               std::size_t numRounds,
                               const std::vector<std::size_t> &x_stabilizers,
                               const std::vector<std::size_t> &z_stabilizers,
                               const std::vector<std::size_t> &obs_matrix_flat,
                               std::size_t num_observables,
                               bool measure_in_x_basis);

/// \entry_point_kernel
/// @brief Host-launchable variant of memory_circuit_hw; allocates qubit
/// registers internally.
__qpu__ void memory_circuit(const code::stabilizer_round &stabilizer_round,
                            const code::one_qubit_encoding &statePrep,
                            std::size_t numData, std::size_t numAncx,
                            std::size_t numAncz, std::size_t numRounds,
                            const std::vector<std::size_t> &x_stabilizers,
                            const std::vector<std::size_t> &z_stabilizers,
                            const std::vector<std::size_t> &obs_matrix_flat,
                            std::size_t num_observables,
                            bool measure_in_x_basis);

/// \pure_device_kernel
/// @brief Runs memory_circuit_hw independently on each of numLogical logical
/// qubits, tiled across slices of the caller-allocated registers
/// (numLogical * numData qubits each).
__qpu__ void memory_circuit_multi(
    cudaq::qview<> data, cudaq::qview<> xstab_anc, cudaq::qview<> zstab_anc,
    const code::stabilizer_round &stabilizer_round,
    const code::one_qubit_encoding &statePrep, std::size_t numData,
    std::size_t numAncx, std::size_t numAncz, std::size_t numRounds,
    const std::vector<std::size_t> &x_stabilizers,
    const std::vector<std::size_t> &z_stabilizers,
    const std::vector<std::size_t> &obs_matrix_flat,
    std::size_t num_observables, bool measure_in_x_basis,
    std::size_t num_logical);

/// @brief Reshapes a flat numLogical*numData readout into per-logical slices.
template <typename T>
std::vector<std::vector<T>>
memory_circuit_multi_readout(const std::vector<T> &flat_readout,
                             std::size_t numLogical, std::size_t numData) {
  std::vector<std::vector<T>> per_logical(numLogical);
  for (std::size_t i = 0; i < numLogical; ++i)
    per_logical[i].assign(flat_readout.begin() + i * numData,
                          flat_readout.begin() + (i + 1) * numData);
  return per_logical;
}
} // namespace cudaq::qec

// Kernel implementations are inline so nvq++ translation units get definitions
// by including this header. CUDAQ_QEC_MEMORY_CIRCUIT_DECLARATIONS_ONLY
// suppresses them for host-only TUs
#if !defined(CUDAQ_QEC_MEMORY_CIRCUIT_DECLARATIONS_ONLY)
/// \cond INTERNAL

namespace cudaq::qec::detail {
__qpu__ void
enqueue_syndromes_if(std::int64_t decoder_id,
                     const std::vector<cudaq::measure_result> &syndromes) {}
__qpu__ void reset_decoder_if(std::int64_t decoder_id) {}
} // namespace cudaq::qec::detail

namespace cudaq::qec {
#include "cudaq/qec/device/memory_circuit_impl.h"

__qpu__ void memory_circuit_hw(cudaq::qview<> data, cudaq::qview<> xstab_anc,
                               cudaq::qview<> zstab_anc,
                               const code::stabilizer_round &stabilizer_round,
                               const code::one_qubit_encoding &statePrep,
                               std::size_t num_rounds,
                               const std::vector<std::size_t> &x_stabilizers,
                               const std::vector<std::size_t> &z_stabilizers,
                               const std::vector<std::size_t> &obs_matrix_flat,
                               std::size_t num_observables,
                               bool measure_in_x_basis) {
  cudaq::qec::memory_circuit_hw_impl(
      data, xstab_anc, zstab_anc, stabilizer_round, statePrep, num_rounds,
      x_stabilizers, z_stabilizers, obs_matrix_flat, num_observables,
      measure_in_x_basis, /*enqueue_decoder_id=*/-1);
}

__qpu__ void memory_circuit(const code::stabilizer_round &stabilizer_round,
                            const code::one_qubit_encoding &statePrep,
                            std::size_t numData, std::size_t numAncx,
                            std::size_t numAncz, std::size_t numRounds,
                            const std::vector<std::size_t> &x_stabilizers,
                            const std::vector<std::size_t> &z_stabilizers,
                            const std::vector<std::size_t> &obs_matrix_flat,
                            std::size_t num_observables,
                            bool measure_in_x_basis) {
  cudaq::qvector data(numData), xstab_anc(numAncx), zstab_anc(numAncz);
  cudaq::qec::memory_circuit_hw_impl(
      data, xstab_anc, zstab_anc, stabilizer_round, statePrep, numRounds,
      x_stabilizers, z_stabilizers, obs_matrix_flat, num_observables,
      measure_in_x_basis, /*enqueue_decoder_id=*/-1);
}

__qpu__ void memory_circuit_multi(
    cudaq::qview<> data, cudaq::qview<> xstab_anc, cudaq::qview<> zstab_anc,
    const code::stabilizer_round &stabilizer_round,
    const code::one_qubit_encoding &statePrep, std::size_t numData,
    std::size_t numAncx, std::size_t numAncz, std::size_t num_rounds,
    const std::vector<std::size_t> &x_stabilizers,
    const std::vector<std::size_t> &z_stabilizers,
    const std::vector<std::size_t> &obs_matrix_flat,
    std::size_t num_observables, bool measure_in_x_basis,
    std::size_t numLogical) {
  for (std::size_t i = 0; i < numLogical; ++i)
    cudaq::qec::memory_circuit_hw_impl(
        data.slice(i * numData, numData), xstab_anc.slice(i * numAncx, numAncx),
        zstab_anc.slice(i * numAncz, numAncz), stabilizer_round, statePrep,
        num_rounds, x_stabilizers, z_stabilizers, obs_matrix_flat,
        num_observables, measure_in_x_basis, /*enqueue_decoder_id=*/-1);
}
} // namespace cudaq::qec
/// \endcond
#endif // CUDAQ_QEC_MEMORY_CIRCUIT_DECLARATIONS_ONLY
