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
///
/// @brief Hardware-compatible memory circuit for quantum error correction,
/// operating on caller-allocated qubit registers.
///
/// The `_hw` variant takes pre-allocated views so that it can lower to
/// hardware QIR profiles: those profiles require compile-time-constant
/// allocation sizes, which callers get by allocating the registers in their
/// entry-point kernel (whose arguments `cudaq::run` specializes to constants).
/// (It runs equally well on simulators; `memory_circuit` is the host-launchable
/// simulator convenience that allocates internally and forwards here.)
///
/// The kernel measures the data qubits itself (the readout is needed for the
/// detectors/observables it emits and for the realtime enqueue). Callers that
/// need the readout re-measure their `data` register after this returns; the
/// qubits are already collapsed, so the re-measurement is deterministic and
/// consistent.
///
/// @param data Data-qubit register (one per data qubit of the code)
/// @param xstab_anc X-stabilizer ancilla register
/// @param zstab_anc Z-stabilizer ancilla register
/// @param stabilizer_round Function pointer to the stabilizer round
/// implementation
/// @param statePrep Function pointer to the state preparation implementation
/// @param numRounds Number of rounds to execute the memory circuit
/// @param x_stabilizers Vector of indices for X stabilizers
/// @param z_stabilizers Vector of indices for Z stabilizers
/// @param obs_matrix_flat Row-major flattened logical observable matrix
///        (num_observables × data.size() entries, values 0/1).
/// @param num_observables Number of rows in the observable matrix (k).
/// @param measure_in_x_basis Performing X- or Z-memory circuit
///
/// This is the PLAIN family: it never streams measurements to a realtime
/// decoder and its binary needs no decoding shim. The realtime family --
/// identical kernels that additionally take an enqueue decoder id -- lives in
/// cudaq::qec::realtime (cudaq/qec/device/memory_circuit_realtime.h).
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
///
/// @brief Host-launchable memory circuit: allocates the qubit registers itself
/// (`numData`/`numAncx`/`numAncz`) and forwards to `memory_circuit_hw`. This is
/// the kernel `sample_memory_circuit` and `dem_from_memory_circuit` launch. The
/// runtime-sized allocation restricts it to simulator targets; hardware-bound
/// callers use `memory_circuit_hw` from their own entry-point kernel.
/// Parameters otherwise match `memory_circuit_hw`.
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
///
/// @brief Multi-logical-qubit wrapper around `memory_circuit_hw`. Runs an
/// independent memory experiment on each of `numLogical` logical qubits, laid
/// out on slices of the caller-allocated registers: logical qubit `i` uses
/// `data[i*numData, (i+1)*numData)` and the matching ancilla slices, so the
/// registers must hold `numLogical * numData` (resp. `numLogical * numAncx`,
/// `numLogical * numAncz`) qubits. Parameters otherwise match
/// `memory_circuit_hw`.
///
/// Callers that need the readout re-measure their `data` register after this
/// returns (deterministic, as above); `memory_circuit_multi_readout` reshapes
/// that flat readout by logical qubit.
__qpu__ void memory_circuit_multi(
    cudaq::qview<> data, cudaq::qview<> xstab_anc, cudaq::qview<> zstab_anc,
    const code::stabilizer_round &stabilizer_round,
    const code::one_qubit_encoding &statePrep, std::size_t numData,
    std::size_t numAncx, std::size_t numAncz, std::size_t numRounds,
    const std::vector<std::size_t> &x_stabilizers,
    const std::vector<std::size_t> &z_stabilizers,
    const std::vector<std::size_t> &obs_matrix_flat,
    std::size_t num_observables, bool measure_in_x_basis,
    std::size_t numLogical);

/// @brief Host-side helper that reshapes a flat `numLogical * numData`
/// row-major readout (e.g. a caller's re-measurement of the `data` register
/// after `memory_circuit_multi`) into a per-logical-qubit `vector<vector<>>`,
/// so `result[i]` is logical qubit `i`'s `numData` values.
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

//===----------------------------------------------------------------------===//
// Kernel implementations
//
// The implementations are part of this header so that a translation unit
// compiled with nvq++ gets compiling kernels just by including
// cudaq/qec/device/memory_circuit.h. This is the PLAIN family: the decoding
// helpers below are unconditional no-ops, so the TU makes no reference to the
// realtime decoding API and its binary needs no decoding shim. The realtime
// family -- the same kernels instantiated in cudaq::qec::realtime with
// helpers that forward to the decoding API -- lives in
// cudaq/qec/device/memory_circuit_realtime.h; using it is the opt-in and
// requires linking a decoding shim. The shared kernel bodies live in
// memory_circuit_impl.h, included into each family's namespace.
//
// Host-only translation units must not compile kernel definitions: CMake
// consumers of the cudaq-qec target inherit
// CUDAQ_QEC_MEMORY_CIRCUIT_DECLARATIONS_ONLY automatically (a PUBLIC compile
// definition; nvq++ device-code TUs do not), and non-CMake host code defines
// it manually before including this header.
//===----------------------------------------------------------------------===//
#if !defined(CUDAQ_QEC_MEMORY_CIRCUIT_DECLARATIONS_ONLY)
/// \cond INTERNAL

// No-op decoding helpers for the plain family: enqueue_decoder_id is accepted
// (the signatures are shared with the realtime family) but ignored.
namespace cudaq::qec::detail {
__qpu__ void
enqueue_syndromes_if(std::int64_t decoder_id,
                     const std::vector<cudaq::measure_result> &syndromes) {}
__qpu__ void reset_decoder_if(std::int64_t decoder_id) {}
} // namespace cudaq::qec::detail

namespace cudaq::qec {
#include "cudaq/qec/device/memory_circuit_impl.h"

// Plain public kernel: forwards to the shared body with enqueueing disabled.
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

// Host-launchable entry point: allocates the registers and forwards to
// memory_circuit_hw. The runtime-sized allocation restricts it to simulator
// targets (sample_memory_circuit / dem_from_memory_circuit launch this one);
// hardware-bound callers allocate in their own entry-point kernel and use
// memory_circuit_hw directly.
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
  cudaq::qec::memory_circuit_hw(data, xstab_anc, zstab_anc, stabilizer_round,
                                statePrep, numRounds, x_stabilizers,
                                z_stabilizers, obs_matrix_flat, num_observables,
                                measure_in_x_basis);
}

// Multi-logical-qubit wrapper around `memory_circuit_hw`: an independent
// memory experiment on each of `numLogical` logical qubits, each on its own
// slice of the caller-allocated registers.
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
  for (std::size_t i = 0; i < numLogical; ++i) {
    cudaq::qec::memory_circuit_hw(
        data.slice(i * numData, numData), xstab_anc.slice(i * numAncx, numAncx),
        zstab_anc.slice(i * numAncz, numAncz), stabilizer_round, statePrep,
        num_rounds, x_stabilizers, z_stabilizers, obs_matrix_flat,
        num_observables, measure_in_x_basis);
  }
}
} // namespace cudaq::qec
/// \endcond
#endif // CUDAQ_QEC_MEMORY_CIRCUIT_DECLARATIONS_ONLY
