/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/qec/device/memory_circuit.h"
#include "cudaq/qec/realtime/decoding.h"

// Realtime family of the memory-circuit kernels: identical to the plain
// cudaq::qec family (same shared bodies, see memory_circuit_impl.h) except
// that the decoding helpers forward to the realtime decoding API, so a
// decoder id >= 0 streams each round's stabilizer measurements and the final
// data readout to that decoder. Using these kernels is the realtime opt-in:
// the binary must link exactly one decoding shim (realtime/simulation,
// realtime/simulation-cqr, or realtime/quantinuum); linking none is a link
// error. No compiler flags are involved -- plain applications simply use
// cudaq::qec::memory_circuit* from memory_circuit.h instead.

namespace cudaq::qec::realtime {

__qpu__ void memory_circuit_hw(cudaq::qview<> data, cudaq::qview<> xstab_anc,
                               cudaq::qview<> zstab_anc,
                               const code::stabilizer_round &stabilizer_round,
                               const code::one_qubit_encoding &statePrep,
                               std::size_t numRounds,
                               const std::vector<std::size_t> &x_stabilizers,
                               const std::vector<std::size_t> &z_stabilizers,
                               const std::vector<std::size_t> &obs_matrix_flat,
                               std::size_t num_observables,
                               bool measure_in_x_basis,
                               std::int64_t enqueue_decoder_id = -1);

__qpu__ void memory_circuit(const code::stabilizer_round &stabilizer_round,
                            const code::one_qubit_encoding &statePrep,
                            std::size_t numData, std::size_t numAncx,
                            std::size_t numAncz, std::size_t numRounds,
                            const std::vector<std::size_t> &x_stabilizers,
                            const std::vector<std::size_t> &z_stabilizers,
                            const std::vector<std::size_t> &obs_matrix_flat,
                            std::size_t num_observables,
                            bool measure_in_x_basis,
                            std::int64_t enqueue_decoder_id = -1);

__qpu__ void memory_circuit_multi(
    cudaq::qview<> data, cudaq::qview<> xstab_anc, cudaq::qview<> zstab_anc,
    const code::stabilizer_round &stabilizer_round,
    const code::one_qubit_encoding &statePrep, std::size_t numData,
    std::size_t numAncx, std::size_t numAncz, std::size_t numRounds,
    const std::vector<std::size_t> &x_stabilizers,
    const std::vector<std::size_t> &z_stabilizers,
    const std::vector<std::size_t> &obs_matrix_flat,
    std::size_t num_observables, bool measure_in_x_basis,
    std::size_t numLogical,
    const std::vector<std::int64_t> &enqueue_decoder_ids = {});

} // namespace cudaq::qec::realtime

#if !defined(CUDAQ_QEC_MEMORY_CIRCUIT_DECLARATIONS_ONLY)
/// \cond INTERNAL

namespace cudaq::qec::realtime {

// Forwarding decoding helpers: shadow cudaq::qec::detail's no-ops for this
// family's instantiation of the shared kernel bodies.
namespace detail {
__qpu__ void
enqueue_syndromes_if(std::int64_t decoder_id,
                     const std::vector<cudaq::measure_result> &syndromes) {
  if (decoder_id >= 0)
    cudaq::qec::decoding::enqueue_syndromes(
        static_cast<std::uint64_t>(decoder_id), syndromes);
}
__qpu__ void reset_decoder_if(std::int64_t decoder_id) {
  if (decoder_id >= 0)
    cudaq::qec::decoding::reset_decoder(static_cast<std::uint64_t>(decoder_id));
}
} // namespace detail

#include "cudaq/qec/device/memory_circuit_impl.h"

// Realtime public kernel: forwards to this family's shared body, which
// streams to decoder `enqueue_decoder_id` when it is >= 0.
__qpu__ void memory_circuit_hw(cudaq::qview<> data, cudaq::qview<> xstab_anc,
                               cudaq::qview<> zstab_anc,
                               const code::stabilizer_round &stabilizer_round,
                               const code::one_qubit_encoding &statePrep,
                               std::size_t num_rounds,
                               const std::vector<std::size_t> &x_stabilizers,
                               const std::vector<std::size_t> &z_stabilizers,
                               const std::vector<std::size_t> &obs_matrix_flat,
                               std::size_t num_observables,
                               bool measure_in_x_basis,
                               std::int64_t enqueue_decoder_id) {
  cudaq::qec::realtime::memory_circuit_hw_impl(
      data, xstab_anc, zstab_anc, stabilizer_round, statePrep, num_rounds,
      x_stabilizers, z_stabilizers, obs_matrix_flat, num_observables,
      measure_in_x_basis, enqueue_decoder_id);
}

// Host-launchable entry point (realtime family): allocates the registers and
// forwards to the realtime memory_circuit_hw.
__qpu__ void memory_circuit(const code::stabilizer_round &stabilizer_round,
                            const code::one_qubit_encoding &statePrep,
                            std::size_t numData, std::size_t numAncx,
                            std::size_t numAncz, std::size_t numRounds,
                            const std::vector<std::size_t> &x_stabilizers,
                            const std::vector<std::size_t> &z_stabilizers,
                            const std::vector<std::size_t> &obs_matrix_flat,
                            std::size_t num_observables,
                            bool measure_in_x_basis,
                            std::int64_t enqueue_decoder_id) {
  cudaq::qvector data(numData), xstab_anc(numAncx), zstab_anc(numAncz);
  cudaq::qec::realtime::memory_circuit_hw(
      data, xstab_anc, zstab_anc, stabilizer_round, statePrep, numRounds,
      x_stabilizers, z_stabilizers, obs_matrix_flat, num_observables,
      measure_in_x_basis, enqueue_decoder_id);
}

// Multi-logical-qubit wrapper (realtime family): an independent memory
// experiment on each of `numLogical` logical qubits; logical qubit `i` resets
// and enqueues to decoder `enqueue_decoder_ids[i]`.
__qpu__ void memory_circuit_multi(
    cudaq::qview<> data, cudaq::qview<> xstab_anc, cudaq::qview<> zstab_anc,
    const code::stabilizer_round &stabilizer_round,
    const code::one_qubit_encoding &statePrep, std::size_t numData,
    std::size_t numAncx, std::size_t numAncz, std::size_t num_rounds,
    const std::vector<std::size_t> &x_stabilizers,
    const std::vector<std::size_t> &z_stabilizers,
    const std::vector<std::size_t> &obs_matrix_flat,
    std::size_t num_observables, bool measure_in_x_basis,
    std::size_t numLogical,
    const std::vector<std::int64_t> &enqueue_decoder_ids) {
  bool enqueue = enqueue_decoder_ids.size() == numLogical;
  for (std::size_t i = 0; i < numLogical; ++i) {
    std::int64_t enqueue_decoder_id = -1;
    if (enqueue)
      enqueue_decoder_id = enqueue_decoder_ids[i];

    detail::reset_decoder_if(enqueue_decoder_id);

    cudaq::qec::realtime::memory_circuit_hw(
        data.slice(i * numData, numData), xstab_anc.slice(i * numAncx, numAncx),
        zstab_anc.slice(i * numAncz, numAncz), stabilizer_round, statePrep,
        num_rounds, x_stabilizers, z_stabilizers, obs_matrix_flat,
        num_observables, measure_in_x_basis, enqueue_decoder_id);
  }
}

} // namespace cudaq::qec::realtime
/// \endcond
#endif // CUDAQ_QEC_MEMORY_CIRCUIT_DECLARATIONS_ONLY
