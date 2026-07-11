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
/// @param enqueue_decoder_id When >= 0, enqueue each round's stabilizer
///        measurements followed by the final data-qubit readout (in
///        chronological measurement order) to the realtime decoder with this
///        ID, so the decoder forms the same detectors as
///        `dem_from_memory_circuit`. A negative value (the default) disables
///        enqueueing. Enqueueing requires linking a realtime decoding shim
///        (see cudaq/qec/realtime/decoding.h); in a binary without a shim the
///        decoding calls bind to libcudaq-qec's default no-op implementations,
///        so this argument is effectively ignored.
///
///        The same kernel can therefore be used both to generate the DEM (via
///        `dem_from_memory_circuit`) and to run the live experiment: the host
///        decoding entry points no-op while a circuit is being analyzed for its
///        DEM, so the enqueue is harmless during extraction.
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
                            bool measure_in_x_basis,
                            std::int64_t enqueue_decoder_id = -1);

/// \pure_device_kernel
///
/// @brief Multi-logical-qubit wrapper around `memory_circuit_hw`. Runs an
/// independent memory experiment on each of `numLogical` logical qubits, laid
/// out on slices of the caller-allocated registers: logical qubit `i` uses
/// `data[i*numData, (i+1)*numData)` and the matching ancilla slices, so the
/// registers must hold `numLogical * numData` (resp. `numLogical * numAncx`,
/// `numLogical * numAncz`) qubits. `enqueue_decoder_ids` selects the realtime
/// decoder for each logical qubit (need not be consecutive): logical qubit `i`
/// resets and enqueues to decoder `enqueue_decoder_ids[i]`. Pass an empty
/// vector to disable all realtime decoding. Parameters otherwise match
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
    std::size_t numLogical,
    const std::vector<std::int64_t> &enqueue_decoder_ids = {});

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
// cudaq/qec/device/memory_circuit.h (link a realtime decoding shim to stream
// syndromes; without one, the decoding calls bind to libcudaq-qec's default
// no-ops defined in lib/device/memory_circuit.cpp). Host-only translation
// units must not compile kernel definitions: CMake consumers of the cudaq-qec
// target inherit CUDAQ_QEC_MEMORY_CIRCUIT_DECLARATIONS_ONLY automatically (a
// PUBLIC compile definition; nvq++ device-code TUs do not), and non-CMake host
// code defines it manually before including this header. The copy shipped in libcudaq-qec is compiled with
// CUDAQ_QEC_DISABLE_REALTIME_DECODING (set in lib/device/CMakeLists.txt),
// which turns the decoder calls off so the library's registered copy is
// deterministically decoding-free.
//===----------------------------------------------------------------------===//
#if !defined(CUDAQ_QEC_MEMORY_CIRCUIT_DECLARATIONS_ONLY)
/// \cond INTERNAL

#if !defined(CUDAQ_QEC_DISABLE_REALTIME_DECODING)
#include "cudaq/qec/realtime/decoding.h"
#endif

// Helper kernels shared by the memory-circuit bodies: forward to the realtime
// decoding API when a decoder id is selected. Without a decoding shim linked,
// the calls bind to libcudaq-qec's default no-op implementations (see
// lib/device/memory_circuit.cpp). The cudaq-qec build itself compiles with
// CUDAQ_QEC_DISABLE_REALTIME_DECODING (set in lib/device/CMakeLists.txt) so
// the library's registered copy of the kernels is deterministically
// decoding-free; the copy an application compiles (by including
// memory_circuit.h) keeps the calls.
#if defined(CUDAQ_QEC_DISABLE_REALTIME_DECODING)
namespace cudaq::qec::detail {
__qpu__ void
enqueue_syndromes_if(std::int64_t decoder_id,
                     const std::vector<cudaq::measure_result> &syndromes) {}
__qpu__ void reset_decoder_if(std::int64_t decoder_id) {}
} // namespace cudaq::qec::detail
#else
namespace cudaq::qec::detail {
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
} // namespace cudaq::qec::detail
#endif

namespace cudaq::qec {

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
                            bool measure_in_x_basis,
                            std::int64_t enqueue_decoder_id) {
  cudaq::qvector data(numData), xstab_anc(numAncx), zstab_anc(numAncz);
  memory_circuit_hw(data, xstab_anc, zstab_anc, stabilizer_round, statePrep,
                    numRounds, x_stabilizers, z_stabilizers, obs_matrix_flat,
                    num_observables, measure_in_x_basis, enqueue_decoder_id);
}

// Multi-logical-qubit wrapper around `memory_circuit_hw`. Runs an independent
// memory experiment on each of `numLogical` logical qubits, each on its own
// slice of the caller-allocated registers (logical qubit `i` uses
// `data[i*numData, (i+1)*numData)` and the matching ancilla slices).
// `enqueue_decoder_ids` selects the realtime decoder for each logical qubit
// (need not be consecutive): logical qubit `i` resets and enqueues to decoder
// `enqueue_decoder_ids[i]`. Pass an empty vector to disable all realtime
// decoding (matching `memory_circuit`'s convention).
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

    memory_circuit_hw(
        data.slice(i * numData, numData), xstab_anc.slice(i * numAncx, numAncx),
        zstab_anc.slice(i * numAncz, numAncz), stabilizer_round, statePrep,
        num_rounds, x_stabilizers, z_stabilizers, obs_matrix_flat,
        num_observables, measure_in_x_basis, enqueue_decoder_id);
  }
}

} // namespace cudaq::qec
/// \endcond
#endif // CUDAQ_QEC_MEMORY_CIRCUIT_DECLARATIONS_ONLY
