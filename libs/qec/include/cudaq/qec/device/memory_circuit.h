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

/// \entry_point_kernel
///
/// @brief Execute a memory circuit for quantum error correction
/// @param stabilizer_round Function pointer to the stabilizer round
/// implementation
/// @param statePrep Function pointer to the state preparation implementation
/// @param numData Number of data qubits in the code
/// @param numAncx Number of ancilla x qubits in the code
/// @param numAncz Number of ancilla z qubits in the code
/// @param numRounds Number of rounds to execute the memory circuit
/// @param x_stabilizers Vector of indices for X stabilizers
/// @param z_stabilizers Vector of indices for Z stabilizers
/// @param obs_matrix_flat Row-major flattened logical observable matrix
///        (num_observables × numData entries, values 0/1).
/// @param num_observables Number of rows in the observable matrix (k).
/// @param measure_in_x_basis Performing X- or Z-memory circuit
/// @param enqueue_syndromes When true, enqueue each round's stabilizer
///        measurements to the realtime decoder identified by @p decoder_id.
///        Enqueueing requires the kernel to be compiled with
///        `-DCUDAQ_QEC_ENABLE_REALTIME_DECODING` and linked against a realtime
///        decoding shim (see cudaq/qec/realtime/decoding.h); in the default
///        (offline) build this argument is ignored.
/// @param decoder_id ID of the realtime decoder to enqueue syndromes to (only
///        used when @p enqueue_syndromes is true).
__qpu__ void memory_circuit(const code::stabilizer_round &stabilizer_round,
                            const code::one_qubit_encoding &statePrep,
                            std::size_t numData, std::size_t numAncx,
                            std::size_t numAncz, std::size_t numRounds,
                            const std::vector<std::size_t> &x_stabilizers,
                            const std::vector<std::size_t> &z_stabilizers,
                            const std::vector<std::size_t> &obs_matrix_flat,
                            std::size_t num_observables, bool measure_in_x_basis,
                            bool enqueue_syndromes = false,
                            std::uint64_t decoder_id = 0);
} // namespace cudaq::qec
