/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
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
/// @param fb_indices CSR herald-column indices for the detector inlined
///        feedback layout. `fb_indices[fb_offsets[j] .. fb_offsets[j+1])`
///        lists the herald record columns XORed into record j's cross-round
///        detector (records of the earlier round) and into its final boundary
///        detector (records of the last round).
/// @param fb_offsets CSR row offsets for the detector inlined feedback layout.
///        Size is 0 (no feedback; legacy `cudaq::detectors` structure) or
///        numCols + 1 with numCols = numAncx + numAncz.
/// @param obs_fb_indices CSR record-column indices for the observable inlined
///        feedback layout. `obs_fb_indices[obs_fb_offsets[m] ..
///        obs_fb_offsets[m+1])` lists the record columns XORed into logical
///        observable m on every round.
/// @param obs_fb_offsets CSR row offsets for the observable inlined feedback
///        layout. Size is 0 (no feedback) or num_observables + 1.
__qpu__ void memory_circuit(const code::stabilizer_round &stabilizer_round,
                            const code::one_qubit_encoding &statePrep,
                            std::size_t numData, std::size_t numAncx,
                            std::size_t numAncz, std::size_t numRounds,
                            const std::vector<std::size_t> &x_stabilizers,
                            const std::vector<std::size_t> &z_stabilizers,
                            const std::vector<std::size_t> &obs_matrix_flat,
                            std::size_t num_observables,
                            bool measure_in_x_basis,
                            const std::vector<std::size_t> &fb_indices,
                            const std::vector<std::size_t> &fb_offsets,
                            const std::vector<std::size_t> &obs_fb_indices,
                            const std::vector<std::size_t> &obs_fb_offsets);
} // namespace cudaq::qec
