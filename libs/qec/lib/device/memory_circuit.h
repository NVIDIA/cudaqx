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
/// @brief Execute a memory circuit for quantum error correction, mz on data
/// qubits
/// @param stabilizer_round Function pointer to the stabilizer round
/// implementation
/// @param statePrep Function pointer to the state preparation implementation
/// @param numData Number of data qubits in the code
/// @param numAncx Number of ancilla x qubits in the code
/// @param numAncz Number of ancilla z qubits in the code
/// @param numRounds Number of rounds to execute the memory circuit
/// @param x_stabilizers Vector of indices for X stabilizers
/// @param z_stabilizers Vector of indices for Z stabilizers
__qpu__ void memory_circuit_mz(const code::stabilizer_round &stabilizer_round,
                               const code::one_qubit_encoding &statePrep,
                               std::size_t numData, std::size_t numAncx,
                               std::size_t numAncz, std::size_t numRounds,
                               const std::vector<std::size_t> &x_stabilizers,
                               const std::vector<std::size_t> &z_stabilizers);
/// \entry_point_kernel
///
/// @brief Execute a memory circuit for quantum error correction, mx on data
/// qubits
/// @param stabilizer_round Function pointer to the stabilizer round
/// implementation
/// @param statePrep Function pointer to the state preparation implementation
/// @param numData Number of data qubits in the code
/// @param numAncx Number of ancilla x qubits in the code
/// @param numAncz Number of ancilla z qubits in the code
/// @param numRounds Number of rounds to execute the memory circuit
/// @param x_stabilizers Vector of indices for X stabilizers
/// @param z_stabilizers Vector of indices for Z stabilizers
__qpu__ void memory_circuit_mx(const code::stabilizer_round &stabilizer_round,
                               const code::one_qubit_encoding &statePrep,
                               std::size_t numData, std::size_t numAncx,
                               std::size_t numAncz, std::size_t numRounds,
                               const std::vector<std::size_t> &x_stabilizers,
                               const std::vector<std::size_t> &z_stabilizers);

} // namespace cudaq::qec
