/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "memory_circuit.h"

namespace cudaq::qec {

__qpu__ void __memory_circuit_stabs(
    cudaq::qview<> data, cudaq::qview<> xstab_anc, cudaq::qview<> zstab_anc,
    const code::stabilizer_round &stabilizer_round,
    const code::one_qubit_encoding &statePrep, std::size_t numRounds,
    const std::vector<std::size_t> &x_stabilizers,
    const std::vector<std::size_t> &z_stabilizers) {
  // Create the logical patch
  patch logical(data, xstab_anc, zstab_anc);

  // Prepare the initial state fault tolerantly
  statePrep({data, xstab_anc, zstab_anc});

  // Generate syndrome data
  for (std::size_t round = 0; round < numRounds; round++) {
    // Run the stabilizer round, generate the syndrome measurements
    auto syndrome = stabilizer_round(logical, x_stabilizers, z_stabilizers);
  }
}

__qpu__ void memory_circuit_mz(const code::stabilizer_round &stabilizer_round,
                               const code::one_qubit_encoding &statePrep,
                               std::size_t numData, std::size_t numAncx,
                               std::size_t numAncz, std::size_t numRounds,
                               const std::vector<std::size_t> &x_stabilizers,
                               const std::vector<std::size_t> &z_stabilizers) {

  // Allocate the data and ancilla qubits
  cudaq::qvector data(numData), xstab_anc(numAncx), zstab_anc(numAncz);

  // Persists ancilla measures
  __memory_circuit_stabs(data, xstab_anc, zstab_anc, stabilizer_round,
                         statePrep, numRounds, x_stabilizers, z_stabilizers);

  auto dataResults = mz(data);
}

__qpu__ void memory_circuit_mx(const code::stabilizer_round &stabilizer_round,
                               const code::one_qubit_encoding &statePrep,
                               std::size_t numData, std::size_t numAncx,
                               std::size_t numAncz, std::size_t numRounds,
                               const std::vector<std::size_t> &x_stabilizers,
                               const std::vector<std::size_t> &z_stabilizers) {

  // Allocate the data and ancilla qubits
  cudaq::qvector data(numData), xstab_anc(numAncx), zstab_anc(numAncz);

  // Persists ancilla measures
  __memory_circuit_stabs(data, xstab_anc, zstab_anc, stabilizer_round,
                         statePrep, numRounds, x_stabilizers, z_stabilizers);

  h(data);
  auto dataResults = mz(data);
}

} // namespace cudaq::qec
