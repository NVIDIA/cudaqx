/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq.h"
#include <utility>
#include <vector>

namespace cudaq::solvers::stateprep {

/// @brief Generate CEO operator pool and extract Pauli words and coefficients
/// @details The CEO pool is constructed using qubit excitation operators that
/// are coupled to reduce the circuit implementation cost. The excitations
/// preserve the particle number and Sz quantum numbers, but do not include Z
/// parity strings. For more details, see the CEO paper
/// (https://arxiv.org/abs/2407.08696).
/// @param norbitals Number of spatial orbitals
/// @return Pair of lists: [Pauli words grouped by excitation], [coefficients
/// grouped by excitation]
std::pair<std::vector<std::vector<cudaq::pauli_word>>,
          std::vector<std::vector<double>>>
get_ceo_pauli_lists(std::size_t norbitals);

/// \pure_device_kernel
///
/// @brief Apply CEO ansatz to a qubit register using grouped Pauli words and
/// coefficients
/// @param qubits Qubit register
/// @param thetas Vector of rotation angles (one per excitation group)
/// @param pauliWordsList Pauli words grouped by excitation
/// @param coefficientsList Coefficients grouped by excitation
__qpu__ void
ceo(cudaq::qview<> qubits, const std::vector<double> &thetas,
    const std::vector<std::vector<cudaq::pauli_word>> &pauliWordsList,
    const std::vector<std::vector<double>> &coefficientsList);

} // namespace cudaq::solvers::stateprep
