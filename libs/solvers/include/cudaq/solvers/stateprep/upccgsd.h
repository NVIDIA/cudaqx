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

/// @brief Generate UpCCGSD operator pool (Python-style unique singles/doubles)
/// and extract Pauli words and coefficients.
/// @details The UpCCGSD pool is constructed using spin-preserving generalized
/// singles and paired αβ→αβ generalized doubles in an interleaved spin-orbital
/// ordering (α0, β0, α1, β1, ..., α_{N-1}, β_{N-1}).
/// @param norbitals Number of spin orbitals (qubits)
/// @param only_singles If true, include only single excitations
/// @param only_doubles If true, include only double excitations
/// @return Pair of lists: [Pauli words grouped by excitation],
/// [coefficients grouped by excitation]
std::pair<std::vector<std::vector<cudaq::pauli_word>>,
          std::vector<std::vector<double>>>
get_upccgsd_pauli_lists(std::size_t norbitals, bool only_singles = false,
                        bool only_doubles = false);

/// \pure_device_kernel
///
/// @brief Apply UpCCGSD ansatz to a qubit register using grouped Pauli words
/// and coefficients.
/// @param qubits Qubit register
/// @param thetas Vector of rotation angles (one per excitation group)
/// @param pauliWordsList Pauli words grouped by excitation
/// @param coefficientsList Coefficients grouped by excitation
__qpu__ void
upccgsd(cudaq::qview<> qubits, const std::vector<double> &thetas,
        const std::vector<std::vector<cudaq::pauli_word>> &pauliWordsList,
        const std::vector<std::vector<double>> &coefficientsList);

} // namespace cudaq::solvers::stateprep
