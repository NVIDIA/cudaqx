/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq/qis/pauli_word.h"
#include "cudaq/qis/qkernel.h"
#include "cudaq/qis/qubit_qis.h"
#include "cudaq/qis/qview.h"
#include <functional>

namespace cudaq {

/// @brief CUDA-Q quantum kernel function for the ADAPT-VQE algorithm
///
/// This function represents a quantum kernel that implements the core
/// quantum operations of the ADAPT-VQE (Adaptive Derivative-Assembled
/// Pseudo-Trotter Variational Quantum Eigensolver) algorithm.
///
/// @param numQubits The number of qubits in the quantum system
/// @param statePrep A function to prepare the initial quantum state
/// @param thetas Vector of rotation angles for the variational circuit
/// @param coefficients Vector of coefficients for the Hamiltonian terms
/// @param trotterOpList Vector of Pauli words representing the Trotter
/// operators
/// @param poolIndices Vector of indices indicating which terms belong to which
/// operator
///
/// @note This is a CUDA-Q quantum kernel function and should be executed within
/// the CUDA-Q framework. It applies the ADAPT-VQE circuit construction based on
/// the provided parameters and operators.
///
/// @see ADAPT-VQE algorithm for more details on the method
void adapt_kernel(std::size_t numQubits,
                  const cudaq::qkernel<void(cudaq::qvector<> &)> &statePrep,
                  const std::vector<double> &thetas,
                  const std::vector<double> &coefficients,
                  const std::vector<cudaq::pauli_word> &trotterOpList,
                  const std::vector<size_t> poolIndices);

} // namespace cudaq
