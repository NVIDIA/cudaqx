/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cuda-qx/core/heterogeneous_map.h"
#include "cudaq.h"
#include "cudaq/solvers/operators/block_encoding.h"
#include <utility>
#include <vector>

using namespace cudaqx;

namespace cudaq::solvers {

/// @brief Result from Quantum Exact Lanczos (QEL) algorithm
/// @details Encapsulates all data produced by a QEL execution, including
/// the Krylov Hamiltonian matrix, overlap matrix, collected moments,
/// and metadata about the computation. Users can diagonalize the matrices
/// using their preferred linear algebra library (NumPy, SciPy, etc.)
struct qel_result {
  /// Krylov Hamiltonian matrix (flattened, row-major order)
  /// Dimension: krylov_dimension × krylov_dimension
  std::vector<double> hamiltonian_matrix;

  /// Krylov overlap matrix (flattened, row-major order)
  /// Dimension: krylov_dimension × krylov_dimension
  std::vector<double> overlap_matrix;

  /// Collected moments: μₖ = ⟨ψ|Tₖ(H)|ψ⟩
  std::vector<double> moments;

  /// Dimension of the Krylov subspace used
  int krylov_dimension;

  /// Constant term from the Hamiltonian (to be added to eigenvalues)
  double constant_term;

  /// Normalization constant (1-norm of Hamiltonian coefficients)
  double normalization;

  /// Number of ancilla qubits used
  std::size_t num_ancilla;

  /// Number of system qubits
  std::size_t num_system;
};

/// @brief Run the Quantum Exact Lanczos (QEL) algorithm
/// @details Uses block encoding and amplitude amplification to compute
/// the ground state energy of a quantum Hamiltonian. The algorithm builds
/// a Krylov subspace by collecting moments μₖ = ⟨ψ|Tₖ(H)|ψ⟩ where Tₖ are
/// Chebyshev polynomials, then solves a generalized eigenvalue problem.
///
/// @param hamiltonian The target Hamiltonian as a spin_op
/// @param initial_state Quantum kernel to prepare the initial state (e.g., HF
/// state)
/// @param n_electrons Number of electrons in the system (for HF initialization)
/// @param options Additional options for the algorithm. Supported Keys:
///   - "krylov_dim" (int): Krylov subspace dimension [default: 10]
///   - "shots" (int): Number of measurement shots (-1 for exact) [default: -1]
///   - "verbose" (bool): Enable detailed output logging [default: false]
/// @return qel_result containing the Krylov matrices and moments for
/// diagonalization
qel_result
quantum_exact_lanczos(const cudaq::spin_op &hamiltonian, std::size_t num_qubits,
                      std::size_t n_electrons,
                      heterogeneous_map options = heterogeneous_map());

} // namespace cudaq::solvers
