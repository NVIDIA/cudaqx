/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq.h"
#include <cstddef>
#include <vector>

namespace cudaq::solvers {

/// @brief Block encoding implementation using Pauli LCU (Linear Combination of
/// Unitaries)
///
/// A block encoding represents a quantum subroutine that encodes a target
/// operator (typically a Hamiltonian) into a unitary operator acting on
/// an extended Hilbert space with ancilla qubits. This is a fundamental
/// building block for algorithms like Quantum Eigenvalue Learning (QEL),
/// Quantum Singular Value Transformation (QSVT), and Hamiltonian simulation.
///
/// The block encoding U satisfies: (⟨0|_anc ⊗ I_sys) U (|0⟩_anc ⊗ I_sys) = H/α
/// where α is the normalization constant and H is the target operator.
///
/// This implementation is optimized for Hamiltonians expressed as sums of Pauli
/// strings (e.g., molecular Hamiltonians from quantum chemistry). It uses:
/// - PREPARE: State preparation tree with controlled rotations
/// - SELECT: Controlled Pauli operations indexed by ancilla state
///
/// The encoding uses log₂(# terms) ancilla qubits and achieves α = ||H||₁.
class pauli_lcu {
private:
  // Flattened data for GPU kernels
  std::vector<double> state_prep_angles; // Rotation angles for PREPARE tree
  std::vector<int> term_controls;        // Binary control values per term
  std::vector<int> term_ops;     // Flattened [code, qubit, code, qubit, ...]
  std::vector<int> term_lengths; // Number of Pauli ops per term
  std::vector<int> term_signs;   // Sign (+1 or -1) per term

  // Metadata
  std::size_t n_anc; // Number of ancilla qubits
  std::size_t n_sys; // Number of system qubits
  double alpha;      // Normalization (1-norm)

  /// @brief Compute rotation angles for state preparation tree
  /// @param probs Probability distribution (must be power of 2 length)
  /// @return Vector of rotation angles for each node in the tree
  static std::vector<double> compute_angles(const std::vector<double> &probs);

public:
  /// @brief Construct a Pauli LCU block encoding from a spin operator
  /// @param hamiltonian The target Hamiltonian as a spin_op
  /// @param num_qubits Number of system qubits (must match Hamiltonian support)
  explicit pauli_lcu(const cudaq::spin_op &hamiltonian, std::size_t num_qubits);

  /// @brief Get the number of ancilla qubits
  /// @return Number of ancilla qubits
  std::size_t num_ancilla() const { return n_anc; }

  /// @brief Get the number of system qubits
  /// @return Number of system qubits
  std::size_t num_system() const { return n_sys; }

  /// @brief Get the normalization constant (1-norm of coefficients)
  /// @details The block encoding satisfies ||H|| ≤ α = ||H||₁
  /// @return Normalization constant (1-norm)
  double normalization() const { return alpha; }

  /// @brief Apply PREPARE: encode coefficients into ancilla superposition
  /// @details Prepares a superposition state on the ancilla qubits that
  /// encodes the coefficients of the Hamiltonian terms
  /// @param ancilla View of ancilla qubits
  void prepare(cudaq::qview<> ancilla) const;

  /// @brief Apply PREPARE†: uncompute the ancilla superposition
  /// @details Adjoint of the PREPARE operation
  /// @param ancilla View of ancilla qubits
  void unprepare(cudaq::qview<> ancilla) const;

  /// @brief Apply SELECT: controlled Pauli operations
  /// @details Applies the appropriate Hamiltonian term conditioned on the
  /// ancilla register state
  /// @param ancilla View of ancilla qubits (control register)
  /// @param system View of system qubits (target register)
  void select(cudaq::qview<> ancilla, cudaq::qview<> system) const;

  /// @brief Apply the full block encoding: PREPARE → SELECT → PREPARE†
  /// @details Applies the complete block encoding unitary operation
  /// @param ancilla View of ancilla qubits
  /// @param system View of system qubits
  void apply(cudaq::qview<> ancilla, cudaq::qview<> system) const {
    prepare(ancilla);
    select(ancilla, system);
    unprepare(ancilla);
  }

  /// @brief Get the state preparation angles (for debugging/testing)
  const std::vector<double> &get_angles() const { return state_prep_angles; }

  /// @brief Get the flattened term controls (for debugging/testing)
  const std::vector<int> &get_term_controls() const { return term_controls; }

  /// @brief Get the flattened term ops (for debugging/testing)
  const std::vector<int> &get_term_ops() const { return term_ops; }

  /// @brief Get the term lengths (for debugging/testing)
  const std::vector<int> &get_term_lengths() const { return term_lengths; }

  /// @brief Get the term signs (for debugging/testing)
  const std::vector<int> &get_term_signs() const { return term_signs; }
};

} // namespace cudaq::solvers
