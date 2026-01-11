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

/// @brief Abstract base class for block encoding implementations
///
/// A block encoding represents a quantum subroutine that encodes a target
/// operator (typically a Hamiltonian) into a unitary operator acting on
/// an extended Hilbert space with ancilla qubits. This is the fundamental
/// building block for algorithms like Quantum Eigenvalue Learning (QEL),
/// Quantum Singular Value Transformation (QSVT), and Hamiltonian simulation.
///
/// The block encoding U satisfies: (⟨0|_anc ⊗ I_sys) U (|0⟩_anc ⊗ I_sys) = H/α
/// where α is the normalization constant and H is the target operator.
class block_encoding {
public:
  virtual ~block_encoding() = default;

  /// @brief Get the number of ancilla qubits required
  /// @return Number of ancilla qubits
  virtual std::size_t num_ancilla() const = 0;

  /// @brief Get the number of system qubits
  /// @return Number of system qubits
  virtual std::size_t num_system() const = 0;

  /// @brief Get the normalization constant (alpha)
  /// @details The block encoding satisfies ||H|| ≤ α, where α is typically
  /// the 1-norm or 2-norm of the Hamiltonian
  /// @return Normalization constant
  virtual double normalization() const = 0;

  /// @brief Apply the PREPARE operation
  /// @details Prepares a superposition state on the ancilla qubits that
  /// encodes the coefficients of the Hamiltonian terms
  /// @param ancilla View of ancilla qubits
  virtual void prepare(cudaq::qview<> ancilla) const = 0;

  /// @brief Apply the PREPARE† (adjoint/uncomputation) operation
  /// @param ancilla View of ancilla qubits
  virtual void unprepare(cudaq::qview<> ancilla) const = 0;

  /// @brief Apply the SELECT operation
  /// @details Applies the appropriate Hamiltonian term conditioned on the
  /// ancilla register state
  /// @param ancilla View of ancilla qubits (control register)
  /// @param system View of system qubits (target register)
  virtual void select(cudaq::qview<> ancilla, cudaq::qview<> system) const = 0;

  /// @brief Apply the full block encoding: PREPARE → SELECT → PREPARE†
  /// @param ancilla View of ancilla qubits
  /// @param system View of system qubits
  virtual void apply(cudaq::qview<> ancilla, cudaq::qview<> system) const {
    prepare(ancilla);
    select(ancilla, system);
    unprepare(ancilla);
  }
};

/// @brief Block encoding implementation using Pauli LCU (Linear Combination of
/// Unitaries)
///
/// This implementation is optimized for Hamiltonians expressed as sums of Pauli
/// strings (e.g., molecular Hamiltonians from quantum chemistry). It uses:
/// - PREPARE: State preparation tree with controlled rotations
/// - SELECT: Controlled Pauli operations indexed by ancilla state
///
/// The encoding uses log₂(# terms) ancilla qubits and achieves α = ||H||₁.
class pauli_lcu : public block_encoding {
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
  std::size_t num_ancilla() const override { return n_anc; }

  /// @brief Get the number of system qubits
  std::size_t num_system() const override { return n_sys; }

  /// @brief Get the normalization constant (1-norm of coefficients)
  double normalization() const override { return alpha; }

  /// @brief Apply PREPARE: encode coefficients into ancilla superposition
  void prepare(cudaq::qview<> ancilla) const override;

  /// @brief Apply PREPARE†: uncompute the ancilla superposition
  void unprepare(cudaq::qview<> ancilla) const override;

  /// @brief Apply SELECT: controlled Pauli operations
  void select(cudaq::qview<> ancilla, cudaq::qview<> system) const override;

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
