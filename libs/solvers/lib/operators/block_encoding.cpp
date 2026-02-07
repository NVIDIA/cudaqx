/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/solvers/operators/block_encoding.h"
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace cudaq::solvers {

// ============================================================================
// QUANTUM KERNELS (Stateless functors for CUDA-Q compiler)
// ============================================================================

/// @brief Kernel for PREPARE operation - state preparation tree
struct pauli_prepare_kernel {
  void operator()(cudaq::qview<> anc,
                  const std::vector<double> &angles) const __qpu__ {
    if (anc.size() == 0)
      return;

    // Root rotation
    ry(angles[0], anc[0]);

    // Tree layers (controlled rotations)
    int angle_idx = 1;
    for (std::size_t layer = 1; layer < anc.size(); ++layer) {
      int num_branches = 1 << layer; // 2^layer

      for (int i = 0; i < num_branches; ++i) {
        // Setup X gates to match control pattern
        // We use binary representation: if bit is 0, flip that qubit
        for (int bit = 0; bit < static_cast<int>(layer); ++bit) {
          if (!((i >> bit) & 1)) {
            x(anc[layer - 1 - bit]);
          }
        }

        // Controlled rotation - explicitly handle cases up to 20 ancilla
        // This covers large molecular Hamiltonians (up to 2^20 = 1,048,576
        // terms) NOTE: Cannot use qview with ry<cudaq::ctrl> unlike Pauli gates
        if (layer == 1) {
          ry<cudaq::ctrl>(angles[angle_idx], anc[0], anc[1]);
        } else if (layer == 2) {
          ry<cudaq::ctrl>(angles[angle_idx], anc[0], anc[1], anc[2]);
        } else if (layer == 3) {
          ry<cudaq::ctrl>(angles[angle_idx], anc[0], anc[1], anc[2], anc[3]);
        } else if (layer == 4) {
          ry<cudaq::ctrl>(angles[angle_idx], anc[0], anc[1], anc[2], anc[3],
                          anc[4]);
        } else if (layer == 5) {
          ry<cudaq::ctrl>(angles[angle_idx], anc[0], anc[1], anc[2], anc[3],
                          anc[4], anc[5]);
        } else if (layer == 6) {
          ry<cudaq::ctrl>(angles[angle_idx], anc[0], anc[1], anc[2], anc[3],
                          anc[4], anc[5], anc[6]);
        } else if (layer == 7) {
          ry<cudaq::ctrl>(angles[angle_idx], anc[0], anc[1], anc[2], anc[3],
                          anc[4], anc[5], anc[6], anc[7]);
        } else if (layer == 8) {
          ry<cudaq::ctrl>(angles[angle_idx], anc[0], anc[1], anc[2], anc[3],
                          anc[4], anc[5], anc[6], anc[7], anc[8]);
        } else if (layer == 9) {
          ry<cudaq::ctrl>(angles[angle_idx], anc[0], anc[1], anc[2], anc[3],
                          anc[4], anc[5], anc[6], anc[7], anc[8], anc[9]);
        } else if (layer == 10) {
          ry<cudaq::ctrl>(angles[angle_idx], anc[0], anc[1], anc[2], anc[3],
                          anc[4], anc[5], anc[6], anc[7], anc[8], anc[9],
                          anc[10]);
        } else if (layer == 11) {
          ry<cudaq::ctrl>(angles[angle_idx], anc[0], anc[1], anc[2], anc[3],
                          anc[4], anc[5], anc[6], anc[7], anc[8], anc[9],
                          anc[10], anc[11]);
        } else if (layer == 12) {
          ry<cudaq::ctrl>(angles[angle_idx], anc[0], anc[1], anc[2], anc[3],
                          anc[4], anc[5], anc[6], anc[7], anc[8], anc[9],
                          anc[10], anc[11], anc[12]);
        } else if (layer == 13) {
          ry<cudaq::ctrl>(angles[angle_idx], anc[0], anc[1], anc[2], anc[3],
                          anc[4], anc[5], anc[6], anc[7], anc[8], anc[9],
                          anc[10], anc[11], anc[12], anc[13]);
        } else if (layer == 14) {
          ry<cudaq::ctrl>(angles[angle_idx], anc[0], anc[1], anc[2], anc[3],
                          anc[4], anc[5], anc[6], anc[7], anc[8], anc[9],
                          anc[10], anc[11], anc[12], anc[13], anc[14]);
        } else if (layer == 15) {
          ry<cudaq::ctrl>(angles[angle_idx], anc[0], anc[1], anc[2], anc[3],
                          anc[4], anc[5], anc[6], anc[7], anc[8], anc[9],
                          anc[10], anc[11], anc[12], anc[13], anc[14], anc[15]);
        } else if (layer == 16) {
          ry<cudaq::ctrl>(angles[angle_idx], anc[0], anc[1], anc[2], anc[3],
                          anc[4], anc[5], anc[6], anc[7], anc[8], anc[9],
                          anc[10], anc[11], anc[12], anc[13], anc[14], anc[15],
                          anc[16]);
        } else if (layer == 17) {
          ry<cudaq::ctrl>(angles[angle_idx], anc[0], anc[1], anc[2], anc[3],
                          anc[4], anc[5], anc[6], anc[7], anc[8], anc[9],
                          anc[10], anc[11], anc[12], anc[13], anc[14], anc[15],
                          anc[16], anc[17]);
        } else if (layer == 18) {
          ry<cudaq::ctrl>(angles[angle_idx], anc[0], anc[1], anc[2], anc[3],
                          anc[4], anc[5], anc[6], anc[7], anc[8], anc[9],
                          anc[10], anc[11], anc[12], anc[13], anc[14], anc[15],
                          anc[16], anc[17], anc[18]);
        } else if (layer == 19) {
          ry<cudaq::ctrl>(angles[angle_idx], anc[0], anc[1], anc[2], anc[3],
                          anc[4], anc[5], anc[6], anc[7], anc[8], anc[9],
                          anc[10], anc[11], anc[12], anc[13], anc[14], anc[15],
                          anc[16], anc[17], anc[18], anc[19]);
        } else {
          // Hamiltonian with > 2^20 terms - extremely rare
          // This would require > 20 ancilla qubits
          // Users should consider Hamiltonian truncation for such large systems
          throw std::runtime_error("Block encoding requires > 20 ancilla "
                                   "qubits. Consider Hamiltonian truncation.");
        }
        angle_idx++;

        // Uncompute X gates
        for (int bit = 0; bit < static_cast<int>(layer); ++bit) {
          if (!((i >> bit) & 1)) {
            x(anc[layer - 1 - bit]);
          }
        }
      }
    }
  }
};

/// @brief Kernel for SELECT operation - controlled Pauli application
struct pauli_select_kernel {
  void operator()(cudaq::qview<> anc, cudaq::qview<> sys,
                  const std::vector<int> &ctrls, const std::vector<int> &ops,
                  const std::vector<int> &lens,
                  const std::vector<int> &signs) const __qpu__ {
    int ptr_ctrl = 0;
    int ptr_op = 0;
    int n_anc = anc.size();

    // Iterate through every term in the Hamiltonian
    for (std::size_t i = 0; i < lens.size(); ++i) {
      int n_ops = lens[i];
      int sign = signs[i];

      // 1. Activate control pattern (match ancilla state to term index)
      for (int b = 0; b < n_anc; ++b) {
        int bit_val = ctrls[ptr_ctrl++];
        if (bit_val == 0)
          x(anc[b]);
      }

      // 2. Apply Pauli operators to system qubits
      // Format: [code, qubit_idx, code, qubit_idx, ...]
      // Codes: 1=X, 2=Y, 3=Z
      for (int k = 0; k < n_ops; ++k) {
        int code = ops[ptr_op++];
        int q_idx = ops[ptr_op++];

        if (code == 1)
          x<cudaq::ctrl>(anc, sys[q_idx]); // Controlled-X
        else if (code == 2)
          y<cudaq::ctrl>(anc, sys[q_idx]); // Controlled-Y
        else if (code == 3)
          z<cudaq::ctrl>(anc, sys[q_idx]); // Controlled-Z
      }

      // 3. Apply sign correction (if coefficient is negative)
      if (sign < 0) {
        // Controlled-Z on last ancilla, controlled by all others
        if (n_anc == 1) {
          z(anc[0]);
        } else if (n_anc == 2) {
          z<cudaq::ctrl>(anc[0], anc[1]);
        } else if (n_anc == 3) {
          z<cudaq::ctrl>(anc[0], anc[1], anc[2]);
        } else if (n_anc == 4) {
          z<cudaq::ctrl>(anc[0], anc[1], anc[2], anc[3]);
        } else if (n_anc == 5) {
          z<cudaq::ctrl>(anc[0], anc[1], anc[2], anc[3], anc[4]);
        } else if (n_anc == 6) {
          z<cudaq::ctrl>(anc[0], anc[1], anc[2], anc[3], anc[4], anc[5]);
        } else if (n_anc == 7) {
          z<cudaq::ctrl>(anc[0], anc[1], anc[2], anc[3], anc[4], anc[5],
                         anc[6]);
        } else if (n_anc == 8) {
          z<cudaq::ctrl>(anc[0], anc[1], anc[2], anc[3], anc[4], anc[5], anc[6],
                         anc[7]);
        } else if (n_anc == 9) {
          z<cudaq::ctrl>(anc[0], anc[1], anc[2], anc[3], anc[4], anc[5], anc[6],
                         anc[7], anc[8]);
        } else if (n_anc == 10) {
          z<cudaq::ctrl>(anc[0], anc[1], anc[2], anc[3], anc[4], anc[5], anc[6],
                         anc[7], anc[8], anc[9]);
        }
        // Note: Hamiltonians with > 1024 terms are very rare in practice
      }

      // 4. Deactivate control pattern (uncompute X gates)
      int back_ptr = ptr_ctrl - 1;
      for (int b_rev = 0; b_rev < n_anc; ++b_rev) {
        int anc_idx = (n_anc - 1) - b_rev;
        int bit_val = ctrls[back_ptr--];
        if (bit_val == 0)
          x(anc[anc_idx]);
      }
    }
  }
};

// ============================================================================
// CLASSICAL UTILITIES
// ============================================================================

std::vector<double>
pauli_lcu::compute_angles(const std::vector<double> &probs) {
  if (probs.empty())
    return {};

  std::size_t n_leaves = probs.size();
  if ((n_leaves & (n_leaves - 1)) != 0)
    throw std::runtime_error(
        "compute_angles: probability vector size must be a power of 2");

  int n_qubits = std::log2(n_leaves);
  std::vector<double> angles;

  // Build tree layer by layer
  for (int layer = 0; layer < n_qubits; ++layer) {
    int n_nodes = 1 << layer; // Number of nodes at this layer
    for (int node = 0; node < n_nodes; ++node) {
      int step = n_leaves / (1 << (layer + 1));
      int start_idx = node * step * 2;

      // Sum probabilities for current subtree
      double total_p = 0.0;
      for (int k = 0; k < 2 * step; ++k)
        total_p += probs[start_idx + k];

      if (total_p < 1e-12) {
        angles.push_back(0.0);
      } else {
        // Sum of right branch
        double p_1 = 0.0;
        for (int k = step; k < 2 * step; ++k)
          p_1 += probs[start_idx + k];

        // Compute rotation angle: θ = 2*arcsin(sqrt(p_right / p_total))
        angles.push_back(2.0 * std::asin(std::sqrt(p_1 / total_p)));
      }
    }
  }

  return angles;
}

// ============================================================================
// PAULI LCU IMPLEMENTATION
// ============================================================================

pauli_lcu::pauli_lcu(const cudaq::spin_op &hamiltonian,
                     std::size_t num_qubits) {
  n_sys = num_qubits;

  // Count terms and compute 1-norm
  std::vector<double> coeffs;
  std::vector<cudaq::pauli_word> words;

  alpha = 0.0;
  std::size_t num_terms = 0;

  // Extract terms from spin_op
  for (const auto &term : hamiltonian) {
    auto coeff = term.evaluate_coefficient();
    double abs_coeff = std::abs(coeff);

    // Skip very small terms (numerical noise)
    if (abs_coeff < 1e-12)
      continue;

    // For now, we only support real Hamiltonians
    if (std::abs(coeff.imag()) > 1e-10)
      throw std::runtime_error(
          "pauli_lcu: complex Hamiltonians not yet supported");

    coeffs.push_back(coeff.real());
    words.push_back(term.get_pauli_word(num_qubits));
    alpha += abs_coeff;
    num_terms++;
  }

  if (num_terms == 0)
    throw std::runtime_error("pauli_lcu: Hamiltonian has no terms");

  // Determine number of ancilla qubits needed
  n_anc = static_cast<std::size_t>(std::ceil(std::log2(num_terms)));

  // Normalize coefficients to probabilities
  std::vector<double> probs;
  for (double c : coeffs)
    probs.push_back(std::abs(c) / alpha);

  // Pad to power of 2 for binary tree
  std::size_t padded_size = 1ULL << n_anc;
  while (probs.size() < padded_size)
    probs.push_back(0.0);

  // Compute state preparation angles
  state_prep_angles = compute_angles(probs);

  // Flatten terms for SELECT kernel
  for (std::size_t idx = 0; idx < num_terms; ++idx) {
    // Binary representation of index for control pattern
    for (std::size_t b = 0; b < n_anc; ++b) {
      // MSB to LSB ordering
      int bit_val = (idx >> (n_anc - 1 - b)) & 1;
      term_controls.push_back(bit_val);
    }

    // Parse Pauli word string: format is "XYZII..." where position i is qubit i
    std::string word_str = words[idx].str();
    std::vector<std::pair<int, int>> ops_for_term; // (code, qubit_idx)

    // Each character position corresponds to a qubit
    for (std::size_t q_idx = 0; q_idx < word_str.size(); ++q_idx) {
      char pauli_char = word_str[q_idx];

      if (pauli_char == 'I') {
        // Identity - skip
        continue;
      } else if (pauli_char == 'X') {
        ops_for_term.push_back({1, static_cast<int>(q_idx)});
      } else if (pauli_char == 'Y') {
        ops_for_term.push_back({2, static_cast<int>(q_idx)});
      } else if (pauli_char == 'Z') {
        ops_for_term.push_back({3, static_cast<int>(q_idx)});
      }
      // Other characters are ignored
    }

    // Flatten ops
    for (const auto &[code, q_idx] : ops_for_term) {
      term_ops.push_back(code);
      term_ops.push_back(q_idx);
    }

    term_lengths.push_back(ops_for_term.size());
    term_signs.push_back((coeffs[idx] < 0) ? -1 : 1);
  }
}

// ============================================================================
// QUANTUM OPERATION DISPATCHERS
// ============================================================================

void pauli_lcu::prepare(cudaq::qview<> ancilla) const {
  if (ancilla.size() != n_anc)
    throw std::runtime_error("pauli_lcu::prepare: ancilla size mismatch");
  pauli_prepare_kernel{}(ancilla, state_prep_angles);
}

void pauli_lcu::unprepare(cudaq::qview<> ancilla) const {
  if (ancilla.size() != n_anc)
    throw std::runtime_error("pauli_lcu::unprepare: ancilla size mismatch");
  cudaq::adjoint(pauli_prepare_kernel{}, ancilla, state_prep_angles);
}

void pauli_lcu::select(cudaq::qview<> ancilla, cudaq::qview<> system) const {
  if (ancilla.size() != n_anc)
    throw std::runtime_error("pauli_lcu::select: ancilla size mismatch");
  if (system.size() != n_sys)
    throw std::runtime_error("pauli_lcu::select: system size mismatch");

  pauli_select_kernel{}(ancilla, system, term_controls, term_ops, term_lengths,
                        term_signs);
}

} // namespace cudaq::solvers
