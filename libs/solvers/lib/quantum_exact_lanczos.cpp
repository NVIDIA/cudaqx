/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/solvers/quantum_exact_lanczos.h"
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace cudaq::solvers {

// ============================================================================
// QUANTUM KERNELS FOR QEL
// ============================================================================

/// @brief Reflection operator for amplitude amplification
/// @details Implements: PREPARE† → X → Multi-controlled-Z → X → PREPARE
struct qel_reflection_kernel {
  void operator()(cudaq::qview<> anc, const pauli_lcu &encoding) const __qpu__ {
    // PREPARE†
    encoding.unprepare(anc);

    // X on all ancilla
    for (std::size_t i = 0; i < anc.size(); ++i) {
      x(anc[i]);
    }

    // Multi-controlled Z (controlled by all but last, target is last)
    std::size_t n_anc = anc.size();
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
      z<cudaq::ctrl>(anc[0], anc[1], anc[2], anc[3], anc[4], anc[5], anc[6]);
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

    // X on all ancilla (uncompute)
    for (std::size_t i = 0; i < anc.size(); ++i) {
      x(anc[i]);
    }

    // PREPARE
    encoding.prepare(anc);
  }
};

/// @brief State preparation for odd moment measurement
struct qel_state_prep_kernel {
  void operator()(cudaq::qview<> anc, cudaq::qview<> sys,
                  const pauli_lcu &encoding, int k_half) const __qpu__ {
    // Apply k iterations of SELECT + REFLECT
    encoding.prepare(anc);
    for (int i = 0; i < k_half; ++i) {
      encoding.select(anc, sys);
      qel_reflection_kernel{}(anc, encoding);
    }
  }
};

/// @brief State preparation for even moment measurement
struct qel_measure_even_kernel {
  void operator()(cudaq::qview<> anc, cudaq::qview<> sys,
                  const pauli_lcu &encoding, int k_half) const __qpu__ {
    // Apply k iterations of SELECT + REFLECT
    encoding.prepare(anc);
    for (int i = 0; i < k_half; ++i) {
      encoding.select(anc, sys);
      qel_reflection_kernel{}(anc, encoding);
    }
    // Unprepare for measurement in |0⟩ basis
    encoding.unprepare(anc);
  }
};

// ============================================================================
// OBSERVABLE CONSTRUCTION
// ============================================================================

/// @brief Build projector |0⟩⟨0| on ancilla qubits
cudaq::spin_op build_ancilla_projector(std::size_t n_anc) {
  // P_0 = (I + Z) / 2 for each qubit
  cudaq::spin_op projector = 0.5 * (cudaq::spin::i(0) + cudaq::spin::z(0));
  for (std::size_t q = 1; q < n_anc; ++q) {
    projector *= 0.5 * (cudaq::spin::i(q) + cudaq::spin::z(q));
  }

  return projector;
}

/// @brief Build R observable for even moments: R = 2*P_0 - I
cudaq::spin_op build_R_observable(std::size_t n_anc) {
  auto P_zero = build_ancilla_projector(n_anc);
  return 2.0 * P_zero - cudaq::spin::i(0);
}

/// @brief Build U observable for odd moments
/// @details U = Σᵢ sign(cᵢ) * P_i ⊗ Pᵢ where P_i projects onto ancilla state
/// |i⟩
cudaq::spin_op build_U_observable(const pauli_lcu &encoding) {
  std::size_t n_anc = encoding.num_ancilla();
  std::size_t n_sys = encoding.num_system();

  const auto &controls = encoding.get_term_controls();
  const auto &ops = encoding.get_term_ops();
  const auto &lengths = encoding.get_term_lengths();
  const auto &signs = encoding.get_term_signs();

  cudaq::spin_op U_op;
  bool first_term = true;

  int ctrl_ptr = 0;
  int ops_ptr = 0;

  for (std::size_t term_idx = 0; term_idx < lengths.size(); ++term_idx) {
    // Build ancilla projector for this term's index
    cudaq::spin_op anc_proj;
    bool first_anc = true;

    for (std::size_t b = 0; b < n_anc; ++b) {
      int bit_val = controls[ctrl_ptr++];
      cudaq::spin_op proj_bit =
          (bit_val == 0)
              ? 0.5 * (cudaq::spin::i(b) + cudaq::spin::z(b))  // |0⟩⟨0|
              : 0.5 * (cudaq::spin::i(b) - cudaq::spin::z(b)); // |1⟩⟨1|

      if (first_anc) {
        anc_proj = proj_bit;
        first_anc = false;
      } else {
        anc_proj = anc_proj * proj_bit;
      }
    }

    // Build system Pauli operator
    cudaq::spin_op sys_pauli;
    bool first_pauli = true;

    int n_ops = lengths[term_idx];
    for (int k = 0; k < n_ops; ++k) {
      int code = ops[ops_ptr++];
      int qubit = ops[ops_ptr++] + n_anc; // Offset by ancilla qubits

      cudaq::spin_op pauli_op;
      if (code == 1)
        pauli_op = cudaq::spin::x(qubit);
      else if (code == 2)
        pauli_op = cudaq::spin::y(qubit);
      else if (code == 3)
        pauli_op = cudaq::spin::z(qubit);

      if (first_pauli) {
        sys_pauli = pauli_op;
        first_pauli = false;
      } else {
        sys_pauli = sys_pauli * pauli_op;
      }
    }

    // If no Pauli operators, use identity
    if (first_pauli) {
      sys_pauli = cudaq::spin::i(n_anc);
    }

    // Combine with sign
    double sign = signs[term_idx];
    cudaq::spin_op term = sign * anc_proj * sys_pauli;

    if (first_term) {
      U_op = term;
      first_term = false;
    } else {
      U_op = U_op + term;
    }
  }

  return U_op;
}

// ============================================================================
// KRYLOV MATRIX CONSTRUCTION
// ============================================================================

/// @brief Build Krylov Hamiltonian and overlap matrices from moments
/// @details Constructs H and S matrices that can be diagonalized by the user
/// to extract eigenvalues: H|v⟩ = E·S|v⟩
std::pair<std::vector<double>, std::vector<double>>
build_krylov_matrices(const std::vector<double> &moments, int krylov_dim) {

  std::vector<double> H_mat(krylov_dim * krylov_dim);
  std::vector<double> S_mat(krylov_dim * krylov_dim);

  for (int i = 0; i < krylov_dim; ++i) {
    for (int j = 0; j < krylov_dim; ++j) {
      int idx = i * krylov_dim + j; // Row-major indexing

      S_mat[idx] = 0.5 * (moments[i + j] + moments[std::abs(i - j)]);
      H_mat[idx] =
          0.25 * (moments[i + j + 1] + moments[std::abs(i + j - 1)] +
                  moments[std::abs(i - j + 1)] + moments[std::abs(i - j - 1)]);
    }
  }

  return {H_mat, S_mat};
}

// ============================================================================
// MAIN QEL ALGORITHM
// ============================================================================

qel_result quantum_exact_lanczos(const cudaq::spin_op &hamiltonian,
                                 std::size_t num_qubits,
                                 std::size_t n_electrons,
                                 heterogeneous_map options) {

  // Extract options
  int krylov_dim = options.get("krylov_dim", 10);
  int shots = options.get("shots", -1);
  bool verbose = options.get("verbose", false);

  // Create block encoding
  pauli_lcu encoding(hamiltonian, num_qubits);

  std::size_t n_anc = encoding.num_ancilla();
  std::size_t n_sys = encoding.num_system();
  double one_norm = encoding.normalization();

  // Extract constant term from Hamiltonian
  double constant_term = 0.0;
  for (const auto &term : hamiltonian) {
    auto word = term.get_pauli_word(num_qubits);
    // Check if this is an identity term (all characters are 'I')
    bool is_identity = (word.find_first_not_of('I') == std::string::npos);
    if (is_identity) {
      constant_term = term.evaluate_coefficient().real();
      break;
    }
  }

  if (verbose) {
    std::cout << "\n=== Quantum Exact Lanczos ===" << std::endl;
    std::cout << "System qubits: " << n_sys << std::endl;
    std::cout << "Ancilla qubits: " << n_anc << std::endl;
    std::cout << "Electrons: " << n_electrons << std::endl;
    std::cout << "1-Norm (α): " << one_norm << std::endl;
    std::cout << "Constant term: " << constant_term << " Ha" << std::endl;
    std::cout << "Krylov dimension: " << krylov_dim << std::endl;
  }

  // Build observables
  auto R_op = build_R_observable(n_anc);
  auto U_op = build_U_observable(encoding);

  // Collect moments
  std::vector<double> moments;
  moments.reserve(2 * krylov_dim);

  if (verbose) {
    std::cout << "\nCollecting moments..." << std::endl;
  }

  for (int k = 0; k < 2 * krylov_dim; ++k) {
    int m = k / 2;

    // Create quantum kernel for this moment
    auto measure_kernel = [&, m, n_anc, n_sys, n_electrons]() __qpu__ {
      cudaq::qvector<> anc(n_anc);
      cudaq::qvector<> sys(n_sys);

      // Initialize Hartree-Fock state
      for (std::size_t i = 0; i < n_electrons; ++i) {
        x(sys[i]);
      }

      if (k % 2 == 0) {
        // Even moment
        qel_measure_even_kernel{}(anc, sys, encoding, m);
      } else {
        // Odd moment
        qel_state_prep_kernel{}(anc, sys, encoding, m);
      }
    };

    // Measure
    cudaq::spin_op obs = (k % 2 == 0) ? R_op : U_op;
    auto result = cudaq::observe(shots, measure_kernel, obs);
    double moment = result.expectation();
    moments.push_back(moment);

    if (verbose) {
      std::cout << "  k=" << k << ": " << moment << std::endl;
    }
  }

  // Build Krylov matrices
  if (verbose) {
    std::cout << "\nBuilding Krylov matrices..." << std::endl;
  }

  auto [H_mat, S_mat] = build_krylov_matrices(moments, krylov_dim);

  if (verbose) {
    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Hamiltonian matrix: " << krylov_dim << "×" << krylov_dim
              << std::endl;
    std::cout << "Overlap matrix: " << krylov_dim << "×" << krylov_dim
              << std::endl;
    std::cout << "Total moments collected: " << moments.size() << std::endl;
    std::cout << "\nTo extract eigenvalues, solve: H|v⟩ = E·S|v⟩" << std::endl;
    std::cout << "Then convert: E_physical = E_scaled * α + constant"
              << std::endl;
  }

  // Return result
  return qel_result{H_mat,         S_mat,    moments, krylov_dim,
                    constant_term, one_norm, n_anc,   n_sys};
}

} // namespace cudaq::solvers
