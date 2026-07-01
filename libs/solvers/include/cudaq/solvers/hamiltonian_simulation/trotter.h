/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq.h"
#include "cudaq/qis/pauli_word.h"
#include "cudaq/spin_op.h"

#include <cstddef>
#include <vector>

namespace cudaq::solvers {

inline constexpr int first_order_trotter = 1;
inline constexpr int second_order_trotter = 2;
inline constexpr int fourth_order_trotter = 4;

inline constexpr double forest_ruth_w1 = 1.3512071919596578;
inline constexpr double forest_ruth_w0 = -1.7024143839193153;

/// @brief Flattened Hamiltonian representation used by Trotter routines.
/// @details Identity terms are stored separately. For H = c I + H',
/// apply_trotter() implements the product-formula circuit for H' and omits the
/// phase exp(-i c t). That phase is globally unobservable for a single
/// unconditioned state evolution, but it can matter for controlled evolution,
/// overlaps, phase estimation, Krylov/QEL moments, and other interference-based
/// algorithms. Callers in those contexts must account for identity_coefficient.
struct trotter_terms {
  std::vector<double> coefficients;
  std::vector<cudaq::pauli_word> words;
  double identity_coefficient = 0.0;
  std::size_t num_qubits = 0;
};

/// @brief Extract real non-identity Pauli terms from a spin operator.
/// @details The returned identity_coefficient is not consumed by
/// apply_trotter(). It is returned so higher-level algorithms can preserve the
/// omitted exp(-i * identity_coefficient * time) phase when that phase becomes
/// observable as a relative phase.
/// @throws std::invalid_argument for coefficients with non-negligible imaginary
/// parts.
trotter_terms make_trotter_terms(const cudaq::spin_op &hamiltonian,
                                 double coefficient_tolerance = 1e-12);

/// @brief Apply Suzuki-Trotter evolution to a live quantum register.
/// @details This QPU-facing primitive consumes flattened Pauli terms rather
/// than cudaq::spin_op so that kernels only receive device-lowerable data.
/// Identity terms are intentionally omitted. Thus, if the original Hamiltonian
/// was H = c I + H', this applies a product-formula approximation to
/// exp(-i H' t), not the full exp(-i H t). The omitted exp(-i c t) phase is
/// harmless for ordinary expectation values of a single unconditioned evolved
/// state, but must be tracked or reintroduced by controlled/interference-based
/// algorithms.
__qpu__ void apply_trotter(const std::vector<double> &coefficients,
                           const std::vector<cudaq::pauli_word> &words,
                           double time, std::size_t steps, int order,
                           cudaq::qview<> qubits);

} // namespace cudaq::solvers
