/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/solvers/hamiltonian_simulation/trotter.h"

#include <cmath>
#include <stdexcept>

namespace cudaq::solvers {

trotter_terms make_trotter_terms(const cudaq::spin_op &hamiltonian,
                                 double coefficient_tolerance) {
  if (coefficient_tolerance < 0.0)
    throw std::invalid_argument(
        "trotter error - coefficient tolerance must be non-negative.");

  trotter_terms terms;
  terms.num_qubits = hamiltonian.num_qubits();

  for (const auto &term : hamiltonian) {
    const auto coefficient = term.evaluate_coefficient();
    if (std::abs(coefficient.imag()) > coefficient_tolerance)
      throw std::invalid_argument(
          "trotter error - only real Hamiltonian coefficients are supported.");

    const auto real_coefficient = coefficient.real();
    if (term.is_identity()) {
      // Preserve identity contributions for callers that need the omitted
      // exp(-i c t) phase in controlled or interference-based algorithms.
      terms.identity_coefficient += real_coefficient;
      continue;
    }

    terms.coefficients.push_back(real_coefficient);
    terms.words.push_back(term.get_pauli_word(terms.num_qubits));
  }

  return terms;
}

__qpu__ void apply_trotter(const std::vector<double> &coefficients,
                           const std::vector<cudaq::pauli_word> &words,
                           double time, std::size_t steps, int order,
                           cudaq::qview<> qubits) {
  if (steps == 0 || coefficients.size() != words.size())
    return;

  if (order != 1 && order != 2 && order != 4)
    return;

  const double dt = time / static_cast<double>(steps);
  for (std::size_t step = 0; step < steps; ++step) {
    if (order == 1) {
      for (std::size_t i = 0; i < words.size(); ++i)
        exp_pauli(-dt * coefficients[i], qubits, words[i]);
    } else if (order == 4) {
      for (std::size_t i = 0; i < words.size(); ++i)
        exp_pauli(-0.5 * 1.3512071919596578 * dt * coefficients[i], qubits,
                  words[i]);
      for (std::size_t i = words.size(); i > 0; --i)
        exp_pauli(-0.5 * 1.3512071919596578 * dt * coefficients[i - 1], qubits,
                  words[i - 1]);

      for (std::size_t i = 0; i < words.size(); ++i)
        exp_pauli(-0.5 * -1.7024143839193153 * dt * coefficients[i], qubits,
                  words[i]);
      for (std::size_t i = words.size(); i > 0; --i)
        exp_pauli(-0.5 * -1.7024143839193153 * dt * coefficients[i - 1], qubits,
                  words[i - 1]);

      for (std::size_t i = 0; i < words.size(); ++i)
        exp_pauli(-0.5 * 1.3512071919596578 * dt * coefficients[i], qubits,
                  words[i]);
      for (std::size_t i = words.size(); i > 0; --i)
        exp_pauli(-0.5 * 1.3512071919596578 * dt * coefficients[i - 1], qubits,
                  words[i - 1]);
    } else {
      for (std::size_t i = 0; i < words.size(); ++i)
        exp_pauli(-0.5 * dt * coefficients[i], qubits, words[i]);
      for (std::size_t i = words.size(); i > 0; --i)
        exp_pauli(-0.5 * dt * coefficients[i - 1], qubits, words[i - 1]);
    }
  }
}

} // namespace cudaq::solvers
