/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq/solvers/stateprep/uccgsd.h"

namespace cudaq::solvers::stateprep {

// Helper for single excitation operator pool
void addGeneralizedSingleExcitation(std::vector<cudaq::spin_op> &ops, std::size_t p, std::size_t q) {
  if (p > q) {
    cudaq::spin_op_term parity;
    for (std::size_t i = q + 1; i < p; ++i)
      parity *= cudaq::spin::z(i);
    std::complex<double> c = {0.5, 0};
    ops.emplace_back(-c * cudaq::spin::x(q) * parity * cudaq::spin::y(p) +
                     c * cudaq::spin::y(q) * parity * cudaq::spin::x(p));
  }
}

// Helper for double excitation operator pool
void addGeneralizedDoubleExcitation(std::vector<cudaq::spin_op> &ops, std::size_t p, std::size_t q, std::size_t r, std::size_t s) {
  if (p > q && q > r && r > s) {
    cudaq::spin_op_term parity_a, parity_b;
    for (std::size_t i = q + 1; i < p; ++i)
      parity_a *= cudaq::spin::z(i);
    for (std::size_t i = s + 1; i < r; ++i)
      parity_b *= cudaq::spin::z(i);
    std::complex<double> c = {1.0 / 8.0, 0};
    cudaq::spin_op op_term_temp =
        c * cudaq::spin::y(s) * parity_b * cudaq::spin::x(r) * cudaq::spin::x(q) * parity_a * cudaq::spin::x(p);
    op_term_temp += c * cudaq::spin::x(s) * parity_b * cudaq::spin::y(r) * cudaq::spin::x(q) * parity_a * cudaq::spin::x(p);
    op_term_temp += c * cudaq::spin::y(s) * parity_b * cudaq::spin::y(r) * cudaq::spin::y(q) * parity_a * cudaq::spin::x(p);
    op_term_temp += c * cudaq::spin::y(s) * parity_b * cudaq::spin::y(r) * cudaq::spin::x(q) * parity_a * cudaq::spin::y(p);
    op_term_temp -= c * cudaq::spin::x(s) * parity_b * cudaq::spin::x(r) * cudaq::spin::y(q) * parity_a * cudaq::spin::x(p);
    op_term_temp -= c * cudaq::spin::x(s) * parity_b * cudaq::spin::x(r) * cudaq::spin::x(q) * parity_a * cudaq::spin::y(p);
    op_term_temp -= c * cudaq::spin::x(s) * parity_b * cudaq::spin::y(r) * cudaq::spin::y(q) * parity_a * cudaq::spin::y(p);
    op_term_temp -= c * cudaq::spin::y(s) * parity_b * cudaq::spin::x(r) * cudaq::spin::y(q) * parity_a * cudaq::spin::y(p);
    ops.emplace_back(op_term_temp);
  }
}

std::pair<std::vector<std::vector<cudaq::pauli_word>>,
          std::vector<std::vector<double>>>
get_uccgsd_pauli_lists(std::size_t nelectrons,
                       std::size_t norbitals,
                       bool only_singles,
                       bool only_doubles) {

  std::vector<cudaq::spin_op> ops;
  // Generate operator pool
  if (!only_singles && !only_doubles) {
    for (std::size_t p = 1; p < norbitals; ++p)
      for (std::size_t q = 0; q < p; ++q)
        addGeneralizedSingleExcitation(ops, p, q);
    for (std::size_t p = 3; p < norbitals; ++p)
      for (std::size_t q = 2; q < p; ++q)
        for (std::size_t r = 1; r < q; ++r)
          for (std::size_t s = 0; s < r; ++s)
            addGeneralizedDoubleExcitation(ops, p, q, r, s);
  } else if (only_singles) {
    for (std::size_t p = 1; p < norbitals; ++p)
      for (std::size_t q = 0; q < p; ++q)
        addGeneralizedSingleExcitation(ops, p, q);
  } else if (only_doubles) {
    for (std::size_t p = 3; p < norbitals; ++p)
      for (std::size_t q = 2; q < p; ++q)
        for (std::size_t r = 1; r < q; ++r)
          for (std::size_t s = 0; s < r; ++s)
            addGeneralizedDoubleExcitation(ops, p, q, r, s);
  }

  std::vector<std::vector<cudaq::pauli_word>> pauliWordsList;
  std::vector<std::vector<double>> coefficientsList;

  for (const auto &op : ops) {
    std::vector<cudaq::pauli_word> words;
    std::vector<double> coeffs;
    for (const auto &term : op) {
      words.push_back(term.get_pauli_word(norbitals));
      coeffs.push_back(term.evaluate_coefficient().real());
    }
    pauliWordsList.push_back(words);
    coefficientsList.push_back(coeffs);
  }

  return {pauliWordsList, coefficientsList};
}

__qpu__ void uccgsd(cudaq::qview<> qubits,
                                   const std::vector<double>& thetas,
                                   const std::vector<std::vector<cudaq::pauli_word>>& pauliWordsList) {
  for (std::size_t i = 0; i < pauliWordsList.size(); ++i) {
    // Use the same theta for all terms in this group
    double theta = thetas[i];
    const auto& words = pauliWordsList[i];
    for (std::size_t j = 0; j < words.size(); ++j) {
      exp_pauli(theta , qubits, words[j]);
    }
  }
}

} // namespace cudaq::solvers::stateprep