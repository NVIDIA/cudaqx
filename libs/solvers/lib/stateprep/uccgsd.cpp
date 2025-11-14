/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq/solvers/stateprep/uccgsd.h"
#include "cudaq/solvers/operators/uccgsd_excitation_utils.h"
#include <algorithm>
#include <tuple>

namespace cudaq::solvers::stateprep {

// Wrapper functions for backward compatibility with existing stateprep API
// These delegate to the shared utility functions

void addGeneralizedSingleExcitation(std::vector<cudaq::spin_op> &ops,
                                    std::size_t p, std::size_t q) {
  cudaq::solvers::addUCCGSDSingleExcitation(ops, p, q);
}

void addGeneralizedDoubleExcitation(std::vector<cudaq::spin_op> &ops,
                                    std::size_t p, std::size_t q, std::size_t r,
                                    std::size_t s) {
  cudaq::solvers::addUCCGSDDoubleExcitation(ops, p, q, r, s);
}

std::vector<std::pair<std::size_t, std::size_t>>
uccgsd_unique_singles(std::size_t norbitals) {
  return cudaq::solvers::generate_uccgsd_singles(norbitals);
}

std::vector<std::pair<std::pair<std::size_t, std::size_t>,
                      std::pair<std::size_t, std::size_t>>>
uccgsd_unique_doubles(std::size_t norbitals) {
  return cudaq::solvers::generate_uccgsd_doubles(norbitals);
}

// New function: Python-style unique singles/doubles pool
std::pair<std::vector<std::vector<cudaq::pauli_word>>,
          std::vector<std::vector<double>>>
get_uccgsd_pauli_lists(std::size_t norbitals, bool only_singles,
                       bool only_doubles) {

  std::vector<cudaq::spin_op> ops;

  if (!only_singles && !only_doubles) {
    // Add all singles
    for (auto [p, q] : uccgsd_unique_singles(norbitals))
      addGeneralizedSingleExcitation(ops, p, q);
    // Add all doubles
    for (auto pair : uccgsd_unique_doubles(norbitals)) {
      auto [pq, rs] = pair;
      addGeneralizedDoubleExcitation(ops, pq.first, pq.second, rs.first,
                                     rs.second);
    }
  } else if (only_singles) {
    for (auto [p, q] : uccgsd_unique_singles(norbitals))
      addGeneralizedSingleExcitation(ops, p, q);
  } else if (only_doubles) {
    for (auto pair : uccgsd_unique_doubles(norbitals)) {
      auto [pq, rs] = pair;
      addGeneralizedDoubleExcitation(ops, pq.first, pq.second, rs.first,
                                     rs.second);
    }
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

__qpu__ void
uccgsd(cudaq::qview<> qubits, const std::vector<double> &thetas,
       const std::vector<std::vector<cudaq::pauli_word>> &pauliWordsList,
       const std::vector<std::vector<double>> &coefficientsList) {
  for (std::size_t i = 0; i < pauliWordsList.size(); ++i) {
    // Use the same theta for all terms in this group
    double theta = thetas[i];
    const auto &words = pauliWordsList[i];
    const auto &coeffs = coefficientsList[i];
    for (std::size_t j = 0; j < words.size(); ++j) {
      exp_pauli(theta * coeffs[j], qubits, words[j]);
    }
  }
}

} // namespace cudaq::solvers::stateprep
