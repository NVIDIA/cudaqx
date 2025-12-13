/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq/solvers/stateprep/ceo.h"
#include "cudaq/solvers/operators/operator_pools/ceo_operator_pool.h"
#include <algorithm>
#include <tuple>

namespace cudaq::solvers::stateprep {

// Pauli lists and coefficient lists for the CEO operator pool
// Each Pauli list / coefficient list is for a single CEO operator even
// though there are several operators for the same double excitation indices.
std::pair<std::vector<std::vector<cudaq::pauli_word>>,
          std::vector<std::vector<double>>>
get_ceo_pauli_lists(std::size_t norbitals) {

  std::vector<cudaq::spin_op> ops;
  std::size_t numQubits = 2 * norbitals;

  // Use ceo::generate to fill in the ops vector
  // to avoid code redundancy 
  cudaq::solvers::heterogeneous_map config;
  config.insert("num-orbitals", norbitals);
  ops = cudaq::solvers::ceo::generate(config);

  std::vector<std::vector<cudaq::pauli_word>> pauliWordsList;
  std::vector<std::vector<double>> coefficientsList;

  for (const auto &op : ops) {
    std::vector<cudaq::pauli_word> words;
    std::vector<double> coeffs;
    for (const auto &term : op) {
      words.push_back(term.get_pauli_word(numQubits));
      coeffs.push_back(term.evaluate_coefficient().real());
    }
    pauliWordsList.push_back(words);
    coefficientsList.push_back(coeffs);
  }

  return {pauliWordsList, coefficientsList};
}

// Note: The CEO paper (https://arxiv.org/abs/2407.08696) presents an 
// optimized implementation for the CEO doubles operators, that is 
// more efficient than the simple kernel below.
// TODO: Implement the optimized kernel.
__qpu__ void
ceo(cudaq::qview<> qubits, const std::vector<double> &thetas,
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


