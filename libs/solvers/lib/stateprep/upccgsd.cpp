/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq/solvers/stateprep/upccgsd.h"
#include "cudaq/solvers/operators/uccgsd_excitation_utils.h"
#include <algorithm>
#include <tuple>
#include <stdexcept>

namespace cudaq::solvers::stateprep {

// Wrapper helpers that reuse the shared JW excitation utilities


static void addUpCCGSDSingleExcitation(std::vector<cudaq::spin_op> &ops,
                                       std::size_t p, std::size_t q) {
  // Reuse the UCCGSD single-excitation builder
  cudaq::solvers::addUCCGSDSingleExcitation(ops, p, q);
}

static void addUpCCGSDDoubleExcitation(std::vector<cudaq::spin_op> &ops,
                                       std::size_t p, std::size_t q,
                                       std::size_t r, std::size_t s) {
  // Reuse the UCCGSD double-excitation builder
  cudaq::solvers::addUCCGSDDoubleExcitation(ops, p, q, r, s);
}

// UpCCGSD unique singles: spin-preserving generalized singles
// norbitals is interpreted as the number of spin orbitals (i.e. qubits).
std::vector<std::pair<std::size_t, std::size_t>>
upccgsd_unique_singles(std::size_t norbitals) {
  // Start from all generalized singles over spin orbitals
  auto allSingles = cudaq::solvers::generate_uccgsd_singles(norbitals);

  std::vector<std::pair<std::size_t, std::size_t>> filtered;
  filtered.reserve(allSingles.size());

  for (auto [p, q] : allSingles) {
    // Interleaved mapping: even -> α, odd -> β
    // Keep only spin-preserving singles (α→α or β→β)
    if ((p % 2) == (q % 2))
      filtered.emplace_back(p, q);
  }

  return filtered;
}

// UpCCGSD unique doubles: paired αβ@p -> αβ@q generalized doubles
// norbitals is the number of spin orbitals (qubits) arranged interleaved as
// (α0, β0, α1, β1, ..., α_{N-1}, β_{N-1}), so N_spatial = norbitals / 2.
std::vector<std::pair<std::pair<std::size_t, std::size_t>,
                      std::pair<std::size_t, std::size_t>>>
upccgsd_unique_doubles(std::size_t norbitals) {
  if (norbitals % 2 != 0)
    throw std::invalid_argument(
        "upccgsd_unique_doubles expects an even number of spin orbitals.");

  const std::size_t numSpatialOrbitals = norbitals / 2;

  std::vector<std::pair<std::pair<std::size_t, std::size_t>,
                        std::pair<std::size_t, std::size_t>>>
      doubles;

  for (std::size_t p = 0; p < numSpatialOrbitals; ++p) {
    for (std::size_t q = p + 1; q < numSpatialOrbitals; ++q) {
      // Interleaved spin-orbital indices for spatial orbitals p and q
      const std::size_t p_alpha = 2 * p;
      const std::size_t p_beta  = 2 * p + 1;
      const std::size_t q_alpha = 2 * q;
      const std::size_t q_beta  = 2 * q + 1;

      // Sort each pair so that first > second, as required by the helper.
      std::size_t P, Qp;
      if (q_beta > q_alpha) {
        P  = q_beta;
        Qp = q_alpha;
      } else {
        P  = q_alpha;
        Qp = q_beta;
      }

      std::size_t R, S;
      if (p_beta > p_alpha) {
        R = p_beta;
        S = p_alpha;
      } else {
        R = p_alpha;
        S = p_beta;
      }

      doubles.push_back({{P, Qp}, {R, S}});
    }
  }

  return doubles;
}

// Python-style UpCCGSD pool: build Pauli-word lists + coefficients
// norbitals = # spin orbitals / qubits (same as uccgsd stateprep).
std::pair<std::vector<std::vector<cudaq::pauli_word>>,
          std::vector<std::vector<double>>>
get_upccgsd_pauli_lists(std::size_t norbitals, bool only_singles,
                        bool only_doubles) {

  std::vector<cudaq::spin_op> ops;

  if (!only_singles && !only_doubles) {
    // Add all UpCCGSD singles
    for (auto [p, q] : upccgsd_unique_singles(norbitals))
      addUpCCGSDSingleExcitation(ops, p, q);
    // Add all UpCCGSD paired doubles
    for (auto pair : upccgsd_unique_doubles(norbitals)) {
      auto [pq, rs] = pair;
      addUpCCGSDDoubleExcitation(ops, pq.first, pq.second, rs.first, rs.second);
    }
  } else if (only_singles) {
    for (auto [p, q] : upccgsd_unique_singles(norbitals))
      addUpCCGSDSingleExcitation(ops, p, q);
  } else if (only_doubles) {
    for (auto pair : upccgsd_unique_doubles(norbitals)) {
      auto [pq, rs] = pair;
      addUpCCGSDDoubleExcitation(ops, pq.first, pq.second, rs.first, rs.second);
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
    pauliWordsList.push_back(std::move(words));
    coefficientsList.push_back(std::move(coeffs));
  }

  return {pauliWordsList, coefficientsList};
}

// QPU kernel: exp(i θ_k H_k) with H_k expanded into Pauli words
__qpu__ void
upccgsd(cudaq::qview<> qubits, const std::vector<double> &thetas,
        const std::vector<std::vector<cudaq::pauli_word>> &pauliWordsList,
        const std::vector<std::vector<double>> &coefficientsList) {
  for (std::size_t i = 0; i < pauliWordsList.size(); ++i) {
    // Use the same theta for all terms in this excitation
    double theta = thetas[i];
    const auto &words = pauliWordsList[i];
    const auto &coeffs = coefficientsList[i];
    for (std::size_t j = 0; j < words.size(); ++j) {
      exp_pauli(theta * coeffs[j], qubits, words[j]);
    }
  }
}

} // namespace cudaq::solvers::stateprep
