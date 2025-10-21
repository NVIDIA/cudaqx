/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq/solvers/stateprep/uccgsd.h"
#include <set>
#include <tuple>
#include <algorithm>

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
  
  if (p > q && r > s) {
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

// Helper: unique unordered singles (p, q) with p > q
std::vector<std::pair<std::size_t, std::size_t>>
uccgsd_unique_singles(std::size_t norbitals) {
  std::vector<std::pair<std::size_t, std::size_t>> singles;
  for (std::size_t p = 0; p < norbitals; ++p)
    for (std::size_t q = 0; q < p; ++q)
      singles.emplace_back(p, q);
  return singles;
}

// Helper: unique unordered doubles ((p,q),(r,s)), pairs sorted, deduplicated
std::vector<std::pair<std::pair<std::size_t, std::size_t>,
                      std::pair<std::size_t, std::size_t>>>
uccgsd_unique_doubles(std::size_t norbitals) {
  std::set<std::pair<std::pair<std::size_t, std::size_t>,
                     std::pair<std::size_t, std::size_t>>> doubles;
  for (std::size_t a = 0; a < norbitals; ++a)
    for (std::size_t b = a + 1; b < norbitals; ++b)
      for (std::size_t c = b + 1; c < norbitals; ++c)
        for (std::size_t d = c + 1; d < norbitals; ++d) {
          std::array<std::size_t, 4> arr = {a, b, c, d};
          // All 3 unique pairings
          std::vector<std::pair<std::pair<std::size_t, std::size_t>,
                                std::pair<std::size_t, std::size_t>>> pairings = {
            {{arr[0], arr[1]}, {arr[2], arr[3]}},
            {{arr[0], arr[2]}, {arr[1], arr[3]}},
            {{arr[0], arr[3]}, {arr[1], arr[2]}}
          };
          for (auto &pairing : pairings) {
            auto p1 = pairing.first, p2 = pairing.second;
            if (p1.first < p1.second) std::swap(p1.first, p1.second);
            if (p2.first < p2.second) std::swap(p2.first, p2.second);
            auto sorted_pairing = std::minmax(p1, p2);
            doubles.insert({sorted_pairing.first, sorted_pairing.second});
          }
        }
  return std::vector<std::pair<std::pair<std::size_t, std::size_t>,
                               std::pair<std::size_t, std::size_t>>>(doubles.begin(), doubles.end());
}

// New function: Python-style unique singles/doubles pool
std::pair<std::vector<std::vector<cudaq::pauli_word>>,
          std::vector<std::vector<double>>>
get_uccgsd_pauli_lists(std::size_t norbitals,
                           bool only_singles,
                           bool only_doubles) {

  std::vector<cudaq::spin_op> ops;

  if (!only_singles && !only_doubles) {
    // Add all singles
    for (auto [p, q] : uccgsd_unique_singles(norbitals))
      addGeneralizedSingleExcitation(ops, p, q);
    // Add all doubles
    for (auto pair : uccgsd_unique_doubles(norbitals)) {
      auto [pq, rs] = pair;
      addGeneralizedDoubleExcitation(ops, pq.first, pq.second, rs.first, rs.second);
    }
  } else if (only_singles) {
    for (auto [p, q] : uccgsd_unique_singles(norbitals))
      addGeneralizedSingleExcitation(ops, p, q);
  } else if (only_doubles) {
    for (auto pair : uccgsd_unique_doubles(norbitals)) {
      auto [pq, rs] = pair;
      addGeneralizedDoubleExcitation(ops, pq.first, pq.second, rs.first, rs.second);
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

__qpu__ void uccgsd(cudaq::qview<> qubits,
                                   const std::vector<double>& thetas,
                                   const std::vector<std::vector<cudaq::pauli_word>>& pauliWordsList,
                                   const std::vector<std::vector<double>>& coefficientsList) {
  for (std::size_t i = 0; i < pauliWordsList.size(); ++i) {
    // Use the same theta for all terms in this group
    double theta = thetas[i];
    const auto& words = pauliWordsList[i];
    const auto& coeffs = coefficientsList[i];
    for (std::size_t j = 0; j < words.size(); ++j) {
      exp_pauli(theta * coeffs[j], qubits, words[j]);
    }
  }
}

} // namespace cudaq::solvers::stateprep