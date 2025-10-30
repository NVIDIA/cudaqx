/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/solvers/operators/uccgsd_excitation_utils.h"
#include <algorithm>
#include <array>
#include <set>

namespace cudaq::solvers {

std::vector<std::pair<std::size_t, std::size_t>>
generate_uccgsd_singles(std::size_t numQubits) {
  std::vector<std::pair<std::size_t, std::size_t>> singles;
  singles.reserve(numQubits * (numQubits - 1) / 2);
  for (std::size_t p = 1; p < numQubits; ++p)
    for (std::size_t q = 0; q < p; ++q)
      singles.emplace_back(p, q);
  return singles;
}

std::vector<std::pair<std::pair<std::size_t, std::size_t>,
                      std::pair<std::size_t, std::size_t>>>
generate_uccgsd_doubles(std::size_t numQubits) {
  std::set<std::pair<std::pair<std::size_t, std::size_t>,
                     std::pair<std::size_t, std::size_t>>>
      doubles;

  // Iterate over all combinations of 4 distinct qubits
  for (std::size_t a = 0; a < numQubits; ++a)
    for (std::size_t b = a + 1; b < numQubits; ++b)
      for (std::size_t c = b + 1; c < numQubits; ++c)
        for (std::size_t d = c + 1; d < numQubits; ++d) {
          std::array<std::size_t, 4> arr = {a, b, c, d};

          // Generate all 3 unique pairings of the 4 indices
          std::vector<std::pair<std::pair<std::size_t, std::size_t>,
                                std::pair<std::size_t, std::size_t>>>
              pairings = {{{arr[0], arr[1]}, {arr[2], arr[3]}},
                          {{arr[0], arr[2]}, {arr[1], arr[3]}},
                          {{arr[0], arr[3]}, {arr[1], arr[2]}}};

          // Normalize and deduplicate each pairing
          for (auto &pairing : pairings) {
            auto p1 = pairing.first, p2 = pairing.second;

            // Ensure within each pair: first > second
            if (p1.first < p1.second)
              std::swap(p1.first, p1.second);
            if (p2.first < p2.second)
              std::swap(p2.first, p2.second);

            // Order the two pairs
            auto sorted_pairing = std::minmax(p1, p2);
            doubles.insert({sorted_pairing.first, sorted_pairing.second});
          }
        }

  return std::vector<std::pair<std::pair<std::size_t, std::size_t>,
                               std::pair<std::size_t, std::size_t>>>(
      doubles.begin(), doubles.end());
}

void addUCCGSDSingleExcitation(std::vector<cudaq::spin_op> &ops,
                               std::size_t p, std::size_t q) {
  if (p > q) {
    // Compute parity string (Z operators between q and p)
    cudaq::spin_op_term parity;
    for (std::size_t i = q + 1; i < p; ++i)
      parity *= cudaq::spin::z(i);

    std::complex<double> c = {0.5, 0.0};

    // Single excitation: Y_q * Z_parity * X_p - X_q * Z_parity * Y_p
    ops.emplace_back(c * cudaq::spin::y(q) * parity * cudaq::spin::x(p) -
                     c * cudaq::spin::x(q) * parity * cudaq::spin::y(p));
  }
}

void addUCCGSDDoubleExcitation(std::vector<cudaq::spin_op> &ops,
                               std::size_t p, std::size_t q, std::size_t r,
                               std::size_t s) {
  if (p > q && r > s) {
    // Compute parity strings
    cudaq::spin_op_term parity_a, parity_b;
    for (std::size_t i = q + 1; i < p; ++i)
      parity_a *= cudaq::spin::z(i);
    for (std::size_t i = s + 1; i < r; ++i)
      parity_b *= cudaq::spin::z(i);

    std::complex<double> c = {0.125, 0.0};

    // Build the 8-term double excitation operator
    cudaq::spin_op temp_op;

    // Positive terms
    temp_op = c * cudaq::spin::y(s) * parity_b * cudaq::spin::x(r) *
              cudaq::spin::x(q) * parity_a * cudaq::spin::x(p);
    temp_op += c * cudaq::spin::x(s) * parity_b * cudaq::spin::y(r) *
               cudaq::spin::x(q) * parity_a * cudaq::spin::x(p);
    temp_op += c * cudaq::spin::y(s) * parity_b * cudaq::spin::y(r) *
               cudaq::spin::y(q) * parity_a * cudaq::spin::x(p);
    temp_op += c * cudaq::spin::y(s) * parity_b * cudaq::spin::y(r) *
               cudaq::spin::x(q) * parity_a * cudaq::spin::y(p);

    // Negative terms
    temp_op -= c * cudaq::spin::x(s) * parity_b * cudaq::spin::x(r) *
               cudaq::spin::y(q) * parity_a * cudaq::spin::x(p);
    temp_op -= c * cudaq::spin::x(s) * parity_b * cudaq::spin::x(r) *
               cudaq::spin::x(q) * parity_a * cudaq::spin::y(p);
    temp_op -= c * cudaq::spin::x(s) * parity_b * cudaq::spin::y(r) *
               cudaq::spin::y(q) * parity_a * cudaq::spin::y(p);
    temp_op -= c * cudaq::spin::y(s) * parity_b * cudaq::spin::x(r) *
               cudaq::spin::y(q) * parity_a * cudaq::spin::y(p);

    ops.emplace_back(temp_op);
  }
}

} // namespace cudaq::solvers

