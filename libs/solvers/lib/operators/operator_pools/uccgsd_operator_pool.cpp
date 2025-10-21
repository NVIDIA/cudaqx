/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq/solvers/operators/operator_pools/uccgsd_operator_pool.h"

using namespace cudaqx;

namespace cudaq::solvers {

std::vector<cudaq::spin_op>
uccgsd::generate(const heterogeneous_map &config) const {

  auto numOrbitals = config.get<std::size_t>({"num-orbitals", "num_orbitals"});

  auto numQubits = 2 * numOrbitals;

  // For UCCGSD, we do not use alpha/beta/mixed excitations, but generate all singles and doubles
  std::vector<cudaq::spin_op> ops;

  auto addSingleExcitation = [](std::vector<cudaq::spin_op> &ops,
                                std::size_t p, std::size_t q) {
    if (p > q) {
      cudaq::spin_op_term parity;
      for (std::size_t i = q + 1; i < p; ++i)
        parity *= cudaq::spin::z(i);
      std::complex<double> c = {0.5, 0};
      ops.emplace_back(c * cudaq::spin::y(q) * parity * cudaq::spin::x(p) -
                       c * cudaq::spin::x(q) * parity * cudaq::spin::y(p));
    }
  };

  auto addDoubleExcitation = [](std::vector<cudaq::spin_op> &ops,
                                std::size_t p, std::size_t q,
                                std::size_t r, std::size_t s) {
    
    if (p > q && r > s) {
      cudaq::spin_op_term parity_a, parity_b;
      for (std::size_t i = q + 1; i < p; ++i)
        parity_a *= cudaq::spin::z(i);
      for (std::size_t i = s + 1; i < r; ++i)
        parity_b *= cudaq::spin::z(i);
      std::complex<double> c = {0.125, 0};
      cudaq::spin_op temp_op = c * cudaq::spin::y(s) * parity_b * cudaq::spin::x(r) *
                                cudaq::spin::x(q) * parity_a * cudaq::spin::x(p);
      temp_op += c * cudaq::spin::x(s) * parity_b * cudaq::spin::y(r) *
                 cudaq::spin::x(q) * parity_a * cudaq::spin::x(p);
      temp_op += c * cudaq::spin::y(s) * parity_b * cudaq::spin::y(r) *
                 cudaq::spin::y(q) * parity_a * cudaq::spin::x(p);
      temp_op += c * cudaq::spin::y(s) * parity_b * cudaq::spin::y(r) *
                 cudaq::spin::x(q) * parity_a * cudaq::spin::y(p);
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
  };

  // Generate all single excitations
  for (std::size_t p = 1; p < numQubits; ++p)
    for (std::size_t q = 0; q < p; ++q)
      addSingleExcitation(ops, p, q);

  // Generate all unique unordered double excitations
  std::set<std::pair<std::pair<std::size_t, std::size_t>,
                     std::pair<std::size_t, std::size_t>>> doubles;
  for (std::size_t a = 0; a < numQubits; ++a)
    for (std::size_t b = a + 1; b < numQubits; ++b)
      for (std::size_t c = b + 1; c < numQubits; ++c)
        for (std::size_t d = c + 1; d < numQubits; ++d) {
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
  for (const auto &pair : doubles) {
    auto [pq, rs] = pair;
    addDoubleExcitation(ops, pq.first, pq.second, rs.first, rs.second);
  }

  return ops;
}

} // namespace cudaq::solvers