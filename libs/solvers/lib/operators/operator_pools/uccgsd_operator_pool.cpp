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

      ops.emplace_back( cudaq::spin::y(q) * parity * cudaq::spin::x(p) -
                        cudaq::spin::x(q) * parity * cudaq::spin::y(p));
    }
  };

  auto addDoubleExcitation = [](std::vector<cudaq::spin_op> &ops,
                                std::size_t p, std::size_t q,
                                std::size_t r, std::size_t s) {
    if (p > q && q > r && r > s) {
      cudaq::spin_op_term parity_a, parity_b;
      for (std::size_t i = q + 1; i < p; ++i)
        parity_a *= cudaq::spin::z(i);
      for (std::size_t i = s + 1; i < r; ++i)
        parity_b *= cudaq::spin::z(i);

      cudaq::spin_op temp_op = cudaq::spin::y(s) * parity_b * cudaq::spin::x(r) * 
                                cudaq::spin::x(q) * parity_a * cudaq::spin::x(p);
      temp_op += cudaq::spin::x(s) * parity_b * cudaq::spin::y(r) * 
                cudaq::spin::x(q) * parity_a * cudaq::spin::x(p);
      temp_op += cudaq::spin::y(s) * parity_b * cudaq::spin::y(r) * 
                cudaq::spin::y(q) * parity_a * cudaq::spin::x(p);
      temp_op += cudaq::spin::y(s) * parity_b * cudaq::spin::y(r) * 
                cudaq::spin::x(q) * parity_a * cudaq::spin::y(p);
      temp_op -= cudaq::spin::x(s) * parity_b * cudaq::spin::x(r) * 
                cudaq::spin::y(q) * parity_a * cudaq::spin::x(p);
      temp_op -= cudaq::spin::x(s) * parity_b * cudaq::spin::x(r) * 
                cudaq::spin::x(q) * parity_a * cudaq::spin::y(p);
      temp_op -= cudaq::spin::x(s) * parity_b * cudaq::spin::y(r) * 
                cudaq::spin::y(q) * parity_a * cudaq::spin::y(p);
      temp_op -= cudaq::spin::y(s) * parity_b * cudaq::spin::x(r) * 
                cudaq::spin::y(q) * parity_a * cudaq::spin::y(p);

      ops.emplace_back(temp_op);
    }
  };

  // Generate all single excitations
  for (std::size_t p = 1; p < numQubits; ++p)
    for (std::size_t q = 0; q < p; ++q)
      addSingleExcitation(ops, p, q);

  // Generate all double excitations
  for (std::size_t p = 3; p < numQubits; ++p)
    for (std::size_t q = 2; q < p; ++q)
      for (std::size_t r = 1; r < q; ++r)
        for (std::size_t s = 0; s < r; ++s)
          addDoubleExcitation(ops, p, q, r, s);

  return ops;
}

} // namespace cudaq::solvers