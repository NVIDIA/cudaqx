/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq/solvers/operators/operator_pools/uccgsd_operator_pool.h"
#include "cudaq/solvers/operators/uccgsd_excitation_utils.h"

using namespace cudaqx;

namespace cudaq::solvers {

std::vector<cudaq::spin_op>
uccgsd::generate(const heterogeneous_map &config) const {

  auto numOrbitals = config.get<std::size_t>({"num-orbitals", "num_orbitals"});
  auto numQubits = 2 * numOrbitals;
  if (numOrbitals == 0)
    throw std::invalid_argument("num-orbitals must be > 0");

  // For UCCGSD, we do not use alpha/beta/mixed excitations, but generate all
  // singles and doubles
  std::vector<cudaq::spin_op> ops;

  // Generate all single excitations using shared utility
  auto singles = generate_uccgsd_singles(numQubits);
  for (const auto &[p, q] : singles) {
    addUCCGSDSingleExcitation(ops, p, q);
  }

  // Generate all unique unordered double excitations using shared utility
  auto doubles = generate_uccgsd_doubles(numQubits);
  for (const auto &[pq, rs] : doubles) {
    addUCCGSDDoubleExcitation(ops, pq.first, pq.second, rs.first, rs.second);
  }

  return ops;
}

} // namespace cudaq::solvers
