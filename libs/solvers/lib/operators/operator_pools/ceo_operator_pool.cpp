/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq/solvers/operators/operator_pools/ceo_operator_pool.h"
#include "cudaq/solvers/operators/ceo_excitation_utils.h"

using namespace cudaqx;

namespace cudaq::solvers {

std::vector<cudaq::spin_op>
ceo::generate(const heterogeneous_map &config) const {

  const std::size_t numOrbitals = config.get<std::size_t>({"num-orbitals", "num_orbitals"});
  if (numOrbitals == 0)
    throw std::invalid_argument("num-orbitals must be > 0");

  std::vector<cudaq::spin_op> ops;

  // Similar to UCCSD, we generate the excitations in the following order:
  // single alpha, single beta, doubles mixed, doubles alpha, doubles beta

  // Generate alpha single excitations
  auto alpha_singles = generate_ceo_alpha_singles(numOrbitals);
  for (const auto &[p, q] : alpha_singles) {
    addCEOSingleExcitation(ops, p, q);
  }

  // Generate beta single excitations
  auto beta_singles = generate_ceo_beta_singles(numOrbitals);
  for (const auto &[p, q] : beta_singles) {
    addCEOSingleExcitation(ops, p, q);
  }

  // Generate mixed double excitations
  auto mixed_doubles = generate_ceo_mixed_doubles(numOrbitals);
  for (const auto &[p, q, r, s] : mixed_doubles) {
    addCEODoubleExcitation(ops, p, q, r, s);
  }

  // Generate alpha double excitations
  auto alpha_doubles = generate_ceo_alpha_doubles(numOrbitals);
  for (const auto &[p, q, r, s] : alpha_doubles) {
    addCEODoubleExcitation(ops, p, q, r, s);
  }

  // Generate beta double excitations
  auto beta_doubles = generate_ceo_beta_doubles(numOrbitals);
  for (const auto &[p, q, r, s] : beta_doubles) {
    addCEODoubleExcitation(ops, p, q, r, s);
  }

  return ops;
}

} // namespace cudaq::solvers


