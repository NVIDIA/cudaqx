/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/solvers/operators/operator_pools/upccgsd_operator_pool.h"
#include "cudaq/solvers/operators/uccgsd_excitation_utils.h"

using namespace cudaqx;

namespace cudaq::solvers {

std::vector<cudaq::spin_op>
upccgsd::generate(const heterogeneous_map &config) const {

  // Number of spatial orbitals (same convention as UCCGSD operator pool)
  const std::size_t numOrbitals =
      config.get<std::size_t>({"num-orbitals", "num_orbitals"});
  if (numOrbitals == 0)
    throw std::invalid_argument("num-orbitals must be > 0");

  // Interleaved spin-orbitals: (α0, β0, α1, β1, ..., α_{N-1}, β_{N-1})
  const std::size_t numQubits = 2 * numOrbitals;

  std::vector<cudaq::spin_op> ops;

 
  // 1) Spin-preserving generalized singles on α and β
  //
  // We use generate_uccgsd_singles(numQubits) to get all unordered pairs
  // (p,q) with p > q over spin-orbitals, then restrict to same-spin pairs:
  //   - interleaved mapping means:
  //       even indices  -> α spin
  //       odd indices   -> β spin
  //   - spin-preserving => p % 2 == q % 2
  //
  // Each selected pair is mapped to a JW single-excitation operator via
  // addUCCGSDSingleExcitation.
  auto singles = generate_uccgsd_singles(numQubits);
  for (const auto &[p, q] : singles) {
    // keep only spin-preserving single excitations (α→α or β→β)
    if ((p % 2) == (q % 2)) {
      addUCCGSDSingleExcitation(ops, p, q);
    }
  }

  // 2) Paired generalized doubles (UpCC-style) in interleaved ordering
  //
  // UpCCGSD pair doubles correspond to exciting an αβ pair on spatial orbital p
  // to an αβ pair on spatial orbital q:
  //
  //   | pα, pβ >  ->  | qα, qβ >
  //
  // in a GENERALIZED sense ("G" in UpCCGSD): we consider all unordered pairs
  // of distinct spatial orbitals p < q.
  //
  // Interleaved spin-orbital indices:
  //   spatial orbital o:
  //     o_α -> 2*o
  //     o_β -> 2*o + 1
  //
  // For a given spatial pair (p, q), p < q:
  //   p_alpha = 2*p
  //   p_beta  = 2*p + 1
  //   q_alpha = 2*q
  //   q_beta  = 2*q + 1
  //
  // We then form two spin-orbital *pairs* to pass to addUCCGSDDoubleExcitation:
  //
  //   creation pair  (P, Q) = sorted(q_alpha, q_beta) with P > Q
  //   annihilation pair (R, S) = sorted(p_alpha, p_beta) with R > S
  //
  // This satisfies the helper's requirement p > q and r > s, and reuses
  // the same JW mapping as the UCCGSD double-excitation utility, but now
  // restricted to UpCC-style αβ pair doubles.

  for (std::size_t p = 0; p < numOrbitals; ++p) {
    for (std::size_t q = p + 1; q < numOrbitals; ++q) {
      // Interleaved spin-orbital indices for p and q
      const std::size_t p_alpha = 2 * p;
      const std::size_t p_beta  = 2 * p + 1;
      const std::size_t q_alpha = 2 * q;
      const std::size_t q_beta  = 2 * q + 1;

      const std::size_t P = q_beta;
      const std::size_t Qp = q_alpha;
      const std::size_t R = p_beta;
      const std::size_t S = p_alpha;

      // Use the same double-excitation JW mapping helper as UCCGSD,
      // but restricted to these UpCC-style pair indices.
      addUCCGSDDoubleExcitation(ops, P, Qp, R, S);
    }
  }

  return ops;
}

} // namespace cudaq::solvers
