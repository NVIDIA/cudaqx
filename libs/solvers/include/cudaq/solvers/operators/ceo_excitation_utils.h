/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/spin_op.h"
#include <utility>
#include <vector>

namespace cudaq::solvers {

/// @brief Generate all alpha single excitation index pairs (p, q) with p > q
/// @details Both p and q are alpha spin orbitals (even indices in interleaved
/// convention)
/// @param numOrbitals Number of spatial orbitals
/// @return Vector of (p, q) pairs representing alpha single excitations
std::vector<std::pair<std::size_t, std::size_t>>
generate_ceo_alpha_singles(std::size_t numOrbitals);

/// @brief Generate all beta single excitation index pairs (p, q) with p > q
/// @details Both p and q are beta spin orbitals (odd indices in interleaved
/// convention)
/// @param numOrbitals Number of spatial orbitals
/// @return Vector of (p, q) pairs representing beta single excitations
std::vector<std::pair<std::size_t, std::size_t>>
generate_ceo_beta_singles(std::size_t numOrbitals);

/// @brief Generate all alpha double excitation index tuples
/// @details All four indices are alpha spin orbitals. For p>q>r>s, generates
/// three pairings:
///          (p,q)->(r,s), (p,r)->(q,s), (q,p)->(r,s)
/// Note that the CEO operators have a special structure that is different from
/// either UCCSD or UCCGSD. We are following the index convention from the CEO
/// paper (https://arxiv.org/abs/2407.08696) which is a bit counterintuitive but
/// where (p,q) -> (r,s) generates a different excitation type from (q,p) ->
/// (r,s).
/// @param numOrbitals Number of spatial orbitals
/// @return Vector of (p, q, r, s) tuples representing alpha double excitations
std::vector<std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>>
generate_ceo_alpha_doubles(std::size_t numOrbitals);

/// @brief Generate all beta double excitation index tuples
/// @details All four indices are beta spin orbitals. For p>q>r>s, generates
/// three pairings:
///          (p,q)->(r,s), (p,r)->(q,s), (q,p)->(r,s)
/// See comments for the generate_ceo_alpha_doubles function for more details.
/// @param numOrbitals Number of spatial orbitals
/// @return Vector of (p, q, r, s) tuples representing beta double excitations
std::vector<std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>>
generate_ceo_beta_doubles(std::size_t numOrbitals);

/// @brief Generate all mixed double excitation index tuples
/// @details Mixed spin double excitations. Indices p,q,r,s
/// correspond to the following spins: alpha, beta, alpha, beta respectively ;
/// and we have p > r and q > s.
/// @param numOrbitals Number of spatial orbitals
/// @return Vector of (p, q, r, s) tuples representing mixed double excitations
std::vector<std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>>
generate_ceo_mixed_doubles(std::size_t numOrbitals);

/// @brief Add a CEO single excitation operator to the operator pool
/// @details Generates the operator: 0.5 * (Y_q X_p - X_q Y_p)
/// See the CEO paper (https://arxiv.org/abs/2407.08696) for more details.
/// @param ops Vector to append the operator to
/// @param p Higher qubit index (p > q)
/// @param q Lower qubit index (p > q)
void addCEOSingleExcitation(std::vector<cudaq::spin_op> &ops, std::size_t p,
                            std::size_t q);

/// @brief Add CEO double excitation operators to the operator pool
/// @details Generates two operators for indices (p, q, r, s):
///          A: 0.25 * (X_r X_p X_s Y_q - X_r X_p Y_s X_q + Y_r Y_p X_s Y_q -
///             Y_r Y_p Y_s X_q)
///
///          B: 0.25 * (X_r Y_p X_s X_q + X_r Y_p Y_s Y_q - Y_r X_p X_s X_q -
///             Y_r X_p Y_s Y_q)
///          These operators correspond to the OVP-CEO operators in the CEO
///          paper (https://arxiv.org/abs/2407.08696). They combine qubit
///          excitation operators Q_1 and Q_2 for the same 4 spin-orbitals as A
///          = Q_1 + Q_2 and B = Q_1 - Q_2.
/// @param ops Vector to append the operators to
/// @param p First qubit index
/// @param q Second qubit index
/// @param r Third qubit index
/// @param s Fourth qubit index
void addCEODoubleExcitation(std::vector<cudaq::spin_op> &ops, std::size_t p,
                            std::size_t q, std::size_t r, std::size_t s);

} // namespace cudaq::solvers
