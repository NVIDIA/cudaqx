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

/// @brief Generate all unique single excitation index pairs (p, q) with p > q
/// @param numQubits Number of qubits in the system
/// @return Vector of (p, q) pairs representing single excitations
std::vector<std::pair<std::size_t, std::size_t>>
generate_uccgsd_singles(std::size_t numQubits);

/// @brief Generate all unique double excitation index pairs
/// @details For 4 indices a < b < c < d, generates all 3 unique pairings:
///          (a,b)-(c,d), (a,c)-(b,d), (a,d)-(b,c)
///          Each pair is normalized so that within each pair, first > second,
///          and pairs are ordered.
/// @param numQubits Number of qubits in the system
/// @return Vector of ((p,q), (r,s)) pairs where p>q, r>s, representing double
/// excitations
std::vector<std::pair<std::pair<std::size_t, std::size_t>,
                      std::pair<std::size_t, std::size_t>>>
generate_uccgsd_doubles(std::size_t numQubits);

/// @brief Add a UCCGSD single excitation operator to the operator pool
/// @details Generates the operator: 0.5 * (Y_q * Z_parity * X_p - X_q *
/// Z_parity * Y_p)
///          where Z_parity is the product of Z operators between q and p
/// @param ops Vector to append the operator to
/// @param p Higher qubit index (p > q)
/// @param q Lower qubit index (p > q)
void addUCCGSDSingleExcitation(std::vector<cudaq::spin_op> &ops, std::size_t p,
                               std::size_t q);

/// @brief Add a UCCGSD double excitation operator to the operator pool
/// @details Generates the generalized double excitation operator with 8 terms
/// @param ops Vector to append the operator to
/// @param p First qubit index (p > q)
/// @param q Second qubit index (p > q)
/// @param r Third qubit index (r > s)
/// @param s Fourth qubit index (r > s)
void addUCCGSDDoubleExcitation(std::vector<cudaq::spin_op> &ops, std::size_t p,
                               std::size_t q, std::size_t r, std::size_t s);

} // namespace cudaq::solvers
