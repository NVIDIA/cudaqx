/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cuda-qx/core/graph.h"
#include "cudaq/spin_op.h"

namespace cudaq::solvers {

/// @brief Generates a quantum Hamiltonian for the Maximum Clique problem
///
/// This function constructs a spin Hamiltonian whose ground state corresponds
/// to the maximum clique in the input graph. The Hamiltonian consists of two
/// terms:
/// 1. A node term that rewards including nodes in the clique
/// 2. A penalty term that enforces the clique constraint (all selected nodes
/// must be connected)
///
/// The Hamiltonian takes the form:
/// H = Σ_i w_i/2(Z_i - I) + p/4 Σ_{(i,j) ∉ E} (Z_iZ_j - Z_i - Z_j + I)
/// where:
/// - w_i is the weight of node i
/// - p is the penalty strength
/// - E is the set of edges in the graph
///
/// @param graph The input graph to find the maximum clique in
/// @param penalty The penalty strength for violating clique constraints
/// (default: 4.0)
/// @return cudaq::spin_op The quantum Hamiltonian for the Maximum Clique
/// problem
///
/// @note The penalty parameter should be chosen large enough to ensure that
/// invalid
///       solutions (non-cliques) have higher energy than valid solutions
cudaq::spin_op get_clique_hamiltonian(const cudaqx::graph &graph,
                                      double penalty = 4.0);
} // namespace cudaq::solvers