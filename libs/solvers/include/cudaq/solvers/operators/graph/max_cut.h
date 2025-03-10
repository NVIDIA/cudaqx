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

/// @brief Generates a quantum Hamiltonian for the Maximum Cut problem
///
/// This function constructs a spin Hamiltonian whose ground state corresponds
/// to the maximum cut in the input graph. The Hamiltonian is constructed such
/// that its ground state represents a partition of the graph's vertices into
/// two sets that maximizes the sum of weights of edges crossing between the
/// sets.
///
/// The Hamiltonian takes the form:
/// H = Σ_{(i,j)∈E} w_{ij}/2(Z_iZ_j - I)
/// where:
/// - E is the set of edges in the graph
/// - w_{ij} is the weight of edge (i,j)
/// - Z_i is the Pauli Z operator on qubit i
/// - I is the identity operator
///
/// For an unweighted graph, all w_{ij} = 1.0
///
/// The resulting Hamiltonian has the following properties:
/// - Each qubit represents a vertex in the graph
/// - Z_i = +1 assigns vertex i to one partition
/// - Z_i = -1 assigns vertex i to the other partition
/// - The ground state energy corresponds to the negative of the maximum cut
/// value
///
/// @param graph The input graph to find the maximum cut in
/// @return cudaq::spin_op The quantum Hamiltonian for the MaxCut problem
///
/// @note The Hamiltonian is constructed to be symmetric under global spin flip,
///       reflecting the symmetry of the MaxCut problem under swapping the
///       partitions
cudaq::spin_op get_maxcut_hamiltonian(const cudaqx::graph &graph);
} // namespace cudaq::solvers