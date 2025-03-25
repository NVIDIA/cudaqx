/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/solvers/operators/graph/max_cut.h"

namespace cudaq::solvers {

cudaq::spin_op get_maxcut_hamiltonian(const cudaqx::graph &graph) {
  // Get all nodes to iterate through edges
  auto nodes = graph.get_nodes();
  if (nodes.empty())
    return cudaq::spin_op();

  // Initialize empty spin operator
  cudaq::spin_op hamiltonian;

  // Iterate through all nodes
  for (const auto &u : nodes) {
    // Get neighbors of current node
    auto neighbors = graph.get_neighbors(u);

    // For each neighbor v where v > u to avoid counting edges twice
    for (const auto &v : neighbors) {
      if (v > u) {
        // Get the weight for this edge
        double weight = graph.get_edge_weight(u, v);

        // For each weighted edge (u,v), add w/2(Z_u Z_v - I) to the Hamiltonian
        // This matches the mathematical form: H = Σ w_ij/2(Z_i Z_j - I)
        hamiltonian +=
            weight * 0.5 * (cudaq::spin::z(u) * cudaq::spin::z(v) - 1.0);
      }
    }
  }

  return hamiltonian.canonicalize().trim();
}

} // namespace cudaq::solvers