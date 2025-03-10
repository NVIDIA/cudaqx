/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/solvers/operators/graph/clique.h"

namespace cudaq::solvers {

cudaq::spin_op get_clique_hamiltonian(const cudaqx::graph &graph,
                                      double penalty) {
  // Get all nodes
  auto nodes = graph.get_nodes();
  if (nodes.empty())
    return cudaq::spin_op::identity();

  // Initialize empty spin operator
  cudaq::spin_op hamiltonian(nodes.size());

  // First term: Sum over all nodes
  for (const auto &node : nodes) {
    // Get node degree for weight calculation
    double weight = graph.get_node_weight(node);

    // Add 0.5 * weight * (Z_i - I)
    hamiltonian += 0.5 * weight *
                   (cudaq::spin_op::z(node) - cudaq::spin_op::i(nodes.size() - 1));
  }

  // Second term: Sum over non-edges
  // Get disconnected vertex pairs (non-edges)
  auto non_edges = graph.get_disconnected_vertices();

  // Add penalty terms for non-edges
  for (const auto &non_edge : non_edges) {
    int u = non_edge.first;
    int v = non_edge.second;

    // Add penalty/4 * (Z_u Z_v - Z_u - Z_v + I)
    hamiltonian += penalty / 4.0 *
                   (cudaq::spin_op::z(u) * cudaq::spin_op::z(v) - cudaq::spin_op::z(u) -
                    cudaq::spin_op::z(v) + cudaq::spin_op::i(nodes.size() - 1));
  }

  return hamiltonian - cudaq::spin_op(nodes.size() - 1);
}

} // namespace cudaq::solvers