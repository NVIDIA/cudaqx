/*******************************************************************************
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
// [Begin Documentation]
#include "cudaq.h"
#include "cudaq/solvers/operators.h"
#include "cudaq/solvers/qaoa.h"

// Compile and run with
// nvq++ --enable-mlir -lcudaq-solvers molecular_docking_qaoa.cpp
// ./a.out

int main() {

  // Create the ligand-configuration graph
  cudaqx::graph g;
  std::vector<double> weights{0.6686, 0.6686, 0.6686, 0.1453, 0.1453, 0.1453};
  std::vector<std::pair<int, int>> edges{{0, 1}, {0, 2}, {0, 4}, {0, 5},
                                         {1, 2}, {1, 3}, {1, 5}, {2, 3},
                                         {2, 4}, {3, 4}, {3, 5}, {4, 5}};
  for (std::size_t node = 0; auto weight : weights)
    g.add_node(node++, weight);

  for (auto &edge : edges)
    g.add_edge(edge.first, edge.second);

  // Set some parameters we'll need
  double penalty = 6.0;
  std::size_t numLayers = 3;

  // Create the Clique Hamiltonian
  auto H = cudaq::solvers::get_clique_hamiltonian(g, penalty);

  // Get the number of required variational parameters
  auto numParams = cudaq::solvers::get_num_qaoa_parameters(
      H, numLayers,
      {{"full_parameterization", true}, {"counterdiabatic", true}});

  // Create the initial parameters to begin optimization
  auto initParams = cudaq::random_vector(-M_PI / 8., M_PI / 8., numParams);

  // Run QAOA, specify full parameterization and counterdiabatic
  // Full parameterization uses an optimization parameter for
  // every term in the clique Hamiltonian and the mixer hamiltonian.
  // Specifying counterdiabatic adds extra Ry rotations at the
  // end of each layer.
  auto [opt_value, opt_params, opt_config] = cudaq::solvers::qaoa(
      H, numLayers, initParams,
      {{"full_parameterization", true}, {"counterdiabatic", true}});

  // Print out the results
  std::cout << "Optimal energy: " << opt_value << "\n";
  std::cout << "Sampled states: ";
  opt_config.dump();
  std::cout << "Optimal configuraiton: " << opt_config.most_probable() << "\n";
}