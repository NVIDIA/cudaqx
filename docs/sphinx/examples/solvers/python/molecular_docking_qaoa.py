# ============================================================================ #
# Copyright (c) 2024 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import cudaq, cudaq_solvers as solvers
import networkx as nx, numpy as np

# Create the ligand-configuration graph
G = nx.Graph()
edges = [[0, 1], [0, 2], [0, 4], [0, 5], [1, 2], [1, 3], [1, 5], [2, 3], [2, 4],
         [3, 4], [3, 5], [4, 5]]
weights = [0.6686, 0.6686, 0.6686, 0.1453, 0.1453, 0.1453]
for i, weight in enumerate(weights):
    G.add_node(i, weight=weight)
G.add_edges_from(edges)

# Set some parameters we'll need
penalty = 6.0
num_layers = 3

# Create the Clique Hamiltonian
H = solvers.get_clique_hamiltonian(G, penalty=penalty)

# Get the number of parameters we'll need
parameter_count = solvers.get_num_qaoa_parameters(H,
                                                  num_layers,
                                                  full_parameterization=True,
                                                  counterdiabatic=True)

# Create the initial parameters to begin optimization
init_params = np.random.uniform(-np.pi / 8, np.pi / 8, parameter_count)

# Run QAOA, specify full parameterization and counterdiabatic
# Full parameterization uses an optimization parameter for
# every term in the clique Hamiltonian and the mixer hamiltonian.
# Specifying counterdiabatic adds extra Ry rotations at the
# end of each layer.
opt_value, opt_params, opt_config = solvers.qaoa(H,
                                                 num_layers,
                                                 init_params,
                                                 full_parameterization=True,
                                                 counterdiabatic=True)

# Print the results
print()
print('Optimal energy: ', opt_value)
print('Sampled states: ', opt_config)
print('Optimal Configuration: ', opt_config.most_probable())
