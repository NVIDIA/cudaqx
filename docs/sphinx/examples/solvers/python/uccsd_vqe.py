# ============================================================================ #
# Copyright (c) 2024 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import cudaq, cudaq_solvers as solvers
from scipy.optimize import minimize

# Create the molecular hamiltonian
geometry = [('H', (0., 0., 0.)), ('H', (0., 0., .7474))]
molecule = solvers.create_molecule(geometry, 'sto-3g', 0, 0, casci=True)

# Get the number of qubits and electrons
numQubits = molecule.n_orbitals * 2
numElectrons = molecule.n_electrons
spin = 0
initialX = [-.2] * solvers.stateprep.get_num_uccsd_parameters(
    numElectrons, numQubits)


# Define the UCCSD ansatz
@cudaq.kernel
def ansatz(thetas: list[float]):
    q = cudaq.qvector(numQubits)
    for i in range(numElectrons):
        x(q[i])
    solvers.stateprep.uccsd(q, thetas, numElectrons, spin)


# Run VQE
energy, params, all_data = solvers.vqe(ansatz,
                                       molecule.hamiltonian,
                                       initialX,
                                       optimizer=minimize,
                                       method='L-BFGS-B',
                                       jac='3-point',
                                       tol=1e-4,
                                       options={'disp': True})
print(f'Final <H> = {energy}')
