# ============================================================================ #
# Copyright (c) 2024 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# [Begin Documentation]
import cudaq, cudaq_solvers as solvers

# Run this script with
# python3 adapt_h2.py
#
# In order to leverage CUDA-Q MQPU and distribute the work across
# multiple QPUs (thereby observing a speed-up), set the target and
# use MPI:
#
# cudaq.set_target('nvidia', mqpu=True)
# cudaq.mpi.initialize()
#
# run with
#
# mpiexec -np N and vary N to see the speedup...
# e.g. mpiexec -np 2 python3 adapt_h2_mqpu.py
#
# End the script with
# cudaq.mpi.finalize()

# Create the molecular hamiltonian
geometry = [('H', (0., 0., 0.)), ('H', (0., 0., .7474))]
molecule = solvers.create_molecule(geometry, 'sto-3g', 0, 0, casci=True)

# Create the ADAPT operator pool
operators = solvers.get_operator_pool("spin_complement_gsd",
                                      num_orbitals=molecule.n_orbitals)

# Get the number of electrons so we can
# capture it in the initial state kernel
numElectrons = molecule.n_electrons


# Define the initial Hartree Fock state
@cudaq.kernel
def initState(q: cudaq.qview):
    for i in range(numElectrons):
        x(q[i])


# Run ADAPT-VQE
energy, thetas, ops = solvers.adapt_vqe(initState, molecule.hamiltonian,
                                        operators)

# Print the result.
print("<H> = ", energy)
