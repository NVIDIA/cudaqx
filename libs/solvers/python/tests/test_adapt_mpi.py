# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# ADAPT-VQE MPI work-splitting test.
# Verifies that ADAPT-VQE produces the correct H2 ground-state energy when
# commutator evaluation is distributed across multiple MPI ranks.
#
# Run: mpiexec -np 4 python test_adapt_mpi.py

import sys
import numpy as np
import cudaq
import cudaq_solvers as solvers

cudaq.mpi.initialize()

geometry = [('H', (0., 0., 0.)), ('H', (0., 0., .7474))]
molecule = solvers.create_molecule(geometry, 'sto-3g', 0, 0, casci=True)
operators = solvers.get_operator_pool("spin_complement_gsd",
                                      num_orbitals=molecule.n_orbitals)
numElectrons = molecule.n_electrons


@cudaq.kernel
def initState(q: cudaq.qview):
    for i in range(numElectrons):
        x(q[i])


energy, thetas, ops = solvers.adapt_vqe(initState, molecule.hamiltonian,
                                        operators)

rank = cudaq.mpi.rank()
num_ranks = cudaq.mpi.num_ranks()

if rank == 0:
    print(f"[MPI ADAPT] ranks={num_ranks}, energy={energy:.6f}")
    assert np.isclose(energy, -1.137, atol=1e-3), \
        f"MPI ADAPT energy {energy} does not match expected -1.137"
    assert len(ops) > 0, "Expected at least one operator to be selected"
    print("PASS")

cudaq.mpi.finalize()
