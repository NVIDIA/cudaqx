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
# When invoked by pytest, the test launches itself via mpiexec as a subprocess.
# When invoked by mpiexec, the __name__ == "__main__" block runs the actual
# MPI computation.

import os
import shutil
import subprocess

import pytest

_MPI_CMD = ["mpiexec", "--allow-run-as-root", "--oversubscribe"]


def _mpi_available():
    if shutil.which("mpiexec") is None:
        return False
    # Probe with 2 ranks to verify multi-rank MPI actually works
    # (catches missing PML transports, absent cudaq MPI plugin, etc.)
    try:
        rc = subprocess.run(_MPI_CMD + [
            "-np", "2", "python3", "-c",
            "import cudaq; cudaq.mpi.initialize(); "
            "cudaq.mpi.finalize()"
        ],
                            capture_output=True,
                            timeout=30)
        return rc.returncode == 0
    except subprocess.TimeoutExpired:
        return False


@pytest.mark.skipif(not _mpi_available(),
                    reason="mpiexec or cudaq MPI support not available")
@pytest.mark.parametrize("num_ranks", [2, 4])
def test_adapt_mpi(num_ranks):
    result = subprocess.run(_MPI_CMD +
                            ["-np", str(num_ranks), "python3", __file__],
                            capture_output=True,
                            text=True,
                            timeout=120)
    assert result.returncode == 0, \
        (f"MPI test failed (np={num_ranks}):\n"
         f"--- stdout ---\n{result.stdout}\n"
         f"--- stderr ---\n{result.stderr}")


if __name__ == "__main__":
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
