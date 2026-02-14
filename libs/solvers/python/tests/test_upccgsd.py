import os

import pytest
import numpy as np

import cudaq
import cudaq_solvers as solvers
from scipy.optimize import minimize
import subprocess


def is_nvidia_gpu_available():
    """Check if NVIDIA GPU is available using nvidia-smi command."""
    try:
        result = subprocess.run(["nvidia-smi"],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True)
        if result.returncode == 0 and "GPU" in result.stdout:
            return True
    except FileNotFoundError:
        # The nvidia-smi command is not found, indicating no NVIDIA GPU drivers
        return False
    return False


def test_solvers_upccgsd_exc_list():
    N = 20  # or whatever molecule.n_orbitals * 2 would be
    pauliWordsList, coefficientsList = solvers.stateprep.get_upccgsd_pauli_lists(
        N, only_doubles=False)
    parameter_count = len(coefficientsList)
    M = N / 2
    ideal_count = (3 / 2) * M * (M - 1)
    assert parameter_count == ideal_count
    pauliWordsList, coefficientsList = solvers.stateprep.get_upccgsd_pauli_lists(
        N, only_doubles=True)
    parameter_count = len(coefficientsList)
    M = N / 2
    ideal_count = (1 / 2) * M * (M - 1)
    assert parameter_count == ideal_count


def test_solvers_vqe_upccgsd_h2():

    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., .7474))]
    molecule = solvers.create_molecule(geometry, 'sto-3g', 0, 0, casci=True)

    numQubits = molecule.n_orbitals * 2
    numElectrons = molecule.n_electrons

    # Get grouped Pauli words and coefficients from UpCCGSD pool
    pauliWordsList, coefficientsList = solvers.stateprep.get_upccgsd_pauli_lists(
        numQubits, only_doubles=False)

    # Number of theta parameters = number of excitation groups
    parameter_count = len(coefficientsList)
    assert parameter_count == 3

    @cudaq.kernel
    def ansatz(numQubits: int, numElectrons: int, thetas: list[float],
               pauliWordsList: list[list[cudaq.pauli_word]],
               coefficientsList: list[list[float]]):
        q = cudaq.qvector(numQubits)
        for i in range(numElectrons):
            x(q[i])
        # Apply UpCCGSD circuit with grouped thetas
        solvers.stateprep.upccgsd(q, thetas, pauliWordsList, coefficientsList)

    x0 = [0.0 for _ in range(parameter_count)]

    def cost(theta):

        theta = theta.tolist()

        energy = cudaq.observe(ansatz, molecule.hamiltonian, numQubits,
                               numElectrons, theta, pauliWordsList,
                               coefficientsList).expectation()
        return energy

    res = minimize(cost,
                   x0,
                   method='COBYLA',
                   options={
                       'maxiter': 1000,
                       'rhobeg': 0.1,
                       'disp': True
                   })
    energy = res.fun
    print(energy)
    assert np.isclose(energy, -1.1371, atol=1e-4)


@pytest.mark.skip(reason="PYTHON-REFACTOR")
def test_solvers_adapt_upccgsd_lih():
    geometry = [('Li', (0.3925, 0., 0.)), ('H', (-1.1774, 0., .0))]
    molecule = solvers.create_molecule(geometry,
                                       'sto-3g',
                                       0,
                                       0,
                                       nele_cas=4,
                                       norb_cas=4,
                                       casci=True)

    operators = solvers.get_operator_pool("upccgsd",
                                          num_orbitals=molecule.n_orbitals)

    numElectrons = molecule.n_electrons

    @cudaq.kernel
    def initState(q: cudaq.qview):
        for i in range(numElectrons):
            x(q[i])

    energy, thetas, ops = solvers.adapt_vqe(initState, molecule.hamiltonian,
                                            operators)

    print(energy)
    assert np.isclose(energy, -7.8638, atol=1e-4)
