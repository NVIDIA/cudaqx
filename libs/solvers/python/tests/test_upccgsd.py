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
        N, only_doubles=False
    )
    parameter_count = len(coefficientsList)
    M = N/2
    ideal_count = (3/2) * M * (M-1)
    assert parameter_count == ideal_count
    pauliWordsList, coefficientsList = solvers.stateprep.get_upccgsd_pauli_lists(
        N, only_doubles=True
    )
    parameter_count = len(coefficientsList)
    M = N/2
    ideal_count = (1/2) * M * (M-1)
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

    hf_energy = molecule.energies.get("HF", None)

    if hf_energy is not None:
      assert energy <= hf_energy + 1e-4