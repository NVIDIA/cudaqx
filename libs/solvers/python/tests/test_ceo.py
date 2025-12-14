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


def test_solvers_adapt_ceo_h2():
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., .7474))]
    molecule = solvers.create_molecule(geometry, 'sto-3g', 0, 0, casci=True)
    operators = solvers.get_operator_pool("ceo",
                                          num_orbitals=molecule.n_orbitals)

    numElectrons = molecule.n_electrons

    @cudaq.kernel
    def initState(q: cudaq.qview):
        for i in range(numElectrons):
            x(q[i])

    energy, thetas, ops = solvers.adapt_vqe(initState, molecule.hamiltonian,
                                            operators)

    print(energy)
    assert np.isclose(energy, -1.1371, atol=1e-3)


def test_solvers_adapt_ceo_lih():
    geometry = [('Li', (0.3925, 0., 0.)), ('H', (-1.1774, 0., .0))]
    molecule = solvers.create_molecule(geometry,
                                       'sto-3g',
                                       0,
                                       0,
                                       nele_cas=4,
                                       norb_cas=4,
                                       casci=True)

    operators = solvers.get_operator_pool("ceo",
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


def test_solvers_adapt_ceo_N2():
    geometry = [('N', (0.0, 0.0, 0.5600)), ('N', (0.0, 0.0, -0.5600))]
    molecule = solvers.create_molecule(geometry,
                                       'sto-3g',
                                       0,
                                       0,
                                       nele_cas=4,
                                       norb_cas=4,
                                       casci=True,
                                       verbose=True)

    operators = solvers.get_operator_pool("ceo",
                                          num_orbitals=molecule.n_orbitals)

    numElectrons = molecule.n_electrons

    @cudaq.kernel
    def initState(q: cudaq.qview):
        for i in range(numElectrons):
            x(q[i])

    energy, thetas, ops = solvers.adapt_vqe(initState, molecule.hamiltonian,
                                            operators)

    print(energy)
    assert np.isclose(energy, -107.5421, atol=1e-4)


def test_solvers_vqe_ceo_h2():
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., .7474))]
    molecule = solvers.create_molecule(geometry, 'sto-3g', 0, 0, casci=True)

    numOrbitals = molecule.n_orbitals
    numQubits = 2 * numOrbitals
    numElectrons = molecule.n_electrons

    # Get CEO Pauli lists for VQE
    pauli_word_list, coefficient_list = solvers.stateprep.get_ceo_pauli_lists(
        numOrbitals)

    @cudaq.kernel
    def ansatz(numQubits: int, numElectrons: int, thetas: list[float],
               pauliWordsList: list[list[cudaq.pauli_word]],
               coefficientsList: list[list[float]]):
        q = cudaq.qvector(numQubits)
        for i in range(numElectrons):
            x(q[i])
        # Apply CEO Ansatz 
        solvers.stateprep.ceo(q, thetas, pauliWordsList, coefficientsList)

    parameter_count = len(pauli_word_list)
    x0 = [0.0] * parameter_count

    def cost(theta):

        theta = theta.tolist()
        energy = cudaq.observe(ansatz, molecule.hamiltonian, numQubits,
                               numElectrons, theta, pauli_word_list,
                               coefficient_list).expectation()

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


def test_solvers_vqe_ceo_lih():
    geometry = [('Li', (0.3925, 0., 0.)), ('H', (-1.1774, 0., .0))]
    molecule = solvers.create_molecule(geometry, 
                                       'sto-3g', 
                                       0, 
                                       0, 
                                       nele_cas=4, 
                                       norb_cas=4, 
                                       casci=True)

    numOrbitals = molecule.n_orbitals
    numQubits = 2 * numOrbitals
    numElectrons = molecule.n_electrons

    # Get CEO Pauli lists for VQE
    pauli_word_list, coefficient_list = solvers.stateprep.get_ceo_pauli_lists(
        numOrbitals)

    @cudaq.kernel
    def ansatz(numQubits: int, numElectrons: int, thetas: list[float],
               pauliWordsList: list[list[cudaq.pauli_word]],
               coefficientsList: list[list[float]]):
        q = cudaq.qvector(numQubits)
        for i in range(numElectrons):
            x(q[i])
        # Apply CEO Ansatz 
        solvers.stateprep.ceo(q, thetas, pauliWordsList, coefficientsList)

    parameter_count = len(pauli_word_list)
    x0 = [0.0] * parameter_count

    def cost(theta):

        theta = theta.tolist()
        energy = cudaq.observe(ansatz, molecule.hamiltonian, numQubits,
                               numElectrons, theta, pauli_word_list,
                               coefficient_list).expectation()

        return energy

    res = minimize(cost, 
                   x0, 
                   method='COBYLA', 
                   options={
                    'maxiter': 6000,
                    'rhobeg': 0.1,
                    })

    energy = res.fun

    print(energy)
    assert np.isclose(energy, -7.8638, atol=1e-3)


def test_ceo_operator_pool_generation():
    """Test CEO operator pool generation with different sizes."""
    # Test with 2 orbitals
    ops = solvers.get_operator_pool("ceo", num_orbitals=2)
    assert len(ops) == 4  # 2 singles + 2 mixed doubles

    # Test with 4 orbitals
    ops = solvers.get_operator_pool("ceo", num_orbitals=4)
    assert len(ops) == 96  # 12 singles + 72 mixed doubles, 12 same-spin doubles

