import os

import pytest
import numpy as np

import cudaq
import cudaq_solvers as solvers
from scipy.optimize import minimize

def test_solvers_adapt_uccgsd_h2():
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., .7474))]
    molecule = solvers.create_molecule(geometry, 'sto-3g', 0, 0, casci=True)
    operators = solvers.get_operator_pool("uccgsd",
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


def test_solvers_adapt_uccgsd_lih():
    geometry = [('Li', (0.3925, 0., 0.)), ('H', (-1.1774, 0., .0))]
    molecule = solvers.create_molecule(geometry,
                                       'sto-3g',
                                       0,
                                       0,
                                       nele_cas=4,
                                       norb_cas=4,
                                       casci=True)

    operators = solvers.get_operator_pool("uccgsd",
                                          num_orbitals=molecule.n_orbitals)

    numElectrons = molecule.n_electrons

    from scipy.optimize import minimize

    @cudaq.kernel
    def initState(q: cudaq.qview):
        for i in range(numElectrons):
            x(q[i])

    energy, thetas, ops = solvers.adapt_vqe(initState, molecule.hamiltonian,
                                            operators)

    print(energy)
    assert np.isclose(energy, -7.8638, atol=1e-4)


def test_solvers_adapt_uccgsd_N2():

    geometry = [('N', (0.0, 0.0, 0.5600)), ('N', (0.0, 0.0, -0.5600))]
    molecule = solvers.create_molecule(geometry,
                                       'sto-3g',
                                       0,
                                       0,
                                       nele_cas=4,
                                       norb_cas=4,
                                       ccsd=True,
                                       casci=True,
                                       verbose=True)

    assert molecule.n_orbitals == 4
    assert molecule.n_electrons == 4

    numElectrons = molecule.n_electrons

    operators = solvers.get_operator_pool("uccgsd",
                                          num_orbitals=molecule.n_orbitals)

    @cudaq.kernel
    def initState(q: cudaq.qview):
        for i in range(numElectrons):
            x(q[i])

    energy, thetas, ops = solvers.adapt_vqe(initState, molecule.hamiltonian,
                                            operators)

    print(energy)
    assert np.isclose(energy, -107.5421, atol=1e-4)


def test_solvers_vqe_uccgsd_h2():

    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., .7474))]
    molecule = solvers.create_molecule(geometry, 'sto-3g', 0, 0, casci=True)

    numQubits = molecule.n_orbitals * 2
    numElectrons = molecule.n_electrons

    # Get grouped Pauli words and coefficients from UCCGSD pool
    pauliWordsList, coefficientsList = solvers.stateprep.get_uccgsd_pauli_lists(
        numQubits, only_singles=False, only_doubles=True)

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
        # Apply UCCGSD circuit with grouped thetas
        solvers.stateprep.uccgsd(q, thetas, pauliWordsList, coefficientsList)

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

    assert np.isclose(energy, -1.1371, atol=1e-4)


def test_solvers_vqe_uccgsd_lih():

    geometry = [('Li', (0.3925, 0., 0.)), ('H', (-1.1774, 0., .0))]
    molecule = solvers.create_molecule(geometry,
                                       'sto-3g',
                                       0,
                                       0,
                                       nele_cas=4,
                                       norb_cas=4,
                                       casci=True)

    numQubits = molecule.n_orbitals * 2
    numElectrons = molecule.n_electrons

    # Get grouped Pauli words and coefficients from UCCGSD pool (Singles and Doubles)
    pauliWordsList, coefficientsList = solvers.stateprep.get_uccgsd_pauli_lists(
        numQubits)

    # Number of theta parameters = number of excitation groups
    parameter_count = len(coefficientsList)

    singles = numQubits * (numQubits - 1) / 2
    doubles = (numQubits * (numQubits - 1) * (numQubits - 2) *
               (numQubits - 3)) / 8

    assert parameter_count == singles + doubles
    #assert parameter_count == 238

    @cudaq.kernel
    def ansatz(numQubits: int, numElectrons: int, thetas: list[float],
               pauliWordsList: list[list[cudaq.pauli_word]],
               coefficientsList: list[list[float]]):
        q = cudaq.qvector(numQubits)
        for i in range(numElectrons):
            x(q[i])
        # Apply UCCGSD circuit with grouped thetas
        solvers.stateprep.uccgsd(q, thetas, pauliWordsList, coefficientsList)

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
                       'maxiter': 6000,
                       'rhobeg': 0.1
                   })
    energy = res.fun
    assert np.isclose(energy, -7.8638, atol=1e-3)
