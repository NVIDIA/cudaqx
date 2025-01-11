# ============================================================================ #
# Copyright (c) 2024 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os

import pytest
import numpy as np

import cudaq, cudaq_solvers as solvers

from scipy.optimize import minimize


def test_solvers_uccsd():
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., .7474))]
    molecule = solvers.create_molecule(geometry, 'sto-3g', 0, 0, casci=True)

    numQubits = molecule.n_orbitals * 2
    numElectrons = molecule.n_electrons
    spin = 0

    @cudaq.kernel
    def ansatz(thetas: list[float]):
        q = cudaq.qvector(numQubits)
        for i in range(numElectrons):
            x(q[i])
        solvers.stateprep.uccsd(q, thetas, numElectrons, spin)

    ansatz.compile()

    energy, params, all_data = solvers.vqe(ansatz,
                                           molecule.hamiltonian,
                                           [-.2, -.2, -.2],
                                           optimizer=minimize,
                                           method='L-BFGS-B',
                                           jac='3-point',
                                           tol=1e-4,
                                           options={'disp': True})
    print(energy)
    assert np.isclose(energy, -1.13, 1e-2)


def test_uccsd_active_space():

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

    numQubits = molecule.n_orbitals * 2
    numElectrons = molecule.n_electrons
    spin = 0

    alphasingle, betasingle, mixeddouble, alphadouble, betadouble = solvers.stateprep.get_uccsd_excitations(
        numElectrons, numQubits, spin)
    a_single = [[0, 4], [0, 6], [2, 4], [2, 6]]
    a_double = [[0, 2, 4, 6]]
    assert alphasingle == a_single
    assert alphadouble == a_double

    parameter_count = solvers.stateprep.get_num_uccsd_parameters(
        numElectrons, numQubits, spin)

    @cudaq.kernel
    def ansatz2(thetas: list[float]):
        q = cudaq.qvector(numQubits)
        for i in range(numElectrons):
            x(q[i])
        solvers.stateprep.uccsd(q, thetas, numElectrons, spin)

    ansatz2.compile()

    np.random.seed(42)
    x0 = np.random.normal(-np.pi / 8.0, np.pi / 8.0, parameter_count)

    energy, params, all_data = solvers.vqe(ansatz2,
                                           molecule.hamiltonian,
                                           x0,
                                           optimizer=minimize,
                                           method='COBYLA',
                                           tol=1e-5,
                                           options={'disp': True})

    print(energy)
    assert np.isclose(energy, -107.542, 1e-2)


def test_uccsd_active_space_natorb():

    geometry = [('N', (0.0, 0.0, 0.5600)), ('N', (0.0, 0.0, -0.5600))]
    molecule = solvers.create_molecule(geometry,
                                       'sto-3g',
                                       0,
                                       0,
                                       nele_cas=4,
                                       norb_cas=4,
                                       MP2=True,
                                       ccsd=True,
                                       casci=True,
                                       natorb=True,
                                       integrals_natorb=True,
                                       verbose=True)

    numQubits = molecule.n_orbitals * 2
    numElectrons = molecule.n_electrons
    spin = 0

    parameter_count = solvers.stateprep.get_num_uccsd_parameters(
        numElectrons, numQubits, spin)

    @cudaq.kernel
    def ansatz3(thetas: list[float]):
        q = cudaq.qvector(numQubits)
        for i in range(numElectrons):
            x(q[i])
        solvers.stateprep.uccsd(q, thetas, numElectrons, spin)

    ansatz3.compile()

    np.random.seed(42)
    x0 = np.random.normal(-np.pi / 8.0, np.pi / 8.0, parameter_count)

    energy, params, all_data = solvers.vqe(ansatz3,
                                           molecule.hamiltonian,
                                           x0,
                                           optimizer=minimize,
                                           method='COBYLA',
                                           tol=1e-5,
                                           options={'disp': True})

    print(energy)
    assert np.isclose(energy, -107.6059, 1e-2)
