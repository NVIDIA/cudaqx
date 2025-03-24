# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os

import pytest, pathlib
import numpy as np
import cudaq_solvers as solvers

currentPath = pathlib.Path(__file__).parent.resolve()


def test_operators():
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., .7474))]
    molecule = solvers.create_molecule(geometry,
                                       'sto-3g',
                                       0,
                                       0,
                                       verbose=True,
                                       casci=True)
    print(molecule.hamiltonian.to_string())
    print(molecule.energies)
    assert np.isclose(-1.11, molecule.energies['hf_energy'], atol=1e-2)
    assert np.isclose(-1.13, molecule.energies['fci_energy'], atol=1e-2)
    from scipy.linalg import eigh
    minE = eigh(molecule.hamiltonian.to_matrix(), eigvals_only=True)[0]
    assert np.isclose(-1.13, minE, atol=1e-2)


def test_from_xyz_filename():
    molecule = solvers.create_molecule(str(currentPath) + '/resources/LiH.xyz',
                                       'sto-3g',
                                       0,
                                       0,
                                       verbose=True)
    print(molecule.energies)
    print(molecule.n_orbitals)
    print(molecule.n_electrons)
    assert molecule.n_orbitals == 6
    assert molecule.n_electrons == 4


def test_jordan_wigner():
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., .7474))]
    molecule = solvers.create_molecule(geometry,
                                       'sto-3g',
                                       0,
                                       0,
                                       verbose=True,
                                       casci=True)

    op = solvers.jordan_wigner(molecule.hpq,
                               molecule.hpqrs,
                               core_energy=molecule.energies['nuclear_energy'],
                               tol=1e-15)
    assert molecule.hamiltonian == op
    hpq = np.array(molecule.hpq)
    hpqrs = np.array(molecule.hpqrs)
    hpqJw = solvers.jordan_wigner(hpq, molecule.energies['nuclear_energy'])
    hpqrsJw = solvers.jordan_wigner(hpqrs)
    op2 = hpqJw + hpqrsJw
    # FIXME - revisit
    # print(str(op2))
    # print(str(molecule.hamiltonian))
    assert str(op2) == str(molecule.hamiltonian)

    spin_ham_matrix = molecule.hamiltonian.to_matrix()
    e, c = np.linalg.eig(spin_ham_matrix)
    assert np.isclose(np.min(e), -1.13717, rtol=1e-4)

    spin_ham_matrix = op2.to_matrix()
    e, c = np.linalg.eig(spin_ham_matrix)
    assert np.isclose(np.min(e), -1.13717, rtol=1e-4)


def test_bravyi_kitaev():
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., .7474))]
    molecule = solvers.create_molecule(geometry,
                                       'sto-3g',
                                       0,
                                       0,
                                       verbose=True,
                                       casci=True)

    op = solvers.bravyi_kitaev(molecule.hpq,
                               molecule.hpqrs,
                               core_energy=molecule.energies['nuclear_energy'],
                               tol=1e-15)
    spin_ham_matrix = molecule.hamiltonian.to_matrix()
    e, c = np.linalg.eig(spin_ham_matrix)
    assert np.isclose(np.min(e), -1.13717, rtol=1e-4)

    spin_ham_matrix = op.to_matrix()
    e, c = np.linalg.eig(spin_ham_matrix)
    assert np.isclose(np.min(e), -1.13717, rtol=1e-4)


def test_active_space():

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
    assert np.isclose(molecule.energies['core_energy'], -102.139973, rtol=1e-4)
    assert np.isclose(molecule.energies['R-CCSD'], -107.5421878, rtol=1e-4)
    assert np.isclose(molecule.energies['R-CASCI'], -107.5421983, rtol=1e-4)

    print(molecule.energies)
    print(molecule.n_orbitals)
    print(molecule.n_electrons)


def test_jordan_wigner_as():
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

    op = solvers.jordan_wigner(molecule.hpq, molecule.hpqrs,
                               molecule.energies['core_energy'])

    print(op.to_string())
    assert molecule.hamiltonian == op

    hpq = np.array(molecule.hpq)
    hpqrs = np.array(molecule.hpqrs)
    hpqJw = solvers.jordan_wigner(hpq,
                                  core_energy=molecule.energies['core_energy'],
                                  tolerance=1e-15)
    hpqrsJw = solvers.jordan_wigner(hpqrs)
    op2 = hpqJw + hpqrsJw

    spin_ham_matrix = molecule.hamiltonian.to_matrix()
    e, c = np.linalg.eig(spin_ham_matrix)
    print(np.min(e))
    assert np.isclose(np.min(e), -107.542198, rtol=1e-4)

    spin_ham_matrix = op2.to_matrix()
    e, c = np.linalg.eig(spin_ham_matrix)
    print(np.min(e))
    assert np.isclose(np.min(e), -107.542198, rtol=1e-4)


def test_bravyi_kitaev_as():
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

    op = solvers.bravyi_kitaev(molecule.hpq,
                               molecule.hpqrs,
                               core_energy=molecule.energies['core_energy'],
                               tol=1e-15)
    spin_ham_matrix = molecule.hamiltonian.to_matrix()
    e, c = np.linalg.eig(spin_ham_matrix)
    assert np.isclose(np.min(e), -107.542198, rtol=1e-4)

    spin_ham_matrix = op.to_matrix()
    e, c = np.linalg.eig(spin_ham_matrix)
    assert np.isclose(np.min(e), -107.542198, rtol=1e-4)


def test_as_with_natorb():

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
    assert molecule.n_orbitals == 4
    assert molecule.n_electrons == 4
    assert np.isclose(molecule.energies['R-CCSD'], -107.6059540, rtol=1e-4)
    assert np.isclose(molecule.energies['R-CASCI'], -107.6076127, rtol=1e-4)

    print(molecule.energies)
    print(molecule.n_orbitals)
    print(molecule.n_electrons)


def test_as_with_casscf():

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
                                       casscf=True,
                                       natorb=True,
                                       integrals_casscf=True,
                                       verbose=True)

    assert molecule.n_orbitals == 4
    assert molecule.n_electrons == 4
    assert np.isclose(molecule.energies['R-CASSCF'], -107.607626, rtol=1e-4)

    print(molecule.energies)
    print(molecule.n_orbitals)
    print(molecule.n_electrons)
