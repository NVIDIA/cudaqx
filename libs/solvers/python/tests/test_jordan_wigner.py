# ============================================================================ #
# Copyright (c) 2024 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os
import pytest
import cudaq, cudaq_solvers as solvers
from pyscf import gto, scf, fci
import numpy as np

def extract_words(hamiltonian: cudaq.SpinOperator):
    result = []
    hamiltonian.for_each_term(lambda term: result.append(term.to_string(False)))
    return result

def extract_coefficients(hamiltonian: cudaq.SpinOperator):
    result = []
    hamiltonian.for_each_term(lambda term: result.append(term.get_coefficient()))
    return result

def extract_spin_op_to_dict(op: cudaq.SpinOperator) -> dict:
    d = {}
    coeffs = extract_coefficients(op)
    words = extract_words(op)
    for c,w in zip(coeffs, words):
        d[w] = c
    return d

def jw_molecule_compare_hamiltonians_test(xyz):
    # Compute energy using CUDA-Q/OpenFermion
    of_hamiltonian, data = cudaq.chemistry.create_molecular_hamiltonian(
        xyz, 'sto-3g', 1, 0)

    # Compute energy using CUDA-QX
    molecule = solvers.create_molecule(xyz, 'sto-3g', 0, 0, casci=True)
    cqx_op = solvers.jordan_wigner(molecule.hpq,
                                   molecule.hpqrs,
                                   core_energy=molecule.energies['nuclear_energy'])

    of_hamiltonian_dict = extract_spin_op_to_dict(of_hamiltonian) 
    cqx_op_dict = extract_spin_op_to_dict(cqx_op) 

    for k in of_hamiltonian_dict.keys():
        assert(k in cqx_op_dict.keys())
    for k in cqx_op_dict.keys():
        assert(k in of_hamiltonian_dict.keys())

    for k in of_hamiltonian_dict.keys():
        np.isclose(of_hamiltonian_dict[k], cqx_op_dict[k], 1e-12)

def jw_molecule_test(xyz):
    # Compute FCI energy
    mol = gto.M(atom=xyz, basis='sto-3g', symmetry=False)
    mf = scf.RHF(mol).run()
    fci_energy = fci.FCI(mf).kernel()[0]
    print(f'FCI energy:            {fci_energy}')

    # Compute energy using CUDA-Q/OpenFermion
    of_hamiltonian, data = cudaq.chemistry.create_molecular_hamiltonian(
        xyz, 'sto-3g', 1, 0)
    of_energy = np.min(np.linalg.eigvals(of_hamiltonian.to_matrix()))
    print(f'OpenFermion energy:    {of_energy.real}')

    # Compute energy using CUDA-QX
    molecule = solvers.create_molecule(xyz, 'sto-3g', 0, 0, casci=True)
    op = solvers.jordan_wigner(molecule.hpq,
                               molecule.hpqrs,
                               core_energy=molecule.energies['nuclear_energy'])
    assert op == molecule.hamiltonian
    assert of_hamiltonian == molecule.hamiltonian

    # FIXME - why do we need to call eigvals again if we can just assert the equality checks above?
    cudaqx_eig = np.min(np.linalg.eigvals(op.to_matrix()))
    print(f'CUDA-QX energy:        {cudaqx_eig.real}')
    assert np.isclose(cudaqx_eig, of_energy.real, atol=1e-4)

    num_terms = of_hamiltonian.get_term_count()
    print(f'Number of terms in CUDA-Q/OpenFermion: {num_terms}')
    num_terms = molecule.hamiltonian.get_term_count()
    print(f'Number of terms in CUDA-QX           : {num_terms}')


def test_ground_state():
    xyz = [('H', (0., 0., 0.)), ('H', (0., 0., .7474))]
    jw_molecule_compare_hamiltonians_test(xyz)
    jw_molecule_test(xyz)
    xyz = [('H', (0., 0., 0.)), ('H', (0., 0., .7474)), ('H', (1., 0., 0.)),
           ('H', (1., 0., .7474))]
    jw_molecule_compare_hamiltonians_test(xyz)
    jw_molecule_test(xyz)
    xyz = [('H', (0., 0., 0.)), ('H', (1.0, 0., 0.)),
           ('H', (0.322, 2.592, 0.1)), ('H', (1.2825, 2.292, 0.1))]
    jw_molecule_compare_hamiltonians_test(xyz)
    jw_molecule_test(xyz)
    xyz = [('Li', (0., 0., 0.)), ('H', (0., 0., 1.1774))]
    jw_molecule_compare_hamiltonians_test(xyz)
    jw_molecule_test(xyz)
    xyz = [('O', (0.000000, 0.000000, 0.000000)),
           ('H', (0.757000, 0.586000, 0.000000)),
           ('H', (-0.757000, 0.586000, 0.000000))]
    jw_molecule_compare_hamiltonians_test(xyz)
    # jw_molecule_test(xyz)
