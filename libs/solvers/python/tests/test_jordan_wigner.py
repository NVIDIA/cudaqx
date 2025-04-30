# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
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
    for term in hamiltonian:
        result.append(term.get_pauli_word(hamiltonian.qubit_count))
    return result


def extract_coefficients(hamiltonian: cudaq.SpinOperator):
    result = []
    for term in hamiltonian:
        result.append(term.evaluate_coefficient())
    return result


def extract_spin_op_to_dict(op: cudaq.SpinOperator) -> dict:
    d = {}
    coeffs = extract_coefficients(op)
    words = extract_words(op)
    for c, w in zip(coeffs, words):
        d[w] = c
    return d


def jw_molecule_compare_hamiltonians_test(xyz):
    # Compute energy using CUDA-Q/OpenFermion
    of_hamiltonian, data = cudaq.chemistry.create_molecular_hamiltonian(
        xyz, 'sto-3g', 1, 0)

    # Compute energy using CUDA-QX. Note you must run with
    # OMP_NUM_THREADS=1 if you want this to be bit-for-bit repeatable.
    # This is a PySCF limitation. With OMP_NUM_THREADS>1, the Hamiltonian
    # coefficients will randomly toggle their signs, but the resulting
    # eigenvalues of the Hamiltonian will still be correct.
    molecule = solvers.create_molecule(xyz, 'sto-3g', 0, 0, casci=True)
    cqx_op = solvers.jordan_wigner(
        molecule.hpq,
        molecule.hpqrs,
        core_energy=molecule.energies['nuclear_energy'],
        tol=1e-12)

    of_hamiltonian_dict = extract_spin_op_to_dict(of_hamiltonian)
    cqx_op_dict = extract_spin_op_to_dict(cqx_op)

    for k in of_hamiltonian_dict.keys():
        assert (k in cqx_op_dict.keys())
    for k in cqx_op_dict.keys():
        assert (k in of_hamiltonian_dict.keys())

    for k in of_hamiltonian_dict.keys():
        # Use abs() checks because the sign can mismatch and still keep the same
        # eigenvalues. Also, see OMP_NUM_THREADS note above.
        assert np.isclose(abs(np.real(of_hamiltonian_dict[k])),
                          abs(np.real(cqx_op_dict[k])), 1e-12)
        assert np.isclose(abs(np.imag(of_hamiltonian_dict[k])),
                          abs(np.imag(cqx_op_dict[k])), 1e-12)


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

    # Compute energy using CUDA-QX. Note you must run with
    # OMP_NUM_THREADS=1 if you want this to be bit-for-bit repeatable.
    # This is a PySCF limitation. With OMP_NUM_THREADS>1, the Hamiltonian
    # coefficients will randomly toggle their signs, but the resulting
    # eigenvalues of the Hamiltonian will still be correct.
    molecule = solvers.create_molecule(xyz, 'sto-3g', 0, 0, casci=True)
    op = solvers.jordan_wigner(molecule.hpq,
                               molecule.hpqrs,
                               core_energy=molecule.energies['nuclear_energy'],
                               tol=1e-12)

    # The checks on the matrices below intentionally ignores the sign of the
    # values. (See note above.)
    def split_real_imag_abs(mat):
        mat = np.array(mat)
        return np.abs(np.real(mat)), np.abs(np.imag(mat))

    mop_r, mop_i = split_real_imag_abs(op.to_matrix())
    mh_r, mh_i = split_real_imag_abs(molecule.hamiltonian.to_matrix())
    of_r, of_i = split_real_imag_abs(of_hamiltonian.to_matrix())
    assert np.allclose(mop_r, mh_r, atol=1e-12)
    assert np.allclose(mop_i, mh_i, atol=1e-12)
    assert np.allclose(of_r, mh_r, atol=1e-12)
    assert np.allclose(of_i, mh_i, atol=1e-12)

    cudaqx_eig = np.min(np.linalg.eigvals(op.to_matrix()))
    print(f'CUDA-QX energy:        {cudaqx_eig.real}')
    assert np.isclose(cudaqx_eig, of_energy.real, atol=1e-4)

    num_terms = of_hamiltonian.term_count
    print(f'Number of terms in CUDA-Q/OpenFermion: {num_terms}')
    num_terms = molecule.hamiltonian.term_count
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
    # This is commented out by default because it is a very long test.
    # jw_molecule_test(xyz)
