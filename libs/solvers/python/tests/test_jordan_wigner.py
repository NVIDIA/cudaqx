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

def test_ground_state():
    #xyz = [('H', (0., 0., 0.)), ('H', (0., 0., .7474))]
    #xyz = [('H', (0., 0., 0.)), ('H', (0., 0., .7474)), ('H', (1., 0., 0.)), ('H', (1., 0., .7474))]
    xyz = [('H', (0., 0., 0.)), ('H', (1.0, 0., 0.)), ('H', (0.322, 2.592, 0.1)), ('H', (1.2825, 2.292, 0.1))]
    # xyz = [('Li', (0., 0., 0.)), ('H', (0., 0., 1.1774))]
    #xyz = [('O', (0.000000, 0.000000, 0.000000)), ('H', (0.757000, 0.586000, 0.000000)), ('H', (-0.757000, 0.586000, 0.000000))]

    # Compute FCI energy
    mol = gto.M (atom = xyz, basis = 'sto-3g',symmetry=False)
    mf = scf.RHF(mol).run()
    fci_energy = fci.FCI(mf).kernel()[0]
    print(f'FCI energy: {fci_energy}')

    # Compute energy using CUDA-Q/OpenFermion
    of_hamiltonian, data = cudaq.chemistry.create_molecular_hamiltonian(xyz, 'sto-3g', 1, 0)
    of_energy = np.min(np.linalg.eigvals(of_hamiltonian.to_matrix()))
    print(f'OpenFermion energy: {of_energy.real}')

    # Compute energy using CUDA-QX
    molecule = solvers.create_molecule(xyz, 'sto-3g', 0, 0, casci=True)
    # op = solvers.jordan_wigner(molecule.hpg, molecule.hpqrs, )
    cudaq1_eig = np.min(np.linalg.eigvals(molecule.hamiltonian.to_matrix()))
    print(f'CUDA-QX energy: {cudaq1_eig.real}')
    assert np.isclose(cudaq1_eig, of_energy.real, atol=1e-2)

    num_terms = of_hamiltonian.get_term_count()
    print(num_terms)
    num_terms = molecule.hamiltonian.get_term_count()
    print(num_terms) 

    def extract_pauli_terms(hamiltonian):
        pauli_dict = {}
        for term in hamiltonian:
            term_str = term.to_string()
            # Splitting at the first space
            coeff_string, pauli_str = term_str.split(" ", 1)
            coeff = term.get_coefficient()
            pauli_dict[pauli_str] = coeff
        return pauli_dict

    # Extract Pauli terms and coefficients from both Hamiltonians
    openfermion_terms = extract_pauli_terms(of_hamiltonian)
    cudaqx_terms = extract_pauli_terms(molecule.hamiltonian)
    sorted_of = dict(sorted(openfermion_terms.items()))
    sorted_cudaq = dict(sorted(cudaqx_terms.items()))

    #for (k1, v1), (k2, v2) in zip(sorted_of.items(), sorted_cudaq.items()):
    #if k1 != k2:
    #    pass
        # print(f"Mismatch: {k1} vs {k2}")
    #print(openfermion_terms))
    #print(len(cudaqx_terms))
