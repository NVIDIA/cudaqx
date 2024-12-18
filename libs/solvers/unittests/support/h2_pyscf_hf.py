"""
H2 Hamiltonian Generator
=======================

This script generates ground truth data for the test_bravyi_kitaev.cpp unit tests by creating
and transforming a molecular Hamiltonian for the H2 molecule. The process involves:

1. Creating a molecular Hamiltonian using PySCF (via OpenFermion)
2. Extracting one- and two-body integrals to be used as coefficients
3. Converting to a spin Hamiltonian using the Bravyi-Kitaev transformation

The output includes:
- System information (number of orbitals, electrons)
- Nuclear repulsion energy
- Hartree-Fock energy
- Molecular Hamiltonian terms
- Qubit Hamiltonian terms after Bravyi-Kitaev transformation

Dependencies:
    - numpy
    - openfermion
    - openfermionpyscf

Configuration:
    - Molecule: H2
    - Geometry: H atoms at [0,0,0] and [0,0,0.7474]
    - Basis set: STO-3G
    - Multiplicity: 1 (singlet)
    - Charge: 0 (neutral)
"""

import numpy as np
from openfermion import *
from openfermionpyscf import run_pyscf

geometry = [['H', [0, 0, 0]], ['H', [0, 0, 0.7474]]]
basis = 'sto-3g'
multiplicity = 1
charge = 0
molecule = MolecularData(geometry, basis, multiplicity, charge)
molecule = run_pyscf(molecule, run_scf=True)

# Get the molecular Hamiltonian
molecular_hamiltonian = molecule.get_molecular_hamiltonian()

# Convert to fermion operator
fermion_hamiltonian = get_fermion_operator(molecular_hamiltonian)

# Convert to qubit Hamiltonian using Bravyi-Kitaev transformation
qubit_hamiltonian = bravyi_kitaev(fermion_hamiltonian)


def print_hamiltonian_info():
    print("System Information:")
    print(f"Number of orbitals: {molecule.n_orbitals}")
    print(f"Number of electrons: {molecule.n_electrons}")
    print(f"Nuclear repulsion energy: {molecule.nuclear_repulsion:.8f}")
    print(f"HF energy: {molecule.hf_energy:.8f}")

    print("\nMolecular Hamiltonian terms:")
    print(molecular_hamiltonian)

    print("\nNumber of qubits required:")
    print(count_qubits(qubit_hamiltonian))

    print("\nQubit Hamiltonian terms:")
    for term, coefficient in qubit_hamiltonian.terms.items():
        if abs(coefficient) > 1e-8:  # Filter out near-zero terms
            print(f"{coefficient:.8f} [{' '.join(str(x) for x in term)}]")


# Print the Hamiltonians and system information
print_hamiltonian_info()
