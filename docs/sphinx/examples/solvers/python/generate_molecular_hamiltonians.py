# ============================================================================ #
# Copyright (c) 2024 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# [Begin Documentation]
import cudaq_solvers as solvers

# Generate active space Hamiltonian using RHF molecular orbitals

geometry = [('N', (0.0, 0.0, 0.5600)), ('N', (0.0, 0.0, -0.5600))]
molecule = solvers.create_molecule(geometry,
                                   'sto-3g',
                                   0,
                                   0,
                                   nele_cas=2,
                                   norb_cas=3,
                                   verbose=True)

print('N2 RHF Hamiltonian')
print('Energies : ', molecule.energies)
print('No. of orbitals: ', molecule.n_orbitals)
print('No. of electrons: ', molecule.n_electrons)

# Generate active space Hamiltonian using UHF molecular orbitals

geometry = [('N', (0.0, 0.0, 0.5600)), ('N', (0.0, 0.0, -0.5600))]
molecule = solvers.create_molecule(geometry,
                                   'sto-3g',
                                   0,
                                   0,
                                   nele_cas=2,
                                   norb_cas=3,
                                   UR=True,
                                   verbose=True)

print('N2 UHF Hamiltonian')
print('Energies : ', molecule.energies)
print('No. of orbitals: ', molecule.n_orbitals)
print('No. of electrons: ', molecule.n_electrons)

# Generate active space Hamiltonian using natural orbitals from MP2

geometry = [('N', (0.0, 0.0, 0.5600)), ('N', (0.0, 0.0, -0.5600))]
molecule = solvers.create_molecule(geometry,
                                   'sto-3g',
                                   0,
                                   0,
                                   nele_cas=2,
                                   norb_cas=3,
                                   MP2=True,
                                   integrals_natorb=True,
                                   verbose=True)

print('N2 Natural Orbitals from MP2 Hamiltonian')
print('Energies: ', molecule.energies)
print('No. of orbitals: ', molecule.n_orbitals)
print('No. of electrons: ', molecule.n_electrons)

# Generate active space Hamiltonian using casscf orbitals,
# where the active space of the casscf was defined from HF molecular orbitals

geometry = [('N', (0.0, 0.0, 0.5600)), ('N', (0.0, 0.0, -0.5600))]
molecule = solvers.create_molecule(geometry,
                                   'sto-3g',
                                   0,
                                   0,
                                   nele_cas=2,
                                   norb_cas=3,
                                   casscf=True,
                                   integrals_casscf=True,
                                   verbose=True)

print('N2 Active Space Hamiltonian Using CASSF Orbitals - HF orbitals')
print('Energies: ', molecule.energies)
print('No. of orbitals: ', molecule.n_orbitals)
print('No. of electrons: ', molecule.n_electrons)

# Generate active space Hamiltonian using casscf orbitals,
# where the active space of the casscf was defined from the MP2 natural orbitals.

geometry = [('N', (0.0, 0.0, 0.5600)), ('N', (0.0, 0.0, -0.5600))]
molecule = solvers.create_molecule(geometry,
                                   'sto-3g',
                                   0,
                                   0,
                                   nele_cas=2,
                                   norb_cas=3,
                                   MP2=True,
                                   natorb=True,
                                   casscf=True,
                                   integrals_casscf=True,
                                   verbose=True)

print('N2 Active Space Hamiltonian Using CASSF Orbitals - MP2 natural orbitals')
print('N2 HF Hamiltonian')
print('Energies: ', molecule.energies)
print('No. of orbitals: ', molecule.n_orbitals)
print('No. of electrons: ', molecule.n_electrons)

# For open-shell systems: Generate active space Hamiltonian using ROHF molecular orbitals
geometry = [('N', (0.0, 0.0, 0.5600)), ('N', (0.0, 0.0, -0.5600))]
molecule = solvers.create_molecule(geometry,
                                    'sto-3g',
                                    1,
                                    1,
                                    nele_cas=3,
                                    norb_cas=3,
                                    ccsd=True,
                                    verbose=True)

print('N2+ ROHF Hamiltonian')
print('Energies : ', molecule.energies)
print('No. of orbitals: ', molecule.n_orbitals)
print('No. of electrons: ', molecule.n_electrons)

# For open-shell systems: Generate active space Hamiltonian using UHF molecular orbitals
geometry = [('N', (0.0, 0.0, 0.5600)), ('N', (0.0, 0.0, -0.5600))]
molecule = solvers.create_molecule(geometry,
                                    'sto-3g',
                                    1,
                                    1,
                                    nele_cas=3,
                                    norb_cas=3,
                                    ccsd=True,
                                    UR=True,
                                    verbose=True)

print('N2+ UHF Hamiltonian')
print('Energies : ', molecule.energies)
print('No. of orbitals: ', molecule.n_orbitals)
print('No. of electrons: ', molecule.n_electrons)
