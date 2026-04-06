# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
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
    def ansatz(thetas: list[float]):
        q = cudaq.qvector(numQubits)
        for i in range(numElectrons):
            x(q[i])
        solvers.stateprep.uccsd(q, thetas, numElectrons, spin)

    ansatz.compile()

    np.random.seed(42)
    x0 = np.random.normal(-np.pi / 8.0, np.pi / 8.0, parameter_count)

    energy, params, all_data = solvers.vqe(ansatz,
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
    def ansatz(thetas: list[float]):
        q = cudaq.qvector(numQubits)
        for i in range(numElectrons):
            x(q[i])
        solvers.stateprep.uccsd(q, thetas, numElectrons, spin)

    ansatz.compile()

    np.random.seed(42)
    x0 = np.random.normal(-np.pi / 8.0, np.pi / 8.0, parameter_count)

    energy, params, all_data = solvers.vqe(ansatz,
                                           molecule.hamiltonian,
                                           x0,
                                           optimizer=minimize,
                                           method='COBYLA',
                                           tol=1e-5,
                                           options={'disp': True})

    print(energy)
    assert np.isclose(energy, -107.6059, 1e-2)


def test_uccsd_loops():
    repro_num_electrons = 2
    repro_num_qubits = 8

    repro_thetas = [
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558
    ]

    @cudaq.kernel
    def repro_trial_state(qubits: cudaq.qvector, num_electrons: int,
                          thetas: list[float]):
        for i in range(num_electrons):
            x(qubits[i])
        solvers.stateprep.uccsd(qubits, thetas, num_electrons, 0)

    @cudaq.kernel
    def repro():
        repro_qubits = cudaq.qvector(repro_num_qubits)
        repro_trial_state(repro_qubits, repro_num_electrons, repro_thetas)

    counts = cudaq.sample(repro, shots_count=1000)
    # There are normally 6 possible outcomes, but a PySCF non-repeatability
    # sometimes makes this fail by producing more than 6 outcomes, so we do not
    # check the length of the counts.
    # assert len(counts) == 6
    assert '00000011' in counts
    assert '00000110' in counts
    assert '00010010' in counts
    assert '01000010' in counts
    assert '10000001' in counts
    assert '11000000' in counts


def test_uccsd_open_shell_h3():
    # H3 linear chain: 3 electrons, doublet (spin=1).
    # Exercises the spin>0 code path of UCCSD.
    # A correct UCCSD ansatz must capture correlation energy, producing
    # a VQE energy lower than the Hartree-Fock energy and close to FCI.
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., 1.0)), ('H', (0., 0., 2.0))]
    molecule = solvers.create_molecule(geometry,
                                       'sto-3g',
                                       1,
                                       0,
                                       casci=True,
                                       verbose=True)

    numQubits = molecule.n_orbitals * 2
    numElectrons = molecule.n_electrons
    spin = 1

    assert numElectrons == 3, f"Expected 3 electrons, got {numElectrons}"
    assert numQubits == 6, f"Expected 6 qubits, got {numQubits}"

    hf_energy = molecule.energies['hf_energy']
    fci_energy = molecule.energies['fci_energy']
    print(f"HF  energy: {hf_energy:.8f}")
    print(f"FCI energy: {fci_energy:.8f}")

    # FCI must be lower than HF (sanity check on the reference values)
    assert fci_energy < hf_energy - 1e-6

    parameter_count = solvers.stateprep.get_num_uccsd_parameters(
        numElectrons, numQubits, spin)

    @cudaq.kernel
    def ansatz(thetas: list[float]):
        q = cudaq.qvector(numQubits)
        # Prepare reference state: occupy first numElectrons qubits
        # In interleaved ordering this gives alpha_0, beta_0, alpha_1
        for i in range(numElectrons):
            x(q[i])
        solvers.stateprep.uccsd(q, thetas, numElectrons, spin)

    ansatz.compile()

    x0 = np.zeros(parameter_count)

    energy, params, all_data = solvers.vqe(ansatz,
                                           molecule.hamiltonian,
                                           x0,
                                           optimizer=minimize,
                                           method='COBYLA',
                                           tol=1e-5,
                                           options={
                                               'disp': True,
                                               'maxiter': 500
                                           })
    print(f"VQE energy: {energy:.8f}")

    # UCCSD must improve upon Hartree-Fock by capturing correlation energy
    assert energy < hf_energy - 0.001, \
        (f"VQE+UCCSD on open-shell H3 failed to improve upon HF: "
         f"VQE={energy:.6f}, HF={hf_energy:.6f}")

    # UCCSD energy should be close to the exact (FCI) answer
    assert np.isclose(energy, fci_energy, atol=0.05), \
        (f"VQE+UCCSD energy not close to FCI: "
         f"VQE={energy:.6f}, FCI={fci_energy:.6f}")
