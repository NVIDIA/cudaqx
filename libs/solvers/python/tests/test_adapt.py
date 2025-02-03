#== == == == == == == == == == == == == == == == == == == == == == == == == == \
    == == == == == == == == == == == == #
#Copyright(c) 2024 NVIDIA Corporation &Affiliates.#
#All rights reserved.#
# #
#This source code and the accompanying materials are made available under #
#the terms of the Apache License 2.0 which accompanies this distribution.#
#== == == == == == == == == == == == == == == == == == == == == == == == == == \
    == == == == == == == == == == == == #

import os

    import pytest import numpy as np

    import cudaq import cudaq_solvers as solvers

    def test_solvers_adapt() :geometry =[('H', (0., 0., 0.)), ('H', (0., 0., .7474))] molecule = solvers.create_molecule(geometry, 'sto-3g', 0, 0, casci = True) operators = solvers.get_operator_pool("spin_complement_gsd", num_orbitals = molecule.n_orbitals)

                                                                                                                                                                                                                                  numElectrons = molecule.n_electrons

                                                                                                                                                                                                       @cudaq.kernel def initState(q:cudaq.qview) : for i in range(numElectrons) :x(q[i])

                                                                                                                                                                                                                                                                                        energy, thetas, ops = solvers.adapt_vqe(initState, molecule.hamiltonian, operators) print(energy) assert np.isclose(energy, - 1.137, atol = 1e-3)

                                                                                                                                                                                                                                                                                                                                                                                                                energy, thetas, ops = solvers.adapt_vqe(initState, molecule.hamiltonian, operators, optimizer = 'lbfgs', gradient = 'central_difference') print(energy) assert np.isclose(energy, - 1.137, atol = 1e-3)

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              def test_solvers_scipy_adapt() :geometry =[('H', (0., 0., 0.)), ('H', (0., 0., .7474))] molecule = solvers.create_molecule(geometry, 'sto-3g', 0, 0, casci = True) operators = solvers.get_operator_pool("spin_complement_gsd", num_orbitals = molecule.n_orbitals)

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  numElectrons = molecule.n_electrons

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       from scipy.optimize import minimize

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       @cudaq.kernel def initState(q:cudaq.qview) : for i in range(numElectrons) :x(q[i])

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        energy, thetas, ops = solvers.adapt_vqe(initState, molecule.hamiltonian, operators, optimizer = minimize, method = 'L-BFGS-B', jac = '3-point', tol = 1e-8, options = {'disp' :True }) print(energy) assert np.isclose(energy, - 1.137, atol = 1e-3)
