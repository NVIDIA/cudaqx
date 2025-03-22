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

import cudaq
from cudaq import spin

import cudaq_solvers as solvers


def test_solvers_vqe():

    @cudaq.kernel
    def ansatz(theta: float):
        q = cudaq.qvector(2)
        x(q[0])
        ry(theta, q[1])
        x.ctrl(q[1], q[0])

    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

    # Can specify optimizer and gradient and tol
    energy, params, all_data = solvers.vqe(lambda thetas: ansatz(thetas[0]),
                                           hamiltonian, [0.],
                                           optimizer='lbfgs',
                                           gradient='parameter_shift',
                                           tol=1e-7)
    assert np.isclose(-1.74, energy, atol=1e-2)
    all_data[0].result.dump()

    # For gradient-based optimizer, can pick up default gradient (parameter_shift)
    energy, params, all_data = solvers.vqe(lambda thetas: ansatz(thetas[0]),
                                           hamiltonian, [0.],
                                           optimizer='lbfgs',
                                           verbose=True)
    assert np.isclose(-1.74, energy, atol=1e-2)

    # Can pick up default optimizer (cobyla)
    energy, params, all_data = solvers.vqe(lambda thetas: ansatz(thetas[0]),
                                           hamiltonian, [0.],
                                           verbose=True)
    assert np.isclose(-1.74, energy, atol=1e-2)

    cudaq.set_random_seed(22)

    # Can pick up default optimizer (cobyla)
    energy, params, all_data = solvers.vqe(lambda thetas: ansatz(thetas[0]),
                                           hamiltonian, [0.],
                                           verbose=True,
                                           shots=10000,
                                           max_iterations=10)
    assert energy > -2 and energy < -1.5
    print(energy)
    all_data[0].result.dump()
    counts = all_data[0].result.counts()
    assert 5 == len(counts.register_names)
    assert 4 == len(counts.get_register_counts('XX'))
    assert 4 == len(counts.get_register_counts('YY'))
    assert 1 == len(counts.get_register_counts('ZI'))
    assert 1 == len(counts.get_register_counts('IZ'))


def test_scipy_optimizer():

    @cudaq.kernel
    def ansatz(theta: float):
        q = cudaq.qvector(2)
        x(q[0])
        ry(theta, q[1])
        x.ctrl(q[1], q[0])

    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

    from scipy.optimize import minimize

    exp_vals = []

    def callback(xk):
        exp_vals.append(cudaq.observe(ansatz, hamiltonian, xk[0]).expectation())

    # Can specify optimizer and gradient
    energy, params, all_data = solvers.vqe(lambda thetas: ansatz(thetas[0]),
                                           hamiltonian, [0.],
                                           optimizer=minimize,
                                           callback=callback,
                                           method='L-BFGS-B',
                                           jac='3-point',
                                           tol=1e-4,
                                           options={'disp': True})
    assert np.isclose(-1.74, energy, atol=1e-2)
    print(exp_vals)
