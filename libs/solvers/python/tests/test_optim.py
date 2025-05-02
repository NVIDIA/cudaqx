# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os

import pytest
import numpy as np

import cudaq_solvers as solvers


def objective(x: list[float]):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2, [
        -2 * (1 - x[0]) + 400 * (x[0]**3 - x[1] * x[0]), 200 * (x[1] - x[0]**2)
    ]


def test_lbfgs():
    opt, params = solvers.optim.optimize(objective, [0., 0.], method='lbfgs')
    assert np.isclose(0.0, opt, atol=1e-6)
    assert np.isclose(1.0, params[0], atol=1e-6)
    assert np.isclose(1.0, params[1], atol=1e-6)


def test_cobyla():
    opt, params = solvers.optim.optimize(objective, [1., 1.], method='cobyla')
    assert np.isclose(0.0, opt, atol=1e-6)
    assert np.isclose(1.0, params[0], atol=1e-6)
    assert np.isclose(1.0, params[1], atol=1e-6)


def test_version():
    assert "CUDA-Q Solvers" in solvers.__version__
