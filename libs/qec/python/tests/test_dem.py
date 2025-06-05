# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import pytest
import numpy as np
import cudaq
import cudaq_qec as qec


# Helper function to convert a binary matrix to a convenient string
def mat_to_str(mat):
    s = ''
    for row in mat:
        for col in row:
            if col == 1:
                s += '1'
            else:
                s += '.'
        s += '\n'
    return s


# Use the fixture to set the target
@pytest.fixture(scope="module", autouse=True)
def set_target():
    cudaq.set_target("stim")
    yield
    cudaq.reset_target()


def test_dem_from_memory_circuit():
    code = qec.get_code('steane')
    # observables = code.get_pauli_observables_matrix()
    p = 0.01
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(p), 1)
    statePrep = qec.operation.prep0
    nRounds = 2

    dem = qec.z_dem_from_memory_circuit(code, statePrep, nRounds, noise)
    expected_detector_error_matrix = """
1111...1..............
.1.111..111...........
..11.11..1.1111.......
.......111.1.1.1111...
..........1.11..1.111.
..............1..11.11
"""
    # print(mat_to_str(dem.detector_error_matrix), end='')
    assert '\n' + mat_to_str(
        dem.detector_error_matrix) == expected_detector_error_matrix

    # Round to 4 decimal places
    # print(np.round(dem.error_rates, 4))
    expected_error_rates = [
        0.0132, 0.0053, 0.008, 0.0053, 0.0211, 0.0053, 0.0132, 0.0106, 0.0053,
        0.0053, 0.0106, 0.0053, 0.0053, 0.0053, 0.0106, 0.0237, 0.0159, 0.0159,
        0.0185, 0.0341, 0.0211, 0.0443
    ]
    assert np.allclose(dem.error_rates, expected_error_rates, atol=1e-4)

    expected_observables_flips_matrix = '....11......1......111\n'
    # print(mat_to_str(dem.observables_flips_matrix), end='')
    assert mat_to_str(
        dem.observables_flips_matrix) == expected_observables_flips_matrix


if __name__ == "__main__":
    pytest.main()
