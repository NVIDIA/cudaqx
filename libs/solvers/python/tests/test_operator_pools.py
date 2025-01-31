# ============================================================================ #
# Copyright (c) 2024 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import pytest
import numpy as np
import cudaq_solvers as solvers


def test_generate_with_default_config():
    operators = solvers.get_operator_pool("uccsd",
                                          num_qubits=4,
                                          num_electrons=2)
    assert operators
    assert len(operators) == 2 + 1

    for op in operators:
        assert op.get_qubit_count() == 4


def test_generate_with_custom_coefficients():
    operators = solvers.get_operator_pool("uccsd",
                                          num_qubits=4,
                                          num_electrons=2)

    assert operators
    assert len(operators) == (2 + 1)

    for i, op in enumerate(operators):
        assert op.get_qubit_count() == 4
        expected_coeff = 1.0
        for term in op:
            assert np.isclose(abs(term.get_coefficient().imag), expected_coeff)


def test_generate_with_odd_electrons():
    operators = solvers.get_operator_pool("uccsd",
                                          num_qubits=6,
                                          num_electrons=3,
                                          spin=1)

    assert operators
    assert len(operators) == 2 * 2 + 4

    for op in operators:
        assert op.get_qubit_count() == 6


def test_generate_with_large_system():
    operators = solvers.get_operator_pool("uccsd",
                                          num_qubits=20,
                                          num_electrons=10)

    assert operators
    assert len(operators) == 875

    for op in operators:
        assert op.get_qubit_count() == 20


def test_uccsd_operator_pool_correctness():
    # Generate the UCCSD operator pool
    pool = solvers.get_operator_pool("uccsd", num_qubits=4, num_electrons=2)

    # Convert SpinOperators to strings
    pool_strings = [op.to_string(False) for op in pool]

    # Expected result
    expected_pool = ["XZYIYZXI", "IXZYIYZX", "XXXYXXYXXYXXXYYYYXXXYXYYYYXYYYYX"]

    # Assert that the generated pool matches the expected result
    assert pool_strings == expected_pool, f"Expected {expected_pool}, but got {pool_strings}"

    # Check that all operators contain only valid characters (I, X, Y, Z)
    valid_chars = set('IXYZ')
    for op_string in pool_strings:
        assert set(op_string).issubset(
            valid_chars), f"Operator {op_string} contains invalid characters"


def test_generate_with_invalid_config():
    # Missing required parameters
    with pytest.raises(RuntimeError):
        pool = solvers.get_operator_pool("uccsd")
