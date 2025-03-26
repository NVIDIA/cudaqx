# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
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
        assert op.get_qubit_count() <= 4


def test_generate_with_custom_coefficients():
    operators = solvers.get_operator_pool("uccsd",
                                          num_qubits=4,
                                          num_electrons=2)

    assert operators
    assert len(operators) == (2 + 1)

    for i, op in enumerate(operators):
        assert op.get_qubit_count() <= 4
        expected_coeff = [0.5, 0.125]
        for term in op:
            assert (abs(term.get_coefficient().real) in expected_coeff)


def test_generate_with_odd_electrons():
    operators = solvers.get_operator_pool("uccsd",
                                          num_qubits=6,
                                          num_electrons=3,
                                          spin=1)

    assert operators
    assert len(operators) == 2 * 2 + 4

    for op in operators:
        assert op.get_qubit_count() <= 6


def test_generate_with_large_system():
    operators = solvers.get_operator_pool("uccsd",
                                          num_qubits=20,
                                          num_electrons=10)

    assert operators
    assert len(operators) == 875

    for op in operators:
        assert op.get_qubit_count() <= 20


def test_uccsd_operator_pool_correctness():
    pool = solvers.get_operator_pool("uccsd", num_qubits=4, num_electrons=2)

    temp_data = [[], [], []]
    data_counter = 0
    for op in pool:
        op.for_each_term(lambda term: temp_data[data_counter].append(
            (term.get_pauli_word(4), term.get_coefficient())))
        data_counter += 1

    # Assert
    expected_operators = [["XZYI", "YZXI"], ["IXZY", "IYZX"],
                          [
                              "YYYX", "YXXX", "XXYX", "YYXY", "XYYY", "XXXY",
                              "YXYY", "XYXX"
                          ]]
    expected_coefficients = [[complex(-0.5, 0),
                              complex(0.5, 0)],
                             [complex(-0.5, 0),
                              complex(0.5, 0)],
                             [
                                 complex(-0.125, 0),
                                 complex(-0.125, 0),
                                 complex(0.125, 0),
                                 complex(-0.125, 0),
                                 complex(0.125, 0),
                                 complex(0.125, 0),
                                 complex(0.125, 0),
                                 complex(-0.125, 0)
                             ]]

    valid_chars = set('IXYZ')
    assert len(temp_data) == len(
        expected_operators
    ), f"Number of generated operators ({len(temp_data)}) does not match expected count ({len(expected_operators)})"

    for i in range(len(temp_data)):
        for j in range(len(temp_data[i])):
            op_string = temp_data[i][j][0]
            op_coeff = temp_data[i][j][1]
            # Check operator length
            assert len(
                op_string
            ) <= 4, f"Operator {op_string} does not have the expected length of <= 4"
            index = expected_operators[i].index(op_string)

            assert op_coeff == expected_coefficients[i][index], \
                f"Coefficient mismatch at index {i}, {index}: expected {expected_coefficients[i][index]}, got {op_coeff}"

            assert set(op_string).issubset(valid_chars), \
                f"Operator {op_string} contains invalid characters"
