# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import numpy as np
import pytest

import cudaq
from cudaq import spin
import cudaq_solvers as solvers

FOREST_RUTH_W1 = 1.3512071919596578
FOREST_RUTH_W0 = -1.7024143839193153


def _apply_pauli_to_vector(word, vector):
    word = str(word)
    result = np.zeros_like(vector, dtype=np.complex128)
    n_qubits = len(word)
    for basis, amplitude in enumerate(vector):
        target = basis
        phase = 1.0 + 0.0j
        for qubit, op in enumerate(word):
            bit = (basis >> qubit) & 1
            if op == "I":
                continue
            if op == "X":
                target ^= 1 << qubit
            elif op == "Y":
                target ^= 1 << qubit
                phase *= -1.0j if bit else 1.0j
            elif op == "Z":
                phase *= -1.0 if bit else 1.0
            else:
                raise ValueError(op)
        result[target] += phase * amplitude
    return result


def _apply_pauli_rotation(vector, word, angle):
    return (np.cos(angle) * vector -
            1.0j * np.sin(angle) * _apply_pauli_to_vector(word, vector))


def _second_order_step(vector, coefficients, words, tau):
    state = vector
    for coefficient, word in zip(coefficients, words):
        state = _apply_pauli_rotation(state, word, 0.5 * tau * coefficient)
    for coefficient, word in reversed(list(zip(coefficients, words))):
        state = _apply_pauli_rotation(state, word, 0.5 * tau * coefficient)
    return state


def _simulate_trotter(coefficients,
                      words,
                      identity,
                      num_qubits,
                      time,
                      steps,
                      order,
                      ket,
                      include_identity=True):
    if steps == 0:
        raise ValueError("steps must be greater than zero")
    if order not in (1, 2, 4):
        raise ValueError("order must be one of {1, 2, 4}")
    if len(coefficients) != len(words):
        raise ValueError("coefficients and words must have equal length")
    if ket.size != 2**num_qubits:
        raise ValueError("ket length does not match num_qubits")

    state = np.array(ket, dtype=np.complex128, copy=True)
    dt = time / steps
    for _ in range(steps):
        if order == 1:
            for coefficient, word in zip(coefficients, words):
                state = _apply_pauli_rotation(state, word, dt * coefficient)
        elif order == 2:
            state = _second_order_step(state, coefficients, words, dt)
        else:
            state = _second_order_step(state, coefficients, words,
                                       FOREST_RUTH_W1 * dt)
            state = _second_order_step(state, coefficients, words,
                                       FOREST_RUTH_W0 * dt)
            state = _second_order_step(state, coefficients, words,
                                       FOREST_RUTH_W1 * dt)

    if include_identity and identity != 0.0:
        state *= np.exp(-1.0j * identity * time)
    return state


def _pauli_matrix(word):
    dim = 2**len(word)
    matrix = np.zeros((dim, dim), dtype=np.complex128)
    for basis in range(dim):
        vector = np.zeros(dim, dtype=np.complex128)
        vector[basis] = 1.0
        matrix[:, basis] = _apply_pauli_to_vector(word, vector)
    return matrix


def _exact_evolve(coefficients, words, identity, time, ket):
    matrix = identity * np.eye(ket.size, dtype=np.complex128)
    for coefficient, word in zip(coefficients, words):
        matrix += coefficient * _pauli_matrix(str(word))
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return eigenvectors @ (np.exp(-1.0j * time * eigenvalues) *
                           (eigenvectors.conj().T @ ket))


def _two_qubit_product_state(rx_angle, ry_angle):
    q0 = np.array([np.cos(0.5 * rx_angle), -1.0j * np.sin(0.5 * rx_angle)],
                  dtype=np.complex128)
    q1 = np.array(
        [np.cos(0.5 * ry_angle), np.sin(0.5 * ry_angle)], dtype=np.complex128)
    return np.array(
        [q0[basis & 1] * q1[(basis >> 1) & 1] for basis in range(4)],
        dtype=np.complex128)


def _phase_align_error(actual, expected):
    overlap = np.vdot(expected, actual)
    if abs(overlap) > 0.0:
        actual = actual * np.exp(-1.0j * np.angle(overlap))
    return np.linalg.norm(actual - expected)


def test_no_public_host_statevector_trotter_api():
    assert not hasattr(solvers, "trotter")


def test_make_trotter_terms_extracts_python_spin_operator():
    hamiltonian = (0.7 * spin.x(0) + 0.4 * spin.z(1) -
                   0.2 * cudaq.SpinOperator.from_word("II"))

    coefficients, words, identity, num_qubits = solvers.make_trotter_terms(
        hamiltonian)

    by_word = {
        str(word): coefficient
        for coefficient, word in zip(coefficients, words)
    }
    assert num_qubits == 2
    assert identity == pytest.approx(-0.2)
    assert by_word["XI"] == pytest.approx(0.7)
    assert by_word["IZ"] == pytest.approx(0.4)


def test_make_trotter_terms_accepts_single_python_spin_term():
    coefficients, words, identity, num_qubits = solvers.make_trotter_terms(
        0.5 * spin.y(2))

    assert coefficients == pytest.approx([0.5])
    assert [str(word) for word in words] == ["IIY"]
    assert identity == pytest.approx(0.0)
    assert num_qubits == 3


def test_product_formula_reference_improves_with_order():
    hamiltonian = (0.7 * spin.x(0) + 0.4 * spin.z(1) +
                   0.31 * spin.x(0) * spin.z(1) + 0.23 * spin.y(0) * spin.y(1))
    coefficients, words, identity, num_qubits = solvers.make_trotter_terms(
        hamiltonian)

    rng = np.random.default_rng(7)
    ket = rng.normal(size=4) + 1.0j * rng.normal(size=4)
    ket = ket / np.linalg.norm(ket)

    time = 0.8
    steps = 2
    exact = _exact_evolve(coefficients, words, identity, time, ket)

    first = _simulate_trotter(coefficients, words, identity, num_qubits, time,
                              steps, 1, ket)
    second = _simulate_trotter(coefficients, words, identity, num_qubits, time,
                               steps, 2, ket)
    fourth = _simulate_trotter(coefficients, words, identity, num_qubits, time,
                               steps, 4, ket)

    first_error = _phase_align_error(first, exact)
    second_error = _phase_align_error(second, exact)
    fourth_error = _phase_align_error(fourth, exact)

    assert second_error < first_error
    assert fourth_error < second_error


def test_apply_trotter_kernel_interop_with_flattened_terms():
    hamiltonian = spin.x(0)
    coefficients, words, identity, num_qubits = solvers.make_trotter_terms(
        hamiltonian)
    assert identity == 0.0
    assert num_qubits == 1

    @cudaq.kernel
    def evolve(coeffs: list[float], paulis: list[cudaq.pauli_word], t: float):
        q = cudaq.qvector(1)
        solvers.hamiltonian_simulation.apply_trotter(coeffs, paulis, t, 1, 2, q)

    state = np.asarray(cudaq.get_state(evolve, coefficients, words, 0.25),
                       dtype=np.complex128)
    expected = _simulate_trotter(coefficients, words, identity, num_qubits,
                                 0.25, 1, 2,
                                 np.array([1.0, 0.0], dtype=np.complex128))

    np.testing.assert_allclose(state, expected, atol=1e-12)


def test_apply_trotter_kernel_matches_reference_for_orders():
    hamiltonian = (0.7 * spin.x(0) + 0.4 * spin.z(1) +
                   0.31 * spin.x(0) * spin.z(1) + 0.23 * spin.y(0) * spin.y(1))
    coefficients, words, identity, num_qubits = solvers.make_trotter_terms(
        hamiltonian)
    ket = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)

    @cudaq.kernel
    def evolve(coeffs: list[float], paulis: list[cudaq.pauli_word], t: float,
               steps: int, order: int):
        q = cudaq.qvector(2)
        solvers.hamiltonian_simulation.apply_trotter(coeffs, paulis, t, steps,
                                                     order, q)

    for order in (1, 2, 4):
        state = np.asarray(cudaq.get_state(evolve, coefficients, words, 0.8, 3,
                                           order),
                           dtype=np.complex128)
        expected = _simulate_trotter(coefficients, words, identity, num_qubits,
                                     0.8, 3, order, ket)
        np.testing.assert_allclose(state, expected, atol=1e-6)


def test_apply_trotter_kernel_orders_track_exact_evolution():
    hamiltonian = (0.37 * spin.x(0) - 0.22 * spin.z(1) +
                   0.19 * spin.x(0) * spin.x(1) + 0.41 * spin.y(0) * spin.y(1) +
                   0.13 * spin.z(0) * spin.x(1))
    coefficients, words, identity, num_qubits = solvers.make_trotter_terms(
        hamiltonian)
    assert identity == pytest.approx(0.0)
    assert num_qubits == 2

    time = 0.7
    steps = 4
    rx_angle = 0.37
    ry_angle = -0.52
    ket = _two_qubit_product_state(rx_angle, ry_angle)
    exact = _exact_evolve(coefficients, words, identity, time, ket)

    @cudaq.kernel
    def evolve(coeffs: list[float], paulis: list[cudaq.pauli_word], t: float,
               n_steps: int, order: int, theta0: float, theta1: float):
        q = cudaq.qvector(2)
        rx(theta0, q[0])
        ry(theta1, q[1])
        solvers.hamiltonian_simulation.apply_trotter(coeffs, paulis, t, n_steps,
                                                     order, q)

    errors = {}
    for order in (1, 2, 4):
        state = np.asarray(cudaq.get_state(evolve, coefficients, words, time,
                                           steps, order, rx_angle, ry_angle),
                           dtype=np.complex128)
        errors[order] = _phase_align_error(state, exact)

    assert errors[2] < errors[1]
    assert errors[4] < errors[2]
    assert errors[1] < 2.0e-2
    assert errors[2] < 6.0e-4
    assert errors[4] < 5.0e-6


def test_apply_trotter_kernel_handles_four_qubit_hamiltonian_with_many_terms():
    hamiltonian = (0.11 * spin.x(0) - 0.17 * spin.y(1) + 0.23 * spin.z(2) -
                   0.29 * spin.x(3) + 0.31 * spin.x(0) * spin.x(1) +
                   0.37 * spin.y(1) * spin.z(2) - 0.41 * spin.z(0) * spin.x(3) +
                   0.43 * spin.x(0) * spin.y(2) * spin.z(3) -
                   0.47 * spin.y(0) * spin.y(1) * spin.x(2) +
                   0.53 * spin.z(0) * spin.x(1) * spin.y(2) * spin.z(3))
    coefficients, words, identity, num_qubits = solvers.make_trotter_terms(
        hamiltonian)

    assert num_qubits == 4
    assert len(coefficients) > 8
    assert len(coefficients) == len(words)

    ket = np.zeros(16, dtype=np.complex128)
    ket[0] = 1.0

    @cudaq.kernel
    def evolve(coeffs: list[float], paulis: list[cudaq.pauli_word], t: float,
               steps: int, order: int):
        q = cudaq.qvector(4)
        solvers.hamiltonian_simulation.apply_trotter(coeffs, paulis, t, steps,
                                                     order, q)

    state = np.asarray(cudaq.get_state(evolve, coefficients, words, 0.37, 2, 2),
                       dtype=np.complex128)
    expected = _simulate_trotter(coefficients, words, identity, num_qubits,
                                 0.37, 2, 2, ket)

    np.testing.assert_allclose(state, expected, atol=1e-6)
