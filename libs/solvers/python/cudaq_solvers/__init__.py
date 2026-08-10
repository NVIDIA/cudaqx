# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

try:
    from ._pycudaqx_solvers_the_suffix_matters_cudaq_solvers import *
    from ._pycudaqx_solvers_the_suffix_matters_cudaq_solvers import __version__
    from ._pycudaqx_solvers_the_suffix_matters_cudaq_solvers import make_trotter_terms as _cpp_make_trotter_terms
except ImportError as _e:
    from ._system_dep_check import raise_if_missing_system_dep
    raise_if_missing_system_dep(_e, "cudaq-solvers")
    raise


def _maybe_call(value):
    return value() if callable(value) else value


def _is_python_spin_operator(value):
    return hasattr(value, "term_count") and hasattr(value, "__iter__")


def _is_python_spin_term(value):
    return hasattr(value, "evaluate_coefficient") and hasattr(
        value, "get_pauli_word")


def _term_coefficient(term, coefficient_tolerance):
    coefficient = term.evaluate_coefficient()
    if abs(coefficient.imag) > coefficient_tolerance:
        raise ValueError(
            "trotter error - only real Hamiltonian coefficients are supported.")
    return float(coefficient.real)


def _term_qubit_extent(term):
    max_degree = _maybe_call(getattr(term, "max_degree", -1))
    return max_degree + 1 if max_degree >= 0 else 0


def make_trotter_terms(hamiltonian, coefficient_tolerance=1e-12):
    """Return flattened terms for Suzuki-Trotter circuit primitives.

    Returns ``(coefficients, words, identity_coefficient, num_qubits)`` where
    ``words`` are padded Pauli strings suitable for CUDA-Q kernel arguments.

    ``apply_trotter`` omits identity terms. For ``H = c I + H'``, it applies a
    product-formula approximation to ``exp(-i H' t)`` and leaves the phase
    ``exp(-i c t)`` to the caller. This phase cancels in ordinary expectation
    values of one unconditioned evolved state, but it can matter for controlled
    evolution, overlaps, phase estimation, Krylov/QEL moments, and other
    interference-based algorithms.
    """
    if coefficient_tolerance < 0.0:
        raise ValueError(
            "trotter error - coefficient tolerance must be non-negative.")

    if _is_python_spin_term(hamiltonian):
        num_qubits = _term_qubit_extent(hamiltonian)
        terms = [hamiltonian]
    elif _is_python_spin_operator(hamiltonian):
        num_qubits = int(_maybe_call(getattr(hamiltonian, "qubit_count", 0)))
        terms = list(hamiltonian)
    else:
        return _cpp_make_trotter_terms(hamiltonian, coefficient_tolerance)

    coefficients = []
    words = []
    identity_coefficient = 0.0

    for term in terms:
        coefficient = _term_coefficient(term, coefficient_tolerance)
        term_extent = _term_qubit_extent(term)
        num_qubits = max(num_qubits, term_extent)
        if term.is_identity():
            identity_coefficient += coefficient
            continue
        words.append(term.get_pauli_word(num_qubits))
        coefficients.append(coefficient)

    return coefficients, words, identity_coefficient, num_qubits


try:
    from .gqe_algorithm.gqe import gqe
except ImportError:

    def gqe(*args, **kwargs):
        raise ImportError(
            "Failed to load GQE solver due to missing dependencies. "
            "Recommend installing the required dependencies with: "
            "'pip install cudaq-solvers[gqe]'")
