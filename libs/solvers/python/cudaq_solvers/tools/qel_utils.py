# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Shared helpers for Quantum Exact Lanczos (QEL) examples/tests.

These are intentionally dependency-light (NumPy only) so they're safe to use
from unit tests.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
from cudaq import spin


def pauli_string_to_spin_operator(pauli_str: str):
    """Convert a Pauli string into a `cudaq.spin` operator (no coefficient).

    Supported formats:
    - "Z0", "X0Y1Z2", "X0X1Y2Y3"
    - "Z0 Z1" (whitespace is ignored)

    Note: Identity ("I") is handled by returning `spin.i(0)`.
    """
    import re

    if pauli_str is None:
        raise ValueError("pauli_str must be a string, got None")

    s = pauli_str.strip()
    if not s or s == "I":
        return spin.i(0)

    matches = re.findall(r"([XYZ])(\d+)", s)
    if not matches:
        raise ValueError(f"Invalid Pauli string: {pauli_str!r}")

    op = None
    for pauli_char, qubit_str in matches:
        qubit_idx = int(qubit_str)

        if pauli_char == "X":
            term = spin.x(qubit_idx)
        elif pauli_char == "Y":
            term = spin.y(qubit_idx)
        elif pauli_char == "Z":
            term = spin.z(qubit_idx)
        else:
            # Should be unreachable due to regex, but keep as a safety net.
            raise ValueError(f"Invalid Pauli character: {pauli_char!r}")

        op = term if op is None else (op * term)

    return op


def pauli_term_to_spin(pauli_str: str, coeff: float):
    """Convert (pauli_str, coeff) into a scaled `cudaq.spin` operator."""
    return float(coeff) * pauli_string_to_spin_operator(pauli_str)


def build_spin_hamiltonian(terms: Sequence[Tuple[str, float]]):
    """Build a Hamiltonian from a list of (pauli_str, coeff) terms."""
    if not terms:
        raise ValueError("terms must be a non-empty sequence")

    ham = pauli_term_to_spin(*terms[0])
    for pauli_str, coeff in terms[1:]:
        ham += pauli_term_to_spin(pauli_str, coeff)
    return ham


def solve_generalized_eigenvalues_filtered(
    H: np.ndarray,
    S: np.ndarray,
    *,
    threshold: float = 1e-4,
    verbose: bool = False,
    krylov_dimension: int | None = None,
) -> np.ndarray:
    """Solve the generalized eigenproblem H v = E S v via S-filtering.

    The overlap matrix S can become (nearly) singular due to loss of Krylov
    orthogonality. This helper filters small/negative eigenvalues of S,
    projects H into the retained subspace, and solves the standard symmetric
    eigenproblem.

    Returns:
      Sorted eigenvalues (ascending).
    """
    evals_S, evecs_S = np.linalg.eigh(S)

    if verbose:
        print(f"\nS eigenvalues before filtering: {evals_S}")

    keep_indices = np.asarray(
        [i for i, e in enumerate(evals_S) if e > threshold])
    if keep_indices.size == 0:
        raise AssertionError("S matrix collapsed, no positive eigenvalues")

    if verbose:
        kd = krylov_dimension if krylov_dimension is not None else S.shape[0]
        print(
            f"Keeping {keep_indices.size}/{kd} eigenvectors (threshold={threshold})"
        )
        print(f"Filtered S eigenvalues: {evals_S[keep_indices]}")

    # Project onto the subspace with positive S eigenvalues.
    S_filtered = np.diag(evals_S[keep_indices])
    V_filtered = evecs_S[:, keep_indices]

    H_proj = V_filtered.T @ H @ V_filtered
    S_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(S_filtered)))
    H_final = S_inv_sqrt @ H_proj @ S_inv_sqrt

    eigenvalues = np.linalg.eigvalsh(H_final)
    if verbose:
        print(f"\nFinal eigenvalues: {eigenvalues}")
    return eigenvalues


def ground_state_energy_from_qel(
    eigenvalues: np.ndarray,
    *,
    normalization: float,
    constant_term: float = 0.0,
) -> float:
    """Convert QEL's scaled eigenvalues to an energy estimate."""
    return float(eigenvalues[0]) * float(normalization) + float(constant_term)
