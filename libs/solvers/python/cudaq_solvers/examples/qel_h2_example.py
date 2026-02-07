#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""
Example: Quantum Exact Lanczos for H2 Molecule

This example demonstrates how to use the Quantum Exact Lanczos (QEL) algorithm
to find the ground state energy of the H2 molecule using CUDA-Q Solvers.

QEL uses block encoding and amplitude amplification to build a Krylov subspace
via quantum moment collection, then solves a generalized eigenvalue problem
classically to extract eigenvalues.
"""

import numpy as np
from scipy import linalg as la
from cudaq import spin
import cudaq_solvers as solvers


def create_h2_hamiltonian():
    """
    Create H2 Hamiltonian at equilibrium geometry (0.7414 Å).
    
    Returns:
        cudaq.SpinOperator: H2 Hamiltonian in qubit basis
    """
    # Molecular data for H2 at equilibrium
    # Computed using Jordan-Wigner transformation
    hamiltonian = (
        -1.0523732 +  # Constant
        0.39793742 * spin.z(0) -  # Single-qubit terms
        0.39793742 * spin.z(1) - 0.01128010 * spin.z(2) +
        0.01128010 * spin.z(3) + 0.18093120 * spin.x(0) * spin.x(1) *
        spin.y(2) * spin.y(3)  # 4-qubit term
    )

    return hamiltonian


def run_qel_basic():
    """Basic QEL example with minimal parameters."""
    print("=" * 70)
    print("Example 1: Basic QEL for H2")
    print("=" * 70)

    # Create Hamiltonian
    h2 = create_h2_hamiltonian()

    # Run QEL with default parameters
    result = solvers.quantum_exact_lanczos(
        h2,
        num_qubits=4,  # H2 requires 4 qubits in minimal basis
        n_electrons=2,  # 2 electrons in H2
        krylov_dim=5  # Small Krylov subspace
    )

    print(f"\nQEL completed:")
    print(f"  Krylov dimension: {result.krylov_dimension}")
    print(f"  System qubits: {result.num_system}")
    print(f"  Ancilla qubits: {result.num_ancilla}")
    print(f"  Normalization (α): {result.normalization:.6f}")
    print(f"  Constant term: {result.constant_term:.6f}")

    # Extract matrices
    H = result.get_hamiltonian_matrix()
    S = result.get_overlap_matrix()

    print(f"\n  Hamiltonian matrix shape: {H.shape}")
    print(f"  Overlap matrix shape: {S.shape}")

    # Solve generalized eigenvalue problem
    eigenvalues = la.eigh(H, S, eigvals_only=True)

    # Convert to physical energies
    energies = eigenvalues * result.normalization + result.constant_term

    print(f"\n  All eigenvalues (scaled): {eigenvalues}")
    print(f"  All energies (Ha): {energies}")

    # Filter to valid Chebyshev range [-1, 1]
    mask = np.abs(eigenvalues) <= 1.0
    physical_energies = energies[mask]

    if len(physical_energies) > 0:
        ground_energy = physical_energies.min()
        print(f"\n  Ground state energy: {ground_energy:.6f} Ha")
        print(f"  Expected energy: -1.137 Ha")
        print(f"  Error: {abs(ground_energy - (-1.137)):.6f} Ha")

    return result


def run_qel_with_filtering():
    """QEL example with overlap matrix regularization."""
    print("\n" + "=" * 70)
    print("Example 2: QEL with Overlap Matrix Regularization")
    print("=" * 70)

    h2 = create_h2_hamiltonian()

    # Run with larger Krylov dimension
    result = solvers.quantum_exact_lanczos(
        h2,
        num_qubits=4,
        n_electrons=2,
        krylov_dim=8,
        verbose=True  # Enable verbose output
    )

    # Extract matrices
    H = result.get_hamiltonian_matrix()
    S = result.get_overlap_matrix()

    # Add small regularization to avoid numerical issues
    epsilon = 1e-12
    S_reg = S + epsilon * np.eye(result.krylov_dimension)

    print(f"\n  Added regularization: ε = {epsilon}")

    # Solve with regularized overlap
    eigenvalues = la.eigh(H, S_reg, eigvals_only=True)
    energies = eigenvalues * result.normalization + result.constant_term

    # Filter to Chebyshev range
    mask = np.abs(eigenvalues) <= 1.0
    physical_energies = energies[mask]

    print(f"\n  Total eigenvalues: {len(eigenvalues)}")
    print(f"  Physical eigenvalues (|λ| ≤ 1): {len(physical_energies)}")

    if len(physical_energies) > 0:
        sorted_energies = np.sort(physical_energies)
        print(f"\n  Lowest 3 energies:")
        for i, E in enumerate(sorted_energies[:3]):
            print(f"    E_{i}: {E:.6f} Ha")

        ground_energy = sorted_energies[0]
        print(f"\n  Ground state: {ground_energy:.6f} Ha")

    return result


def run_qel_convergence_study():
    """Study convergence with respect to Krylov dimension."""
    print("\n" + "=" * 70)
    print("Example 3: QEL Convergence Study")
    print("=" * 70)

    h2 = create_h2_hamiltonian()
    krylov_dimensions = [3, 5, 7, 10]

    print(f"\n{'Krylov Dim':<12} {'Ground State (Ha)':<20} {'Error (Ha)'}")
    print("-" * 50)

    exact_energy = -1.137  # Reference value

    for kdim in krylov_dimensions:
        result = solvers.quantum_exact_lanczos(h2,
                                               num_qubits=4,
                                               n_electrons=2,
                                               krylov_dim=kdim,
                                               verbose=False)

        # Extract and solve
        H = result.get_hamiltonian_matrix()
        S = result.get_overlap_matrix() + 1e-12 * np.eye(kdim)

        eigenvalues = la.eigh(H, S, eigvals_only=True)
        energies = eigenvalues * result.normalization + result.constant_term

        # Filter and find ground state
        mask = np.abs(eigenvalues) <= 1.0
        physical_energies = energies[mask]

        if len(physical_energies) > 0:
            ground = physical_energies.min()
            error = abs(ground - exact_energy)
            print(f"{kdim:<12} {ground:<20.6f} {error:.6e}")


def explore_block_encoding():
    """Explore the PauliLCU block encoding."""
    print("\n" + "=" * 70)
    print("Example 4: Exploring Block Encoding")
    print("=" * 70)

    h2 = create_h2_hamiltonian()

    # Create block encoding
    encoding = solvers.PauliLCU(h2, num_qubits=4)

    print(f"\nBlock Encoding Properties:")
    print(f"  Type: {type(encoding)}")
    print(f"  Is BlockEncoding: {isinstance(encoding, solvers.BlockEncoding)}")
    print(f"  Is PauliLCU: {isinstance(encoding, solvers.PauliLCU)}")
    print(f"  System qubits: {encoding.num_system}")
    print(f"  Ancilla qubits: {encoding.num_ancilla}")
    print(f"  Normalization: {encoding.normalization:.6f}")

    # Get internal data (for debugging)
    angles = encoding.get_angles()
    controls = encoding.get_term_controls()
    ops = encoding.get_term_ops()
    lengths = encoding.get_term_lengths()
    signs = encoding.get_term_signs()

    print(f"\n  State preparation angles: {len(angles)} total")
    print(f"  First 3 angles: {angles[:3]}")

    print(f"\n  Binary control patterns: {len(controls)} terms")
    print(f"  Control patterns: {controls}")

    print(f"\n  Term lengths: {lengths}")
    print(f"  Total operators: {len(ops)}")
    print(f"  Signs: {signs}")


def compare_with_exact_diagonalization():
    """Compare QEL with exact diagonalization (for small systems)."""
    print("\n" + "=" * 70)
    print("Example 5: Comparison with Exact Diagonalization")
    print("=" * 70)

    # Small system for exact comparison
    h_small = 0.5 * spin.z(0) + 0.3 * spin.z(1) + 0.2 * spin.x(0) * spin.x(1)

    # Run QEL
    result = solvers.quantum_exact_lanczos(h_small,
                                           num_qubits=2,
                                           n_electrons=1,
                                           krylov_dim=4,
                                           verbose=False)

    H = result.get_hamiltonian_matrix()
    S = result.get_overlap_matrix() + 1e-12 * np.eye(4)

    eigenvalues_qel = la.eigh(H, S, eigvals_only=True)
    energies_qel = eigenvalues_qel * result.normalization + result.constant_term

    mask = np.abs(eigenvalues_qel) <= 1.0
    qel_ground = energies_qel[mask].min()

    print(f"\n  QEL Ground State: {qel_ground:.6f} Ha")
    print(f"  (For full comparison, would need exact diagonalization code)")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("CUDA-Q Solvers: Quantum Exact Lanczos (QEL) Examples")
    print("=" * 70)

    # Run examples
    run_qel_basic()
    run_qel_with_filtering()
    run_qel_convergence_study()
    explore_block_encoding()
    compare_with_exact_diagonalization()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)

    print("\nKey Takeaways:")
    print("  1. QEL returns Krylov matrices (H, S) for user to diagonalize")
    print("  2. Block encoding (PauliLCU) handles Hamiltonian preparation")
    print("  3. Larger Krylov dimensions generally improve accuracy")
    print("  4. Eigenvalues should be filtered to |λ| ≤ 1 (Chebyshev range)")
    print("  5. Overlap matrix regularization helps numerical stability")


if __name__ == "__main__":
    main()
