#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Tests for Quantum Exact Lanczos (QEL) algorithm Python bindings."""

import sys
import os
# Add build directory to path if running tests directly
if 'cudaq_solvers' not in sys.modules:
    build_path = os.path.join(os.path.dirname(__file__),
                              '../../../../build/python')
    if os.path.exists(build_path):
        sys.path.insert(0, build_path)

import pytest
import numpy as np
import subprocess
from cudaq import spin
import cudaq_solvers as solvers
from cudaq_solvers.tools.qel_utils import (
    build_spin_hamiltonian, ground_state_energy_from_qel,
    solve_generalized_eigenvalues_filtered)
try:
    from scipy.linalg import eigh
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def is_nvidia_gpu_available():
    """Check if NVIDIA GPU is available using nvidia-smi command."""
    try:
        result = subprocess.run(["nvidia-smi"],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True)
        if result.returncode == 0 and "GPU" in result.stdout:
            return True
    except FileNotFoundError:
        # The nvidia-smi command is not found, indicating no NVIDIA GPU drivers
        return False
    return False


def test_pauli_lcu_simple():
    """Test PauliLCU with a simple Hamiltonian."""
    # H = 0.5*X + 0.3*Z
    h = 0.5 * spin.x(0) + 0.3 * spin.z(0)

    encoding = solvers.PauliLCU(h, num_qubits=1)

    # Check basic properties
    assert encoding.num_ancilla == 1, "Expected 1 ancilla for 2 terms"
    assert encoding.num_system == 1, "Expected 1 system qubit"
    assert abs(encoding.normalization - 0.8) < 1e-10, "Expected norm of 0.8"
    assert isinstance(encoding, solvers.PauliLCU)

    # Check debug methods return numpy arrays
    angles = encoding.get_angles()
    assert isinstance(angles, np.ndarray)
    assert len(angles) == 1

    controls = encoding.get_term_controls()
    assert isinstance(controls, np.ndarray)

    ops = encoding.get_term_ops()
    assert isinstance(ops, np.ndarray)

    lengths = encoding.get_term_lengths()
    assert isinstance(lengths, np.ndarray)

    signs = encoding.get_term_signs()
    assert isinstance(signs, np.ndarray)


def test_pauli_lcu_larger():
    """Test PauliLCU with a larger Hamiltonian."""
    # H2 Hamiltonian (simplified)
    h2 = (-1.0523732 + 0.39793742 * spin.z(0) - 0.39793742 * spin.z(1) -
          0.01128010 * spin.z(2) + 0.01128010 * spin.z(3) +
          0.18093120 * spin.x(0) * spin.x(1) * spin.y(2) * spin.y(3))

    encoding = solvers.PauliLCU(h2, num_qubits=4)

    # Should need log2(6 terms) = 3 ancilla
    assert encoding.num_ancilla == 3
    assert encoding.num_system == 4
    assert encoding.normalization > 0.0


def test_qel_result_structure():
    """Test QELResult structure and methods."""
    result = solvers.QELResult()

    # Set test values
    result.krylov_dimension = 5
    result.num_system = 4
    result.num_ancilla = 3
    result.normalization = 2.0
    result.constant_term = -1.05
    result.moments = [0.1, 0.2, 0.3, 0.4, 0.5]
    result.hamiltonian_matrix = list(range(25))  # 5x5 flattened
    result.overlap_matrix = list(range(25, 50))

    # Check properties
    assert result.krylov_dimension == 5
    assert result.num_system == 4
    assert result.num_ancilla == 3
    assert abs(result.normalization - 2.0) < 1e-10
    assert abs(result.constant_term - (-1.05)) < 1e-10

    # Check helper methods return properly shaped arrays
    H = result.get_hamiltonian_matrix()
    S = result.get_overlap_matrix()
    moments = result.get_moments()

    assert isinstance(H, np.ndarray)
    assert isinstance(S, np.ndarray)
    assert isinstance(moments, np.ndarray)

    assert H.shape == (5, 5)
    assert S.shape == (5, 5)
    assert len(moments) == 5

    # Check __repr__
    repr_str = repr(result)
    assert "QELResult" in repr_str
    assert "krylov_dimension=5" in repr_str


def test_quantum_exact_lanczos_function_exists():
    """Test that quantum_exact_lanczos function is properly exposed."""
    assert hasattr(solvers, 'quantum_exact_lanczos')
    assert callable(solvers.quantum_exact_lanczos)

    # Check docstring exists
    doc = solvers.quantum_exact_lanczos.__doc__
    assert doc is not None
    assert len(doc) > 100
    assert "Quantum Exact Lanczos" in doc


def test_quantum_exact_lanczos_simple_hamiltonian():
    """Test QEL with a simple Hamiltonian."""
    # Simple 2-qubit Hamiltonian
    h = 0.5 * spin.z(0) + 0.3 * spin.z(1) + 0.2 * spin.x(0) * spin.x(1)

    # Run with small Krylov dimension
    result = solvers.quantum_exact_lanczos(h,
                                           num_qubits=2,
                                           n_electrons=1,
                                           krylov_dim=3,
                                           shots=-1,
                                           verbose=False)

    # Check result structure
    assert result.krylov_dimension == 3
    assert result.num_system == 2
    assert result.num_ancilla >= 1

    # Check matrices are the right size
    H = result.get_hamiltonian_matrix()
    S = result.get_overlap_matrix()

    assert H.shape == (3, 3)
    assert S.shape == (3, 3)

    # Moments should have 2*krylov_dim entries
    moments = result.get_moments()
    assert len(moments) == 6


def test_quantum_exact_lanczos_default_parameters():
    """Test QEL with default parameters."""
    h = 0.7 * spin.z(0)

    # Should work with only required parameters
    result = solvers.quantum_exact_lanczos(h, num_qubits=1, n_electrons=0)

    # Default krylov_dim is 10
    assert result.krylov_dimension == 10
    assert result.num_system == 1


@pytest.mark.skipif(not HAS_SCIPY,
                    reason="SciPy required for eigenvalue solving")
def test_quantum_exact_lanczos_h2_molecule():
    """Test QEL with full H2 Hamiltonian from lanczos_h2_complete.py."""
    # H2 Hamiltonian terms (from lanczos_h2_complete.py lines 21-36)
    # FCI ground state: -1.137284 Ha
    # Constant term: -0.09706627 Ha
    terms = [
        ("Z0", 0.17141283),
        ("Z1", 0.17141283),
        ("Z2", -0.22343154),
        ("Z3", -0.22343154),
        ("Z0Z1", 0.16868898),
        ("Z0Z2", 0.12062523),
        ("Z1Z2", 0.16592785),
        ("Z0Z3", 0.16592785),
        ("Z1Z3", 0.12062523),
        ("Z2Z3", 0.17441288),
        ("X0X1Y2Y3", -0.04530262),
        ("X0Y1Y2X3", 0.04530262),
        ("Y0X1X2Y3", 0.04530262),
        ("Y0Y1X2X3", -0.04530262),
    ]

    CONSTANT_TERM = -0.09706627
    FCI_ENERGY = -1.137284

    # Build the Hamiltonian (WITHOUT constant term)
    # The constant term is kept separate to avoid inflating the 1-norm
    # This matches the approach in lanczos_h2_complete.py
    h2_hamiltonian = build_spin_hamiltonian(terms)

    # Run QEL algorithm
    print("\nRunning QEL for H2 molecule...")
    result = solvers.quantum_exact_lanczos(
        h2_hamiltonian,
        num_qubits=4,
        n_electrons=2,
        krylov_dim=5,
        shots=-1,  # Exact simulation
        verbose=True)

    # Check result metadata
    assert result.num_system == 4, "Expected 4 system qubits"
    assert result.krylov_dimension == 5, "Expected Krylov dimension 5"

    # Get Krylov matrices
    H = result.get_hamiltonian_matrix()
    S = result.get_overlap_matrix()
    moments = result.get_moments()

    print(f"\nH matrix shape: {H.shape}")
    print(f"S matrix shape: {S.shape}")
    print(f"Number of moments: {len(moments)}")

    # Check shapes
    assert H.shape == (5, 5), "H matrix should be 5x5"
    assert S.shape == (5, 5), "S matrix should be 5x5"
    assert len(moments) == 10, "Should have 10 moments (2*krylov_dim)"

    eigenvalues = solve_generalized_eigenvalues_filtered(
        H,
        S,
        threshold=1e-4,
        verbose=True,
        krylov_dimension=result.krylov_dimension,
    )

    # Ground state energy: eigenvalue * normalization + constant_term
    # The QEL eigenvalues are scaled by the 1-norm during block encoding
    # NOTE: Constant term was not added to Hamiltonian, so we add it manually here
    estimated_energy_scaled = eigenvalues[0]
    estimated_energy = ground_state_energy_from_qel(
        eigenvalues,
        normalization=result.normalization,
        constant_term=CONSTANT_TERM)

    print(f"\nScaled eigenvalue: {estimated_energy_scaled:.6f}")
    print(f"1-Norm: {result.normalization:.6f}")
    print(f"Constant: {CONSTANT_TERM:.6f} Ha (from data, not QEL)")
    print(f"Estimated ground state energy: {estimated_energy:.6f} Ha")
    print(f"Expected FCI energy:           {FCI_ENERGY:.6f} Ha")
    error = abs(estimated_energy - FCI_ENERGY)
    print(
        f"Absolute error:                {error:.6f} Ha ({error*1000:.2f} mHa)")

    # Check accuracy (allow reasonable tolerance for quantum simulation)
    # Using exact simulation (shots=-1), we should get excellent accuracy
    assert error < 0.01, f"Error too large: {error:.6f} Ha (expected < 0.01 Ha)"

    print("\n✅ H2 QEL test passed!")


@pytest.mark.skipif(not HAS_SCIPY,
                    reason="SciPy required for eigenvalue solving")
@pytest.mark.skipif(not is_nvidia_gpu_available(),
                    reason="NVIDIA GPU not found")
def test_quantum_exact_lanczos_lih_molecule():
    """Test QEL with LiH Hamiltonian from saved data file."""
    # Import LiH Hamiltonian data (no PySCF/OpenFermion required)
    try:
        from . import lih_hamiltonian_data
    except ImportError:
        try:
            import lih_hamiltonian_data
        except ImportError:
            pytest.skip("LiH Hamiltonian data file not found")

    terms = lih_hamiltonian_data.PAULI_TERMS
    CONSTANT_TERM = lih_hamiltonian_data.CONSTANT_TERM
    FCI_ENERGY = lih_hamiltonian_data.FCI_ENERGY
    NUM_QUBITS = lih_hamiltonian_data.NUM_QUBITS
    NUM_ELECTRONS = lih_hamiltonian_data.NUM_ELECTRONS

    print(f"\nLiH Molecule:")
    print(f"  Qubits: {NUM_QUBITS}")
    print(f"  Electrons: {NUM_ELECTRONS}")
    print(f"  Pauli terms: {len(terms)}")
    print(f"  Target FCI energy: {FCI_ENERGY:.6f} Ha")

    # Build the Hamiltonian (WITHOUT constant term)
    # The constant term is kept separate to avoid inflating the 1-norm
    # This matches the approach in lih_complete.py
    lih_hamiltonian = build_spin_hamiltonian(terms)

    # Run QEL algorithm with Krylov dimension 8 (matches Python script)
    print("\nRunning QEL for LiH molecule...")
    result = solvers.quantum_exact_lanczos(
        lih_hamiltonian,
        num_qubits=NUM_QUBITS,
        n_electrons=NUM_ELECTRONS,
        krylov_dim=8,  # Increased from 5 to match Python implementation
        shots=-1,  # Exact simulation
        verbose=True)

    # Check result metadata
    assert result.num_system == NUM_QUBITS
    assert result.krylov_dimension == 8

    # Get Krylov matrices
    H = result.get_hamiltonian_matrix()
    S = result.get_overlap_matrix()
    moments = result.get_moments()

    print(f"\nH matrix shape: {H.shape}")
    print(f"S matrix shape: {S.shape}")
    print(f"Number of moments: {len(moments)}")

    # Check shapes
    assert H.shape == (8, 8), "H matrix should be 8x8"
    assert S.shape == (8, 8), "S matrix should be 8x8"
    assert len(moments) == 16, "Should have 16 moments (2*krylov_dim)"

    eigenvalues = solve_generalized_eigenvalues_filtered(
        H,
        S,
        threshold=1e-4,
        verbose=True,
        krylov_dimension=result.krylov_dimension,
    )

    # Ground state energy: eigenvalue * normalization + constant_term
    # The QEL eigenvalues are scaled by the 1-norm during block encoding
    # NOTE: Constant term was not added to Hamiltonian, so we add it manually here
    estimated_energy_scaled = eigenvalues[0]
    estimated_energy = ground_state_energy_from_qel(
        eigenvalues,
        normalization=result.normalization,
        constant_term=CONSTANT_TERM)

    print(f"\nScaled eigenvalue: {estimated_energy_scaled:.6f}")
    print(f"1-Norm: {result.normalization:.6f}")
    print(f"Constant: {CONSTANT_TERM:.6f} Ha (from data, not QEL)")
    print(f"Estimated ground state energy: {estimated_energy:.6f} Ha")
    print(f"Expected FCI energy:           {FCI_ENERGY:.6f} Ha")
    error = abs(estimated_energy - FCI_ENERGY)
    print(
        f"Absolute error:                {error:.6f} Ha ({error*1000:.2f} mHa)")

    # Check accuracy - should match Python script accuracy (~0.3 mHa)
    assert error < 0.01, f"Error too large: {error:.6f} Ha (expected < 0.01 Ha = 10 mHa)"

    print("\n✅ LiH QEL test passed!")


def test_quantum_exact_lanczos_n2_molecule():
    """Test QEL with N2 Hamiltonian from saved data file."""
    # Import N2 Hamiltonian data (no PySCF/OpenFermion required)
    try:
        from . import n2_hamiltonian_data
    except ImportError:
        try:
            import n2_hamiltonian_data
        except ImportError:
            pytest.skip("N2 Hamiltonian data file not found")

    terms = n2_hamiltonian_data.PAULI_TERMS
    CONSTANT_TERM = n2_hamiltonian_data.CONSTANT_TERM
    CASCI_ENERGY = n2_hamiltonian_data.CASCI_ENERGY  # Target energy from CASCI
    NUM_QUBITS = n2_hamiltonian_data.NUM_QUBITS
    NUM_ELECTRONS = n2_hamiltonian_data.NUM_ELECTRONS

    print(f"\nN2 Molecule:")
    print(f"  Qubits: {NUM_QUBITS}")
    print(f"  Electrons: {NUM_ELECTRONS}")
    print(f"  Pauli terms: {len(terms)}")
    print(f"  Target CASCI energy: {CASCI_ENERGY:.6f} Ha")

    # Build the Hamiltonian (WITHOUT constant term)
    # The constant term is kept separate to avoid inflating the 1-norm
    n2_hamiltonian = build_spin_hamiltonian(terms)

    # Run QEL algorithm with Krylov dimension 8
    print("\nRunning QEL for N2 molecule...")
    result = solvers.quantum_exact_lanczos(
        n2_hamiltonian,
        num_qubits=NUM_QUBITS,
        n_electrons=NUM_ELECTRONS,
        krylov_dim=8,
        shots=-1,  # Exact simulation
        verbose=True)

    # Check result metadata
    assert result.num_system == NUM_QUBITS
    assert result.krylov_dimension == 8

    # Get Krylov matrices
    H = result.get_hamiltonian_matrix()
    S = result.get_overlap_matrix()
    moments = result.get_moments()

    print(f"\nH matrix shape: {H.shape}")
    print(f"S matrix shape: {S.shape}")
    print(f"Number of moments: {len(moments)}")

    # Check shapes
    assert H.shape == (8, 8), "H matrix should be 8x8"
    assert S.shape == (8, 8), "S matrix should be 8x8"
    assert len(moments) == 16, "Should have 16 moments (2*krylov_dim)"

    eigenvalues = solve_generalized_eigenvalues_filtered(
        H,
        S,
        threshold=1e-4,
        verbose=True,
        krylov_dimension=result.krylov_dimension,
    )

    # Ground state energy: eigenvalue * normalization + constant_term
    # The QEL eigenvalues are scaled by the 1-norm during block encoding
    # NOTE: Constant term was not added to Hamiltonian, so we add it manually here
    estimated_energy_scaled = eigenvalues[0]
    estimated_energy = ground_state_energy_from_qel(
        eigenvalues,
        normalization=result.normalization,
        constant_term=CONSTANT_TERM)

    print(f"\nScaled eigenvalue: {estimated_energy_scaled:.6f}")
    print(f"1-Norm: {result.normalization:.6f}")
    print(f"Constant: {CONSTANT_TERM:.6f} Ha (from data, not QEL)")
    print(f"Estimated ground state energy: {estimated_energy:.6f} Ha")
    print(f"Expected CASCI energy:         {CASCI_ENERGY:.6f} Ha")
    error = abs(estimated_energy - CASCI_ENERGY)
    print(
        f"Absolute error:                {error:.6f} Ha ({error*1000:.2f} mHa)")

    # Check accuracy - should be reasonably close
    assert error < 0.01, f"Error too large: {error:.6f} Ha (expected < 0.01 Ha = 10 mHa)"

    print("\n✅ N2 QEL test passed!")


def test_quantum_exact_lanczos_h2o_molecule():
    """Test QEL with H2O using pre-extracted Hamiltonian data."""
    # Import H2O Hamiltonian data
    from h2o_hamiltonian_data import (PAULI_TERMS, CONSTANT_TERM, NUM_QUBITS,
                                      NUM_ELECTRONS, CASCI_ENERGY)

    terms = PAULI_TERMS

    print(f"\nH2O Molecule:")
    print(f"  Qubits: {NUM_QUBITS}")
    print(f"  Electrons: {NUM_ELECTRONS}")
    print(f"  Pauli terms: {len(terms)}")
    print(f"  Target CASCI energy: {CASCI_ENERGY:.6f} Ha")

    # Build the Hamiltonian (WITHOUT constant term)
    # The constant term is kept separate to avoid inflating the 1-norm
    h2o_hamiltonian = build_spin_hamiltonian(terms)

    print("\nRunning QEL for H2O molecule...")

    # Run QEL
    result = solvers.quantum_exact_lanczos(h2o_hamiltonian,
                                           num_qubits=NUM_QUBITS,
                                           n_electrons=NUM_ELECTRONS,
                                           krylov_dim=8,
                                           shots=-1,
                                           verbose=True)

    # Get Krylov matrices
    H = result.get_hamiltonian_matrix()
    S = result.get_overlap_matrix()
    moments = result.get_moments()

    print(f"\nH matrix shape: {H.shape}")
    print(f"S matrix shape: {S.shape}")
    print(f"Number of moments: {len(moments)}")

    # Check shapes
    assert H.shape == (8, 8), "H matrix should be 8x8"
    assert S.shape == (8, 8), "S matrix should be 8x8"
    assert len(moments) == 16, "Should have 16 moments (2*krylov_dim)"

    eigenvalues = solve_generalized_eigenvalues_filtered(
        H,
        S,
        threshold=1e-4,
        verbose=True,
        krylov_dimension=result.krylov_dimension,
    )

    # Ground state energy: eigenvalue * normalization + constant_term
    # The QEL eigenvalues are scaled by the 1-norm during block encoding
    # NOTE: Constant term was not added to Hamiltonian, so we add it manually here
    estimated_energy_scaled = eigenvalues[0]
    estimated_energy = ground_state_energy_from_qel(
        eigenvalues,
        normalization=result.normalization,
        constant_term=CONSTANT_TERM)

    print(f"\nScaled eigenvalue: {estimated_energy_scaled:.6f}")
    print(f"1-Norm: {result.normalization:.6f}")
    print(f"Constant: {CONSTANT_TERM:.6f} Ha (from data, not QEL)")
    print(f"Estimated ground state energy: {estimated_energy:.6f} Ha")
    print(f"Expected CASCI energy:         {CASCI_ENERGY:.6f} Ha")
    error = abs(estimated_energy - CASCI_ENERGY)
    print(
        f"Absolute error:                {error:.6f} Ha ({error*1000:.2f} mHa)")

    # Check accuracy - should be reasonably close
    assert error < 0.01, f"Error too large: {error:.6f} Ha (expected < 0.01 Ha = 10 mHa)"

    print("\n✅ H2O QEL test passed!")


@pytest.mark.skipif(not HAS_SCIPY,
                    reason="SciPy required for eigenvalue solving")
def test_quantum_exact_lanczos_benzene_molecule():
    """Test QEL on Benzene (C6H6) molecule with (4e, 4o) active space."""
    try:
        from . import benzene_hamiltonian_data
    except ImportError:
        import benzene_hamiltonian_data

    NUM_QUBITS = benzene_hamiltonian_data.NUM_QUBITS
    NUM_ELECTRONS = benzene_hamiltonian_data.NUM_ELECTRONS
    CONSTANT_TERM = benzene_hamiltonian_data.CONSTANT_TERM
    CASCI_ENERGY = benzene_hamiltonian_data.CASCI_ENERGY

    terms = benzene_hamiltonian_data.PAULI_TERMS

    print(f"\n{'='*70}")
    print(f"Testing Benzene (C6H6) molecule - (4e, 4o) active space")
    print(f"{'='*70}")
    print(f"Number of qubits: {NUM_QUBITS}")
    print(f"Number of electrons: {NUM_ELECTRONS}")
    print(f"Number of Pauli terms: {len(terms)}")
    print(f"Constant term: {CONSTANT_TERM:.6f} Ha")
    print(f"Expected CASCI energy: {CASCI_ENERGY:.6f} Ha")

    # Build the Hamiltonian (WITHOUT constant term)
    # The constant term is kept separate to avoid inflating the 1-norm
    benzene_hamiltonian = build_spin_hamiltonian(terms)

    print("\nRunning QEL for Benzene molecule...")

    # Run QEL
    result = solvers.quantum_exact_lanczos(benzene_hamiltonian,
                                           num_qubits=NUM_QUBITS,
                                           n_electrons=NUM_ELECTRONS,
                                           krylov_dim=8,
                                           shots=-1,
                                           verbose=True)

    # Get Krylov matrices
    H = result.get_hamiltonian_matrix()
    S = result.get_overlap_matrix()
    moments = result.get_moments()

    print(f"\nH matrix shape: {H.shape}")
    print(f"S matrix shape: {S.shape}")
    print(f"Number of moments: {len(moments)}")

    # Check shapes
    assert H.shape == (8, 8), "H matrix should be 8x8"
    assert S.shape == (8, 8), "S matrix should be 8x8"
    assert len(moments) == 16, "Should have 16 moments (2*krylov_dim)"

    eigenvalues = solve_generalized_eigenvalues_filtered(
        H,
        S,
        threshold=1e-4,
        verbose=True,
        krylov_dimension=result.krylov_dimension,
    )

    # Ground state energy: eigenvalue * normalization + constant_term
    # The QEL eigenvalues are scaled by the 1-norm during block encoding
    # NOTE: Constant term was not added to Hamiltonian, so we add it manually here
    estimated_energy_scaled = eigenvalues[0]
    estimated_energy = ground_state_energy_from_qel(
        eigenvalues,
        normalization=result.normalization,
        constant_term=CONSTANT_TERM)

    print(f"\nScaled eigenvalue: {estimated_energy_scaled:.6f}")
    print(f"1-Norm: {result.normalization:.6f}")
    print(f"Constant: {CONSTANT_TERM:.6f} Ha (from data, not QEL)")
    print(f"Estimated ground state energy: {estimated_energy:.6f} Ha")
    print(f"Expected CASCI energy:         {CASCI_ENERGY:.6f} Ha")
    error = abs(estimated_energy - CASCI_ENERGY)
    print(
        f"Absolute error:                {error:.6f} Ha ({error*1000:.2f} mHa)")

    # Check accuracy - should be reasonably close
    assert error < 0.01, f"Error too large: {error:.6f} Ha (expected < 0.01 Ha = 10 mHa)"

    print("\n✅ Benzene QEL test passed!")


def test_pauli_lcu_interface():
    """Test that PauliLCU has all expected methods."""
    h = 0.7 * spin.z(0) + 0.3 * spin.x(0)
    encoding = solvers.PauliLCU(h, num_qubits=1)

    # Check type
    assert isinstance(encoding, solvers.PauliLCU)

    # Check all block encoding methods exist
    assert hasattr(encoding, 'prepare')
    assert hasattr(encoding, 'unprepare')
    assert hasattr(encoding, 'select')
    assert hasattr(encoding, 'apply')
    assert hasattr(encoding, 'num_ancilla')
    assert hasattr(encoding, 'num_system')
    assert hasattr(encoding, 'normalization')

    # Check PauliLCU-specific methods
    assert hasattr(encoding, 'get_angles')
    assert hasattr(encoding, 'get_term_controls')
    assert hasattr(encoding, 'get_term_ops')
    assert hasattr(encoding, 'get_term_lengths')
    assert hasattr(encoding, 'get_term_signs')


if __name__ == "__main__":
    # Allow running tests individually for debugging
    test_pauli_lcu_simple()
    test_pauli_lcu_larger()
    test_qel_result_structure()
    test_quantum_exact_lanczos_function_exists()
    test_quantum_exact_lanczos_simple_hamiltonian()
    test_quantum_exact_lanczos_default_parameters()
    if HAS_SCIPY:
        test_quantum_exact_lanczos_h2_molecule()
    else:
        print("Skipping H2 molecule test (SciPy not available)")
    test_pauli_lcu_interface()

    print("\nAll tests passed!")
