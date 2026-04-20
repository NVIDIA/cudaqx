/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "bindings/utils/type_casters.h"
#include "cuda-qx/core/kwargs_utils.h"
#include "cudaq/solvers/operators/block_encoding.h"
#include "cudaq/solvers/quantum_exact_lanczos.h"

namespace py = pybind11;
using namespace cudaqx;

namespace cudaq::solvers {

void bindBlockEncoding(py::module &mod) {

  // ============================================================================
  // PAULI LCU BLOCK ENCODING
  // ============================================================================

  py::class_<pauli_lcu>(
      mod, "PauliLCU",
      R"(Block encoding using Pauli Linear Combination of Unitaries.

This implementation is optimized for Hamiltonians expressed as sums of Pauli
strings (e.g., molecular Hamiltonians from quantum chemistry). It uses:
  - PREPARE: State preparation tree with controlled rotations
  - SELECT: Controlled Pauli operations indexed by ancilla state

The encoding uses log₂(# terms) ancilla qubits and achieves α = ||H||₁.

Example:
    >>> from cudaq import spin
    >>> import cudaq_solvers as solvers
    >>> 
    >>> # Define Hamiltonian
    >>> h = 0.5 * spin.x(0) + 0.3 * spin.z(0)
    >>> 
    >>> # Create block encoding
    >>> encoding = solvers.PauliLCU(h, num_qubits=1)
    >>> 
    >>> print(f"Ancilla qubits needed: {encoding.num_ancilla}")
    >>> print(f"Normalization: {encoding.normalization}")
    >>> 
    >>> # Use in quantum kernel
    >>> @cudaq.kernel
    >>> def my_circuit():
    >>>     anc = cudaq.qvector(encoding.num_ancilla)
    >>>     sys = cudaq.qvector(encoding.num_system)
    >>>     encoding.apply(anc, sys))")
      .def(py::init<const cudaq::spin_op &, std::size_t>(),
           py::arg("hamiltonian"), py::arg("num_qubits"),
           R"(Initialize Pauli LCU block encoding.

Args:
    hamiltonian: Target Hamiltonian as a SpinOperator
    num_qubits: Number of system qubits

Raises:
    RuntimeError: If Hamiltonian contains complex coefficients
    RuntimeError: If Hamiltonian has no terms)")
      .def_property_readonly("num_ancilla", &pauli_lcu::num_ancilla,
                             "Number of ancilla qubits: ⌈log₂(# terms)⌉")
      .def_property_readonly("num_system", &pauli_lcu::num_system,
                             "Number of system qubits")
      .def_property_readonly("normalization", &pauli_lcu::normalization,
                             "Normalization constant: α = ||H||₁ (1-norm)")
      .def("prepare", &pauli_lcu::prepare, py::arg("ancilla"),
           R"(Apply the PREPARE operation to ancilla qubits.
          
Prepares a superposition state on the ancilla qubits that
encodes the coefficients of the Hamiltonian terms.

Args:
    ancilla: View of ancilla qubits)")
      .def("unprepare", &pauli_lcu::unprepare, py::arg("ancilla"),
           R"(Apply the PREPARE† (adjoint/uncomputation) operation.

Args:
    ancilla: View of ancilla qubits)")
      .def("select", &pauli_lcu::select, py::arg("ancilla"), py::arg("system"),
           R"(Apply the SELECT operation.
          
Applies the appropriate Hamiltonian term conditioned on the
ancilla register state.

Args:
    ancilla: View of ancilla qubits (control register)
    system: View of system qubits (target register))")
      .def("apply", &pauli_lcu::apply, py::arg("ancilla"), py::arg("system"),
           R"(Apply the full block encoding: PREPARE → SELECT → PREPARE†.

Args:
    ancilla: View of ancilla qubits
    system: View of system qubits)")
      .def(
          "get_angles",
          [](const pauli_lcu &self) {
            const auto &angles = self.get_angles();
            return py::array_t<double>(angles.size(), angles.data());
          },
          "Get state preparation angles as NumPy array (for debugging)")
      .def(
          "get_term_controls",
          [](const pauli_lcu &self) {
            const auto &controls = self.get_term_controls();
            return py::array_t<int>(controls.size(), controls.data());
          },
          "Get binary control patterns as NumPy array (for debugging)")
      .def(
          "get_term_ops",
          [](const pauli_lcu &self) {
            const auto &ops = self.get_term_ops();
            return py::array_t<int>(ops.size(), ops.data());
          },
          "Get flattened Pauli operations as NumPy array (for debugging)")
      .def(
          "get_term_lengths",
          [](const pauli_lcu &self) {
            const auto &lengths = self.get_term_lengths();
            return py::array_t<int>(lengths.size(), lengths.data());
          },
          "Get number of operators per term as NumPy array (for debugging)")
      .def(
          "get_term_signs",
          [](const pauli_lcu &self) {
            const auto &signs = self.get_term_signs();
            return py::array_t<int>(signs.size(), signs.data());
          },
          "Get sign of each coefficient as NumPy array (for debugging)");

  // ============================================================================
  // QUANTUM EXACT LANCZOS RESULT
  // ============================================================================

  py::class_<qel_result>(mod, "QELResult",
                         R"(Result from Quantum Exact Lanczos algorithm.

Contains Krylov matrices that can be diagonalized to extract eigenvalues.
Solve the generalized eigenvalue problem H|v⟩ = E·S|v⟩ to get scaled
eigenvalues, then convert to physical energies:
    E_physical = E_scaled * normalization + constant_term

Example:
    >>> import numpy as np
    >>> from scipy import linalg as la
    >>> 
    >>> result = quantum_exact_lanczos(hamiltonian, ...)
    >>> 
    >>> # Reshape flat matrices
    >>> H = result.hamiltonian_matrix.reshape(
    >>>     result.krylov_dimension, result.krylov_dimension
    >>> )
    >>> S = result.overlap_matrix.reshape(
    >>>     result.krylov_dimension, result.krylov_dimension
    >>> )
    >>> 
    >>> # Solve generalized eigenvalue problem
    >>> eigenvalues = la.eigh(H, S, eigvals_only=True)
    >>> 
    >>> # Convert to physical energies
    >>> energies = eigenvalues * result.normalization + result.constant_term
    >>> ground_state = energies.min())")
      .def(py::init<>())
      .def_readwrite(
          "hamiltonian_matrix", &qel_result::hamiltonian_matrix,
          "Krylov H matrix (flattened, row-major, krylov_dim × krylov_dim)")
      .def_readwrite(
          "overlap_matrix", &qel_result::overlap_matrix,
          "Krylov S matrix (flattened, row-major, krylov_dim × krylov_dim)")
      .def_readwrite("moments", &qel_result::moments,
                     "Collected moments: μₖ = ⟨ψ|Tₖ(H)|ψ⟩")
      .def_readwrite("krylov_dimension", &qel_result::krylov_dimension,
                     "Dimension of the Krylov subspace")
      .def_readwrite("constant_term", &qel_result::constant_term,
                     "Constant term from Hamiltonian (add to eigenvalues)")
      .def_readwrite("normalization", &qel_result::normalization,
                     "Normalization constant α = ||H||₁")
      .def_readwrite("num_ancilla", &qel_result::num_ancilla,
                     "Number of ancilla qubits used")
      .def_readwrite("num_system", &qel_result::num_system,
                     "Number of system qubits")
      .def(
          "get_hamiltonian_matrix",
          [](const qel_result &self) {
            int dim = self.krylov_dimension;
            auto np_array = py::array_t<double>({dim, dim});
            auto buf = np_array.request();
            double *ptr = static_cast<double *>(buf.ptr);
            std::copy(self.hamiltonian_matrix.begin(),
                      self.hamiltonian_matrix.end(), ptr);
            return np_array;
          },
          "Get Hamiltonian matrix as 2D NumPy array")
      .def(
          "get_overlap_matrix",
          [](const qel_result &self) {
            int dim = self.krylov_dimension;
            auto np_array = py::array_t<double>({dim, dim});
            auto buf = np_array.request();
            double *ptr = static_cast<double *>(buf.ptr);
            std::copy(self.overlap_matrix.begin(), self.overlap_matrix.end(),
                      ptr);
            return np_array;
          },
          "Get overlap matrix as 2D NumPy array")
      .def(
          "get_moments",
          [](const qel_result &self) {
            return py::array_t<double>(self.moments.size(),
                                       self.moments.data());
          },
          "Get moments as NumPy array")
      .def("__repr__", [](const qel_result &self) {
        std::ostringstream oss;
        oss << "QELResult(krylov_dimension=" << self.krylov_dimension
            << ", num_system=" << self.num_system
            << ", num_ancilla=" << self.num_ancilla
            << ", normalization=" << self.normalization
            << ", constant_term=" << self.constant_term << ")";
        return oss.str();
      });

  // ============================================================================
  // QUANTUM EXACT LANCZOS FUNCTION
  // ============================================================================

  mod.def(
      "quantum_exact_lanczos",
      [](const cudaq::spin_op &hamiltonian, std::size_t num_qubits,
         std::size_t n_electrons, py::kwargs options) {
        heterogeneous_map opts;
        opts.insert("krylov_dim", getValueOr<int>(options, "krylov_dim", 10));
        opts.insert("shots", getValueOr<int>(options, "shots", -1));
        opts.insert("verbose", getValueOr<bool>(options, "verbose", false));

        return quantum_exact_lanczos(hamiltonian, num_qubits, n_electrons,
                                     opts);
      },
      py::arg("hamiltonian"), py::arg("num_qubits"), py::arg("n_electrons"),
      R"(Run Quantum Exact Lanczos algorithm to compute ground state energy.

Uses block encoding and amplitude amplification to build a Krylov subspace
via quantum moment collection. Returns matrices for classical eigenvalue
extraction.

Args:
    hamiltonian: Target Hamiltonian as SpinOperator
    num_qubits: Number of system qubits
    n_electrons: Number of electrons (for Hartree-Fock initialization)

Keyword Args:
    krylov_dim (int): Dimension of Krylov subspace (default: 10)
    shots (int): Number of measurement shots, -1 for exact (default: -1)
    verbose (bool): Enable detailed output (default: False)

Returns:
    QELResult: Contains Krylov matrices and metadata

Example (basic usage):
    >>> from cudaq import spin
    >>> import cudaq_solvers as solvers
    >>> import numpy as np
    >>> from scipy import linalg as la
    >>> 
    >>> # H2 Hamiltonian
    >>> h2 = -1.05 + 0.4*spin.z(0) - 0.4*spin.z(1) + 0.18*spin.x(0)*spin.x(1)
    >>> 
    >>> # Run QEL
    >>> result = solvers.quantum_exact_lanczos(
    >>>     h2, num_qubits=2, n_electrons=2, krylov_dim=5
    >>> )
    >>> 
    >>> # Extract ground state energy
    >>> H = result.get_hamiltonian_matrix()
    >>> S = result.get_overlap_matrix()
    >>> eigenvalues = la.eigh(H, S, eigvals_only=True)
    >>> energies = eigenvalues * result.normalization + result.constant_term
    >>> ground_state = energies.min()
    >>> print(f"Ground state: {ground_state:.6f} Ha")

Example (with filtering):
    >>> result = solvers.quantum_exact_lanczos(h2, 2, 2, krylov_dim=8)
    >>> 
    >>> H = result.get_hamiltonian_matrix()
    >>> S = result.get_overlap_matrix() + 1e-12 * np.eye(8)  # Regularization
    >>> 
    >>> eigenvalues = la.eigh(H, S, eigvals_only=True)
    >>> 
    >>> # Filter to Chebyshev range
    >>> mask = np.abs(eigenvalues) <= 1.0
    >>> physical_evals = eigenvalues[mask] * result.normalization + result.constant_term
    >>> ground_state = physical_evals.min()

Notes:
    - Requires ⌈log₂(# terms)⌉ ancilla qubits
    - Supports Hamiltonians with up to 1024 terms (10 ancilla)
    - Returns matrices for user to diagonalize with preferred library
    - Initial state is Hartree-Fock (first n_electrons qubits in |1⟩))");
}

} // namespace cudaq::solvers
