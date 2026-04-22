/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include <limits>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include "cudaq/python/PythonCppInterop.h"

#include "cudaq/solvers/adapt.h"
#include "cudaq/solvers/qaoa.h"
#include "cudaq/solvers/stateprep/ceo.h"
#include "cudaq/solvers/stateprep/uccgsd.h"
#include "cudaq/solvers/stateprep/upccgsd.h"

#include "cudaq/solvers/stateprep/uccsd.h"
#include "cudaq/solvers/version.h"
#include "cudaq/solvers/vqe.h"

#include "cudaq/solvers/operators/graph/clique.h"
#include "cudaq/solvers/operators/graph/max_cut.h"
#include "cudaq/solvers/operators/molecule.h"
#include "cudaq/solvers/operators/molecule/fermion_compiler.h"
#include "cudaq/solvers/operators/operator_pool.h"

#include "bindings/utils/type_casters.h"
#include "cuda-qx/core/kwargs_utils.h"

namespace nb = nanobind;

using namespace cudaqx;

namespace cudaq::solvers {

cudaqx::graph convert_networkx_graph(nb::object nx_graph) {
  cudaqx::graph g;

  // Get nodes from NetworkX graph
  nb::list nodes(nx_graph.attr("nodes")());

  // Add nodes and their weights if present
  for (const auto &node : nodes) {
    int node_id = nb::cast<int>(node);

    // Try to get node weight if it exists
    try {
      nb::dict node_data = nb::cast<nb::dict>(nx_graph.attr("nodes")[node]);
      if (node_data.contains("weight")) {
        double weight = nb::cast<double>(node_data["weight"]);
        g.add_node(node_id, weight);
      } else {
        g.add_node(node_id);
      }
    } catch (...) {
      // If no node attributes, add with default weight
      g.add_node(node_id);
    }
  }

  // Get edges from NetworkX graph
  nb::list edges(nx_graph.attr("edges")());

  // Add edges and their weights if present
  for (const auto &edge : edges) {
    nb::tuple edge_tuple(edge);
    int u = nb::cast<int>(edge_tuple[0]);
    int v = nb::cast<int>(edge_tuple[1]);

    // Try to get edge weight if it exists
    try {
      nb::dict edge_data = nb::cast<nb::dict>(nx_graph.attr("edges")[edge]);
      if (edge_data.contains("weight")) {
        double weight = nb::cast<double>(edge_data["weight"]);
        g.add_edge(u, v, weight);
      } else {
        g.add_edge(u, v);
      }
    } catch (...) {
      // If no edge attributes, add with default weight
      g.add_edge(u, v);
    }
  }

  return g;
}

/// @class PythonOptimizer
/// @brief A nanobind wrapper around SciPy's function optimization.
///
/// This class provides an interface to use SciPy's optimization functions
/// within a C++ environment using nanobind. It inherits from the
/// `optim::optimizer` class and overrides its methods to utilize SciPy's
/// optimization capabilities.
class PythonOptimizer : public optim::optimizer {
private:
  nb::callable minimize;
  nb::kwargs kwargs;
  std::vector<double> initParams;

public:
  using optimizer::optimize;

  /// @brief Constructor for PythonOptimizer
  /// @param optCallback The SciPy optimization function (e.g.,
  /// scipy.optimize.minimize)
  /// @param kw Keyword arguments to pass to the optimization function
  /// @param init Initial parameters for optimization (optional)
  PythonOptimizer(nb::callable optCallback, nb::kwargs kw,
                  const std::vector<double> init = {})
      : minimize(optCallback), kwargs(kw), initParams(init) {}

  /// @brief always false
  bool requiresGradients() const override { return false; }

  /// @brief Performs optimization using the SciPy minimize function
  /// @param dim Dimension of the optimization problem
  /// @param opt_function The function to be optimized
  /// @param options Additional options for the optimizer (not used in this
  /// implementation)
  /// @return A tuple containing the optimal function value and the optimal
  /// parameters
  optimization_result optimize(std::size_t dim,
                               const optim::optimizable_function &opt_function,
                               const heterogeneous_map &options) override {
    if (kwargs.contains("gradient"))
      kwargs.attr("pop")("gradient");

    if (kwargs.contains("optimizer"))
      kwargs.attr("pop")("optimizer");

    if (kwargs.contains("verbose"))
      kwargs.attr("pop")("verbose");

    if (kwargs.contains("shots"))
      kwargs.attr("pop")("shots");

    if (initParams.empty())
      initParams.resize(dim);

    double value = 0.0;
    std::vector<double> parameters(dim);
    auto result = minimize(nb::cpp_function([&](const std::vector<double> &x) {
                             std::vector<double> dx(x.size());
                             value = opt_function(x, dx);
                             parameters = x;
                             return value;
                           }),
                           initParams, **kwargs);
    return std::make_tuple(value, parameters);
  }
};

void addStatePrepKernels(nb::module_ &mod) {
  cudaq::python::addDeviceKernelInterop<
      cudaq::qview<>, const std::vector<double> &, std::size_t, std::size_t>(
      mod, "stateprep", "uccsd",
      "Unitary Coupled Cluster Singles Doubles Ansatz. Takes as input the "
      "qubits to apply the ansatz on, the rotational parameters, the number of "
      "electrons in the system, and the total spin (the number of unpaired "
      "electrons).");

  // Add this for UCCGSD
  cudaq::python::addDeviceKernelInterop<
      cudaq::qview<>, const std::vector<double> &,
      const std::vector<std::vector<cudaq::pauli_word>> &,
      const std::vector<std::vector<double>> &>(
      mod, "stateprep", "uccgsd",
      "Unitary Coupled Cluster Generalized Singles Doubles Ansatz. "
      "Takes as input the qubits, grouped rotational parameters, grouped Pauli "
      "words, "
      "and grouped coefficients.");
  cudaq::python::addDeviceKernelInterop<
      cudaq::qview<>, const std::vector<double> &,
      const std::vector<std::vector<cudaq::pauli_word>> &,
      const std::vector<std::vector<double>> &>(
      mod, "stateprep", "upccgsd",
      "Unitary Coupled Cluster Generalized Singles and Paired Doubles Ansatz. "
      "Takes as input the qubits, grouped rotational parameters, grouped Pauli "
      "words, "
      "and grouped coefficients.");
  cudaq::python::addDeviceKernelInterop<cudaq::qview<>, double, std::size_t,
                                        std::size_t>(
      mod, "stateprep", "single_excitation",
      "Perform a single fermionic excitation.");
  cudaq::python::addDeviceKernelInterop<cudaq::qview<>, double, std::size_t,
                                        std::size_t, std::size_t, std::size_t>(
      mod, "stateprep", "double_excitation",
      "Perform a double fermionic excitation.");
  auto stateprep = nb::cast<nb::module_>(mod.attr("stateprep"));
  stateprep.def("get_num_uccsd_parameters",
                &cudaq::solvers::stateprep::get_num_uccsd_parameters,
                nb::arg("num_electrons"), nb::arg("num_qubits"),
                nb::arg("spin") = 0,
                "Calculate the number of UCCSD parameters\n\n"
                "Args:\n"
                "    num_electrons (int): Number of electrons\n"
                "    num_qubits (int): Number of qubits\n"
                "    spin (int): Spin value. Optional, defaults to 0.\n\n"
                "Returns:\n"
                "    int: Number of UCCSD parameters");
  stateprep.def("get_uccsd_excitations",
                &cudaq::solvers::stateprep::get_uccsd_excitations, "");

  stateprep.def("get_uccgsd_pauli_lists",
                &cudaq::solvers::stateprep::get_uccgsd_pauli_lists,
                nb::arg("num_qubits"), nb::arg("only_singles") = false,
                nb::arg("only_doubles") = false,
                R"(
  Generate UCCGSD operator pool (Python-style unique singles/doubles) and extract Pauli words and coefficients grouped by excitation.

  Args:
      num_qubits (int): Number of spin orbitals (qubits).
      only_doubles (bool): If True, only double excitations.

  Returns:
      Tuple[List[List[PauliWord]], List[List[float]]]: Pauli words and coefficients grouped by excitation.
  )");
  stateprep.def("get_upccgsd_pauli_lists",
                &cudaq::solvers::stateprep::get_upccgsd_pauli_lists,
                nb::arg("num_qubits"), nb::arg("only_doubles") = false,
                R"(
  Generate UpCCGSD operator pool (Python-style unique singles/doubles) and extract Pauli words and coefficients grouped by excitation.

  Args:
      num_qubits (int): Number of spin orbitals (qubits).
      only_doubles (bool): If True, only double excitations.

  Returns:
      Tuple[List[List[PauliWord]], List[List[float]]]: Pauli words and coefficients grouped by excitation.
  )");

  // Add CEO state preparation
  cudaq::python::addDeviceKernelInterop<
      cudaq::qview<>, const std::vector<double> &,
      const std::vector<std::vector<cudaq::pauli_word>> &,
      const std::vector<std::vector<double>> &>(
      mod, "stateprep", "ceo",
      "CEO (Coupled Exchange Operator) Ansatz. "
      "Takes as input the qubits, grouped rotational parameters, grouped Pauli "
      "words, "
      "and grouped coefficients.");

  stateprep.def("get_ceo_pauli_lists",
                &cudaq::solvers::stateprep::get_ceo_pauli_lists,
                nb::arg("num_orbitals"),
                R"(
  Generate CEO operator pool (spin-dependent generalized singles/doubles coupled qubit excitation operators) and extract Pauli words and coefficients grouped by excitation.

  Args:
      num_orbitals (int): Number of spatial orbitals.

  Returns:
      Tuple[List[List[PauliWord]], List[List[float]]]: Pauli words and coefficients grouped by excitation.
  )");
}

// Helper function to convert tensor to numpy array
template <typename T = std::complex<double>>
nb::object tensor_to_numpy(const cudaqx::tensor<T> &tensor_data) {
  const auto &shape = tensor_data.shape();
  std::size_t total_size = tensor_data.size();

  T *data_copy = new T[total_size];
  std::copy(tensor_data.data(), tensor_data.data() + total_size, data_copy);

  std::vector<size_t> nb_shape(shape.begin(), shape.end());
  nb::capsule owner(data_copy,
                    [](void *p) noexcept { delete[] static_cast<T *>(p); });

  // No explicit strides: nanobind computes C-order element strides
  // automatically. Passing byte strides here would be wrong — nanobind
  // multiplies stored strides by itemsize in the buffer protocol, so only
  // element strides must be stored.
  return nb::cast(nb::ndarray<nb::numpy, T>(data_copy, shape.size(),
                                            nb_shape.data(), owner));
}

void bindOperators(nb::module_ &mod) {

  mod.def(
      "jordan_wigner",
      [](nb::ndarray<nb::numpy, std::complex<double>> hpq,
         nb::ndarray<nb::numpy, std::complex<double>> hpqrs, double core_energy,
         nb::kwargs options) {
        auto *hpqData = static_cast<std::complex<double> *>(hpq.data());
        auto *hpqrsData = static_cast<std::complex<double> *>(hpqrs.data());

        std::vector<std::size_t> hpq_shape, hpqrs_shape;
        for (size_t d = 0; d < hpq.ndim(); d++)
          hpq_shape.push_back(hpq.shape(d));
        for (size_t d = 0; d < hpqrs.ndim(); d++)
          hpqrs_shape.push_back(hpqrs.shape(d));

        cudaqx::tensor hpqT, hpqrsT;
        hpqT.borrow(hpqData, hpq_shape);
        hpqrsT.borrow(hpqrsData, hpqrs_shape);

        return fermion_compiler::get("jordan_wigner")
            ->generate(core_energy, hpqT, hpqrsT, hetMapFromKwargs(options));
      },
      nb::arg("hpq"), nb::arg("hpqrs"), nb::arg("core_energy") = 0.0,
      nb::arg("**kwargs"),
      R"#(
Perform the Jordan-Wigner transformation on fermionic operators.

This function applies the Jordan-Wigner transformation to convert fermionic operators
(represented by one- and two-body integrals) into qubit operators.

Parameters:
-----------
hpq : numpy.ndarray
    A 2D complex numpy array representing the one-body integrals.
    Shape should be (N, N) where N is the number of spin molecular orbitals.
hpqrs : numpy.ndarray
    A 4D complex numpy array representing the two-body integrals.
    Shape should be (N, N, N, N) where N is the number of spin molecular orbitals.
core_energy : float, optional
    The core energy of the system when using active space Hamiltonian, nuclear energy otherwise. Default is 0.0.
tolerance : float, optional
    The threshold value for ignoring small coefficients.
    Can also be specified using 'tol'.
    Coefficients with absolute values smaller than this tolerance are considered as zero.
    Default is 1e-15.

Returns:
--------
cudaq.SpinOperator
    A qubit operator (spin operator) resulting from the Jordan-Wigner transformation.

Raises:
-------
ValueError
    If the input arrays have incorrect shapes or types.
RuntimeError
    If the Jordan-Wigner transformation fails for any reason.

Examples:
---------
>>> import numpy as np
>>> h1 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
>>> h2 = np.zeros((2, 2, 2, 2), dtype=np.complex128)
>>> h2[0, 1, 1, 0] = h2[1, 0, 0, 1] = 0.5
>>> qubit_op = jordan_wigner(h1, h2, core_energy=0.1, tolerance=1e-14)

Notes:
------
- The input arrays `hpq` and `hpqrs` must be contiguous and in row-major order.
- This function uses the "jordan_wigner" fermion compiler internally to perform
  the transformation.
- The resulting qubit operator can be used directly in quantum algorithms or
  further manipulated using CUDA Quantum operations.
)#");

  mod.def(
      "jordan_wigner",
      [](nb::ndarray<nb::numpy, std::complex<double>> buffer,
         double core_energy, nb::kwargs options) {
        auto *data = static_cast<std::complex<double> *>(buffer.data());
        std::size_t size = 1;
        std::vector<std::size_t> shape_vec;
        for (size_t d = 0; d < buffer.ndim(); d++) {
          shape_vec.push_back(buffer.shape(d));
          size *= buffer.shape(d);
        }
        if (buffer.ndim() == 2) {
          std::size_t dim = buffer.shape(0);
          cudaqx::tensor hpq, hpqrs({dim, dim, dim, dim});
          hpq.borrow(data, shape_vec);
          return fermion_compiler::get("jordan_wigner")
              ->generate(core_energy, hpq, hpqrs, hetMapFromKwargs(options));
        }

        std::size_t dim = buffer.shape(0);
        cudaqx::tensor hpq({dim, dim}), hpqrs;
        hpqrs.borrow(data, shape_vec);
        return fermion_compiler::get("jordan_wigner")
            ->generate(core_energy, hpq, hpqrs, hetMapFromKwargs(options));
      },
      nb::arg("hpq"), nb::arg("core_energy") = 0.0, nb::arg("**kwargs"),
      R"#(
Perform the Jordan-Wigner transformation on fermionic operators.

This function applies the Jordan-Wigner transformation to convert fermionic operators
(represented by either one-body or two-body integrals) into qubit operators.

Parameters:
-----------
hpq : numpy.ndarray
    A complex numpy array representing either:
    - One-body integrals: 2D array with shape (N, N)
    - Two-body integrals: 4D array with shape (N, N, N, N)
    where N is the number of orbitals.
core_energy : float, optional
    The core energy of the system. Default is 0.0.
tolerance : float, optional
    The threshold value for ignoring small coefficients.
    Can also be specified using 'tol'.
    Coefficients with absolute values smaller than this tolerance are considered as zero.
    Default is 1e-15.

Returns:
--------
cudaq.SpinOperator
    A qubit operator (spin operator) resulting from the Jordan-Wigner transformation.

Raises:
-------
ValueError
    If the input array has an incorrect shape or type.
RuntimeError
    If the Jordan-Wigner transformation fails for any reason.

Examples:
---------
>>> import numpy as np
>>> # One-body integrals
>>> h1 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
>>> qubit_op1 = jordan_wigner(h1, core_energy=0.1, tolerance=1e-14)

>>> # Two-body integrals
>>> h2 = np.zeros((2, 2, 2, 2), dtype=np.complex128)
>>> h2[0, 1, 1, 0] = h2[1, 0, 0, 1] = 0.5
>>> qubit_op2 = jordan_wigner(h2)

Notes:
------
- The input array must be contiguous and in row-major order.
- This function automatically detects whether the input represents one-body or
  two-body integrals based on its shape.
- For one-body integrals input, a zero-initialized two-body tensor is used internally.
- For two-body integrals input, a zero-initialized one-body tensor is used internally.
- This function uses the "jordan_wigner" fermion compiler internally to perform
  the transformation.
- The resulting qubit operator can be used directly in quantum algorithms or
  further manipulated using CUDA Quantum operations.
)#");

  mod.def(
      "bravyi_kitaev",
      [](nb::ndarray<nb::numpy, std::complex<double>> hpq,
         nb::ndarray<nb::numpy, std::complex<double>> hpqrs, double core_energy,
         nb::kwargs options) {
        auto *hpqData = static_cast<std::complex<double> *>(hpq.data());
        auto *hpqrsData = static_cast<std::complex<double> *>(hpqrs.data());

        std::vector<std::size_t> hpq_shape, hpqrs_shape;
        for (size_t d = 0; d < hpq.ndim(); d++)
          hpq_shape.push_back(hpq.shape(d));
        for (size_t d = 0; d < hpqrs.ndim(); d++)
          hpqrs_shape.push_back(hpqrs.shape(d));

        cudaqx::tensor hpqT, hpqrsT;
        hpqT.borrow(hpqData, hpq_shape);
        hpqrsT.borrow(hpqrsData, hpqrs_shape);

        return fermion_compiler::get("bravyi_kitaev")
            ->generate(core_energy, hpqT, hpqrsT, hetMapFromKwargs(options));
      },
      nb::arg("hpq"), nb::arg("hpqrs"), nb::arg("core_energy") = 0.0,
      nb::arg("**kwargs"),
      R"#(
Perform the Bravyi-Kitaev transformation on fermionic operators.

This function applies the Bravyi-Kitaev transformation to convert fermionic operators
(represented by one- and two-body integrals) into qubit operators.

Parameters:
-----------
hpq : numpy.ndarray
    A 2D complex numpy array representing the one-body integrals.
    Shape should be (N, N) where N is the number of spin molecular orbitals.
hpqrs : numpy.ndarray
    A 4D complex numpy array representing the two-body integrals.
    Shape should be (N, N, N, N) where N is the number of spin molecular orbitals.
core_energy : float, optional
    The core energy of the system when using active space Hamiltonian, nuclear energy otherwise. Default is 0.0.
tolerance : float, optional
    The threshold value for ignoring small coefficients.
    Can also be specified using 'tol'.
    Coefficients with absolute values smaller than this tolerance are considered as zero.
    Default is 1e-15.

Returns:
--------
cudaq.SpinOperator
    A qubit operator (spin operator) resulting from the Bravyi-Kitaev transformation.

Raises:
-------
ValueError
    If the input arrays have incorrect shapes or types.
RuntimeError
    If the Bravyi-Kitaev transformation fails for any reason.

Examples:
---------
>>> import numpy as np
>>> h1 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
>>> h2 = np.zeros((2, 2, 2, 2), dtype=np.complex128)
>>> h2[0, 1, 1, 0] = h2[1, 0, 0, 1] = 0.5
>>> qubit_op = bravyi_kitaev(h1, h2, core_energy=0.1, tolerance=1e-14)

Notes:
------
- The input arrays `hpq` and `hpqrs` must be contiguous and in row-major order.
- This function uses the "bravyi_kitaev" fermion compiler internally to perform
  the transformation.
- The resulting qubit operator can be used directly in quantum algorithms or
  further manipulated using CUDA Quantum operations.
)#");

  mod.def(
      "bravyi_kitaev",
      [](nb::ndarray<nb::numpy, std::complex<double>> buffer,
         double core_energy, nb::kwargs options) {
        auto *data = static_cast<std::complex<double> *>(buffer.data());
        std::size_t size = 1;
        std::vector<std::size_t> shape_vec;
        for (size_t d = 0; d < buffer.ndim(); d++) {
          shape_vec.push_back(buffer.shape(d));
          size *= buffer.shape(d);
        }
        if (buffer.ndim() == 2) {
          std::size_t dim = buffer.shape(0);
          cudaqx::tensor hpq, hpqrs({dim, dim, dim, dim});
          hpq.borrow(data, shape_vec);
          return fermion_compiler::get("bravyi_kitaev")
              ->generate(core_energy, hpq, hpqrs, hetMapFromKwargs(options));
        }

        std::size_t dim = buffer.shape(0);
        cudaqx::tensor hpq({dim, dim}), hpqrs;
        hpqrs.borrow(data, shape_vec);
        return fermion_compiler::get("bravyi_kitaev")
            ->generate(core_energy, hpq, hpqrs, hetMapFromKwargs(options));
      },
      nb::arg("hpq"), nb::arg("core_energy") = 0.0, nb::arg("**kwargs"),
      R"#(
Perform the Bravyi-Kitaev transformation on fermionic operators.

This function applies the Bravyi-Kitaev transformation to convert fermionic operators
(represented by either one-body or two-body integrals) into qubit operators.

Parameters:
-----------
hpq : numpy.ndarray
    A complex numpy array representing either:
    - One-body integrals: 2D array with shape (N, N)
    - Two-body integrals: 4D array with shape (N, N, N, N)
    where N is the number of orbitals.
core_energy : float, optional
    The core energy of the system. Default is 0.0.
tolerance : float, optional
    The threshold value for ignoring small coefficients.
    Can also be specified using 'tol'.
    Coefficients with absolute values smaller than this tolerance are considered as zero.
    Default is 1e-15.

Returns:
--------
cudaq.SpinOperator
    A qubit operator (spin operator) resulting from the Bravyi-Kitaev transformation.

Raises:
-------
ValueError
    If the input array has an incorrect shape or type.
RuntimeError
    If the Bravyi-Kitaev transformation fails for any reason.

Examples:
---------
>>> import numpy as np
>>> # One-body integrals
>>> h1 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
>>> qubit_op1 = bravyi_kitaev(h1, core_energy=0.1, tolerance=1e-14)

>>> # Two-body integrals
>>> h2 = np.zeros((2, 2, 2, 2), dtype=np.complex128)
>>> h2[0, 1, 1, 0] = h2[1, 0, 0, 1] = 0.5
>>> qubit_op2 = bravyi_kitaev(h2)

Notes:
------
- The input array must be contiguous and in row-major order.
- This function automatically detects whether the input represents one-body or
  two-body integrals based on its shape.
- For one-body integrals input, a zero-initialized two-body tensor is used internally.
- For two-body integrals input, a zero-initialized one-body tensor is used internally.
- This function uses the "bravyi_kitaev" fermion compiler internally to perform
  the transformation.
- The resulting qubit operator can be used directly in quantum algorithms or
  further manipulated using CUDA Quantum operations.
)#");

  nb::class_<molecular_hamiltonian>(mod, "MolecularHamiltonian")
      .def_ro("energies", &molecular_hamiltonian::energies,
              R"#(
        Dictionary of energies from classical computation.
        )#")
      .def_ro("hamiltonian", &molecular_hamiltonian::hamiltonian,
              R"#(
        :class:`cudaq.SpinOperator`: The qubit representation of the molecular Hamiltonian.

        This is the full electronic Hamiltonian of the molecule, transformed into
        qubit operators using a specific mapping (e.g., Jordan-Wigner).
        )#")
      .def_ro("n_electrons", &molecular_hamiltonian::n_electrons,
              R"#(
        int: The number of electrons in the molecule.

        This represents the total number of electrons in the molecular system,
        which is crucial for determining the filling of orbitals and the overall
        electronic structure.
        )#")
      .def_ro("n_orbitals", &molecular_hamiltonian::n_orbitals,
              R"#(
        int: The number of molecular orbitals.

        This is the total number of molecular orbitals considered in the
        calculation, which determines the size of the Hamiltonian and the
        complexity of the quantum simulation.
        )#")
      .def_prop_ro(
          "hpq",
          [](const molecular_hamiltonian &self) {
            return tensor_to_numpy(self.hpq);
          },
          R"#(
        numpy.ndarray: One-electron integrals.

        A 2D complex array of shape (n_orbitals, n_orbitals), where n_orbitals is the
        number of spin molecular orbitals, representing
        the one-electron integrals in the molecular orbital basis. These
        include kinetic energy and electron-nuclear attraction terms.
        )#")
      .def_prop_ro(
          "hpqrs",
          [](const molecular_hamiltonian &self) {
            return tensor_to_numpy(self.hpqrs);
          },
          R"#(
        numpy.ndarray: Two-electron integrals.

        A 4D complex array of shape (n_orbitals, n_orbitals, n_orbitals, n_orbitals),
        where n_orbitals is the number of spin molecular orbitals,
        representing the two-electron integrals in the molecular orbital basis.
        These describe electron-electron interactions.
        )#");

  auto creator = [](molecular_geometry &molGeom, const std::string basis,
                    int spin, int charge, nb::kwargs options) {
    molecule_options inOptions;
    inOptions.type = getValueOr<std::string>(options, "type", "gas_phase");
    constexpr std::size_t kNotSet = std::numeric_limits<std::size_t>::max();
    std::optional<std::size_t> nele_cas =
        getValueOr<std::size_t>(options, "nele_cas", kNotSet);
    inOptions.nele_cas = (nele_cas == kNotSet) ? std::nullopt : nele_cas;
    std::optional<std::size_t> norb_cas =
        getValueOr<std::size_t>(options, "norb_cas", kNotSet);
    inOptions.norb_cas = (norb_cas == kNotSet) ? std::nullopt : norb_cas;
    inOptions.symmetry = getValueOr<bool>(options, "symmetry", false);
    inOptions.memory = getValueOr<double>(options, "memory", 4000.);
    inOptions.cycles = getValueOr<std::size_t>(options, "cycles", 100);
    inOptions.initguess =
        getValueOr<std::string>(options, "initguess", "minao");
    inOptions.UR = getValueOr<bool>(options, "UR", false);
    inOptions.MP2 = getValueOr<bool>(options, "MP2", false);
    inOptions.natorb = getValueOr<bool>(options, "natorb", false);
    inOptions.casci = getValueOr<bool>(options, "casci", false);
    inOptions.ccsd = getValueOr<bool>(options, "ccsd", false);
    inOptions.casscf = getValueOr<bool>(options, "casscf", false);
    inOptions.integrals_natorb =
        getValueOr<bool>(options, "integrals_natorb", false);
    inOptions.integrals_casscf =
        getValueOr<bool>(options, "integrals_casscf", false);
    inOptions.verbose = getValueOr<bool>(options, "verbose", false);

    // We are already running in a specific Python environment. Get the fully
    // qualified path to the executable and populate it here so that we don't
    // accidentally use the wrong Python environment for any child processes
    // that may be spawned.
    inOptions.python_path = []() {
      auto sys = nb::module_::import_("sys");
      return nb::cast<std::string>(sys.attr("executable"));
    }();

    if (inOptions.verbose)
      inOptions.dump();
    return create_molecule(molGeom, basis, spin, charge, inOptions);
  };

  mod.def(
      "create_molecule",
      [&](nb::list geometry, const std::string basis, int spin, int charge,
          nb::kwargs options) {
        std::vector<atom> atoms;
        for (auto el : geometry) {
          if (!nb::isinstance<nb::tuple>(el))
            throw std::runtime_error(
                "geometry must be a list of tuples ('NAME', (X, Y, Z))");
          auto casted = nb::cast<nb::tuple>(el);
          if (!nb::isinstance<nb::tuple>(casted[1]))
            throw std::runtime_error(
                "geometry must be a list of tuples ('NAME', (X, Y, Z))");

          auto name = nb::cast<std::string>(casted[0]);
          auto coords = nb::cast<nb::tuple>(casted[1]);
          atoms.push_back(
              atom{name,
                   {nb::cast<double>(coords[0]), nb::cast<double>(coords[1]),
                    nb::cast<double>(coords[2])}});
        }
        molecular_geometry molGeom(atoms);

        return creator(molGeom, basis, spin, charge, options);
      },
      nb::arg("geometry"), nb::arg("basis"), nb::arg("spin"), nb::arg("charge"),
      nb::arg("**kwargs"),
      R"#(Create a molecular hamiltonian from an XYZ file and additional parameters.

This function generates a molecular hamiltonian based on the geometry specified in an XYZ file
and additional quantum chemical parameters.

Parameters:
-----------
geometry : list of tuples
    List of tuples representing the molecular geometry. Each tuple should be in the format
    ('ELEMENT', (X, Y, Z)), where 'ELEMENT' is the element symbol and X, Y, Z are coordinates.
basis : str
    The basis set to be used for the molecular calculation (e.g., "sto-3g", "6-31g").
spin : int
    The spin multiplicity of the molecule (2S + 1, where S is the total spin).
charge : int
    The total charge of the molecule.
options : dict
    Additional keyword arguments for customizing the molecular model creation.
    These may include method-specific parameters or computational settings.

Returns:
--------
object
    A molecular hamiltonian encoding the spin_op, one and two body overlap integrals, and energies relevant for the model.

Raises:
-------
RuntimeError
    If the molecular model creation fails for any other reason.


)#");

  mod.def(
      "create_molecule",
      [&](const std::string &xyz_file, const std::string basis, int spin,
          int charge, nb::kwargs options) {
        auto geom = molecular_geometry::from_xyz(xyz_file);
        return creator(geom, basis, spin, charge, options);
      },
      nb::arg("xyz_file"), nb::arg("basis"), nb::arg("spin"), nb::arg("charge"),
      nb::arg("**kwargs"),
      R"#(Create a molecular hamiltonian from an XYZ file and additional parameters.)#");

  mod.def(
      "get_operator_pool",
      [](const std::string &name, nb::kwargs config) {
        return operator_pool::get(name)->generate(hetMapFromKwargs(config));
      },
      R"#(Get and generate an operator pool based on the specified name and configuration.

This function retrieves an operator pool implementation by name and generates
a set of operators using the provided configuration.

Parameters:
-----------
name : str
    The name of the operator pool implementation to use.
config : dict
    Keyword arguments representing the configuration for operator pool generation.
    Supported value types:
    - int: Converted to std::size_t in C++.
    - list: Converted to std::vector<double> in C++.

Returns:
--------
list
    A list of generated operators (:class:`cudaq.SpinOperator` objects).

Raises:
-------
RuntimeError
    If the specified operator pool implementation is not found.
TypeError
    If an unsupported configuration value type is provided.

Examples:
---------
>>> ops = get_operator_pool("uccsd", n_qubits=4, n_electrons=2)
>>> ops = get_operator_pool("custom_pool", cutoff=1e-5, parameters=[0.1, 0.2, 0.3])

Notes:
------
The function internally converts Python types to C++ types and uses the
cudaq::operator_pool extension point system to retrieve and generate the
operator pool. Only integer and list configuration values are currently supported.
)#");
}

void bindSolvers(nb::module_ &mod) {

  addStatePrepKernels(mod);

  auto solvers = mod; //.def_submodule("solvers");
  bindOperators(solvers);

  nb::enum_<observe_execution_type>(
      solvers, "ObserveExecutionType",
      R"#(An enumeration representing different types of execution in an optimization process.

This enum defines the various types of operations that can occur during an
optimization iteration, specifically distinguishing between function evaluations
and gradient computations.

Usage:
------
This enum is typically used in conjunction with optimization algorithms and
observation mechanisms to indicate the nature of a particular step or evaluation
in the optimization process.

Examples:
---------
>>> def callback(iteration):
...     if iteration.type == ObserveExecutionType.function:
...         print("Function evaluation")
...     elif iteration.type == ObserveExecutionType.gradient:
...         print("Gradient computation")

>>> # In an optimization loop
>>> for step in optimization_steps:
...     if step.type == ObserveExecutionType.function:
...         # Process function evaluation
...     elif step.type == ObserveExecutionType.gradient:
...         # Process gradient information

Notes:
------
- The distinction between function evaluations and gradient computations is
  particularly important for gradient-based optimization methods.
- Some optimization algorithms may only use function evaluations (gradient-free methods),
  while others rely heavily on gradient information.
- This enum can be used for logging, debugging, or implementing custom behaviors
  based on the type of operation being performed during optimization.
)#")
      .value(
          "function", observe_execution_type::function,
          R"#(Represents a standard function evaluation of the objective function.

This typically involves computing the value of the objective function
at a given point in the parameter space.)#")
      .value("gradient", observe_execution_type::gradient,
             R"#(Represents a gradient computation.

This involves calculating the partial derivatives of the objective)#");

  nb::class_<observe_iteration>(
      solvers, "ObserveIteration",
      R"#(A class representing a single iteration of an optimization process.

This class encapsulates the state of an optimization iteration, including
the current parameter values, the result of the objective function evaluation,
and the type of iteration)#")
      .def_ro(
          "parameters", &observe_iteration::parameters,
          R"#(The current values of the optimization parameters at this iteration.
These represent the point in the parameter space being evaluated.)#")
      .def_ro(
          "result", &observe_iteration::result,
          R"#(The value of the objective function evaluated at the current parameters.
For minimization problems, lower values indicate better solutions.)#")
      .def_ro(
          "type", &observe_iteration::type,
          R"#(A string indicating the type or purpose of this iteration. Common types might include:
- 'function': A standard function evaluation
- 'gradient': An iteration where gradients were computed
The exact set of possible types may depend on the specific optimization algorithm used.)#");

  solvers.def(
      "vqe",
      [](const nb::callable &kernel, cudaq::spin_op op,
         std::vector<double> initial_parameters, nb::kwargs options) {
        heterogeneous_map optOptions;
        optOptions.insert("shots",
                          cudaqx::getValueOr<int>(options, "shots", -1));
        if (options.contains("max_iterations"))
          optOptions.insert(
              "max_iterations",
              cudaqx::getValueOr<int>(options, "max_iterations", -1));
        // in case the provided optimizer is not a scipy one
        optOptions.insert("tol", getValueOr<double>(options, "tol", 1e-12));

        optOptions.insert("verbose",
                          cudaqx::getValueOr<bool>(options, "verbose", false));

        // Handle the case where the user has provided a SciPy optimizer
        if (options.contains("optimizer") &&
            nb::isinstance<nb::callable>(options["optimizer"])) {
          auto func = nb::cast<nb::callable>(options["optimizer"]);
          if (nb::cast<std::string>(func.attr("__name__")) != "minimize")
            throw std::runtime_error(
                "Invalid functional optimizer provided (only "
                "scipy.optimize.minimize supported).");
          PythonOptimizer opt(func, options, initial_parameters);
          auto result =
              cudaq::solvers::vqe([&](std::vector<double> x) { kernel(x); }, op,
                                  opt, initial_parameters, optOptions);
          return nb::make_tuple(result.energy, result.optimal_parameters,
                                result.iteration_data);
        }

        auto optimizerName =
            cudaqx::getValueOr<std::string>(options, "optimizer", "cobyla");
        auto optimizer = cudaq::optim::optimizer::get(optimizerName);
        auto kernelWrapper = [&](std::vector<double> x) { kernel(x); };

        if (!optimizer->requiresGradients()) {
          auto result = cudaq::solvers::vqe(kernelWrapper, op, *optimizer,
                                            initial_parameters, optOptions);
          return nb::make_tuple(result.energy, result.optimal_parameters,
                                result.iteration_data);
        }

        auto gradientName = cudaqx::getValueOr<std::string>(options, "gradient",
                                                            "parameter_shift");
        auto gradient =
            cudaq::observe_gradient::get(gradientName, kernelWrapper, op);

        auto result = cudaq::solvers::vqe(kernelWrapper, op, *optimizer.get(),
                                          *gradient.get(), initial_parameters,
                                          optOptions);
        return nb::make_tuple(result.energy, result.optimal_parameters,
                              result.iteration_data);
      },
      nb::arg("kernel"), nb::arg("spin_op"), nb::arg("initial_parameters"),
      nb::arg("**kwargs"), R"#(
Execute the Variational Quantum Eigensolver (VQE) algorithm.

This function implements the VQE algorithm, a hybrid quantum-classical algorithm
used to find the ground state energy of a given Hamiltonian using a parameterized
quantum circuit.

Parameters:
-----------
kernel : callable
    A function representing the parameterized quantum circuit (ansatz).
    It should take a list of parameters as input and prepare the quantum state.

spin_op : cudaq.SpinOperator
    The Hamiltonian operator for which to find the ground state energy.

initial_parameters : List[float]
    Initial values for the variational parameters of the quantum circuit.

options : dict
    Additional options for the VQE algorithm. Supported options include:
    - shots : int, optional, Number of measurement shots. Default is -1 (use maximum available).
    - max_iterations : int, optional Maximum number of optimization iterations. Default is -1 (no limit).
    - verbose : bool, optional Whether to print verbose output. Default is False.
    - optimizer : str, optional Name of the classical optimizer to use. Default is 'cobyla'.
    - gradient : str, optional Method for gradient computation (for gradient-based optimizers). Default is 'parameter_shift'.
    - tol (double): Tolerance value for the optimizer. Default 1e-12.

Returns:
--------
Tuple[float, List[float], List[ObserveIteration]]
    A tuple containing:
    1. The optimized ground state energy.
    2. The optimal variational parameters.
    3. A list of ObserveIteration objects containing data from each iteration.

Raises:
-------
RuntimeError
    If an invalid optimizer or gradient method is specified.

Examples:
---------
>>> def ansatz(params):
...     # Define your quantum circuit here
...     pass
>>> hamiltonian = cudaq.SpinOperator(...)  # Define your Hamiltonian
>>> initial_params = [0.1, 0.2, 0.3]
>>> energy, opt_params, iterations = vqe(ansatz, hamiltonian, initial_params,
...                                      optimizer='cobyla', shots=1000)
>>> print(f"Ground state energy: {energy}")
>>> print(f"Optimal parameters: {opt_params}")

Notes:
------
- The function automatically selects between gradient-free and gradient-based
  optimization based on the chosen optimizer.
- For gradient-based optimization, the 'parameter_shift' method is used by default,
  but can be changed using the 'gradient' option.
- The ObserveIteration objects in the returned list contain detailed information
  about each optimization step, useful for analysis and visualization.
- The performance of VQE heavily depends on the choice of ansatz, initial parameters,
  and optimization method.

)#");

  solvers.def(
      "adapt_vqe",
      [](nb::object initialStateKernel, cudaq::spin_op op,
         const std::vector<cudaq::spin_op> &pool, nb::kwargs options) {
        cudaq::python::CppPyKernelDecorator initialStateKernelWrapper(
            initialStateKernel);
        auto fptr = initialStateKernelWrapper.getDirectKernelCall<
            cudaq::qkernel<void(cudaq::qvector<> &)>>();
        heterogeneous_map optOptions;
        optOptions.insert("max_iter", getValueOr<int>(options, "max_iter", 30));
        optOptions.insert(
            "grad_norm_tolerance",
            getValueOr<double>(options, "grad_norm_tolerance", 1e-5));
        optOptions.insert(
            "grad_norm_diff_tolerance",
            getValueOr<double>(options, "grad_norm_diff_tolerance", 1e-5));
        optOptions.insert(
            "threshold_energy",
            getValueOr<double>(options, "threshold_energy", 1e-6));
        optOptions.insert("initial_theta",
                          getValueOr<double>(options, "initial_theta", 0.0));
        optOptions.insert("verbose",
                          getValueOr<bool>(options, "verbose", false));
        optOptions.insert("shots", getValueOr<int>(options, "shots", -1));
        optOptions.insert("tol", getValueOr<double>(options, "tol", 1e-12));
        optOptions.insert(
            "dynamic_start",
            getValueOr<std::string>(options, "dynamic_start", "cold"));

        // Handle the case where the user has provided a SciPy optimizer
        if (options.contains("optimizer") &&
            nb::isinstance<nb::callable>(options["optimizer"])) {
          auto func = nb::cast<nb::callable>(options["optimizer"]);
          if (nb::cast<std::string>(func.attr("__name__")) != "minimize")
            throw std::runtime_error(
                "Invalid functional optimizer provided (only "
                "scipy.optimize.minimize supported).");
          PythonOptimizer opt(func, options);
          return cudaq::solvers::adapt_vqe(fptr, op, pool, opt, optOptions);
        }

        auto optimizerName =
            cudaqx::getValueOr<std::string>(options, "optimizer", "cobyla");
        auto optimizer = cudaq::optim::optimizer::get(optimizerName);
        auto gradName =
            cudaqx::getValueOr<std::string>(options, "gradient", "");

        // FIXME Convert options from kwargs
        return cudaq::solvers::adapt_vqe(fptr, op, pool, *optimizer, gradName,
                                         optOptions);
      },
      R"(
    Perform ADAPT-VQE (Adaptive Derivative-Assembled Pseudo-Trotter Variational Quantum Eigensolver) optimization.

    Args:
        initialStateKernel (object): Python object representing the initial state kernel.
        op (cudaq.SpinOperator): The Hamiltonian operator to be optimized.
        pool (list of cudaq.SpinOperator): Pool of operators for ADAPT-VQE.
        options: Additional options for the optimization process.

    Keyword Args:
        optimizer (str): Optional name of the optimizer to use. Defaults to cobyla.
        gradient (str): Optional name of the gradient method to use. Defaults to empty.

    Options Dictionary:
        The following keys are supported in the options dictionary:
        - max_iter (int): Maximum number of iterations. Default: 30
        - grad_norm_tolerance (float): Convergence tolerance for gradient norm. Default: 1e-5
        - grad_norm_diff_tolerance (float): Tolerance for difference between gradient norms. Default: 1e-5
        - threshold_energy (float): Energy convergence threshold. Default: 1e-6
        - initial_theta (float): Initial value for theta parameter. Default: 0.0
        - verbose (bool): Enable detailed output logging. Default: False
        - shots (int): Number of measurement shots (-1 for exact simulation). Default: -1
        - tol (double): Tolerance value for the optimizer. Default 1e-12
        - dynamic_start (string): Optimization mode for the theta parameters at each iteration. It can be either "warm", or "cold". Default: "cold"


    Returns:
        The result of the ADAPT-VQE optimization.

    Note:
        This function wraps the C++ implementation of ADAPT-VQE in CUDA-QX.
        It compiles and registers the initial state kernel, sets up the optimizer,
        and performs the ADAPT-VQE optimization using the provided parameters.
  )");

  // Bind the qaoa_result struct
  nb::class_<cudaq::solvers::qaoa_result>(
      solvers, "QAOAResult",
      "The QAOAResult encodes the optimal value, optimal parameters, and final "
      "sampled state as a cudaq.SampleResult.")
      .def(nb::init<>())
      .def_rw("optimal_value", &cudaq::solvers::qaoa_result::optimal_value)
      .def_rw("optimal_parameters",
              &cudaq::solvers::qaoa_result::optimal_parameters)
      .def_rw("optimal_config", &cudaq::solvers::qaoa_result::optimal_config)
      // Add tuple interface
      .def("__len__", [](const cudaq::solvers::qaoa_result &) { return 3; })
      .def("__getitem__",
           [](const cudaq::solvers::qaoa_result &r, size_t i) -> nb::object {
             switch (i) {
             case 0:
               return nb::cast(r.optimal_value);
             case 1:
               return nb::cast(r.optimal_parameters);
             case 2:
               return nb::cast(r.optimal_config);
             default:
               throw nb::index_error();
             }
           })
      // Enable iteration protocol
      .def("__iter__", [](const cudaq::solvers::qaoa_result &r) -> nb::object {
        return nb::make_tuple(r.optimal_value, r.optimal_parameters,
                              r.optimal_config)
            .attr("__iter__")();
      });

  // Bind QAOA functions using lambdas
  solvers.def(
      "qaoa",
      [](const cudaq::spin_op &problemHamiltonian,
         const cudaq::spin_op &referenceHamiltonian, std::size_t numLayers,
         const std::vector<double> &initialParameters, nb::kwargs options) {
        if (initialParameters.empty())
          throw std::runtime_error("qaoa initial parameters empty.");
        // Handle the case where the user has provided a SciPy optimizer
        if (options.contains("optimizer") &&
            nb::isinstance<nb::callable>(options["optimizer"])) {
          auto func = nb::cast<nb::callable>(options["optimizer"]);
          if (nb::cast<std::string>(func.attr("__name__")) != "minimize")
            throw std::runtime_error(
                "Invalid functional optimizer provided (only "
                "scipy.optimize.minimize supported).");
          PythonOptimizer opt(func, options);
          return cudaq::solvers::qaoa(problemHamiltonian, referenceHamiltonian,
                                      opt, numLayers, initialParameters,
                                      hetMapFromKwargs(options));
        }

        auto optimizerName =
            cudaqx::getValueOr<std::string>(options, "optimizer", "cobyla");
        auto optimizer = cudaq::optim::optimizer::get(optimizerName);

        return cudaq::solvers::qaoa(problemHamiltonian, referenceHamiltonian,
                                    *optimizer, numLayers, initialParameters,
                                    hetMapFromKwargs(options));
      },
      nb::arg("problemHamiltonian"), nb::arg("referenceHamiltonian"),
      nb::arg("numLayers"), nb::arg("initialParameters"), nb::arg("**kwargs"));

  solvers.def(
      "qaoa",
      [](const cudaq::spin_op &problemHamiltonian, std::size_t numLayers,
         const std::vector<double> &initialParameters, nb::kwargs options) {
        if (initialParameters.empty())
          throw std::runtime_error("qaoa initial parameters empty.");
        // Handle the case where the user has provided a SciPy optimizer
        if (options.contains("optimizer") &&
            nb::isinstance<nb::callable>(options["optimizer"])) {
          auto func = nb::cast<nb::callable>(options["optimizer"]);
          if (nb::cast<std::string>(func.attr("__name__")) != "minimize")
            throw std::runtime_error(
                "Invalid functional optimizer provided (only "
                "scipy.optimize.minimize supported).");
          PythonOptimizer opt(func, options);
          return cudaq::solvers::qaoa(problemHamiltonian, opt, numLayers,
                                      initialParameters,
                                      hetMapFromKwargs(options));
        }

        auto optimizerName =
            cudaqx::getValueOr<std::string>(options, "optimizer", "cobyla");
        auto optimizer = cudaq::optim::optimizer::get(optimizerName);
        return cudaq::solvers::qaoa(problemHamiltonian, *optimizer, numLayers,
                                    initialParameters,
                                    hetMapFromKwargs(options));
      },
      nb::arg("problemHamiltonian"), nb::arg("numLayers"),
      nb::arg("initialParameters"), nb::arg("**kwargs"));

  solvers.def(
      "get_num_qaoa_parameters",
      [](const cudaq::spin_op &problemHamiltonian, std::size_t numLayers,
         nb::kwargs options) {
        return cudaq::solvers::get_num_qaoa_parameters(
            problemHamiltonian, numLayers, hetMapFromKwargs(options));
      },
      "Return the number of required QAOA rotation parameters.");

  solvers.def(
      "get_maxcut_hamiltonian",
      [](nb::object nx_graph) {
        // Convert NetworkX graph to our internal representation
        cudaqx::graph g = convert_networkx_graph(nx_graph);

        // Generate and return the Hamiltonian
        return cudaq::solvers::get_maxcut_hamiltonian(g);
      },
      "Generate MaxCut Hamiltonian from a NetworkX graph", nb::arg("graph"));

  solvers.def(
      "get_clique_hamiltonian",
      [](nb::object nx_graph, double penalty = 4.0) {
        // Convert NetworkX graph to our internal representation
        cudaqx::graph g = convert_networkx_graph(nx_graph);

        // Generate and return the Hamiltonian
        return cudaq::solvers::get_clique_hamiltonian(g, penalty);
      },
      "Generate Clique Hamiltonian from a NetworkX graph", nb::arg("graph"),
      nb::arg("penalty") = 4.0);

  std::stringstream ss;
  ss << "CUDA-Q Solvers " << cudaq::solvers::getVersion() << " ("
     << cudaq::solvers::getFullRepositoryVersion() << ")";
  solvers.attr("__version__") = ss.str();
}

} // namespace cudaq::solvers
