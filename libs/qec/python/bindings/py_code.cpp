/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include <limits>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>
#include <nanobind/trampoline.h>

#include "cudaq/runtime/logger/logger.h"

#include "common/DeviceCodeRegistry.h"
#include "cudaq/python/PythonCppInterop.h"
#include "cudaq/qec/experiments.h"
#include "cudaq/qec/version.h"

#include "cuda-qx/core/kwargs_utils.h"
#include "type_casters.h"

namespace nb = nanobind;
using namespace cudaqx;

namespace cudaq::qec {

class PyCode : public qec::code {
public:
  NB_TRAMPOLINE(qec::code, 6);

protected:
  // Trampoline methods for pure virtual functions
  std::size_t get_num_data_qubits() const override {
    NB_OVERRIDE_PURE(get_num_data_qubits);
  }

  std::size_t get_num_ancilla_qubits() const override {
    NB_OVERRIDE_PURE(get_num_ancilla_qubits);
  }

  std::size_t get_num_ancilla_x_qubits() const override {
    NB_OVERRIDE_PURE(get_num_ancilla_x_qubits);
  }

  std::size_t get_num_ancilla_z_qubits() const override {
    NB_OVERRIDE_PURE(get_num_ancilla_z_qubits);
  }

  std::size_t get_num_x_stabilizers() const override {
    NB_OVERRIDE_PURE(get_num_x_stabilizers);
  }

  std::size_t get_num_z_stabilizers() const override {
    NB_OVERRIDE_PURE(get_num_z_stabilizers);
  }
};

/// @brief A wrapper class that handles Python-defined quantum error correction
/// codes
/// @details This class serves as a bridge between Python-defined QEC codes and
/// the C++ implementation, managing the conversion of Python QEC code
/// definitions to their C++ counterparts.
class PyCodeHandle : public qec::code {
protected:
  /// @brief Python object representing the registered QEC code
  nb::object pyCode;

  /// @brief Keep the original Python kernels so that we can hand them back to
  /// Python.
  std::unordered_map<qec::operation, nb::object> m_py_operation_encodings;

  // Jitted python kernels are only alive as long as the
  // CppPyKernelDecorator they were compiled from, so we must cache them here
  std::vector<cudaq::python::CppPyKernelDecorator *> cachedDecorators;

public:
  ~PyCodeHandle() {
    for (auto decorator : cachedDecorators)
      delete decorator;
  }

  /// @brief Constructs a PyCodeHandle from a Python QEC code object
  /// @param registeredCode Python object containing the QEC code definition
  /// @throw std::runtime_error if the Python code lacks required attributes
  /// (stabilizers or operation_encodings)
  /// @details Initializes the handle by:
  /// - Validating the presence of required attributes
  /// - Converting Python stabilizers to C++ representation
  /// - Processing operation encodings and registering CUDA-Q kernels
  PyCodeHandle(nb::object registeredCode) : pyCode(registeredCode) {
    if (!nb::hasattr(registeredCode, "stabilizers"))
      throw std::runtime_error(
          "Invalid Python QEC Code. Must have self.stabilizers = "
          "[cudaq.SpinOperator(...)] (qec.Stabilizers(...)). Please provide "
          "the stabilizers.");
    if (!nb::hasattr(registeredCode, "pauli_observables"))
      throw std::runtime_error(
          "Invalid Python QEC Code. Must have self.pauli_observables = "
          "[cudaq.SpinOperator(...)]. Please provide the observables.");
    if (!nb::hasattr(registeredCode, "operation_encodings"))
      throw std::runtime_error(
          "Invalid Python QEC Code. Must have self.operation_encodings = "
          "{...}. Please provide the CUDA-Q kernels for the operation "
          "encodings.");

    if (nb::hasattr(registeredCode, "pauli_observables")) {
      auto obs_terms = nb::cast<std::vector<cudaq::spin_op_term>>(
          registeredCode.attr("pauli_observables"));
      m_pauli_observables.reserve(obs_terms.size());
      for (auto &&term : obs_terms)
        m_pauli_observables.emplace_back(std::move(term));
    }
    // Get the stabilizers. First convert to spin_op_term's and then convert to
    // spin_op's.
    auto stab_terms = nb::cast<std::vector<cudaq::spin_op_term>>(
        registeredCode.attr("stabilizers"));
    m_stabilizers.reserve(stab_terms.size());
    for (auto &term : stab_terms)
      m_stabilizers.emplace_back(std::move(term));

    // Get the CUDA-Q kernels for the operation encodings
    auto opsDict =
        nb::cast<nb::dict>(registeredCode.attr("operation_encodings"));

    // For each CUDA-Q kernel, extract the JIT-ed function pointer
    for (auto [opKey, kernelHandle] : opsDict) {
      nb::object kernel = nb::borrow<nb::object>(kernelHandle);
      auto opKeyEnum = nb::cast<qec::operation>(opKey);

      // Save the original Python kernel object so that we can return a valid
      // CUDA-Q kernel back to Python later.
      m_py_operation_encodings.emplace(opKeyEnum, kernel);

      // Create the kernel interop object
      auto kernInterop = new cudaq::python::CppPyKernelDecorator(kernel);

      // Make sure we cast the function pointer correctly
      if (opKeyEnum == operation::stabilizer_round) {
        encoding fptr = kernInterop->getDirectKernelCall<qkernel<
            std::vector<measure_result>(patch, const std::vector<std::size_t> &,
                                        const std::vector<std::size_t> &)>>();
        operation_encodings.insert({opKeyEnum, std::move(fptr)});
        cachedDecorators.emplace_back(kernInterop);
        continue;
      }

      // FIXME handle other signatures later... this assumes single patch
      // signatures
      encoding fptr = kernInterop->getDirectKernelCall<qkernel<void(patch)>>();
      operation_encodings.insert({opKeyEnum, std::move(fptr)});
      cachedDecorators.emplace_back(kernInterop);
    }
  }

  /// @brief Expose read-only access to stored Python kernels.
  const std::unordered_map<qec::operation, nb::object> &
  get_py_operation_encodings() const {
    return m_py_operation_encodings;
  }

protected:
  // Trampoline methods for pure virtual functions
  std::size_t get_num_data_qubits() const override {
    return nb::cast<std::size_t>(pyCode.attr("get_num_data_qubits")());
  }

  std::size_t get_num_ancilla_qubits() const override {
    return nb::cast<std::size_t>(pyCode.attr("get_num_ancilla_qubits")());
  }

  std::size_t get_num_ancilla_x_qubits() const override {
    return nb::cast<std::size_t>(pyCode.attr("get_num_ancilla_x_qubits")());
  }

  std::size_t get_num_ancilla_z_qubits() const override {
    return nb::cast<std::size_t>(pyCode.attr("get_num_ancilla_z_qubits")());
  }

  std::size_t get_num_x_stabilizers() const override {
    return nb::cast<std::size_t>(pyCode.attr("get_num_x_stabilizers")());
  }

  std::size_t get_num_z_stabilizers() const override {
    return nb::cast<std::size_t>(pyCode.attr("get_num_z_stabilizers")());
  }
};

namespace {
static nb::object get_python_kernel_or_throw(const code &self, operation op) {
  auto *pyHandle = dynamic_cast<const PyCodeHandle *>(&self);
  if (!pyHandle)
    throw std::runtime_error("This code was not registered from Python; no "
                             "Python kernel is available.");

  const auto &pyKernels = pyHandle->get_py_operation_encodings();
  auto it = pyKernels.find(op);
  if (it == pyKernels.end())
    throw std::runtime_error("No Python kernel registered for requested op.");

  return it->second;
}
} // namespace

// Registry to store code factory functions
class PyCodeRegistry {
private:
  static std::unordered_map<std::string, std::function<nb::object(nb::kwargs)>>
      registry;

public:
  static std::vector<std::string> get_keys() {
    std::vector<std::string> keys;
    for (const auto &pair : registry) {
      keys.push_back(pair.first);
    }
    return keys;
  }

  static void register_code(const std::string &name,
                            std::function<nb::object(nb::kwargs)> factory) {
    cudaq::info("Registering Pythonic QEC Code with name {}", name);
    registry[name] = factory;
  }

  static nb::object get_code(const std::string &name, nb::kwargs options) {
    auto it = registry.find(name);
    if (it == registry.end()) {
      throw std::runtime_error("Unknown code: " + name);
    }

    // Constructs the Python type with kwargs...
    return it->second(options);
  }

  static bool contains(const std::string &name) {
    return registry.find(name) != registry.end();
  }
};

std::unordered_map<std::string, std::function<nb::object(nb::kwargs)>>
    PyCodeRegistry::registry;

void bindCode(nb::module_ &mod) {

  auto qecmod = nb::hasattr(mod, "qecrt")
                    ? nb::cast<nb::module_>(mod.attr("qecrt"))
                    : mod.def_submodule("qecrt");

  nb::class_<qec::two_qubit_depolarization, cudaq::kraus_channel>(
      qecmod, "TwoQubitDepolarization",
      R"#(Models the decoherence of the each qubit independently in a two-qubit operation into a mixture "
      of the computational basis states, `|0>` and `|1>`.)#")
      .def(nb::init<double>(), nb::arg("probability"),
           "Initialize the `TwoQubitDepolarizationChannel` with the provided "
           "`probability`.");

  nb::class_<qec::two_qubit_bitflip, cudaq::kraus_channel>(
      qecmod, "TwoQubitBitFlip",
      R"#(Models independent bit flip errors after a two-qubit operation.)#")
      .def(nb::init<double>(), nb::arg("probability"),
           "Initialize the `TwoQubitBitFlip` with the provided "
           "`probability`.");

  nb::enum_<operation>(
      qecmod, "operation",
      "Enumeration of quantum operations for state preparation")
      .value("prep0", operation::prep0, "Prepare qubit in |0⟩ state")
      .value("prep1", operation::prep1, "Prepare qubit in |1⟩ state")
      .value("prepp", operation::prepp, "Prepare qubit in |+⟩ state")
      .value("prepm", operation::prepm, "Prepare qubit in |-⟩ state")
      .value("x", operation::x, "Apply the logical X operation")
      .value("y", operation::y, "Apply the logical Y operation")
      .value("z", operation::z, "Apply the logical Z operation")
      .value("h", operation::h, "Apply the logical H operation")
      .value("s", operation::s, "Apply the logical S operation")
      .value("cx", operation::cx, "Apply the logical CX operation")
      .value("cy", operation::cy, "Apply the logical CY operation")
      .value("cz", operation::cz, "Apply the logical CZ operation")
      .value("stabilizer_round", operation::stabilizer_round,
             "Apply the stabilizer round operation.");

  qecmod.def(
      "get_code",
      [](const std::string &name, nb::kwargs options) -> std::unique_ptr<code> {
        if (PyCodeRegistry::contains(name))
          return std::make_unique<PyCodeHandle>(
              PyCodeRegistry::get_code(name, options));

        if (options.contains("stabilizers")) {
          auto obj = options["stabilizers"];
          if (!nb::isinstance<nb::list>(obj))
            throw std::runtime_error(
                "invalid stabilizers passed to get_code, must be a list of "
                "string pauli words or list of cudaq.SpinOperator.");

          if (nb::isinstance<nb::str>(nb::cast<nb::list>(obj)[0])) {
            options.attr("pop")("stabilizers");
            auto words = nb::cast<std::vector<std::string>>(obj);
            std::vector<cudaq::spin_op_term> ops;
            for (auto &os : words)
              ops.emplace_back(cudaq::spin_op::from_word(os));
            sortStabilizerOps(ops);
            return get_code(name, ops, hetMapFromKwargs(options));
          }

          if (nb::isinstance<cudaq::spin_op_term>(nb::cast<nb::list>(obj)[0])) {
            options.attr("pop")("stabilizers");
            return get_code(name,
                            nb::cast<std::vector<cudaq::spin_op_term>>(obj),
                            hetMapFromKwargs(options));
          }

          throw std::runtime_error(
              "get_code error - invalid stabilizers element type.");
        }

        return get_code(name, hetMapFromKwargs(options));
      },
      "Retrieve a quantum error correction code by name with optional "
      "parameters");

  qecmod.def(
      "get_available_codes",
      []() {
        auto codes = cudaq::qec::get_available_codes();
        auto py_codes = PyCodeRegistry::get_keys();
        codes.insert(codes.end(), py_codes.begin(), py_codes.end());
        return codes;
      },
      "Get a list of all available quantum error correction codes (C++ and "
      "Python).");

  nb::class_<code, PyCode>(qecmod, "Code",
                           "Represents a quantum error correction code")
      .def(nb::init<>())
      .def(
          "get_parity",
          [](code &code) {
            return cudaq::python::copyCUDAQXTensorToPyArray(code.get_parity());
          },
          "Get the parity check matrix of the code")
      .def(
          "get_parity_x",
          [](code &code) {
            return cudaq::python::copyCUDAQXTensorToPyArray(
                code.get_parity_x());
          },
          "Get the X-type parity check matrix of the code")
      .def(
          "get_parity_z",
          [](code &code) {
            return cudaq::python::copyCUDAQXTensorToPyArray(
                code.get_parity_z());
          },
          "Get the Z-type parity check matrix of the code")
      .def(
          "get_pauli_observables_matrix",
          [](code &code) {
            return cudaq::python::copyCUDAQXTensorToPyArray(
                code.get_pauli_observables_matrix());
          },
          "Get a matrix of the Pauli observables of the code")
      .def(
          "get_observables_x",
          [](code &code) {
            return cudaq::python::copyCUDAQXTensorToPyArray(
                code.get_observables_x());
          },
          "Get the Pauli X observables of the code")
      .def(
          "get_observables_z",
          [](code &code) {
            return cudaq::python::copyCUDAQXTensorToPyArray(
                code.get_observables_z());
          },
          "Get the Pauli Z observables of the code")
      .def("get_stabilizers", &code::get_stabilizers,
           "Get the stabilizer generators of the code")
      .def("contains_operation", &code::contains_operation, nb::arg("op"),
           "Return true if this code contains the given operation encoding")
      .def(
          "get_operation_one_qubit",
          [](const code &self, operation op) -> nb::object {
            if (!self.contains_operation(op))
              throw std::runtime_error(
                  "No encoding registered for requested op.");
            return get_python_kernel_or_throw(self, op);
          },
          nb::arg("op"),
          R"pbdoc(
              Get a CUDA-Q Python kernel for a one-qubit logical operation.

              Returns:
                A valid CUDA-Q Python kernel object.
            )pbdoc")
      .def(
          "get_operation_two_qubit",
          [](const code &self, operation op) -> nb::object {
            if (!self.contains_operation(op))
              throw std::runtime_error(
                  "No encoding registered for requested op.");
            return get_python_kernel_or_throw(self, op);
          },
          nb::arg("op"),
          R"pbdoc(
              Get a CUDA-Q Python kernel for a two-qubit logical operation.

              Returns:
                A valid CUDA-Q Python kernel object.
            )pbdoc")
      .def(
          "get_stabilizer_round",
          [](const code &self) -> nb::object {
            if (!self.contains_operation(operation::stabilizer_round))
              throw std::runtime_error(
                  "No stabilizer_round encoding is registered.");
            return get_python_kernel_or_throw(self,
                                              operation::stabilizer_round);
          },
          R"pbdoc(
              Get a CUDA-Q Python kernel for a stabilizer round logical operation.

              Returns:
                A valid CUDA-Q Python kernel object.
            )pbdoc")
      .def("get_num_data_qubits", &code::get_num_data_qubits,
           "Total number of physical data qubits required by the code.")
      .def("get_num_ancilla_qubits", &code::get_num_ancilla_qubits,
           "Total number of ancilla qubits required by the code.")
      .def(
          "get_num_ancilla_x_qubits", &code::get_num_ancilla_x_qubits,
          "Number of X-type ancilla qubits used for X stabilizer measurements.")
      .def(
          "get_num_ancilla_z_qubits", &code::get_num_ancilla_z_qubits,
          "Number of Z-type ancilla qubits used for Z stabilizer measurements.")
      .def("get_num_x_stabilizers", &code::get_num_x_stabilizers,
           "Number of X-type stabilizers.")
      .def("get_num_z_stabilizers", &code::get_num_z_stabilizers,
           "Number of Z-type stabilizers.");

  qecmod.def("code", [&](const std::string &name) {
    auto cppCodes = qec::get_available_codes();
    if (std::find(cppCodes.begin(), cppCodes.end(), name) != cppCodes.end())
      throw std::runtime_error("Invalid Python QEC Code name. " + name +
                               " is already used in the C++ Code registry.");

    return nb::cpp_function([name](nb::object code_class) -> nb::object {
      // Create new class that inherits from both Code and the original
      nb::object base_code = nb::module_::import_("cudaq_qec").attr("Code");
      // Create new type using Python's type() function
      nb::tuple bases = nb::make_tuple(base_code);
      // __dict__ is a read-only mappingproxy; copy to a real dict for
      // PyType_Type.tp_new
      nb::dict namespace_dict;
      namespace_dict.update(code_class.attr("__dict__"));

      if (!nb::hasattr(code_class, "get_num_data_qubits"))
        throw std::runtime_error(
            "Code class must implement get_num_data_qubits method");

      if (!nb::hasattr(code_class, "get_num_ancilla_qubits"))
        throw std::runtime_error(
            "Code class must implement get_num_ancilla_qubits method");

      if (!nb::hasattr(code_class, "get_num_ancilla_x_qubits"))
        throw std::runtime_error(
            "Code class must implement get_num_ancilla_x_qubits method");

      if (!nb::hasattr(code_class, "get_num_ancilla_z_qubits"))
        throw std::runtime_error(
            "Code class must implement get_num_ancilla_z_qubits method");

      if (!nb::hasattr(code_class, "get_num_x_stabilizers"))
        throw std::runtime_error(
            "Code class must implement get_num_x_stabilizers method");

      if (!nb::hasattr(code_class, "get_num_z_stabilizers"))
        throw std::runtime_error(
            "Code class must implement get_num_z_stabilizers method");

      // Use Python's type() so the correct metaclass (nanobind's) is resolved
      nb::object type_fn = nb::module_::import_("builtins").attr("type");
      nb::object new_class =
          type_fn(code_class.attr("__name__"), bases, namespace_dict);

      // Register the new class in the code registry
      PyCodeRegistry::register_code(name, [new_class](nb::kwargs options) {
        nb::object instance = new_class(**options);
        return instance;
      });
      return new_class;
    });
  });

  qecmod.def(
      "generate_random_bit_flips",
      [](std::size_t numBits, double error_probability) {
        auto data = generate_random_bit_flips(numBits, error_probability);
        return cudaq::python::copy1DCUDAQXTensorToPyArray(data);
      },
      "Generate a rank-1 tensor for random bits", nb::arg("numBits"),
      nb::arg("error_probability"));
  qecmod.def(
      "sample_memory_circuit",
      [](code &code, std::size_t numShots, std::size_t numRounds,
         std::optional<cudaq::noise_model> noise = std::nullopt) {
        auto [synd, dataRes] =
            noise ? sample_memory_circuit(code, numShots, numRounds, *noise)
                  : sample_memory_circuit(code, numShots, numRounds);
        return nb::make_tuple(
            cudaq::python::copyCUDAQXTensorToPyArray(synd),
            cudaq::python::copyCUDAQXTensorToPyArray(dataRes));
      },
      "Sample the memory circuit of the code", nb::arg("code"),
      nb::arg("numShots"), nb::arg("numRounds"), nb::arg("noise") = nb::none());
  qecmod.def(
      "sample_memory_circuit",
      [](code &code, operation op, std::size_t numShots, std::size_t numRounds,
         std::optional<cudaq::noise_model> noise = std::nullopt) {
        auto [synd, dataRes] =
            noise ? sample_memory_circuit(code, op, numShots, numRounds, *noise)
                  : sample_memory_circuit(code, op, numShots, numRounds);
        return nb::make_tuple(
            cudaq::python::copyCUDAQXTensorToPyArray(synd),
            cudaq::python::copyCUDAQXTensorToPyArray(dataRes));
      },
      "Sample the memory circuit of the code with a specific initial "
      "operation",
      nb::arg("code"), nb::arg("op"), nb::arg("numShots"), nb::arg("numRounds"),
      nb::arg("noise") = nb::none());

  qecmod.def(
      "dem_from_memory_circuit",
      [](code &code, operation op, std::size_t numRounds,
         std::optional<cudaq::noise_model> noise = std::nullopt) {
        return dem_from_memory_circuit(code, op, numRounds, *noise);
      },
      R"pbdoc(
        Generate a detector error model from a memory circuit.

        This function generates a detector error model from a memory circuit.
        The memory circuit is specified by the code, the initial state preparation
        operation, and the number of stabilizer measurement rounds. The noise
        model is optional and defaults to no noise.

        Args:
            code: The code to generate the detector error model for.
            op: The initial state preparation operation.
            numRounds: The number of stabilizer measurement rounds.
            noise: The noise model to apply to the memory circuit.

        Returns:
            A detector error model.
      )pbdoc",
      nb::arg("code"), nb::arg("op"), nb::arg("numRounds"),
      nb::arg("noise") = nb::none());

  qecmod.def(
      "x_dem_from_memory_circuit",
      [](code &code, operation op, std::size_t numRounds,
         std::optional<cudaq::noise_model> noise = std::nullopt) {
        return x_dem_from_memory_circuit(code, op, numRounds, *noise);
      },
      R"pbdoc(
        Generate a detector error model from a memory circuit in the X basis.

        This function generates a detector error model from a memory circuit in
        the X basis. The memory circuit is specified by the code, the initial
        state preparation operation, and the number of stabilizer measurement
        rounds. The noise model is optional and defaults to no noise.

        Args:
            code: The code to generate the detector error model for.
            op: The initial state preparation operation.
            numRounds: The number of stabilizer measurement rounds.
            noise: The noise model to apply to the memory circuit.

        Returns:
            A detector error model.
      )pbdoc",
      nb::arg("code"), nb::arg("op"), nb::arg("numRounds"),
      nb::arg("noise") = nb::none());

  qecmod.def(
      "z_dem_from_memory_circuit",
      [](code &code, operation op, std::size_t numRounds,
         std::optional<cudaq::noise_model> noise = std::nullopt) {
        return z_dem_from_memory_circuit(code, op, numRounds, *noise);
      },
      R"pbdoc(
        Generate a detector error model from a memory circuit in the Z basis.

        This function generates a detector error model from a memory circuit in
        the Z basis. The memory circuit is specified by the code, the initial
        state preparation operation, and the number of stabilizer measurement
        rounds. The noise model is optional and defaults to no noise.

        Args:
            code: The code to generate the detector error model for.
            op: The initial state preparation operation.
            numRounds: The number of stabilizer measurement rounds.
            noise: The noise model to apply to the memory circuit.

        Returns:
            A detector error model.
      )pbdoc",
      nb::arg("code"), nb::arg("op"), nb::arg("numRounds"),
      nb::arg("noise") = nb::none());

  qecmod.def(
      "sample_code_capacity",
      [](code &code, std::size_t numShots, double errorProb,
         std::optional<int> seed = std::nullopt) {
        if (seed.has_value()) {
          auto [syndromes, dataRes] =
              sample_code_capacity(code, numShots, errorProb, seed.value());
          return nb::make_tuple(
              cudaq::python::copyCUDAQXTensorToPyArray(syndromes),
              cudaq::python::copyCUDAQXTensorToPyArray(dataRes));
        }

        auto [syndromes, dataRes] =
            sample_code_capacity(code, numShots, errorProb);
        return nb::make_tuple(
            cudaq::python::copyCUDAQXTensorToPyArray(syndromes),
            cudaq::python::copyCUDAQXTensorToPyArray(dataRes));
      },
      "Sample syndrome measurements with code capacity noise.", nb::arg("code"),
      nb::arg("numShots"), nb::arg("errorProb"), nb::arg("seed") = nb::none());
  qecmod.def(
      "sample_code_capacity",
      [](const nb::ndarray<nb::numpy, uint8_t> H, std::size_t numShots,
         double errorProb, std::optional<std::size_t> seed = std::nullopt) {
        if (seed.has_value()) {
          auto [syndromes, dataRes] = sample_code_capacity(
              toTensor(H), numShots, errorProb, seed.value());
          return nb::make_tuple(
              cudaq::python::copyCUDAQXTensorToPyArray(syndromes),
              cudaq::python::copyCUDAQXTensorToPyArray(dataRes));
        }

        auto [syndromes, dataRes] =
            sample_code_capacity(toTensor(H), numShots, errorProb);
        return nb::make_tuple(
            cudaq::python::copyCUDAQXTensorToPyArray(syndromes),
            cudaq::python::copyCUDAQXTensorToPyArray(dataRes));
      },
      "Sample syndrome measurements with code capacity noise.", nb::arg("H"),
      nb::arg("numShots"), nb::arg("errorProb"), nb::arg("seed") = nb::none());

  std::stringstream ss;
  ss << "CUDA-Q QEC " << cudaq::qec::getVersion() << " ("
     << cudaq::qec::getFullRepositoryVersion() << ")";
  qecmod.attr("__version__") = ss.str();
}
} // namespace cudaq::qec
