/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include <limits>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "common/Logger.h"
#include "cudaq/python/PythonCppInterop.h"
#include "cudaq/utils/registry.h"

#include "cudaq/qec/experiments.h"

#include "cuda-qx/core/kwargs_utils.h"
#include "type_casters.h"

namespace py = pybind11;
using namespace cudaqx;

namespace cudaq::qec {

class PyCode : public qec::code {
protected:
  // Trampoline methods for pure virtual functions
  std::size_t get_num_data_qubits() const override {
    PYBIND11_OVERRIDE_PURE(std::size_t, qec::code, get_num_data_qubits);
  }

  std::size_t get_num_ancilla_qubits() const override {
    PYBIND11_OVERRIDE_PURE(std::size_t, qec::code, get_num_ancilla_qubits);
  }

  std::size_t get_num_ancilla_x_qubits() const override {
    PYBIND11_OVERRIDE_PURE(std::size_t, qec::code, get_num_ancilla_x_qubits);
  }

  std::size_t get_num_ancilla_z_qubits() const override {
    PYBIND11_OVERRIDE_PURE(std::size_t, qec::code, get_num_ancilla_z_qubits);
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
  py::object pyCode;

public:
  /// @brief Constructs a PyCodeHandle from a Python QEC code object
  /// @param registeredCode Python object containing the QEC code definition
  /// @throw std::runtime_error if the Python code lacks required attributes
  /// (stabilizers or operation_encodings)
  /// @details Initializes the handle by:
  /// - Validating the presence of required attributes
  /// - Converting Python stabilizers to C++ representation
  /// - Processing operation encodings and registering CUDA-Q kernels
  PyCodeHandle(py::object registeredCode) : pyCode(registeredCode) {
    if (!py::hasattr(registeredCode, "stabilizers"))
      throw std::runtime_error(
          "Invalid Python QEC Code. Must have self.stabilizers = "
          "qec.Stabilizers(...). Please provide the stabilizers.");
    if (!py::hasattr(registeredCode, "operation_encodings"))
      throw std::runtime_error(
          "Invalid Python QEC Code. Must have self.operation_encodings = "
          "{...}. Please provide the CUDA-Q kernels for the operation "
          "encodings.");

    // Get the stabilizers
    m_stabilizers =
        registeredCode.attr("stabilizers").cast<std::vector<cudaq::spin_op_term>>();
    // Get the CUDA-Q kernels for the operation encodings
    auto opsDict = registeredCode.attr("operation_encodings").cast<py::dict>();

    // For each CUDA-Q kernel, extract the JIT-ed function pointer
    for (auto &[opKey, kernelHandle] : opsDict) {
      py::object kernel = py::cast<py::object>(kernelHandle);
      auto opKeyEnum = opKey.cast<qec::operation>();

      // Create the kernel interop object
      cudaq::python::CppPyKernelDecorator opInterop(kernel);
      opInterop.compile();

      // Get the kernel name
      auto baseName = kernelHandle.attr("name").cast<std::string>();
      std::string kernelName = "__nvqpp__mlirgen__" + baseName;

      // Extract teh function pointer, register with qkernel system
      auto capsule = kernel.attr("extract_c_function_pointer")(kernelName)
                         .cast<py::capsule>();
      void *ptr = capsule;
      cudaq::registry::__cudaq_registerLinkableKernel(ptr, baseName.c_str(),
                                                      ptr);

      // Make sure we cast the function pointer correctly
      if (opKeyEnum == operation::stabilizer_round) {
        auto *casted = reinterpret_cast<std::vector<cudaq::measure_result> (*)(
            patch, const std::vector<std::size_t> &,
            const std::vector<std::size_t> &)>(ptr);
        operation_encodings.insert(
            {opKeyEnum, cudaq::qkernel<std::vector<cudaq::measure_result>(
                            patch, const std::vector<std::size_t> &,
                            const std::vector<std::size_t> &)>(casted)});
        continue;
      }

      // FIXME handle other signatures later... this assumes single patch
      // signatures
      auto *casted = reinterpret_cast<void (*)(patch)>(ptr);
      operation_encodings.insert(
          {opKeyEnum, cudaq::qkernel<void(patch)>(casted)});
    }
  }

protected:
  // Trampoline methods for pure virtual functions
  std::size_t get_num_data_qubits() const override {
    return pyCode.attr("get_num_data_qubits")().cast<std::size_t>();
  }

  std::size_t get_num_ancilla_qubits() const override {
    return pyCode.attr("get_num_ancilla_qubits")().cast<std::size_t>();
  }

  std::size_t get_num_ancilla_x_qubits() const override {
    return pyCode.attr("get_num_ancilla_x_qubits")().cast<std::size_t>();
  }

  std::size_t get_num_ancilla_z_qubits() const override {
    return pyCode.attr("get_num_ancilla_z_qubits")().cast<std::size_t>();
  }
};

// Registry to store code factory functions
class PyCodeRegistry {
private:
  static std::unordered_map<std::string, std::function<py::object(py::kwargs)>>
      registry;

public:
  static void register_code(const std::string &name,
                            std::function<py::object(py::kwargs)> factory) {
    cudaq::info("Registering Pythonic QEC Code with name {}", name);
    registry[name] = factory;
  }

  static py::object get_code(const std::string &name, py::kwargs options) {
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

std::unordered_map<std::string, std::function<py::object(py::kwargs)>>
    PyCodeRegistry::registry;

template <typename T>
auto copyCUDAQXTensorToPyArray(const cudaqx::tensor<T> &tensor) {
  auto shape = tensor.shape();
  auto rows = shape[0];
  auto cols = shape[1];
  size_t total_size = rows * cols;

  // Allocate new memory and copy the data
  T *data_copy = new T[total_size];
  std::memcpy(data_copy, tensor.data(), total_size * sizeof(T));

  // Create a NumPy array using the buffer protocol
  return py::array_t<T>(
      {rows, cols},                  // Shape of the array
      {cols * sizeof(T), sizeof(T)}, // Strides for row-major layout
      data_copy,                     // Pointer to the data
      py::capsule(data_copy, [](void *p) { delete[] static_cast<T *>(p); }));
}

template <typename T>
auto copy1DCUDAQXTensorToPyArray(const cudaqx::tensor<T> &tensor) {
  auto shape = tensor.shape();
  auto rows = shape[0];
  size_t total_size = rows;

  // Allocate new memory and copy the data
  T *data_copy = new T[total_size];
  std::memcpy(data_copy, tensor.data(), total_size * sizeof(T));

  // Create a NumPy array using the buffer protocol
  return py::array_t<T>(
      {static_cast<py::ssize_t>(rows)}, // Shape of the array
      data_copy,                        // Pointer to the data
      py::capsule(data_copy, [](void *p) { delete[] static_cast<T *>(p); }));
}

void bindCode(py::module &mod) {

  auto qecmod = py::hasattr(mod, "qecrt")
                    ? mod.attr("qecrt").cast<py::module_>()
                    : mod.def_submodule("qecrt");

  py::class_<qec::two_qubit_depolarization, cudaq::kraus_channel>(
      qecmod, "TwoQubitDepolarization",
      R"#(Models the decoherence of the each qubit independently in a two-qubit operation into a mixture "
      of the computational basis states, `|0>` and `|1>`.)#")
      .def(py::init<double>(), py::arg("probability"),
           "Initialize the `TwoQubitDepolarizationChannel` with the provided "
           "`probability`.");

  py::class_<qec::two_qubit_bitflip, cudaq::kraus_channel>(
      qecmod, "TwoQubitBitFlip",
      R"#(Models independent bit flip errors after a two-qubit operation.)#")
      .def(py::init<double>(), py::arg("probability"),
           "Initialize the `TwoQubitBitFlip` with the provided "
           "`probability`.");

  py::enum_<operation>(
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
      [](const std::string &name, py::kwargs options) -> std::unique_ptr<code> {
        if (PyCodeRegistry::contains(name))
          return std::make_unique<PyCodeHandle>(
              PyCodeRegistry::get_code(name, options));

        if (options.contains("stabilizers")) {
          auto obj = options["stabilizers"];
          if (!py::isinstance<py::list>(obj))
            throw std::runtime_error(
                "invalid stabilizers passed to get_code, must be a list of "
                "string pauli words or list of cudaq.SpinOperator.");

          if (py::isinstance<py::str>(obj.cast<py::list>()[0])) {
            options.attr("pop")("stabilizers");
            auto words = obj.cast<std::vector<std::string>>();
            std::vector<cudaq::spin_op_term> ops;
            for (auto &os : words)
              ops.emplace_back(cudaq::spin_op::from_word(os));
            sortStabilizerOps(ops);
            return get_code(name, ops, hetMapFromKwargs(options));
          }

          if (py::isinstance<cudaq::spin_op_term>(obj[0])) {
            options.attr("pop")("stabilizers");
            return get_code(name, obj.cast<std::vector<cudaq::spin_op_term>>(),
                            hetMapFromKwargs(options));
          }

          throw std::runtime_error(
              "get_code error - invalid stabilizers element type.");
        }

        return get_code(name, hetMapFromKwargs(options));
      },
      "Retrieve a quantum error correction code by name with optional "
      "parameters");

  qecmod.def("get_available_codes", &get_available_codes,
             "Get a list of all available quantum error correction codes");

  py::class_<code, PyCode>(qecmod, "Code",
                           "Represents a quantum error correction code")
      .def(py::init<>())
      .def(
          "get_parity",
          [](code &code) {
            return copyCUDAQXTensorToPyArray(code.get_parity());
          },
          "Get the parity check matrix of the code")
      .def(
          "get_parity_x",
          [](code &code) {
            return copyCUDAQXTensorToPyArray(code.get_parity_x());
          },
          "Get the X-type parity check matrix of the code")
      .def(
          "get_parity_z",
          [](code &code) {
            return copyCUDAQXTensorToPyArray(code.get_parity_z());
          },
          "Get the Z-type parity check matrix of the code")
      .def(
          "get_pauli_observables_matrix",
          [](code &code) {
            return copyCUDAQXTensorToPyArray(
                code.get_pauli_observables_matrix());
          },
          "Get a matrix of the Pauli observables of the code")
      .def(
          "get_observables_x",
          [](code &code) {
            return copyCUDAQXTensorToPyArray(code.get_observables_x());
          },
          "Get the Pauli X observables of the code")
      .def(
          "get_observables_z",
          [](code &code) {
            return copyCUDAQXTensorToPyArray(code.get_observables_z());
          },
          "Get the Pauli Z observables of the code")
      .def("get_stabilizers", &code::get_stabilizers,
           "Get the stabilizer generators of the code");

  qecmod.def("code", [&](const std::string &name) {
    auto cppCodes = qec::get_available_codes();
    if (std::find(cppCodes.begin(), cppCodes.end(), name) != cppCodes.end())
      throw std::runtime_error("Invalid Python QEC Code name. " + name +
                               " is already used in the C++ Code registry.");

    return py::cpp_function([name](py::object code_class) -> py::object {
      // Create new class that inherits from both Code and the original
      class py::object base_code = py::module::import("cudaq_qec").attr("Code");
      // Create new type using Python's type() function
      py::tuple bases = py::make_tuple(base_code);
      py::dict namespace_dict = code_class.attr("__dict__");

      if (!py::hasattr(code_class, "get_num_data_qubits"))
        throw std::runtime_error(
            "Code class must implement get_num_data_qubits method");

      if (!py::hasattr(code_class, "get_num_ancilla_qubits"))
        throw std::runtime_error(
            "Code class must implement get_num_ancilla_qubits method");

      if (!py::hasattr(code_class, "get_num_ancilla_x_qubits"))
        throw std::runtime_error(
            "Code class must implement get_num_ancilla_x_qubits method");

      if (!py::hasattr(code_class, "get_num_ancilla_z_qubits"))
        throw std::runtime_error(
            "Code class must implement get_num_ancilla_z_qubits method");

      py::object new_class =
          py::reinterpret_steal<py::object>(PyType_Type.tp_new(
              &PyType_Type,
              py::make_tuple(code_class.attr("__name__"), bases, namespace_dict)
                  .ptr(),
              nullptr));

      // Register the new class in the code registry
      PyCodeRegistry::register_code(name, [new_class](py::kwargs options) {
        py::object instance = new_class(**options);
        return instance;
      });
      return new_class;
    });
  });

  qecmod.def(
      "generate_random_bit_flips",
      [](std::size_t numBits, double error_probability) {
        auto data = generate_random_bit_flips(numBits, error_probability);
        return copy1DCUDAQXTensorToPyArray(data);
      },
      "Generate a rank-1 tensor for random bits", py::arg("numBits"),
      py::arg("error_probability"));
  qecmod.def(
      "sample_memory_circuit",
      [](code &code, std::size_t numShots, std::size_t numRounds,
         std::optional<cudaq::noise_model> noise = std::nullopt) {
        auto [synd, dataRes] =
            noise ? sample_memory_circuit(code, numShots, numRounds, *noise)
                  : sample_memory_circuit(code, numShots, numRounds);
        return py::make_tuple(copyCUDAQXTensorToPyArray(synd),
                              copyCUDAQXTensorToPyArray(dataRes));
      },
      "Sample the memory circuit of the code", py::arg("code"),
      py::arg("numShots"), py::arg("numRounds"),
      py::arg("noise") = std::nullopt);
  qecmod.def(
      "sample_memory_circuit",
      [](code &code, operation op, std::size_t numShots, std::size_t numRounds,
         std::optional<cudaq::noise_model> noise = std::nullopt) {
        auto [synd, dataRes] =
            noise ? sample_memory_circuit(code, op, numShots, numRounds, *noise)
                  : sample_memory_circuit(code, op, numShots, numRounds);
        return py::make_tuple(copyCUDAQXTensorToPyArray(synd),
                              copyCUDAQXTensorToPyArray(dataRes));
      },
      "Sample the memory circuit of the code with a specific initial "
      "operation",
      py::arg("code"), py::arg("op"), py::arg("numShots"), py::arg("numRounds"),
      py::arg("noise") = std::nullopt);

  qecmod.def(
      "sample_code_capacity",
      [](code &code, std::size_t numShots, double errorProb,
         std::optional<int> seed = std::nullopt) {
        if (seed.has_value()) {
          auto [syndromes, dataRes] =
              sample_code_capacity(code, numShots, errorProb, seed.value());
          return py::make_tuple(copyCUDAQXTensorToPyArray(syndromes),
                                copyCUDAQXTensorToPyArray(dataRes));
        }

        auto [syndromes, dataRes] =
            sample_code_capacity(code, numShots, errorProb);
        return py::make_tuple(copyCUDAQXTensorToPyArray(syndromes),
                              copyCUDAQXTensorToPyArray(dataRes));
      },
      "Sample syndrome measurements with code capacity noise.", py::arg("code"),
      py::arg("numShots"), py::arg("errorProb"), py::arg("seed") = py::none());
  qecmod.def(
      "sample_code_capacity",
      [](const py::array_t<uint8_t> H, std::size_t numShots, double errorProb,
         std::optional<std::size_t> seed = std::nullopt) {
        if (seed.has_value()) {
          auto [syndromes, dataRes] = sample_code_capacity(
              toTensor(H), numShots, errorProb, seed.value());
          return py::make_tuple(copyCUDAQXTensorToPyArray(syndromes),
                                copyCUDAQXTensorToPyArray(dataRes));
        }

        auto [syndromes, dataRes] =
            sample_code_capacity(toTensor(H), numShots, errorProb);
        return py::make_tuple(copyCUDAQXTensorToPyArray(syndromes),
                              copyCUDAQXTensorToPyArray(dataRes));
      },
      "Sample syndrome measurements with code capacity noise.", py::arg("H"),
      py::arg("numShots"), py::arg("errorProb"), py::arg("seed") = py::none());
}
} // namespace cudaq::qec
