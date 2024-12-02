/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
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

#include "cudaq/qec/decoder.h"

#include "type_casters.h"
#include "utils.h"

namespace py = pybind11;
using namespace cudaqx;

namespace cudaq::qec {

class PyDecoder : public decoder {
public:
  PyDecoder(const py::array_t<uint8_t> &H) : decoder(toTensor(H)) {}

  decoder_result decode(const std::vector<float_t> &syndrome) override {
    PYBIND11_OVERRIDE_PURE(decoder_result, decoder, decode, syndrome);
  }
};

// Registry to store decoder factory functions
class PyDecoderRegistry {
private:
  static std::unordered_map<
      std::string,
      std::function<py::object(const py::array_t<uint8_t> &, py::kwargs)>>
      registry;

public:
  static void register_decoder(
      const std::string &name,
      std::function<py::object(const py::array_t<uint8_t> &, py::kwargs)>
          factory) {
    cudaq::info("Registering Pythonic Decoder with name {}", name);
    registry[name] = factory;
  }

  static py::object get_decoder(const std::string &name,
                                const py::array_t<uint8_t> &H,
                                py::kwargs options) {
    auto it = registry.find(name);
    if (it == registry.end()) {
      throw std::runtime_error("Unknown decoder: " + name);
    }

    return it->second(H, options);
  }

  static bool contains(const std::string &name) {
    return registry.find(name) != registry.end();
  }
};

std::unordered_map<std::string, std::function<py::object(
                                    const py::array_t<uint8_t> &, py::kwargs)>>
    PyDecoderRegistry::registry;

void bindDecoder(py::module &mod) {

  auto qecmod = py::hasattr(mod, "qecrt")
                    ? mod.attr("qecrt").cast<py::module_>()
                    : mod.def_submodule("qecrt");
  py::class_<decoder_result>(qecmod, "DecoderResult", R"pbdoc(
    A class representing the results of a quantum error correction decoding operation.

    This class encapsulates both the convergence status and the actual decoding result.
)pbdoc")
      .def(py::init<>(), R"pbdoc(
        Default constructor for DecoderResult.

        Creates a new DecoderResult instance with default values.
    )pbdoc")
      .def_readwrite("converged", &decoder_result::converged, R"pbdoc(
        Boolean flag indicating if the decoder converged to a solution.
        
        True if the decoder successfully found a valid correction chain,
        False if the decoder failed to converge or exceeded iteration limits.
    )pbdoc")
      .def_readwrite("result", &decoder_result::result, R"pbdoc(
        The decoded correction chain or recovery operation.
        
        Contains the sequence of corrections that should be applied to recover
        the original quantum state. The format depends on the specific decoder
        implementation.
    )pbdoc")
      // Add tuple interface
      .def("__len__", [](const decoder_result &) { return 2; })
      .def("__getitem__",
           [](const decoder_result &r, size_t i) {
             switch (i) {
             case 0:
               return py::cast(r.converged);
             case 1:
               return py::cast(r.result);
             default:
               throw py::index_error();
             }
           })
      // Enable iteration protocol
      .def("__iter__", [](const decoder_result &r) -> py::object {
        return py::iter(py::make_tuple(r.converged, r.result));
      });

  py::class_<decoder, PyDecoder>(
      qecmod, "Decoder", "Represents a decoder for quantum error correction")
      .def(py::init_alias<const py::array_t<uint8_t> &>())
      .def(
          "decode",
          [](decoder &decoder, const std::vector<float_t> &syndrome) {
            return decoder.decode(syndrome);
          },
          "Decode the given syndrome to determine the error correction",
          py::arg("syndrome"))
      .def("get_block_size", &decoder::get_block_size,
           "Get the size of the code block")
      .def("get_syndrome_size", &decoder::get_syndrome_size,
           "Get the size of the syndrome");

  // Expose decorator function that handles inheritance
  qecmod.def("decoder", [&](const std::string &name) {
    return py::cpp_function([name](py::object decoder_class) -> py::object {
      // Create new class that inherits from both Decoder and the original
      class py::object base_decoder =
          py::module::import("cudaq_qec").attr("Decoder");
      // Create new type using Python's type() function
      py::tuple bases = py::make_tuple(base_decoder);
      py::dict namespace_dict = decoder_class.attr("__dict__");

      if (!py::hasattr(decoder_class, "decode"))
        throw std::runtime_error("Decoder class must implement decode method");

      py::object new_class = py::reinterpret_steal<py::object>(
          PyType_Type.tp_new(&PyType_Type,
                             py::make_tuple(decoder_class.attr("__name__"),
                                            bases, namespace_dict)
                                 .ptr(),
                             nullptr));

      // Register the new class in the decoder registry
      PyDecoderRegistry::register_decoder(
          name, [new_class](const py::array_t<uint8_t> &H, py::kwargs options) {
            py::object instance = new_class(H, **options);
            return instance;
          });
      return new_class;
    });
  });

  qecmod.def(
      "get_decoder",
      [](const std::string &name, const py::array_t<uint8_t> H,
         const py::kwargs options)
          -> std::variant<py::object, std::unique_ptr<decoder>> {
        if (PyDecoderRegistry::contains(name))
          return PyDecoderRegistry::get_decoder(name, H, options);

        py::buffer_info buf = H.request();

        // Create a vector of the array dimensions
        std::vector<std::size_t> shape;
        for (py::ssize_t d : buf.shape) {
          shape.push_back(static_cast<std::size_t>(d));
        }

        // Create a tensor and borrow the NumPy array data
        cudaqx::tensor<uint8_t> tensor_H(shape);
        tensor_H.borrow(static_cast<uint8_t *>(buf.ptr), shape);

        return get_decoder(name, tensor_H, hetMapFromKwargs(options));
      },
      "Get a decoder by name with a given parity check matrix"
      "and optional decoder-specific parameters");
}

} // namespace cudaq::qec
