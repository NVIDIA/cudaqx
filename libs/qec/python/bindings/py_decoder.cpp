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

#include "cudaq/qec/decoder.h"
#include "cudaq/qec/pcm_utils.h"
#include "cudaq/qec/plugin_loader.h"

#include "cuda-qx/core/kwargs_utils.h"
#include "type_casters.h"

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
  // Required by all plugin classes
  auto cleanup_callback = []() {
    // Change the type to the correct plugin type
    cleanup_plugins(PluginType::DECODER);
  };
  // This ensures the correct shutdown sequence
  mod.add_object("_cleanup", py::capsule(cleanup_callback));

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

  py::class_<async_decoder_result>(qecmod, "AsyncDecoderResult",
                                   R"pbdoc(
      A future-like object that holds the result of an asynchronous decoder call.
      Call get() to block until the result is available.
    )pbdoc")
      .def("get", &async_decoder_result::get,
           py::call_guard<py::gil_scoped_release>(),
           "Return the decoder result (blocking until ready)")
      .def("ready", &async_decoder_result::ready,
           py::call_guard<py::gil_scoped_release>(),
           "Return True if the asynchronous decoder result is ready, False "
           "otherwise");

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
      .def(
          "decode_async",
          [](decoder &dec,
             const std::vector<float_t> &syndrome) -> async_decoder_result {
            // Release the GIL while launching asynchronous work.
            py::gil_scoped_release release;
            return async_decoder_result(dec.decode_async(syndrome));
          },
          "Asynchronously decode the given syndrome", py::arg("syndrome"))
      .def(
          "decode_batch",
          [](decoder &decoder,
             const std::vector<std::vector<float_t>> &syndrome) {
            return decoder.decode_batch(syndrome);
          },
          "Decode multiple syndromes and return the results",
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

        if (buf.ndim != 2) {
          throw std::runtime_error(
              "Parity check matrix must be 2-dimensional.");
        }

        if (buf.itemsize != sizeof(uint8_t)) {
          throw std::runtime_error(
              "Parity check matrix must be an array of uint8_t.");
        }

        if (buf.strides[0] == buf.itemsize) {
          throw std::runtime_error(
              "Parity check matrix must be in row-major order, but "
              "column-major order was detected.");
        }

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
      "and optional decoder-specific parameters. Note: the parity check matrix "
      "must be in row-major order.");

  qecmod.def(
      "get_sorted_pcm_column_indices",
      [](const py::array_t<uint8_t> &H, std::uint32_t num_syndromes_per_round) {
        auto tensor_H = pcmToTensor(H);

        return cudaq::qec::get_sorted_pcm_column_indices(
            tensor_H, num_syndromes_per_round);
      },
      "Get the sorted column indices of a parity check matrix.", py::arg("H"),
      py::arg("num_syndromes_per_round") = 0);

  qecmod.def(
      "reorder_pcm_columns",
      [](const py::array_t<uint8_t> &H,
         const py::array_t<uint32_t> &column_order) {
        auto tensor_H = pcmToTensor(H);

        // Use pybind to create a std::vector from the column_order array
        std::vector<std::uint32_t> column_order_vec =
            column_order.cast<std::vector<std::uint32_t>>();

        auto H_new =
            cudaq::qec::reorder_pcm_columns(tensor_H, column_order_vec);

        // Construct a new py_array_t<uint8_t> from H_new.
        // FIXME - is this necessary
        py::array_t<uint8_t> H_new_py(H_new.shape());
        memcpy(H_new_py.mutable_data(), H_new.data(),
               H_new.shape()[0] * H_new.shape()[1] * sizeof(uint8_t));
        return H_new_py;
      },
      "Reorder the columns of a parity check matrix.");

  qecmod.def(
      "sort_pcm_columns",
      [](py::array_t<uint8_t> &H, std::uint32_t num_syndromes_per_round) {
        auto tensor_H = pcmToTensor(H);
        auto H_new =
            cudaq::qec::sort_pcm_columns(tensor_H, num_syndromes_per_round);

        // Construct a new py_array_t<uint8_t> from H_new.
        // FIXME - is this necessary
        py::array_t<uint8_t> H_new_py(H_new.shape());
        memcpy(H_new_py.mutable_data(), H_new.data(),
               H_new.shape()[0] * H_new.shape()[1] * sizeof(uint8_t));
        return H_new_py;
      },
      "Sort the columns of a parity check matrix.", py::arg("H"),
      py::arg("num_syndromes_per_round") = 0);

  qecmod.def(
      "dump_pcm",
      [](const py::array_t<uint8_t> &H) {
        auto tensor_H = pcmToTensor(H);
        tensor_H.dump_bits();
        printf("\n");
        fflush(stdout);
      },
      "Dump the parity check matrix to stdout.");

  qecmod.def(
      "generate_random_pcm",
      [](std::uint32_t n_rounds, std::uint32_t n_errs_per_round,
         std::uint32_t n_syndromes_per_round, std::uint32_t weight,
         std::uint32_t seed) {
        std::mt19937_64 rng(seed);
        if (seed == 0)
          rng = std::mt19937_64(std::random_device()());

        auto H_new = cudaq::qec::generate_random_pcm(n_rounds, n_errs_per_round,
                                                     n_syndromes_per_round,
                                                     weight, std::move(rng));
        // Construct a new py_array_t<uint8_t> from H_new.
        // FIXME - is this necessary
        py::array_t<uint8_t> H_new_py(H_new.shape());
        memcpy(H_new_py.mutable_data(), H_new.data(),
               H_new.shape()[0] * H_new.shape()[1] * sizeof(uint8_t));
        return H_new_py;
      },
      "Generate a random parity check matrix.", py::arg("n_rounds"),
      py::arg("n_errs_per_round"), py::arg("n_syndromes_per_round"),
      py::arg("weight"), py::arg("seed") = 0);

  qecmod.def(
      "get_pcm_for_rounds",
      [](const py::array_t<uint8_t> &H, std::uint32_t num_syndromes_per_round,
         std::uint32_t start_round, std::uint32_t end_round) {
        auto tensor_H = pcmToTensor(H);

        auto H_new = cudaq::qec::get_pcm_for_rounds(
            tensor_H, num_syndromes_per_round, start_round, end_round);

        // Construct a new py_array_t<uint8_t> from H_new.
        // FIXME - is this necessary
        py::array_t<uint8_t> H_new_py(H_new.shape());
        memcpy(H_new_py.mutable_data(), H_new.data(),
               H_new.shape()[0] * H_new.shape()[1] * sizeof(uint8_t));
        return H_new_py;
      },
      "Get a sub-PCM for a range of rounds.", py::arg("H"),
      py::arg("num_syndromes_per_round"), py::arg("start_round"),
      py::arg("end_round"));

  qecmod.def(
      "shuffle_pcm_columns",
      [](const py::array_t<uint8_t> &H, std::uint32_t seed) {
        auto tensor_H = pcmToTensor(H);
        std::mt19937_64 rng(seed);
        if (seed == 0)
          rng = std::mt19937_64(std::random_device()());

        auto H_new = cudaq::qec::shuffle_pcm_columns(tensor_H, std::move(rng));
        // Construct a new py_array_t<uint8_t> from H_new.
        // FIXME - is this necessary
        py::array_t<uint8_t> H_new_py(H_new.shape());
        memcpy(H_new_py.mutable_data(), H_new.data(),
               H_new.shape()[0] * H_new.shape()[1] * sizeof(uint8_t));
        return H_new_py;
      },
      "Shuffle the columns of a parity check matrix.", py::arg("H"),
      py::arg("seed") = 0);

  qecmod.def(
      "simplify_pcm",
      [](const py::array_t<uint8_t> &H, const py::array_t<double> &weights,
         std::uint32_t num_syndromes_per_round) {
        auto tensor_H = pcmToTensor(H);
        auto weights_vec = weights.cast<std::vector<double>>();
        auto [H_new, weights_new] = cudaq::qec::simplify_pcm(
            tensor_H, weights_vec, num_syndromes_per_round);
        // Construct a new py_array_t<uint8_t> from H_new.
        // FIXME - is this necessary
        py::array_t<uint8_t> H_new_py(H_new.shape());
        memcpy(H_new_py.mutable_data(), H_new.data(),
               H_new.shape()[0] * H_new.shape()[1] * sizeof(uint8_t));
        // Construct a new py_array_t<double> from weights_new.
        py::array_t<double> weights_new_py(weights_new.size());
        memcpy(weights_new_py.mutable_data(), weights_new.data(),
               weights_new.size() * sizeof(double));
        return py::make_tuple(H_new_py, weights_new_py);
      },
      "Simplify a parity check matrix.", py::arg("H"), py::arg("weights"),
      py::arg("num_syndromes_per_round"));
}

} // namespace cudaq::qec
