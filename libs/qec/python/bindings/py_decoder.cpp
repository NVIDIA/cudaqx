/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "common/ExecutionContext.h"
#include "common/FmtCore.h"
#include "cudaq/platform.h"
#include "cudaq/qec/decoder.h"
#include "cudaq/qec/detector_error_model.h"
#include "cudaq/qec/pcm_utils.h"
#include "cudaq/qec/plugin_loader.h"
#include "cudaq/runtime/logger/logger.h"
#include <filesystem>
#include <limits>
#include <link.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "cuda-qx/core/kwargs_utils.h"
#include "cuda-qx/core/library_utils.h"
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
      .def_readwrite("opt_results", &decoder_result::opt_results, R"pbdoc(
        Optional additional results from the decoder stored in a heterogeneous map.
        
        This field may be empty if no additional results are available.
    )pbdoc")
      .def("__len__", [](const decoder_result &) { return 3; })
      .def("__getitem__",
           [](const decoder_result &r, size_t i) {
             switch (i) {
             case 0:
               return py::cast(r.converged);
             case 1:
               return py::cast(r.result);
             case 2:
               return py::cast(r.opt_results);
             default:
               throw py::index_error();
             }
           })
      // Enable iteration protocol
      .def("__iter__", [](const decoder_result &r) -> py::object {
        return py::iter(py::make_tuple(r.converged, r.result, r.opt_results));
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
           "Get the size of the syndrome")
      .def("get_version", &decoder::get_version,
           "Get the version of the decoder");

  py::class_<detector_error_model>(qecmod, "DetectorErrorModel",
                                   R"pbdoc(
      A detector error model (DEM) for a quantum error correction circuit. A
      DEM can be created from a QEC circuit and a noise model. It contains
      information about which errors flip which detectors. This is used by the
      decoder to help make predictions about observables flips.
    )pbdoc")
      .def(py::init<>())
      .def_property(
          "detector_error_matrix",
          [](const detector_error_model &self) {
            const auto &t = self.detector_error_matrix;
            // Question: do you need py::cast(&self) here?
            return py::array_t<uint8_t>(
                t.shape(), {t.shape()[1] * sizeof(uint8_t), sizeof(uint8_t)},
                t.data());
          },
          [](detector_error_model &self, const py::array_t<uint8_t> &a) {
            auto borrow = toTensor(a);
            // Make sure that we own the data within the `detector_error_model`
            // as the input array may go out of scope.
            self.detector_error_matrix.copy(borrow.data(), borrow.shape());
          },
          R"pbdoc(
            The detector error matrix is a specific kind of circuit-level parity-check
            matrix where each row represents a detector, and each column represents
            an error mechanism. The entries of this matrix are H[i,j] = 1 if detector
            i is triggered by error mechanism j, and 0 otherwise.
          )pbdoc")
      .def_readwrite("error_rates", &detector_error_model::error_rates,
                     R"pbdoc(
      The list of weights has length equal to the number of columns of the
      detector error matrix, which assigns a likelihood to each error mechanism.
    )pbdoc")
      .def_readwrite("error_ids", &detector_error_model::error_ids, R"pbdoc(
       Error mechanism ID. From a probability perspective, each error mechanism
       ID is independent of all other error mechanism ID. For all errors with
       the *same* ID, only one of them can happen. That is - the errors
       containing the same ID are correlated with each other.
    )pbdoc")
      .def_property(
          "observables_flips_matrix",
          [](const detector_error_model &self) {
            const auto &t = self.observables_flips_matrix;
            // Question: do you need py::cast(&self) here?
            return py::array_t<uint8_t>(
                t.shape(), {t.shape()[1] * sizeof(uint8_t), sizeof(uint8_t)},
                t.data());
          },
          [](detector_error_model &self, const py::array_t<uint8_t> &a) {
            auto borrow = toTensor(a);
            // Make sure that we own the data within the `detector_error_model`
            // as the input array may go out of scope.
            self.observables_flips_matrix.copy(borrow.data(), borrow.shape());
          },
          R"pbdoc(
            The observables flips matrix is a specific kind of circuit-level parity-
            check matrix where each row represents a Pauli observable, and each
            column represents an error mechanism. The entries of this matrix are
            O[i,j] = 1 if Pauli observable i is flipped by error mechanism j, and 0
            otherwise.
          )pbdoc")
      .def("num_detectors", &detector_error_model::num_detectors,
           R"pbdoc(
            The number of detectors in the detector error model
          )pbdoc")
      .def("num_error_mechanisms", &detector_error_model::num_error_mechanisms,
           R"pbdoc(
            The number of error mechanisms in the detector error model
          )pbdoc")
      .def("num_observables", &detector_error_model::num_observables,
           R"pbdoc(
            The number of observables in the detector error model
          )pbdoc")
      .def("canonicalize_for_rounds",
           &detector_error_model::canonicalize_for_rounds,
           R"pbdoc(
            Canonicalize the detector error model for a given number of rounds
          )pbdoc",
           py::arg("num_syndromes_per_round"));

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

        if (name == "tensor_network_decoder") {
          throw std::runtime_error(
              "Decoder 'tensor_network_decoder' is not available. "
              "To enable it, install the python module's dependencies via:\n\n"
              "    pip install cudaq-qec[tensor-network-decoder]\n");
        }

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
      R"pbdoc(
        Get the sorted column indices of a parity check matrix.

        This function returns the column indices of a parity check matrix in
        topological order.

        Args:
            H: A NumPy array representing the parity check matrix
            num_syndromes_per_round: The number of syndrome measurements per round

        Returns:
            A NumPy array containing the sorted column indices

        See Also:
            :cpp:func:`cudaq::qec::get_sorted_pcm_column_indices`: The
            underlying C++ implementation of this function.
      )pbdoc",
      py::arg("H"), py::arg("num_syndromes_per_round") = 0);

  qecmod.def(
      "pcm_is_sorted",
      [](const py::array_t<uint8_t> &H, std::uint32_t num_syndromes_per_round) {
        auto tensor_H = pcmToTensor(H);
        return cudaq::qec::pcm_is_sorted(tensor_H, num_syndromes_per_round);
      },
      R"pbdoc(
        Check if a parity check matrix is sorted.

        This function checks if a parity check matrix is sorted in topological
        order.

        Args:
            H: A NumPy array representing the parity check matrix
            num_syndromes_per_round: The number of syndrome measurements per round

        Returns:
            A boolean indicating if the parity check matrix is sorted

        See Also:
            :cpp:func:`cudaq::qec::pcm_is_sorted`: The underlying C++
            implementation of this function.
      )pbdoc",
      py::arg("H"), py::arg("num_syndromes_per_round") = 0);

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

        // Construct a new py_array_t<uint8_t> from H_new (deep copy)
        return py::array_t<uint8_t>(
                   H_new.shape(),
                   {H_new.shape()[1] * sizeof(uint8_t), sizeof(uint8_t)},
                   H_new.data())
            .attr("copy")();
      },
      R"pbdoc(
        Reorder the columns of a parity check matrix.

        This function reorders the columns of a parity check matrix according to
        the given column order.

        Args:
            H: A NumPy array representing the parity check matrix
            column_order: A NumPy array containing the column order

        Returns:
            A NumPy array containing the reordered parity check matrix

        See Also:
            :cpp:func:`cudaq::qec::reorder_pcm_columns`: The underlying C++
            implementation of this function.
      )pbdoc",
      py::arg("H"), py::arg("column_order"));

  qecmod.def(
      "sort_pcm_columns",
      [](py::array_t<uint8_t> &H, std::uint32_t num_syndromes_per_round) {
        auto tensor_H = pcmToTensor(H);
        auto H_new =
            cudaq::qec::sort_pcm_columns(tensor_H, num_syndromes_per_round);

        // Construct a new py_array_t<uint8_t> from H_new (deep copy)
        return py::array_t<uint8_t>(
                   H_new.shape(),
                   {H_new.shape()[1] * sizeof(uint8_t), sizeof(uint8_t)},
                   H_new.data())
            .attr("copy")();
      },
      R"pbdoc(
        Sort the columns of a parity check matrix.

        This function sorts the columns of a parity check matrix in topological
        order.

        Args:
            H: A NumPy array representing the parity check matrix
            num_syndromes_per_round: The number of syndrome measurements per round

        Returns:
            A NumPy array containing the sorted parity check matrix

        See Also:
            :cpp:func:`cudaq::qec::sort_pcm_columns`: The underlying C++
            implementation of this function.
      )pbdoc",
      py::arg("H"), py::arg("num_syndromes_per_round") = 0);

  qecmod.def(
      "dump_pcm",
      [](const py::array_t<uint8_t> &H) {
        auto tensor_H = pcmToTensor(H);
        tensor_H.dump_bits();
        printf("\n");
        fflush(stdout);
      },
      R"pbdoc(
        Dump the parity check matrix to stdout.

        This function dumps the parity check matrix to stdout.
      )pbdoc",
      py::arg("H"));

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
        // Construct a new py_array_t<uint8_t> from H_new (deep copy)
        return py::array_t<uint8_t>(
                   H_new.shape(),
                   {H_new.shape()[1] * sizeof(uint8_t), sizeof(uint8_t)},
                   H_new.data())
            .attr("copy")();
      },
      R"pbdoc(
        Generate a random parity check matrix.

        This function creates a random parity check matrix for quantum error correction
        with specified parameters controlling the structure and randomness.

        Args:
            n_rounds: Number of measurement rounds in the error correction protocol
            n_errs_per_round: Number of error mechanisms per round
            n_syndromes_per_round: Number of syndrome measurements per round
            weight: The weight parameter controlling the sparsity of the matrix
            seed: Random seed for reproducibility (0 for random seed)

        See Also:
            :cpp:func:`cudaq::qec::generate_random_pcm`: The underlying C++
            implementation of this function.

        Returns:
            A NumPy array containing the generated parity check matrix
      )pbdoc",
      py::arg("n_rounds"), py::arg("n_errs_per_round"),
      py::arg("n_syndromes_per_round"), py::arg("weight"), py::arg("seed") = 0);

  qecmod.def(
      "get_pcm_for_rounds",
      [](const py::array_t<uint8_t> &H, std::uint32_t num_syndromes_per_round,
         std::uint32_t start_round, std::uint32_t end_round,
         bool straddle_end_round, bool straddle_start_round) {
        auto tensor_H = pcmToTensor(H);

        auto [H_new, first_column, last_column] =
            cudaq::qec::get_pcm_for_rounds(
                tensor_H, num_syndromes_per_round, start_round, end_round,
                straddle_start_round, straddle_end_round);

        // Construct a new py_array_t<uint8_t> from H_new (deep copy)
        return py::make_tuple(
            py::array_t<uint8_t>(
                H_new.shape(),
                {H_new.shape()[1] * sizeof(uint8_t), sizeof(uint8_t)},
                H_new.data())
                .attr("copy")(),
            first_column, last_column);
      },
      R"pbdoc(
        Get a sub-PCM for a range of rounds.

        This function returns a sub-parity check matrix for a range of rounds.

        Args:
            H: A NumPy array representing the parity check matrix
            num_syndromes_per_round: The number of syndrome measurements per round
            start_round: The starting round
            end_round: The ending round
            straddle_start_round: Whether to allow error mechanisms that
              straddle the start round (i.e. include prior rounds, too). This
              defaults to false.
            straddle_end_round: Whether to allow error mechanisms that straddle
              the end round (i.e. include future rounds, too). This defaults to
              false.

        Returns:
            A tuple containing the sub-parity check matrix and the first and last
            column indices of the sub-PCM relative to the original PCM.

        See Also:
            :cpp:func:`cudaq::qec::get_pcm_for_rounds`: The underlying C++
            implementation of this function.
      )pbdoc",
      py::arg("H"), py::arg("num_syndromes_per_round"), py::arg("start_round"),
      py::arg("end_round"), py::arg("straddle_start_round") = false,
      py::arg("straddle_end_round") = false);

  qecmod.def(
      "pcm_extend_to_n_rounds",
      [](const py::array_t<uint8_t> &H, std::uint32_t num_syndromes_per_round,
         std::uint32_t n_rounds) {
        auto tensor_H = pcmToTensor(H);
        auto [H_new, column_list] = cudaq::qec::pcm_extend_to_n_rounds(
            tensor_H, num_syndromes_per_round, n_rounds);
        // Construct a new py_array_t<uint8_t> from H_new.
        py::array_t<uint8_t> H_new_py(
            H_new.shape(),
            {H_new.shape()[1] * sizeof(uint8_t), sizeof(uint8_t)},
            H_new.data());
        return py::make_tuple(H_new_py.attr("copy")(), column_list);
      },
      R"pbdoc(
        Extend a parity check matrix to a given number of rounds.

        This function extends a parity check matrix to a given number of rounds.

        Args:
            H: A NumPy array representing the parity check matrix
            num_syndromes_per_round: The number of syndrome measurements per round
            n_rounds: The number of rounds to extend the parity check matrix to

        Returns:
            A tuple containing the extended parity check matrix and the list of
            column indices from the original PCM that were used to form the new
            PCM.

        See Also:
            :cpp:func:`cudaq::qec::pcm_extend_to_n_rounds`: The underlying C++
            implementation of this function.
      )pbdoc",
      py::arg("H"), py::arg("num_syndromes_per_round"), py::arg("n_rounds"));

  qecmod.def(
      "shuffle_pcm_columns",
      [](const py::array_t<uint8_t> &H, std::uint32_t seed) {
        auto tensor_H = pcmToTensor(H);
        std::mt19937_64 rng(seed);
        if (seed == 0)
          rng = std::mt19937_64(std::random_device()());

        auto H_new = cudaq::qec::shuffle_pcm_columns(tensor_H, std::move(rng));
        // Construct a new py_array_t<uint8_t> from H_new (deep copy)
        return py::array_t<uint8_t>(
                   H_new.shape(),
                   {H_new.shape()[1] * sizeof(uint8_t), sizeof(uint8_t)},
                   H_new.data())
            .attr("copy")();
      },
      R"pbdoc(
        Shuffle the columns of a parity check matrix.

        This function shuffles the columns of a parity check matrix.

        Args:
            H: A NumPy array representing the parity check matrix
            seed: Random seed for reproducibility (0 for random seed)

        Returns:
            A NumPy array containing the shuffled parity check matrix

        See Also:
            :cpp:func:`cudaq::qec::shuffle_pcm_columns`: The underlying C++
            implementation of this function.
      )pbdoc",
      py::arg("H"), py::arg("seed") = 0);

  qecmod.def(
      "simplify_pcm",
      [](const py::array_t<uint8_t> &H, const py::array_t<double> &weights,
         std::uint32_t num_syndromes_per_round) {
        auto tensor_H = pcmToTensor(H);
        auto weights_vec = weights.cast<std::vector<double>>();
        auto [H_new, weights_new] = cudaq::qec::simplify_pcm(
            tensor_H, weights_vec, num_syndromes_per_round);
        // Construct a new py_array_t<uint8_t> from H_new.
        py::array_t<uint8_t> H_new_py(
            H_new.shape(),
            {H_new.shape()[1] * sizeof(uint8_t), sizeof(uint8_t)},
            H_new.data());
        // Construct a new py_array_t<double> from weights_new.
        py::array_t<double> weights_new_py(
            {weights_new.size()}, {sizeof(double)}, weights_new.data());
        return py::make_tuple(H_new_py.attr("copy")(),
                              weights_new_py.attr("copy")());
      },
      R"pbdoc(
        Simplify a parity check matrix.

        This function simplifies a parity check matrix by removing duplicate
        columns and 0-weight columns.

        Args:
            H: A NumPy array representing the parity check matrix
            weights: A NumPy array containing the weights of the columns
            num_syndromes_per_round: The number of syndrome measurements per round

        Returns:
            A tuple containing the simplified parity check matrix and the weights

        See Also:
            :cpp:func:`cudaq::qec::simplify_pcm`: The underlying C++
            implementation of this function.
      )pbdoc",
      py::arg("H"), py::arg("weights"), py::arg("num_syndromes_per_round"));

  const auto loadDecoderLibrary = [](const std::string &path) {
    static void *decoderLibHandle = nullptr;

    // Unload if previously-loaded
    // Note: dlclose makes no guarantees about unloading. Hence, we need to
    // manually execute `init_func` of the kernels (should be executed as static
    // initializers in those libs) to make sure they are registered again.
    if (decoderLibHandle)
      dlclose(decoderLibHandle);

    decoderLibHandle = dlopen(path.c_str(), RTLD_GLOBAL | RTLD_NOW);
    if (!decoderLibHandle) {
      // Retrieve the error message
      char *error_msg = dlerror();
      throw std::runtime_error(
          fmt::format("Failed to load decoder library at '{}': {}", path,
                      (error_msg ? std::string(error_msg) : "unknown error.")));
    }

    // List of `init_func` names to call for decoder registration
    // clang-format off
    static const std::vector<std::string> initFuncNames = {
        "function_enqueue_syndromes._ZN5cudaq3qec8decoding17enqueue_syndromesEmRKSt6vectorIbSaIbEEm.init_func",
        "function_get_corrections._ZN5cudaq3qec8decoding15get_correctionsEmmb.init_func",
        "function_reset_decoder._ZN5cudaq3qec8decoding13reset_decoderEm.init_func"};
    // clang-format on
    for (const auto &funcName : initFuncNames) {
      // Use dlsym to get the function pointer
      using InitFuncType = void (*)();
      InitFuncType initFunc = reinterpret_cast<InitFuncType>(
          dlsym(decoderLibHandle, funcName.c_str()));
      if (!initFunc) {
        char *error_msg = dlerror();
        throw std::runtime_error(fmt::format(
            "Failed to locate init function '{}' in decoder library at '{}': "
            "{}",
            funcName, path,
            (error_msg ? std::string(error_msg) : "unknown error.")));
      }
      // Call the init function to register/update the decoder quake code
      initFunc();
    }
  };

  qecmod.def(
      "load_quantinuum_realtime_decoding",
      [&]() {
        const std::filesystem::path path{
            cudaqx::__internal__::getCUDAQXLibraryPath(
                cudaqx::__internal__::CUDAQXLibraryType::QEC)};
        const auto quantinuumLibPath =
            path.parent_path() / "libcudaq-qec-realtime-decoding-quantinuum.so";
        loadDecoderLibrary(quantinuumLibPath);
      },
      R"pbdoc(
        [Internal] Load Quantinuum realtime decoder library.
      )pbdoc");

  qecmod.def(
      "load_simulation_realtime_decoding",
      [&]() {
        const std::filesystem::path path{
            cudaqx::__internal__::getCUDAQXLibraryPath(
                cudaqx::__internal__::CUDAQXLibraryType::QEC)};
        const auto simulationLibPath =
            path.parent_path() / "libcudaq-qec-realtime-decoding-simulation.so";
        loadDecoderLibrary(simulationLibPath);
      },
      R"pbdoc(
        [Internal] Load local simulation realtime decoder library.
      )pbdoc");

  qecmod.def(
      "compute_msm",
      [](std::function<void()> kernel, bool verbose = false) {
        cudaq::ExecutionContext ctx_msm_size("msm_size");
        auto &platform = cudaq::get_platform();
        platform.with_execution_context(ctx_msm_size, kernel);
        if (!ctx_msm_size.msm_dimensions.has_value()) {
          throw std::runtime_error("No MSM dimensions found");
        }
        if (ctx_msm_size.msm_dimensions.value().second == 0) {
          throw std::runtime_error("No MSM dimensions found");
        }
        cudaq::ExecutionContext ctx_msm("msm");
        ctx_msm.msm_dimensions = ctx_msm_size.msm_dimensions;
        platform.with_execution_context(ctx_msm, kernel);

        auto msm_as_strings = ctx_msm.result.sequential_data();
        if (verbose) {
          printf("MSM Dimensions: %ld measurements x %ld error mechanisms\n",
                 ctx_msm.msm_dimensions.value().first,
                 ctx_msm.msm_dimensions.value().second);
          for (std::size_t i = 0; i < ctx_msm.msm_dimensions.value().first;
               i++) {
            for (std::size_t j = 0; j < ctx_msm.msm_dimensions.value().second;
                 j++) {
              printf("%c", msm_as_strings[j][i] == '1' ? '1' : '.');
            }
            printf("\n");
          }
        }
        return std::make_tuple(msm_as_strings, ctx_msm.msm_dimensions.value(),
                               ctx_msm.msm_probabilities.value(),
                               ctx_msm.msm_prob_err_id.value());
      },
      "");
  qecmod.def(
      "construct_mz_table",
      [](const std::vector<std::string> &msm_as_strings) {
        cudaqx::tensor<uint8_t> mzTable(msm_as_strings);
        mzTable = mzTable.transpose();
        return cudaq::python::copyCUDAQXTensorToPyArray(mzTable);
      },
      "");

  qecmod.def(
      "generate_timelike_sparse_detector_matrix",
      [](std::uint32_t num_syndromes_per_round, std::uint32_t num_rounds,
         bool include_first_round) {
        return cudaq::qec::generate_timelike_sparse_detector_matrix(
            num_syndromes_per_round, num_rounds, include_first_round);
      },
      R"pbdoc(
        Generate a sparse detector matrix for a given number of syndromes per round
        and number of rounds. Time-like here means that each round of syndrome measurement
        bits are xor'd against the preceding round.
        
        Args:
            num_syndromes_per_round: The number of syndrome measurements per round
            num_rounds: The number of rounds to generate the sparse detector matrix for
            include_first_round: Whether to include the first round of syndrome measurements

        Returns:
            The detector matrix format is CSR-like, with -1 values indicating the end of each row.
      )pbdoc",
      py::arg("num_syndromes_per_round"), py::arg("num_rounds"),
      py::arg("include_first_round"));

  qecmod.def(
      "pcm_to_sparse_vec",
      [](const py::array_t<uint8_t> &pcm) {
        auto tensor_pcm = pcmToTensor(pcm);
        return cudaq::qec::pcm_to_sparse_vec(tensor_pcm);
      },
      R"pbdoc(
        Return a sparse representation of the PCM.
      )pbdoc",
      py::arg("pcm"));
}

} // namespace cudaq::qec
