/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_decoding.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cudaq/python/PythonCppInterop.h"

#include "cudaq/qis/qubit_qis.h"
namespace py = pybind11;

namespace cudaq::qec::decoding {

// Convert a Python iterable of bool/int/measure_result into
// std::vector<cudaq::measure_result>
static std::vector<cudaq::measure_result>
to_measure_result_vector(const py::handle &iterable) {
  std::vector<cudaq::measure_result> results;
  for (const auto &item : iterable) {
    if (py::isinstance<cudaq::measure_result>(item)) {
      results.emplace_back(py::cast<cudaq::measure_result>(item));
      continue;
    }

    bool b = py::cast<bool>(item);
    results.emplace_back(cudaq::measure_result{b});
  }
  return results;
}

void bindDecoding(py::module &mod) {
  auto qecmod = py::hasattr(mod, "qecrt")
                    ? mod.attr("qecrt").cast<py::module_>()
                    : mod.def_submodule("qecrt");

  // We split the bindings of device kernels into 2 phases:
  // 1. Register the device kernels as valid Python functions: this is done once
  // per module load
  // 2. Register the device kernel Quake code: this is done via
  // `load_device_kernels` function call. As the quake code is populated into
  // the quake registry after an appropriate library is loaded, we need to
  // ensure that the quake code is registered after the library is loaded.
  py::module_ qecSubMod = [&]() {
    const char *subModName = "qec";
    if (py::hasattr(mod, subModName))
      return mod.attr(subModName).cast<py::module_>();
    else
      return mod.def_submodule(subModName);
  }();
  const std::string qecModName = qecSubMod.attr("__name__").cast<std::string>();
  // Define these kernels as part of the qec submodule
  qecSubMod.def(
      "reset_decoder", [](std::uint64_t) {},
      R"pbdoc(Reset the decoder with the given ID.)pbdoc");
  qecSubMod.def(
      "enqueue_syndromes",
      [](std::uint64_t, std::vector<cudaq::measure_result>, std::uint64_t) {},
      R"pbdoc(Reset the decoder with the given ID.)pbdoc"
      R"pbdoc(Enqueue a vector of syndrome bit for realtime decoding.
                Parameters
                - decoder_id: The ID of the decoder.
                - syndromes: A vector of syndrome bits (0 or 1).
                - tag: An optional tag for the enqueue operation.
                        )pbdoc");
  qecSubMod.def(
      "get_corrections", [](std::uint64_t, std::uint64_t, bool) {},
      R"pbdoc(Get the corrections from the decoder.
                Parameters
                - decoder_id: The ID of the decoder.
                - return_size: The number of corrections to return.
                - reset: Whether to reset the decoder after getting corrections.
                        )pbdoc");

  qecmod.def("load_device_kernels", [qecModName]() {
    cudaq::python::registerDeviceKernel(
        qecModName, "reset_decoder",
        cudaq::python::getMangledArgsString<std::uint64_t>());
    cudaq::python::registerDeviceKernel(
        qecModName, "enqueue_syndromes",
        cudaq::python::getMangledArgsString<std::uint64_t,
                                            std::vector<cudaq::measure_result>,
                                            std::uint64_t>());
    cudaq::python::registerDeviceKernel(
        qecModName, "get_corrections",
        cudaq::python::getMangledArgsString<std::uint64_t, std::uint64_t,
                                            bool>());
  });

  qecmod.def("__repr__", [qecmod]() {
    return "<qecrt.decoding (realtime decoding API bindings)>";
  });
}
} // namespace cudaq::qec::decoding
