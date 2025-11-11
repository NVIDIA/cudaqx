/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_patch.h"

#include "cudaq/qec/patch.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace cudaq::qec {
void bindPatch(py::module &mod) {
  auto qecmod = py::hasattr(mod, "qecrt")
                    ? mod.attr("qecrt").cast<py::module_>()
                    : mod.def_submodule("qecrt");

  mod.doc() = "Python bindings for CUDA-Q QEC patch";

  py::class_<cudaq::qec::patch>(
      qecmod, "patch",
      "Represents a logical qubit patch for quantum error correction.\n"
      "Fields are cudaq.qview objects:\n"
      " - data : data-qubit view\n"
      " - ancx : X-stabilizer ancilla view\n"
      " - ancz : Z-stabilizer ancilla view")
      .def_readwrite("data", &cudaq::qec::patch::data,
                     "View of the data qubits in the patch")
      .def_readwrite("ancx", &cudaq::qec::patch::data,
                     "View of the ancilla qubits for X stabilizers")
      .def_readwrite("ancz", &cudaq::qec::patch::data,
                     "View of the ancilla qubits for Z stabilizers")
      .def("__repr__", [](const cudaq::qec::patch &p) {
        std::string s = "<cudaq.qec.patch";
        try {
          s += " data=" + std::to_string(p.data.size());
          s += " ancx=" + std::to_string(p.ancx.size());
          s += " ancz=" + std::to_string(p.ancz.size());
        } catch (...) {
        }
        s += ">";
        return s;
      });
}
} // namespace cudaq::qec
