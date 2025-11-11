/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "common/ObserveResult.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "cudaq/qis/pauli_word.h"
namespace py = pybind11;

namespace pybind11 {
namespace detail {
template <>
struct type_caster<cudaq::spin_op> {
  PYBIND11_TYPE_CASTER(cudaq::spin_op, const_name("SpinOperator"));

  bool load(handle src, bool) {
    if (!src)
      return false;
    auto data = src.attr("serialize")().cast<std::vector<double>>();
    value = cudaq::spin_op(data);
    return true;
  }

  static handle cast(cudaq::spin_op v, return_value_policy /*policy*/,
                     handle /*parent*/) {
    py::object tv_py = py::module::import("cudaq").attr("SpinOperator")(
        v.get_data_representation());
    return tv_py.release();
  }
};

template <>
struct type_caster<cudaq::sample_result> {
  PYBIND11_TYPE_CASTER(cudaq::sample_result, const_name("SampleResult"));

  bool load(handle src, bool) {
    if (!src)
      return false;

    auto data = src.attr("serialize")().cast<std::vector<std::size_t>>();
    value = cudaq::sample_result();
    value.deserialize(data);
    return true;
  }

  static handle cast(cudaq::sample_result v, return_value_policy /*policy*/,
                     handle /*parent*/) {
    py::object tv_py = py::module::import("cudaq").attr("SampleResult")();
    tv_py.attr("deserialize")(v.serialize());
    return tv_py.release();
  }
};

template <>
struct type_caster<cudaq::observe_result> {
  PYBIND11_TYPE_CASTER(cudaq::observe_result, const_name("ObserveResult"));

  bool load(handle src, bool) {
    if (!src)
      return false;

    auto e = src.attr("expectation")().cast<double>();
    value = cudaq::observe_result(e, cudaq::spin_op());
    // etc.
    return true;
  }

  static handle cast(cudaq::observe_result v, return_value_policy /*policy*/,
                     handle /*parent*/) {
    py::object tv_py = py::module::import("cudaq").attr("ObserveResult")(
        v.expectation(), v.get_spin(), v.raw_data());
    return tv_py.release();
  }
};

template <>
struct type_caster<cudaq::pauli_word> {
  PYBIND11_TYPE_CASTER(cudaq::pauli_word, const_name("pauli_word"));

  // Load from Python to C++
  bool load(handle src, bool) {
    if (!src)
      return false;

    try {
      // If it's already a pauli_word object from cudaq module
      if (hasattr(src, "str")) {
        auto str_val = src.attr("str")().cast<std::string>();
        value = cudaq::pauli_word(str_val);
        return true;
      }
      // If it's just a string
      else if (py::isinstance<py::str>(src)) {
        value = cudaq::pauli_word(src.cast<std::string>());
        return true;
      }
    } catch (...) {
      return false;
    }
    return false;
  }

  // Cast from C++ to Python
  static handle cast(const cudaq::pauli_word &v, return_value_policy /*policy*/,
                     handle /*parent*/) {
    py::object pauli_word_class =
        py::module::import("cudaq").attr("pauli_word");
    py::object pauli_word_obj = pauli_word_class(v.str());
    return pauli_word_obj.release();
  }
};
} // namespace detail
} // namespace pybind11
