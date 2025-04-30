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
namespace py = pybind11;

namespace pybind11 {
namespace detail {
template <>
struct type_caster<cudaq::spin_op> {
  PYBIND11_TYPE_CASTER(cudaq::spin_op, const_name("SpinOperator"));

  bool load(handle src, bool) {
    if (!src)
      return false;
    auto className = src.attr("__class__").attr("__name__");
    if (className.cast<std::string>() == "SpinOperator") {
      auto data = src.attr("serialize")().cast<std::vector<double>>();
      auto numQubits = src.attr("qubit_count").cast<std::size_t>();
      value = cudaq::spin_op(data);
    } else if (className.cast<std::string>() == "SpinOperatorTerm") {
      // Construct a SpinOperator from a SpinOperatorTerm
      auto cudaq_module = py::module::import("cudaq");
      auto spinop_cls = cudaq_module.attr("SpinOperator");
      py::object spinop_src = spinop_cls(src);
      auto data = spinop_src.attr("serialize")().cast<std::vector<double>>();
      auto numQubits = spinop_src.attr("qubit_count").cast<std::size_t>();
      value = cudaq::spin_op(data);
    }
    return false;
  }

  static handle cast(cudaq::spin_op v, return_value_policy /*policy*/,
                     handle /*parent*/) {
    py::object tv_py = py::module::import("cudaq").attr("SpinOperator")(
        v.get_data_representation()); // Construct new python obj
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
} // namespace detail
} // namespace pybind11
