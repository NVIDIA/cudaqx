/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "common/ObserveResult.h"
#include "cuda-qx/core/kwargs_utils.h"
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
    auto data = src.attr("serialize")().cast<std::vector<double>>();
    value = cudaq::spin_op(data);
    return true;
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

template <>
struct type_caster<cudaqx::heterogeneous_map> {
  PYBIND11_TYPE_CASTER(cudaqx::heterogeneous_map, const_name("dict"));

  bool load(handle src, bool) {
    if (!src)
      return false;
    try {
      value = cudaqx::hetMapFromKwargs(src.cast<py::dict>());
      return true;
    } catch (...) {
      return false;
    }
  }

  static handle cast(cudaqx::heterogeneous_map v,
                     return_value_policy /*policy*/, handle /*parent*/) {
    py::dict result;
    for (const auto &[key, val] : v) {
      if (auto *bool_val = std::any_cast<bool>(&val)) {
        result[key.c_str()] = *bool_val;
      } else if (auto *int_val = std::any_cast<std::size_t>(&val)) {
        result[key.c_str()] = *int_val;
      } else if (auto *int_val = std::any_cast<int>(&val)) {
        result[key.c_str()] = *int_val;
      } else if (auto *double_val = std::any_cast<double>(&val)) {
        result[key.c_str()] = *double_val;
      } else if (auto *str_val = std::any_cast<std::string>(&val)) {
        result[key.c_str()] = *str_val;
      } else if (auto *vec_val = std::any_cast<std::vector<double>>(&val)) {
        result[key.c_str()] = py::array_t<double>(
            {vec_val->size()}, {sizeof(double)}, vec_val->data());
      }
    }
    return result.release();
  }
};
} // namespace detail
} // namespace pybind11
