/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "common/ObserveResult.h"
#include "cuda-qx/core/heterogeneous_map.h"
#include "cuda-qx/core/kwargs_utils.h"
#include "cuda-qx/core/tensor.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "cudaq/qec/realtime/decoding_config.h"
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
struct type_caster<cudaqx::heterogeneous_map> {
  PYBIND11_TYPE_CASTER(cudaqx::heterogeneous_map, const_name("dict"));

  bool load(handle src, bool) {
    if (!src)
      return false;
    try {
      // If it's already a heterogeneous_map, use it directly
      if (py::isinstance<cudaqx::heterogeneous_map>(src)) {
        value = src.cast<cudaqx::heterogeneous_map>();
        return true;
      }
      // Otherwise convert from dict using hetMapFromKwargs which handles
      // nested lists, nested dicts, arrays, etc.
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
      } else if (auto *int_val = std::any_cast<uint8_t>(&val)) {
        result[key.c_str()] = *int_val;
      } else if (auto *double_val = std::any_cast<double>(&val)) {
        result[key.c_str()] = *double_val;
      } else if (auto *float_val = std::any_cast<float>(&val)) {
        result[key.c_str()] = *float_val;
      } else if (auto *str_val = std::any_cast<std::string>(&val)) {
        result[key.c_str()] = *str_val;
      } else if (auto *vec_vec_val =
                     std::any_cast<std::vector<std::vector<double>>>(&val)) {
        // Convert vector<vector<double>> to Python list of lists
        py::list outer_list;
        for (const auto &inner_vec : *vec_vec_val) {
          py::list inner_list;
          for (double v : inner_vec) {
            inner_list.append(v);
          }
          outer_list.append(inner_list);
        }
        result[key.c_str()] = outer_list;
      } else if (auto *vec_val = std::any_cast<std::vector<double>>(&val)) {
        result[key.c_str()] = py::array_t<double>(
            {vec_val->size()}, {sizeof(double)}, vec_val->data());
      } else if (auto *vec_int_val = std::any_cast<std::vector<int>>(&val)) {
        result[key.c_str()] = py::array_t<int>(
            {vec_int_val->size()}, {sizeof(int)}, vec_int_val->data());
      } else if (auto *hetMap =
                     std::any_cast<cudaqx::heterogeneous_map>(&val)) {
        // Recursively convert nested heterogeneous_map
        result[key.c_str()] = py::cast(*hetMap);
      } else if (auto *srelay_cfg = std::any_cast<
                     cudaq::qec::decoding::config::srelay_bp_config>(&val)) {
        // Convert srelay_bp_config to heterogeneous_map then recursively cast
        // to dict
        result[key.c_str()] = py::cast(srelay_cfg->to_heterogeneous_map());
      } else if (auto *nv_cfg = std::any_cast<
                     cudaq::qec::decoding::config::nv_qldpc_decoder_config>(
                     &val)) {
        result[key.c_str()] = py::cast(nv_cfg->to_heterogeneous_map());
      } else if (auto *multi_cfg = std::any_cast<
                     cudaq::qec::decoding::config::multi_error_lut_config>(
                     &val)) {
        result[key.c_str()] = py::cast(multi_cfg->to_heterogeneous_map());
      } else if (auto *single_cfg = std::any_cast<
                     cudaq::qec::decoding::config::single_error_lut_config>(
                     &val)) {
        result[key.c_str()] = py::cast(single_cfg->to_heterogeneous_map());
      } else if (auto *sw_cfg = std::any_cast<
                     cudaq::qec::decoding::config::sliding_window_config>(
                     &val)) {
        result[key.c_str()] = py::cast(sw_cfg->to_heterogeneous_map());
      } else {
        throw std::runtime_error("Failed to cast from heterogeneous_map to "
                                 "Python dict. Unsupported data type in the '" +
                                 key + "' field.");
      }
    }
    return result.release();
  }
};
} // namespace detail
} // namespace pybind11

namespace cudaq {
namespace python {
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
} // namespace python
} // namespace cudaq
