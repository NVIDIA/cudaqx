/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include <cstring>

#include "common/ObserveResult.h"
#include "cuda-qx/core/heterogeneous_map.h"
#include "cuda-qx/core/kwargs_utils.h"
#include "cuda-qx/core/tensor.h"
#include "cudaq/qec/realtime/decoding_config.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace nanobind {
namespace detail {

template <>
struct type_caster<cudaq::spin_op> {
  NB_TYPE_CASTER(cudaq::spin_op, const_name("SpinOperator"))

  bool from_python(handle src, uint8_t, cleanup_list *) noexcept {
    if (!src)
      return false;
    try {
      auto data = nb::cast<std::vector<double>>(src.attr("serialize")());
      value = cudaq::spin_op(data);
      return true;
    } catch (...) {
      return false;
    }
  }

  static handle from_cpp(cudaq::spin_op v, rv_policy, cleanup_list *) noexcept {
    try {
      nb::object tv_py = nb::module_::import_("cudaq").attr("SpinOperator")(
          v.get_data_representation());
      return tv_py.release();
    } catch (...) {
      return handle();
    }
  }
};

template <>
struct type_caster<cudaq::sample_result> {
  NB_TYPE_CASTER(cudaq::sample_result, const_name("SampleResult"))

  bool from_python(handle src, uint8_t, cleanup_list *) noexcept {
    if (!src)
      return false;
    try {
      auto data = nb::cast<std::vector<std::size_t>>(src.attr("serialize")());
      value = cudaq::sample_result();
      value.deserialize(data);
      return true;
    } catch (...) {
      return false;
    }
  }

  static handle from_cpp(cudaq::sample_result v, rv_policy,
                         cleanup_list *) noexcept {
    try {
      nb::object tv_py = nb::module_::import_("cudaq").attr("SampleResult")();
      tv_py.attr("deserialize")(v.serialize());
      return tv_py.release();
    } catch (...) {
      return handle();
    }
  }
};

template <>
struct type_caster<cudaq::observe_result> {
  NB_TYPE_CASTER(cudaq::observe_result, const_name("ObserveResult"))

  bool from_python(handle src, uint8_t, cleanup_list *) noexcept {
    if (!src)
      return false;
    try {
      auto e = nb::cast<double>(src.attr("expectation")());
      value = cudaq::observe_result(e, cudaq::spin_op());
      return true;
    } catch (...) {
      return false;
    }
  }

  static handle from_cpp(cudaq::observe_result v, rv_policy,
                         cleanup_list *) noexcept {
    try {
      nb::object tv_py = nb::module_::import_("cudaq").attr("ObserveResult")(
          v.expectation(), v.get_spin(), v.raw_data());
      return tv_py.release();
    } catch (...) {
      return handle();
    }
  }
};

template <>
struct type_caster<cudaqx::heterogeneous_map> {
  NB_TYPE_CASTER(cudaqx::heterogeneous_map, const_name("dict"))

  bool from_python(handle src, uint8_t, cleanup_list *) noexcept {
    if (!src)
      return false;
    try {
      if (nb::isinstance<nb::dict>(src)) {
        value = cudaqx::hetMapFromKwargs(nb::cast<nb::kwargs>(src));
        return true;
      }
      return false;
    } catch (...) {
      return false;
    }
  }

  static handle from_cpp(cudaqx::heterogeneous_map v, rv_policy,
                         cleanup_list *) noexcept {
    try {
      nb::dict result;
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
          nb::list outer_list;
          for (const auto &inner_vec : *vec_vec_val) {
            nb::list inner_list;
            for (double dv : inner_vec) {
              inner_list.append(dv);
            }
            outer_list.append(inner_list);
          }
          result[key.c_str()] = outer_list;
        } else if (auto *vec_val = std::any_cast<std::vector<double>>(&val)) {
          size_t n = vec_val->size();
          double *data_copy = new double[n];
          std::copy(vec_val->begin(), vec_val->end(), data_copy);
          size_t shape[] = {n};
          result[key.c_str()] = nb::ndarray<nb::numpy, double>(
              data_copy, 1, shape, nb::capsule(data_copy, [](void *p) noexcept {
                delete[] static_cast<double *>(p);
              }));
        } else if (auto *vec_int_val = std::any_cast<std::vector<int>>(&val)) {
          size_t n = vec_int_val->size();
          int *data_copy = new int[n];
          std::copy(vec_int_val->begin(), vec_int_val->end(), data_copy);
          size_t shape[] = {n};
          result[key.c_str()] = nb::ndarray<nb::numpy, int>(
              data_copy, 1, shape, nb::capsule(data_copy, [](void *p) noexcept {
                delete[] static_cast<int *>(p);
              }));
        } else if (auto *hetMap =
                       std::any_cast<cudaqx::heterogeneous_map>(&val)) {
          // Recursively convert nested heterogeneous_map
          result[key.c_str()] = nb::cast(*hetMap);
        } else if (auto *srelay_cfg = std::any_cast<
                       cudaq::qec::decoding::config::srelay_bp_config>(&val)) {
          result[key.c_str()] = nb::cast(srelay_cfg->to_heterogeneous_map());
        } else if (auto *nv_cfg = std::any_cast<
                       cudaq::qec::decoding::config::nv_qldpc_decoder_config>(
                       &val)) {
          result[key.c_str()] = nb::cast(nv_cfg->to_heterogeneous_map());
        } else if (auto *multi_cfg = std::any_cast<
                       cudaq::qec::decoding::config::multi_error_lut_config>(
                       &val)) {
          result[key.c_str()] = nb::cast(multi_cfg->to_heterogeneous_map());
        } else if (auto *single_cfg = std::any_cast<
                       cudaq::qec::decoding::config::single_error_lut_config>(
                       &val)) {
          result[key.c_str()] = nb::cast(single_cfg->to_heterogeneous_map());
        } else if (auto *sw_cfg = std::any_cast<
                       cudaq::qec::decoding::config::sliding_window_config>(
                       &val)) {
          result[key.c_str()] = nb::cast(sw_cfg->to_heterogeneous_map());
        } else {
          PyErr_SetString(PyExc_RuntimeError,
                          ("Failed to cast from heterogeneous_map to "
                           "Python dict. Unsupported data type in the '" +
                           key + "' field.")
                              .c_str());
          return handle();
        }
      }
      return result.release();
    } catch (const std::exception &e) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
      return handle();
    } catch (...) {
      PyErr_SetString(PyExc_RuntimeError,
                      "Unknown error in heterogeneous_map conversion");
      return handle();
    }
  }
};

} // namespace detail
} // namespace nanobind

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

  size_t arr_shape[] = {rows, cols};
  return nb::ndarray<nb::numpy, T>(data_copy, 2, arr_shape,
                                   nb::capsule(data_copy, [](void *p) noexcept {
                                     delete[] static_cast<T *>(p);
                                   }));
}

template <typename T>
auto copy1DCUDAQXTensorToPyArray(const cudaqx::tensor<T> &tensor) {
  auto shape = tensor.shape();
  auto rows = shape[0];
  size_t total_size = rows;

  // Allocate new memory and copy the data
  T *data_copy = new T[total_size];
  std::memcpy(data_copy, tensor.data(), total_size * sizeof(T));

  size_t arr_shape[] = {rows};
  return nb::ndarray<nb::numpy, T>(data_copy, 1, arr_shape,
                                   nb::capsule(data_copy, [](void *p) noexcept {
                                     delete[] static_cast<T *>(p);
                                   }));
}

} // namespace python
} // namespace cudaq
