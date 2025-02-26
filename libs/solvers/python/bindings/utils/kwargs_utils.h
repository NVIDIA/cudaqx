/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cuda-qx/core/heterogeneous_map.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace cudaqx {

/// @brief Return the value of given type corresponding to the provided
/// key string from the provided options `kwargs` `dict`. Return the `orVal`
/// if the key is not in the `dict`.
template <typename T>
T getValueOr(py::kwargs &options, const std::string &key, const T &orVal) {
  if (options.contains(key))
    for (auto item : options)
      if (item.first.cast<std::string>() == key)
        return item.second.cast<T>();

  return orVal;
}

inline heterogeneous_map hetMapFromKwargs(const py::kwargs &kwargs) {
  cudaqx::heterogeneous_map result;

  for (const auto &item : kwargs) {
    std::string key = py::cast<std::string>(item.first);
    auto value = item.second;

    if (py::isinstance<py::bool_>(value)) {
      result.insert(key, value.cast<bool>());
    } else if (py::isinstance<py::int_>(value)) {
      result.insert(key, value.cast<std::size_t>());
    } else if (py::isinstance<py::float_>(value)) {
      result.insert(key, value.cast<double>());
    } else if (py::isinstance<py::str>(value)) {
      result.insert(key, value.cast<std::string>());
    } else if (py::isinstance<py::array>(value)) {
      py::array np_array = value.cast<py::array>();
      py::buffer_info info = np_array.request();
      auto insert_vector = [&](auto type_tag) {
        using T = decltype(type_tag);
        std::vector<T> vec(static_cast<T *>(info.ptr),
                           static_cast<T *>(info.ptr) + info.size);
        result.insert(key, std::move(vec));
      };
      if (info.format == py::format_descriptor<double>::format()) {
        insert_vector(double{});
      } else if (info.format == py::format_descriptor<float>::format()) {
        insert_vector(float{});
      } else if (info.format == py::format_descriptor<int>::format()) {
        insert_vector(int{});
      } else if (info.format == py::format_descriptor<uint8_t>::format()) {
        insert_vector(uint8_t{});
      } else {
        throw std::runtime_error("Unsupported array data type in kwargs.");
      }
    } else {
      throw std::runtime_error(
          "Invalid python type for mapping kwargs to a heterogeneous_map.");
    }
  }

  return result;
}
} // namespace cudaqx
