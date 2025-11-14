/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cuda-qx/core/heterogeneous_map.h"
#include "cuda-qx/core/tensor.h"
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
    } else if (py::isinstance<py::dict>(value)) {
      // Recursively convert nested dictionary
      result.insert(key, hetMapFromKwargs(value.cast<py::dict>()));
    } else if (py::isinstance<py::list>(value)) {
      // Handle Python lists
      py::list py_list = value.cast<py::list>();
      if (py_list.size() > 0) {
        // Check if it's a nested list (list of lists)
        if (py::isinstance<py::list>(py_list[0])) {
          std::vector<std::vector<double>> vec_vec;
          for (const auto &item : py_list) {
            py::list inner_list = item.cast<py::list>();
            std::vector<double> inner_vec;
            for (const auto &v : inner_list) {
              inner_vec.push_back(v.cast<double>());
            }
            vec_vec.push_back(inner_vec);
          }
          result.insert(key, std::move(vec_vec));
        } else {
          // Single-level list - try to convert to vector<double>
          std::vector<double> vec;
          for (const auto &item : py_list) {
            vec.push_back(item.cast<double>());
          }
          result.insert(key, std::move(vec));
        }
      }
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

template <typename T>
tensor<T> toTensor(const py::array_t<T> &H, bool perform_pcm_checks = false) {
  py::buffer_info buf = H.request();

  if (buf.ndim >= 1 && buf.strides[0] == buf.itemsize) {
    throw std::runtime_error("toTensor: data must be in row-major order, but "
                             "column-major order was detected.");
  }

  if (perform_pcm_checks) {
    if (buf.itemsize != sizeof(uint8_t)) {
      throw std::runtime_error(
          "Parity check matrix must be an array of uint8_t.");
    }

    if (buf.ndim != 2) {
      throw std::runtime_error("Parity check matrix must be 2-dimensional.");
    }
  }

  // Create a vector of the array dimensions
  std::vector<std::size_t> shape;
  for (py::ssize_t d : buf.shape) {
    shape.push_back(static_cast<std::size_t>(d));
  }

  // Create a tensor and borrow the NumPy array data
  cudaqx::tensor<T> tensor_H(shape);
  tensor_H.borrow(static_cast<T *>(buf.ptr), shape);
  return tensor_H;
}

/// @brief Convert a py::array_t<uint8_t> to a tensor<uint8_t>. This is the same
/// as toTensor, but with additional checks.
template <typename T>
tensor<T> pcmToTensor(const py::array_t<T> &H) {
  return toTensor(H, /*perform_pcm_checks=*/true);
}

} // namespace cudaqx
