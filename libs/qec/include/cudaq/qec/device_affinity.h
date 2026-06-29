/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cuda-qx/core/heterogeneous_map.h"
#include <stdexcept>
#include <string>

namespace cudaq::qec {

namespace detail {
/// Returns -1 if key is absent; throws if stored value is negative.
/// Handles both int (YAML path) and std::size_t (Python kwargs path) storage.
inline int read_pin_key(const cudaqx::heterogeneous_map &params,
                        const std::string &key) {
  if (!params.contains(key))
    return -1;
  int value = params.get<int>(key);
  if (value < 0)
    throw std::runtime_error(key + " must be >= 0 (got " +
                             std::to_string(value) + ")");
  return value;
}
} // namespace detail

/// @brief GPU device id for a decoder. -1 = inherit current device (no-op).
inline int read_cuda_device_id(const cudaqx::heterogeneous_map &params) {
  return detail::read_pin_key(params, "cuda_device_id");
}

/// @brief NUMA node id for a decoder. -1 = no binding.
inline int read_numa_node_id(const cudaqx::heterogeneous_map &params) {
  return detail::read_pin_key(params, "numa_node_id");
}

} // namespace cudaq::qec
