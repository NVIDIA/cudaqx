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
#include <vector>

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

/// @brief Public NUMA memory-policy selector. preferred = soft; bind = strict.
enum class mempolicy_mode { preferred, bind };

/// @brief Read "mempolicy": "bind"->bind, "preferred"/absent->preferred, else
/// throw.
inline mempolicy_mode read_mempolicy(const cudaqx::heterogeneous_map &params) {
  if (!params.contains("mempolicy"))
    return mempolicy_mode::preferred;
  const std::string v = params.get<std::string>("mempolicy");
  if (v == "bind")
    return mempolicy_mode::bind;
  if (v == "preferred")
    return mempolicy_mode::preferred;
  throw std::runtime_error(
      "mempolicy must be \"preferred\" or \"bind\" (got \"" + v + "\")");
}

/// @brief Read "cpu_affinity": a list of CPU core ids. Absent -> empty (no
/// override).
inline std::vector<int>
read_cpu_affinity(const cudaqx::heterogeneous_map &params) {
  if (!params.contains("cpu_affinity"))
    return {};
  return params.get<std::vector<int>>("cpu_affinity");
}

/// @brief NUMA node local to a CUDA device (via PCIe locality), or -1 if the
/// device id is negative or the topology can't be resolved. Defined in
/// decoder.cpp.
int numa_node_for_cuda_device(int cuda_device_id);

} // namespace cudaq::qec
