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

#if defined(__linux__)
#include <fstream>
#include <linux/mempolicy.h>
#include <sched.h>
#include <sstream>
#include <sys/syscall.h>
#include <unistd.h>
#else
#include <iostream>
#endif

namespace cudaq::qec {

namespace detail {

inline int read_pin_key(const cudaqx::heterogeneous_map &params,
                        const std::string &key) {
  if (!params.contains(key))
    return -1;
  // kwargs stores Python int as std::size_t; YAML stores std::optional<int>.
  // heterogeneous_map::get<int> resolves both via the related-types fallback.
  int value = params.get<int>(key);
  if (value < 0)
    throw std::runtime_error(key + " must be >= 0 (got " +
                             std::to_string(value) + ")");
  return value;
}

} // namespace detail

/// @brief GPU device id for a decoder. -1 = inherit current device.
inline int read_cuda_device_id(const cudaqx::heterogeneous_map &params) {
  return detail::read_pin_key(params, "cuda_device_id");
}

/// @brief NUMA node id for a decoder. -1 = no binding.
inline int read_numa_node_id(const cudaqx::heterogeneous_map &params) {
  return detail::read_pin_key(params, "numa_node_id");
}

/// @brief Bind the calling thread + its allocations to a NUMA node for the
/// lifetime of the object. node < 0 is a no-op. Linux only.
class ScopedNumaNode {
public:
  explicit ScopedNumaNode(int node) {
    if (node < 0)
      return;
#if defined(__linux__)
    CPU_ZERO(&previous_set_);
    if (sched_getaffinity(0, sizeof(cpu_set_t), &previous_set_) == 0)
      has_previous_affinity_ = true;

    cpu_set_t node_set;
    CPU_ZERO(&node_set);
    if (build_node_cpuset(node, node_set)) {
      sched_setaffinity(0, sizeof(cpu_set_t), &node_set);
      affinity_set_ = true;
    }

    if (node < static_cast<int>(sizeof(unsigned long) * 8)) {
      unsigned long nodemask = 1UL << node;
      syscall(SYS_set_mempolicy, MPOL_BIND, &nodemask,
              static_cast<unsigned long>(sizeof(nodemask) * 8));
      mempolicy_set_ = true;
    }
#else
    static bool warned = false;
    if (!warned) {
      std::cerr << "[cudaq-qec] numa_node_id ignored: NUMA binding is only "
                   "supported on Linux."
                << std::endl;
      warned = true;
    }
#endif
  }

  ~ScopedNumaNode() {
#if defined(__linux__)
    if (mempolicy_set_)
      syscall(SYS_set_mempolicy, MPOL_DEFAULT, nullptr, 0UL);
    if (affinity_set_ && has_previous_affinity_)
      sched_setaffinity(0, sizeof(cpu_set_t), &previous_set_);
#endif
  }

  ScopedNumaNode(const ScopedNumaNode &) = delete;
  ScopedNumaNode &operator=(const ScopedNumaNode &) = delete;

private:
#if defined(__linux__)
  // Parse /sys/devices/system/node/node<N>/cpulist (e.g. "0-7,16-23").
  static bool build_node_cpuset(int node, cpu_set_t &out) {
    std::ifstream f("/sys/devices/system/node/node" + std::to_string(node) +
                    "/cpulist");
    if (!f.is_open())
      return false;
    std::string list;
    std::getline(f, list);
    if (list.empty())
      return false;
    std::stringstream ss(list);
    std::string range;
    bool any = false;
    while (std::getline(ss, range, ',')) {
      auto dash = range.find('-');
      int lo = std::stoi(range.substr(0, dash));
      int hi = (dash == std::string::npos) ? lo
                                           : std::stoi(range.substr(dash + 1));
      for (int c = lo; c <= hi; ++c) {
        CPU_SET(c, &out);
        any = true;
      }
    }
    return any;
  }

  cpu_set_t previous_set_;
  bool has_previous_affinity_ = false;
  bool affinity_set_ = false;
  bool mempolicy_set_ = false;
#endif
};

} // namespace cudaq::qec
