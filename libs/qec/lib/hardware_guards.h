/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Lib-private (NOT installed): RAII guards that place the calling thread on a
// decoder's CUDA device / NUMA node and restore on scope exit. Shared by the
// base decoder entry points and the realtime session. Never include from a
// public header.
#pragma once

#include "hardware_affinity.h"
#include <cuda_runtime.h>

namespace cudaq::qec::detail_affinity {

// RAII: sets the calling thread's CUDA current device, restores on destruction.
// target < 0 = no-op. Throws std::runtime_error if target is out of range of
// the visible device count or cudaSetDevice() fails while switching to target.
// If cudaGetDevice() cannot read the current device, warns and skips the
// switch (the decode proceeds on the caller's current device).
struct CudaDeviceGuard {
  int prev_ = -1;
  bool active_ = false;

  explicit CudaDeviceGuard(int target) {
    if (target < 0)
      return;
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || target >= count)
      throw std::runtime_error(
          "cuda_device_id " + std::to_string(target) +
          " out of range (device_count=" + std::to_string(count) + ")");
    if (cudaGetDevice(&prev_) != cudaSuccess) {
      // Can't determine the current device; warn and skip the switch so the
      // decode proceeds on whatever device the caller is already on.
      cudaq::qec::detail_affinity::affinity_warn(
          "CudaDeviceGuard: cudaGetDevice failed for cuda_device_id " +
          std::to_string(target) + "; skipping device switch");
      return;
    }
    if (prev_ != target) {
      cudaError_t e = cudaSetDevice(target);
      if (e != cudaSuccess)
        throw std::runtime_error("CudaDeviceGuard: cudaSetDevice(" +
                                 std::to_string(target) +
                                 ") failed: " + cudaGetErrorString(e));
      active_ = true;
    }
  }
  ~CudaDeviceGuard() {
    if (!active_)
      return;
    if (cudaSetDevice(prev_) != cudaSuccess)
      cudaq::qec::detail_affinity::affinity_warn(
          "CudaDeviceGuard: failed to restore prior CUDA device " +
          std::to_string(prev_) + "; thread may remain on wrong device");
  }
  CudaDeviceGuard(const CudaDeviceGuard &) = delete;
  CudaDeviceGuard &operator=(const CudaDeviceGuard &) = delete;
};

#if defined(__linux__)
// RAII: binds the calling thread to a NUMA node and restores on destruction.
// node < 0 = no-op. Uses the shared persistent-bind primitive for the set half
// and remembers prior affinity for the restore.
struct NumaGuard {
  bool affinity_set_ = false;
  bool has_prev_affinity_ = false;
  bool mempol_set_ = false;
  cpu_set_t prev_set_{};
  cudaq::qec::detail_affinity::mempolicy_state prev_mempolicy_;

  explicit NumaGuard(int node, cudaq::qec::mempolicy_mode mode =
                                   cudaq::qec::mempolicy_mode::preferred) {
    if (node < 0)
      return;
    CPU_ZERO(&prev_set_);
    has_prev_affinity_ =
        (sched_getaffinity(0, sizeof(prev_set_), &prev_set_) == 0);
    if (node < static_cast<int>(sizeof(unsigned long) * 8)) {
      prev_mempolicy_ = cudaq::qec::detail_affinity::capture_thread_mempolicy();
      mempol_set_ = (prev_mempolicy_.mode >= 0);
      if (!mempol_set_)
        cudaq::qec::detail_affinity::affinity_warn(
            "NumaGuard: prior thread mempolicy unreadable (get_mempolicy "
            "failed); temporary NUMA memory policy skipped for this decode");
    }
    // A temporary guard must never apply a policy it cannot restore: skip the
    // mempolicy half when the capture failed (affinity is still applied, its
    // restore path is independent).
    cudaq::qec::detail_affinity::bind_this_thread_to_numa_node(
        node, mode, /*apply_mempolicy=*/mempol_set_);
    // Arm the affinity restore only after a successful bind; if bind throws the
    // affinity was not changed and there is nothing to undo.
    affinity_set_ = has_prev_affinity_;
  }

  ~NumaGuard() {
    if (mempol_set_)
      cudaq::qec::detail_affinity::restore_thread_mempolicy(prev_mempolicy_);
    // has_prev_affinity_ is implied: it is set iff affinity_set_ is set.
    if (affinity_set_)
      if (sched_setaffinity(0, sizeof(prev_set_), &prev_set_) != 0)
        cudaq::qec::detail_affinity::affinity_warn(
            "NumaGuard restore: failed to reset thread affinity: " +
            std::string(std::strerror(errno)) + "; thread may remain pinned");
  }

  NumaGuard(const NumaGuard &) = delete;
  NumaGuard &operator=(const NumaGuard &) = delete;
};
#else
struct NumaGuard {
  explicit NumaGuard(int node, cudaq::qec::mempolicy_mode mode =
                                   cudaq::qec::mempolicy_mode::preferred) {
    (void)mode;
    if (node < 0)
      return;
    cudaq::qec::detail_affinity::warn_numa_unsupported_once();
  }
};
#endif

} // namespace cudaq::qec::detail_affinity
