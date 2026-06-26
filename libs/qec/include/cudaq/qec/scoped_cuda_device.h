/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace cudaq::qec {

/// @brief RAII guard that sets the calling thread's CUDA current device and
/// restores the previous device on scope exit. Decoder implementations that
/// allocate GPU resources or launch GPU work should instantiate this guard in
/// their constructor and in any decode path that uses the GPU.
///
/// target < 0 = inherit the current device (no-op).
/// Throws std::runtime_error if target is out of range.
class ScopedCudaDevice {
public:
  explicit ScopedCudaDevice(int target) {
    if (target < 0)
      return;
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || target >= count)
      throw std::runtime_error(
          "cuda_device_id " + std::to_string(target) +
          " out of range (device_count=" + std::to_string(count) + ")");
    if (cudaGetDevice(&previous_) != cudaSuccess)
      throw std::runtime_error("cudaGetDevice failed before device switch");
    // No switch needed when already on the target device; nothing to restore.
    if (previous_ != target) {
      if (cudaSetDevice(target) != cudaSuccess)
        throw std::runtime_error("cudaSetDevice failed for device " +
                                 std::to_string(target));
      active_ = true;
    }
  }

  ~ScopedCudaDevice() {
    if (active_)
      cudaSetDevice(previous_);
  }

  ScopedCudaDevice(const ScopedCudaDevice &) = delete;
  ScopedCudaDevice &operator=(const ScopedCudaDevice &) = delete;

private:
  int previous_ = -1;
  bool active_ = false;
};

} // namespace cudaq::qec
