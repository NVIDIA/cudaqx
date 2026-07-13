/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cuda_runtime_api.h>
#include <stdexcept>
#include <string>
#include <string_view>

namespace cudaq::qec::detail_affinity {

/// Persistently select \p target for the calling thread. The device id must
/// already have been validated by decoder::get(). This is the hot-path helper
/// for long-lived decoder workers and shared realtime dispatcher threads.
inline void select_cuda_device(int target, std::string_view context) {
  if (target < 0)
    return;

  int current = -1;
  cudaError_t err = cudaGetDevice(&current);
  if (err != cudaSuccess)
    throw std::runtime_error(
        std::string(context) +
        ": cudaGetDevice() failed: " + cudaGetErrorString(err));
  if (current == target)
    return;

  err = cudaSetDevice(target);
  if (err != cudaSuccess)
    throw std::runtime_error(std::string(context) + ": cudaSetDevice(" +
                             std::to_string(target) +
                             ") failed: " + cudaGetErrorString(err));
}

/// RAII: set the calling thread's CUDA device, restore the previous device on
/// scope exit. No-op for target < 0. Lib-private and header-only so it can be
/// reused by decoder implementation files without adding a public API.
///
/// This guard is for threads that do NOT follow the one-thread-owns-one-
/// decoder persistent pin (e.g. the fresh worker spawned by decode_async).
class CudaDeviceGuard {
public:
  explicit CudaDeviceGuard(int target) {
    if (target < 0)
      return;
    int count = 0;
    const cudaError_t count_status = cudaGetDeviceCount(&count);
    if (count_status != cudaSuccess)
      throw std::runtime_error(
          "CudaDeviceGuard: cudaGetDeviceCount() failed for cuda_device_id " +
          std::to_string(target) + ": " + cudaGetErrorString(count_status));
    if (target >= count)
      throw std::runtime_error("cuda_device_id " + std::to_string(target) +
                               " is out of range: " + std::to_string(count) +
                               " CUDA device(s) visible");
    // If the current device is unreadable, skip restoration rather than
    // restore to a guessed device; the set below still applies.
    if (cudaGetDevice(&prev_) != cudaSuccess)
      prev_ = -1;
    cudaError_t err = cudaSetDevice(target);
    if (err != cudaSuccess)
      throw std::runtime_error("CudaDeviceGuard: cudaSetDevice(" +
                               std::to_string(target) +
                               ") failed: " + cudaGetErrorString(err));
    restore_ = (prev_ >= 0 && prev_ != target);
  }
  ~CudaDeviceGuard() {
    if (restore_)
      (void)cudaSetDevice(prev_);
  }
  CudaDeviceGuard(const CudaDeviceGuard &) = delete;
  CudaDeviceGuard &operator=(const CudaDeviceGuard &) = delete;

private:
  int prev_ = -1;
  bool restore_ = false;
};

} // namespace cudaq::qec::detail_affinity
