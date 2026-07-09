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

namespace cudaq::qec::detail_affinity {

/// RAII: set the calling thread's CUDA device, restore the previous device on
/// scope exit. No-op for target < 0. Lib-private and header-only so decoder
/// plugins built as separate .so files can reuse it (PR2 extends this header
/// with NUMA guards; the nv-qldpc follow-up mirrors its use).
///
/// This guard is for threads that do NOT follow the one-thread-owns-one-
/// decoder persistent pin (e.g. the fresh worker spawned by decode_async).
class CudaDeviceGuard {
public:
  explicit CudaDeviceGuard(int target) {
    if (target < 0)
      return;
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || target >= count)
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
