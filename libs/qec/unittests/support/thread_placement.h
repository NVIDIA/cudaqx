/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
// Test-only, header-only snapshot of the calling thread's hardware placement
// (CPU affinity + NUMA mempolicy mode + optional CUDA device) so tests can
// assert a guard restored everything exactly. Included via a relative path;
// not installed.
#ifndef CUDAQ_QEC_UNITTESTS_SUPPORT_THREAD_PLACEMENT_H
#define CUDAQ_QEC_UNITTESTS_SUPPORT_THREAD_PLACEMENT_H

#include <cuda_runtime.h>
#include <ostream>
#include <sched.h>
#include <string>
#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace cudaq::qec::test {

struct thread_placement {
  cpu_set_t affinity{};
  int mempolicy_mode = -1; // -1: query failed / unavailable (e.g. seccomp)
  int cuda_device = -1;    // -1: not captured (with_cuda=false) or no CUDA

  // NOTE: issues sched_getaffinity + SYS_get_mempolicy itself; call it outside
  // any shim reset/read window. cudaGetDevice only runs when with_cuda is set
  // so CPU-only tests never touch the CUDA runtime.
  static thread_placement capture(bool with_cuda = false) {
    thread_placement p;
    CPU_ZERO(&p.affinity);
    (void)sched_getaffinity(0, sizeof(p.affinity), &p.affinity);
#if defined(__linux__)
    if (syscall(SYS_get_mempolicy, &p.mempolicy_mode, nullptr, 0UL, nullptr,
                0UL) != 0)
      p.mempolicy_mode = -1;
#endif
    if (with_cuda && cudaGetDevice(&p.cuda_device) != cudaSuccess)
      p.cuda_device = -1;
    return p;
  }

  bool operator==(const thread_placement &o) const {
    return CPU_EQUAL(&affinity, &o.affinity) &&
           mempolicy_mode == o.mempolicy_mode && cuda_device == o.cuda_device;
  }
  bool operator!=(const thread_placement &o) const { return !(*this == o); }

  // Names which field differs (for assertion messages).
  std::string describe_difference(const thread_placement &o) const {
    std::string out;
    if (!CPU_EQUAL(&affinity, &o.affinity))
      out += "cpu affinity differs (" + std::to_string(CPU_COUNT(&affinity)) +
             " vs " + std::to_string(CPU_COUNT(&o.affinity)) + " cpus set); ";
    if (mempolicy_mode != o.mempolicy_mode)
      out += "mempolicy_mode differs (" + std::to_string(mempolicy_mode) +
             " vs " + std::to_string(o.mempolicy_mode) + "); ";
    if (cuda_device != o.cuda_device)
      out += "cuda_device differs (" + std::to_string(cuda_device) + " vs " +
             std::to_string(o.cuda_device) + "); ";
    return out.empty() ? "identical" : out;
  }
};

// gtest hook so failed EXPECT_EQ prints something readable.
inline void PrintTo(const thread_placement &p, std::ostream *os) {
  *os << "{cpus_set=" << CPU_COUNT(&p.affinity)
      << ", mempolicy_mode=" << p.mempolicy_mode
      << ", cuda_device=" << p.cuda_device << "}";
}

} // namespace cudaq::qec::test

#endif // CUDAQ_QEC_UNITTESTS_SUPPORT_THREAD_PLACEMENT_H
