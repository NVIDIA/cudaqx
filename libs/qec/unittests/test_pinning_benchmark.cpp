/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq/qec/decoder.h"
#include <chrono>
#include <cstdio>
#include <gtest/gtest.h>
#include <vector>

// Resolved from the LD_PRELOAD shim when preloaded; weak so the binary still
// links (and the test cleanly skips) when it is not.
extern "C" __attribute__((weak)) long cudaqx_affinity_syscall_count();
extern "C" __attribute__((weak)) void cudaqx_affinity_syscall_reset();

namespace {
cudaqx::tensor<uint8_t> makeH() {
  cudaqx::tensor<uint8_t> H({2, 3});
  return H;
}
// One decoder with numa_node_id pinned so the decode-time guard's syscalls fire
// on the unbound path.
std::unique_ptr<cudaq::qec::decoder> makePinnedLut() {
  cudaqx::heterogeneous_map o;
  o.insert("numa_node_id", 0);
  return cudaq::qec::decoder::get("multi_error_lut", makeH(), o);
}
const std::vector<std::vector<cudaq::qec::float_t>> kChunk = {{0.1, 0.1},
                                                              {0.2, 0.2}};
} // namespace

// Deterministic gate: after binding, a decode loop must issue ZERO
// sched_setaffinity calls; the unbound path (canary) must issue > 0 (proving
// the counter is wired and the guard really fires when not bound).
TEST(PinningBenchmark, BoundDecodeLoopIssuesNoAffinitySyscalls) {
  if (!cudaqx_affinity_syscall_count)
    GTEST_SKIP() << "affinity-counter shim not preloaded";
  auto d = makePinnedLut();
  constexpr int N = 200;

  // Canary: unbound -> per-call guard fires.
  cudaqx_affinity_syscall_reset();
  for (int i = 0; i < N; ++i)
    d->decode_batch(kChunk);
  long unbound = cudaqx_affinity_syscall_count();
  EXPECT_GT(unbound, 0) << "unbound decode loop should issue affinity syscalls "
                           "(else the gate is vacuous)";

  // Bound: bind once, then the loop must add none.
  d->bind_current_thread();
  cudaqx_affinity_syscall_reset();
  for (int i = 0; i < N; ++i)
    d->decode_batch(kChunk);
  long bound = cudaqx_affinity_syscall_count();
  EXPECT_EQ(bound, 0)
      << "bound decode loop must not re-issue affinity syscalls";
}

// Loose A/B (report + gross-regression guard only; timing is noisy).
TEST(PinningBenchmark, BoundThroughputNotWorseThanUnbound) {
  auto d = makePinnedLut();
  constexpr int N = 5000;
  auto run = [&] {
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i)
      d->decode_batch(kChunk);
    return std::chrono::duration<double, std::micro>(
               std::chrono::steady_clock::now() - t0)
        .count();
  };
  run(); // warm up
  double unbound_us = run();
  d->bind_current_thread();
  run(); // warm up bound
  double bound_us = run();
  std::printf(
      "[pinning A/B] unbound=%.1fus bound=%.1fus (%d decodes) ratio=%.2f\n",
      unbound_us, bound_us, N, bound_us / unbound_us);
  // Pinning removes per-call guard work; it must not be materially slower.
  EXPECT_LT(bound_us, unbound_us * 1.5);
}
