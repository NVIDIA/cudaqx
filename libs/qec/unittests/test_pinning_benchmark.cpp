/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "support/thread_placement.h"
#include "cudaq/qec/decoder.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <gtest/gtest.h>
#include <thread>
#include <vector>
#if defined(__linux__)
#include <linux/mempolicy.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

// Resolved from the LD_PRELOAD shim when preloaded; weak so the binary still
// links (and the test cleanly skips) when it is not.
extern "C" __attribute__((weak)) long cudaqx_affinity_syscall_count();
extern "C" __attribute__((weak)) void cudaqx_affinity_syscall_reset();
// Per-syscall counters (same shim).
extern "C" __attribute__((weak)) long cudaqx_sched_setaffinity_count();
extern "C" __attribute__((weak)) long cudaqx_sched_getaffinity_count();
extern "C" __attribute__((weak)) long cudaqx_set_mempolicy_count();
extern "C" __attribute__((weak)) long cudaqx_get_mempolicy_count();

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

// ---- invariant-test helpers (shim counters + placement snapshots) ----------
using cudaq::qec::test::thread_placement;

bool shimCountersPresent() {
  return cudaqx_affinity_syscall_reset && cudaqx_sched_setaffinity_count &&
         cudaqx_sched_getaffinity_count && cudaqx_set_mempolicy_count &&
         cudaqx_get_mempolicy_count;
}
struct ShimCounts {
  long setaff, getaff, setmem, getmem;
};
ShimCounts readShimCounts() {
  return {cudaqx_sched_setaffinity_count(), cudaqx_sched_getaffinity_count(),
          cudaqx_set_mempolicy_count(), cudaqx_get_mempolicy_count()};
}

// Pinned decoder with D/O set so ALL entry points (incl. enqueue_syndrome)
// are callable.
std::unique_ptr<cudaq::qec::decoder> makeEntryReadyPinnedLut() {
  auto d = makePinnedLut();
  d->set_D_sparse(std::vector<std::vector<uint32_t>>{{0}, {1}});
  d->set_O_sparse(std::vector<std::vector<uint32_t>>{{0, 1}});
  return d;
}

// Every guarded entry point whose CUDA/NUMA guard runs on the CALLING thread
// (decode_async runs its guard on a worker thread; tested separately).
struct GuardedEntryPoint {
  const char *name;
  void (*call)(cudaq::qec::decoder &);
};
const GuardedEntryPoint kCallerGuardedEntryPoints[] = {
    {"decode_batch", [](cudaq::qec::decoder &d) { d.decode_batch(kChunk); }},
    {"decode_tensor",
     [](cudaq::qec::decoder &d) {
       cudaqx::tensor<uint8_t> syndrome({2}); // zero-initialized, rank-1
       d.decode(syndrome);
     }},
    {"enqueue_syndrome",
     [](cudaq::qec::decoder &d) {
       std::vector<uint8_t> syndrome = {1, 1};
       d.enqueue_syndrome(syndrome);
     }},
};
void callDecodeAsync(cudaq::qec::decoder &d) {
  d.decode_async(kChunk[0]).get();
}
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

TEST(PinningBenchmark, BindOnOneThreadStillGuardsAnotherThread) {
  if (!cudaqx_affinity_syscall_count)
    GTEST_SKIP() << "affinity-counter shim not preloaded";
  auto d = makePinnedLut();
  d->bind_current_thread(); // bind on the main test thread

  long from_other_thread = -1;
  std::thread other([&] {
    cudaqx_affinity_syscall_reset();
    d->decode_batch(kChunk);
    from_other_thread = cudaqx_affinity_syscall_count();
  });
  other.join();
  EXPECT_GT(from_other_thread, 0)
      << "a thread that never called bind_current_thread() must still be "
         "guarded, even though a DIFFERENT thread bound this decoder";
}

// Guards the decode_on_pinned_thread() contract: its one-shot worker thread
// binds itself for the duration of a single decode, but that binding must
// not stick to the decoder object once the worker exits. A subsequent
// decode_batch() from the (unbound) main thread must still be guarded.
TEST(PinningBenchmark, OneShotPinnedDecodeDoesNotStickToObject) {
  if (!cudaqx_affinity_syscall_count)
    GTEST_SKIP() << "affinity-counter shim not preloaded";
  auto d = makePinnedLut();
  d->decode_on_pinned_thread(kChunk[0]);

  cudaqx_affinity_syscall_reset();
  d->decode_batch(kChunk);
  EXPECT_GT(cudaqx_affinity_syscall_count(), 0)
      << "decode_on_pinned_thread()'s one-shot worker binding must not "
         "outlive it; a later decode_batch() from the main thread must "
         "still issue affinity syscalls";
}

TEST(PinningBenchmark, EnqueueSyndromeAppliesGuardOnUnboundThread) {
  if (!cudaqx_affinity_syscall_count)
    GTEST_SKIP() << "affinity-counter shim not preloaded";
  auto d = makePinnedLut();
  d->set_D_sparse(std::vector<std::vector<uint32_t>>{{0}, {1}});
  d->set_O_sparse(std::vector<std::vector<uint32_t>>{{0, 1}});
  cudaqx_affinity_syscall_reset();
  std::vector<uint8_t> syndrome = {1, 1};
  d->enqueue_syndrome(syndrome);
  EXPECT_GT(cudaqx_affinity_syscall_count(), 0)
      << "enqueue_syndrome's internal decode() call must apply the NUMA "
         "guard on an unbound thread, same as decode_batch()";
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

TEST(PinningBenchmark, DecodeBatchRestoresExactPriorMempolicy) {
#if defined(__linux__)
  int mode = -1;
  if (syscall(SYS_get_mempolicy, &mode, nullptr, 0UL, nullptr, 0UL) != 0)
    GTEST_SKIP() << "mempolicy syscalls unavailable (container seccomp?)";
  unsigned long nodemask = 1UL; // node 0
  if (syscall(SYS_set_mempolicy, MPOL_BIND, &nodemask, sizeof(nodemask) * 8) !=
      0)
    GTEST_SKIP() << "set_mempolicy unavailable (container seccomp?)";
  auto d = makePinnedLut(); // numa_node_id=0, unbound (per-call guard runs)
  d->decode_batch(kChunk);
  int mode_after = -1;
  syscall(SYS_get_mempolicy, &mode_after, nullptr, 0UL, nullptr, 0UL);
  syscall(SYS_set_mempolicy, MPOL_DEFAULT, nullptr, 0UL); // cleanup regardless
  EXPECT_EQ(mode_after, MPOL_BIND)
      << "decode_batch's guard must restore the exact prior mempolicy "
         "(MPOL_BIND), not force MPOL_DEFAULT";
#else
  GTEST_SKIP() << "Linux-only";
#endif
}

// The un-restorable-policy scenario: a seccomp profile that blocks
// get_mempolicy while allowing set_mempolicy. A temporary guard that cannot
// capture the prior policy must not change it at all — it could never restore
// it, so the change would silently outlive the decode. The shim's
// CUDAQX_SHIM_FAIL_GET_MEMPOLICY injection simulates the blocked capture.
TEST(PinningInvariants, GuardSkipsMempolicyWhenCaptureFails) {
  if (!shimCountersPresent())
    GTEST_SKIP() << "affinity-counter shim not preloaded";
#if defined(__linux__)
  // Baseline read runs with injection off.
  int before = -1;
  if (syscall(SYS_get_mempolicy, &before, nullptr, 0UL, nullptr, 0UL) != 0)
    GTEST_SKIP() << "mempolicy syscalls unavailable (container seccomp?)";

  auto d = makePinnedLut(); // numa_node_id=0, unbound -> guard runs per call
  setenv("CUDAQX_SHIM_FAIL_GET_MEMPOLICY", "1", 1);
  d->decode_batch(kChunk);
  unsetenv("CUDAQX_SHIM_FAIL_GET_MEMPOLICY");

  int after = -1;
  ASSERT_EQ(syscall(SYS_get_mempolicy, &after, nullptr, 0UL, nullptr, 0UL), 0);
  EXPECT_EQ(after, before)
      << "the guard applied a mempolicy it could not capture; the policy "
         "leaked past the decode (capture-failure must skip set_mempolicy)";
#else
  GTEST_SKIP() << "Linux-only";
#endif
}

// Bound-path invariant, per entry point: after bind_current_thread() each
// caller-thread entry point must issue ZERO placement syscalls (all four
// counters) and leave the calling thread's placement bit-identical.
TEST(PinningInvariants, BoundEntryPointsIssueNoPlacementSyscalls) {
  if (!shimCountersPresent())
    GTEST_SKIP() << "affinity-counter shim not preloaded";
  for (const auto &ep : kCallerGuardedEntryPoints) {
    SCOPED_TRACE(ep.name);
    auto d = makeEntryReadyPinnedLut();
    d->bind_current_thread();
    auto before = thread_placement::capture();
    cudaqx_affinity_syscall_reset();
    ep.call(*d);
    auto c = readShimCounts();
    EXPECT_EQ(c.setaff, 0) << "bound path issued sched_setaffinity";
    EXPECT_EQ(c.getaff, 0) << "bound path issued sched_getaffinity";
    EXPECT_EQ(c.setmem, 0) << "bound path issued set_mempolicy";
    EXPECT_EQ(c.getmem, 0) << "bound path issued get_mempolicy";
    auto after = thread_placement::capture();
    EXPECT_EQ(before, after) << "bound path changed placement: "
                             << before.describe_difference(after);
  }
}

// decode_async decodes on a fresh std::async worker which can never be the
// bound thread, so its guard runs on the WORKER by design (asserted >0 here:
// deleting that guard fails this test). The caller-side invariant is that the
// calling thread's placement is untouched even though it holds the binding.
TEST(PinningInvariants, BoundDecodeAsyncGuardsWorkerAndLeavesCallerPlacement) {
  if (!shimCountersPresent())
    GTEST_SKIP() << "affinity-counter shim not preloaded";
  auto d = makeEntryReadyPinnedLut();
  d->bind_current_thread();
  auto before = thread_placement::capture();
  cudaqx_affinity_syscall_reset();
  callDecodeAsync(*d);
  EXPECT_GT(readShimCounts().setmem, 0)
      << "decode_async's worker-thread guard must still attempt set_mempolicy "
         "(the caller's binding must not leak to the worker)";
  auto after = thread_placement::capture();
  EXPECT_EQ(before, after) << "decode_async changed the CALLING thread's "
                              "placement: "
                           << before.describe_difference(after);
}

// Unbound-path invariant, per entry point (incl. decode_async): the guard must
// actually run (set_mempolicy attempted -- deleting the guard fails this) AND
// fully restore the calling thread's placement afterwards.
TEST(PinningInvariants, UnboundEntryPointsApplyAndFullyRestoreGuard) {
  if (!shimCountersPresent())
    GTEST_SKIP() << "affinity-counter shim not preloaded";
  std::vector<GuardedEntryPoint> entries(std::begin(kCallerGuardedEntryPoints),
                                         std::end(kCallerGuardedEntryPoints));
  entries.push_back({"decode_async", callDecodeAsync});
  for (const auto &ep : entries) {
    SCOPED_TRACE(ep.name);
    auto d = makeEntryReadyPinnedLut(); // numa_node_id=0, never bound
    auto before = thread_placement::capture();
    cudaqx_affinity_syscall_reset();
    ep.call(*d);
    EXPECT_GT(readShimCounts().setmem, 0)
        << "unbound path must attempt set_mempolicy (guard deleted?)";
    auto after = thread_placement::capture();
    EXPECT_EQ(before, after) << "guard applied but not fully restored: "
                             << before.describe_difference(after);
  }
}
