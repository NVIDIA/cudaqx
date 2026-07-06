/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "support/thread_placement.h"
#include "cudaq/qec/decoder.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <gtest/gtest.h>
#include <string>
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
extern "C" __attribute__((weak)) long cudaqx_mbind_count();
extern "C" __attribute__((weak)) long cudaqx_node_sysfs_open_count();

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

#if defined(__linux__)
// Container seccomp commonly blocks the mempolicy syscalls. When
// SYS_get_mempolicy is blocked, NumaGuard cannot capture the prior policy and
// correctly skips set_mempolicy entirely, so tests asserting setmem > 0 must
// skip, not fail (mirrors test_device_affinity.cpp). Read-only: no side
// effects on the thread's policy.
bool get_mempolicy_usable() {
  int mode = -1;
  return syscall(SYS_get_mempolicy, &mode, nullptr, 0UL, nullptr, 0UL) == 0;
}
#else
bool get_mempolicy_usable() { return false; }
#endif

// bind_current_thread() is persistent (no restore). Run test bodies that bind
// on a disposable worker so the gtest main thread's placement stays intact
// for later tests. EXPECT_* record correctly from any thread.
template <typename F>
void run_on_worker(F &&fn) {
  std::exception_ptr err;
  std::thread t([&] {
    try {
      fn();
    } catch (...) {
      err = std::current_exception();
    }
  });
  t.join();
  if (err)
    std::rethrow_exception(err);
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
  run_on_worker(
      [] {
        auto d = makePinnedLut();
        constexpr int N = 200;

        // Canary: unbound -> per-call guard fires.
        cudaqx_affinity_syscall_reset();
        for (int i = 0; i < N; ++i)
          d->decode_batch(kChunk);
        long unbound = cudaqx_affinity_syscall_count();
        EXPECT_GT(unbound, 0)
            << "unbound decode loop should issue affinity syscalls "
               "(else the gate is vacuous)";

        // Bound: bind once, then the loop must add none.
        d->bind_current_thread();
        cudaqx_affinity_syscall_reset();
        for (int i = 0; i < N; ++i)
          d->decode_batch(kChunk);
        long bound = cudaqx_affinity_syscall_count();
        EXPECT_EQ(bound, 0)
            << "bound decode loop must not re-issue affinity syscalls";
      });
}

TEST(PinningBenchmark, BindOnOneThreadStillGuardsAnotherThread) {
  if (!cudaqx_affinity_syscall_count)
    GTEST_SKIP() << "affinity-counter shim not preloaded";
  run_on_worker([] {
    auto d = makePinnedLut();
    d->bind_current_thread(); // bind on the binding worker thread

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
  });
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

// Verify that decode_on_pinned_thread() does not clobber a concurrent
// bind_current_thread() that runs from a third thread.
TEST(PinningBenchmark, OneShotDoesNotClobberConcurrentBind) {
  if (!cudaqx_affinity_syscall_count)
    GTEST_SKIP() << "affinity-counter shim not preloaded";
  auto d = makePinnedLut();

  // Thread B will call bind_current_thread() while the one-shot worker runs.
  std::atomic<bool> b_bound{false};
  std::thread thread_b([&] {
    // Spin until main_worker signals us to bind.
    while (!b_bound.load(std::memory_order_acquire))
      std::this_thread::yield();
    d->bind_current_thread();
  });

  // Run decode_on_pinned_thread from the main thread; signal B to bind midway.
  std::thread main_worker([&] {
    // Signal B just before decode so the race window is as wide as possible.
    b_bound.store(true, std::memory_order_release);
    d->decode_on_pinned_thread(kChunk[0]);
  });

  main_worker.join();
  thread_b.join();

  // After both threads finish, whoever bound last should have their binding
  // intact. At a minimum, is_bound_here() from thread_b must not permanently
  // suppress guards: a subsequent decode from an unbound thread must still
  // issue syscalls.
  cudaqx_affinity_syscall_reset();
  // Run decode from a fresh thread that was never bound.
  std::thread unbound([&] { d->decode_batch(kChunk); });
  unbound.join();
  // An unbound thread must always fire the guard.
  EXPECT_GT(cudaqx_affinity_syscall_count(), 0)
      << "guard must still fire for unbound threads after concurrent-bind race";
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
  run_on_worker([] {
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
  });
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
  run_on_worker([] {
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
  });
}

// decode_async decodes on a fresh std::async worker which can never be the
// bound thread, so its guard runs on the WORKER by design (asserted >0 here:
// deleting that guard fails this test). The caller-side invariant is that the
// calling thread's placement is untouched even though it holds the binding.
TEST(PinningInvariants, BoundDecodeAsyncGuardsWorkerAndLeavesCallerPlacement) {
  if (!shimCountersPresent())
    GTEST_SKIP() << "affinity-counter shim not preloaded";
  if (!get_mempolicy_usable())
    GTEST_SKIP() << "get_mempolicy blocked (container seccomp?); the guard "
                    "correctly skips set_mempolicy then, so setmem > 0 "
                    "cannot hold";
  run_on_worker([] {
    auto d = makeEntryReadyPinnedLut();
    d->bind_current_thread();
    auto before = thread_placement::capture();
    cudaqx_affinity_syscall_reset();
    callDecodeAsync(*d);
    EXPECT_GT(readShimCounts().setmem, 0)
        << "decode_async's worker-thread guard must still attempt "
           "set_mempolicy (the caller's binding must not leak to the worker)";
    auto after = thread_placement::capture();
    EXPECT_EQ(before, after) << "decode_async changed the CALLING thread's "
                                "placement: "
                             << before.describe_difference(after);
  });
}

// Unbound-path invariant, per entry point (incl. decode_async): the guard must
// actually run (set_mempolicy attempted -- deleting the guard fails this) AND
// fully restore the calling thread's placement afterwards.
TEST(PinningInvariants, UnboundEntryPointsApplyAndFullyRestoreGuard) {
  if (!shimCountersPresent())
    GTEST_SKIP() << "affinity-counter shim not preloaded";
  if (!get_mempolicy_usable())
    GTEST_SKIP() << "get_mempolicy blocked (container seccomp?); the guard "
                    "correctly skips set_mempolicy then, so setmem > 0 "
                    "cannot hold";
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

// The tensor-decode entry must construct its guards BEFORE building the
// soft-syndrome temporaries, so those allocations land on the decoder's node.
// Observable: with the shim, the FIRST placement syscall must occur before
// any decode work — assert the guard fires even for a rank-check failure,
// which returns before temporaries are built.
TEST(PinningInvariants, TensorDecodeGuardPrecedesTemporaries) {
  if (!shimCountersPresent())
    GTEST_SKIP() << "affinity-counter shim not preloaded";
  run_on_worker([] {
    auto d = makeEntryReadyPinnedLut();
    cudaqx::tensor<uint8_t> bad({2, 2}); // rank-2: must throw
    cudaqx_affinity_syscall_reset();
    EXPECT_THROW((void)d->decode(bad), std::runtime_error);
    EXPECT_GT(readShimCounts().getaff + readShimCounts().getmem, 0)
        << "guard must be constructed before input processing";
  });
}

// Post-construction sparse setters allocate session-lifetime buffers
// (corrections, msyn accumulators); those buffers must follow the decoder's
// placement like every other allocating entry point, and the guard must
// restore the calling thread's placement afterwards. Both overload shapes of
// each setter allocate independently, so each is exercised on its own.
TEST(PinningInvariants, SparseSettersApplyNumaGuardWhenUnbound) {
  if (!shimCountersPresent())
    GTEST_SKIP() << "affinity-counter shim not preloaded";
  struct SetterCall {
    const char *name;
    void (*call)(cudaq::qec::decoder &);
  };
  const SetterCall kSetters[] = {
      {"set_D_sparse(nested)",
       [](cudaq::qec::decoder &d) {
         d.set_D_sparse(std::vector<std::vector<uint32_t>>{{0}, {1}});
       }},
      {"set_D_sparse(flat)",
       [](cudaq::qec::decoder &d) {
         // -1 terminates a row: {{0}, {1}}.
         d.set_D_sparse(std::vector<int64_t>{0, -1, 1});
       }},
      {"set_O_sparse(nested)",
       [](cudaq::qec::decoder &d) {
         d.set_O_sparse(std::vector<std::vector<uint32_t>>{{0, 1}});
       }},
      {"set_O_sparse(flat)",
       [](cudaq::qec::decoder &d) {
         d.set_O_sparse(std::vector<int64_t>{0, 1});
       }},
  };
  run_on_worker([&] {
    auto d = makePinnedLut(); // numa_node_id=0, unbound -> guard must run
    for (const auto &s : kSetters) {
      SCOPED_TRACE(s.name);
      auto before = thread_placement::capture();
      cudaqx_affinity_syscall_reset();
      s.call(*d);
      auto c = readShimCounts();
      EXPECT_GT(c.setmem + c.setaff, 0)
          << "sparse setters must apply the placement guard";
      auto after = thread_placement::capture();
      EXPECT_EQ(before, after) << "setter guard must restore placement: "
                               << before.describe_difference(after);
    }
  });
}

// Non-virtual guarded single-syndrome entry (used by the Python binding —
// callers that have not bound a thread must self-place).
TEST(PinningInvariants, DecodeGuardedAppliesAndRestores) {
  if (!shimCountersPresent())
    GTEST_SKIP() << "affinity-counter shim not preloaded";
  run_on_worker([] {
    auto d = makePinnedLut();
    auto before = thread_placement::capture();
    cudaqx_affinity_syscall_reset();
    (void)d->decode_guarded({0.1f, 0.1f});
    EXPECT_GT(readShimCounts().setmem + readShimCounts().setaff, 0);
    EXPECT_EQ(before, thread_placement::capture());
    d->bind_current_thread();
    cudaqx_affinity_syscall_reset();
    (void)d->decode_guarded({0.1f, 0.1f});
    EXPECT_EQ(cudaqx_affinity_syscall_count(), 0)
        << "bound thread must skip decode_guarded's guard";
  });
}

// unbind_thread() must forget the registration so guards re-engage — the
// session calls it at teardown before the bound thread's id can be recycled.
TEST(PinningBenchmark, UnbindThreadRestoresGuarding) {
  if (!cudaqx_affinity_syscall_count)
    GTEST_SKIP() << "affinity-counter shim not preloaded";
  run_on_worker([] {
    auto d = makePinnedLut();
    d->bind_current_thread();
    cudaqx_affinity_syscall_reset();
    d->decode_batch(kChunk);
    EXPECT_EQ(cudaqx_affinity_syscall_count(), 0)
        << "bound thread must skip the guard";
    d->unbind_thread();
    cudaqx_affinity_syscall_reset();
    d->decode_batch(kChunk);
    EXPECT_GT(cudaqx_affinity_syscall_count(), 0)
        << "after unbind_thread() the guard must fire again";
  });
}

// ---- Fault-injection negative tests ------------------------------------
// Each test sets AND unsets its own CUDAQX_SHIM_FAIL_* variable so no
// injection state leaks across tests.

// T1.4: a container that blocks EVERY placement syscall (blanket injection).
// The decode must still be correct, and the degradation must be loud.
TEST(PinningNegative, FullyBlockedSyscallsDegradeLoudlyNotWrongly) {
  if (!shimCountersPresent())
    GTEST_SKIP() << "affinity-counter shim not preloaded";
  run_on_worker([] {
    setenv("CUDAQX_SHIM_FAIL_ALL_PLACEMENT", "1", 1);
    auto d = makeEntryReadyPinnedLut();
    testing::internal::CaptureStderr();
    auto results = d->decode_batch(kChunk);
    std::string err = testing::internal::GetCapturedStderr();
    unsetenv("CUDAQX_SHIM_FAIL_ALL_PLACEMENT");
    ASSERT_EQ(results.size(), kChunk.size());
    for (auto &r : results)
      EXPECT_TRUE(r.converged) << "degraded env must not corrupt decode";
    EXPECT_NE(err.find("[cudaq-qec affinity] WARNING"), std::string::npos)
        << "degradation must be loud";
  });
}

// T2.8: when the prior affinity cannot be read, the guard could never restore
// it, so it must not call sched_setaffinity at all (only-if-restorable rule).
TEST(PinningInvariants, BlockedGetaffinityAppliesNoAffinityAtAll) {
  if (!shimCountersPresent())
    GTEST_SKIP() << "affinity-counter shim not preloaded";
  run_on_worker([] {
    auto d = makePinnedLut();
    auto before = thread_placement::capture();
    setenv("CUDAQX_SHIM_FAIL_GETAFFINITY", "1", 1);
    cudaqx_affinity_syscall_reset();
    auto results = d->decode_batch(kChunk);
    unsetenv("CUDAQX_SHIM_FAIL_GETAFFINITY");
    EXPECT_EQ(readShimCounts().setaff, 0)
        << "unreadable prior affinity must mean NO sched_setaffinity";
    auto after = thread_placement::capture();
    EXPECT_EQ(before, after);
    ASSERT_EQ(results.size(), kChunk.size());
    EXPECT_TRUE(results[0].converged);
  });
}

// T2.9: sched_setaffinity blocked (locked cpuset). The independent mempolicy
// half must still run, the failure must be loud, and nothing may leak.
TEST(PinningInvariants, BlockedSetaffinityStillAppliesMempolicyHalf) {
  if (!shimCountersPresent())
    GTEST_SKIP() << "affinity-counter shim not preloaded";
  if (!get_mempolicy_usable())
    GTEST_SKIP() << "get_mempolicy blocked (container seccomp?); the guard "
                    "correctly skips set_mempolicy then, so setmem > 0 "
                    "cannot hold";
  run_on_worker([] {
    auto d = makePinnedLut();
    auto before = thread_placement::capture();
    setenv("CUDAQX_SHIM_FAIL_SETAFFINITY", "1", 1);
    cudaqx_affinity_syscall_reset();
    testing::internal::CaptureStderr();
    auto results = d->decode_batch(kChunk);
    std::string err = testing::internal::GetCapturedStderr();
    unsetenv("CUDAQX_SHIM_FAIL_SETAFFINITY");
    EXPECT_GT(readShimCounts().setmem, 0)
        << "mempolicy half must still run when sched_setaffinity is blocked";
    auto after = thread_placement::capture();
    EXPECT_EQ(before, after)
        << "blocked sched_setaffinity leaked a placement change: "
        << before.describe_difference(after);
    ASSERT_EQ(results.size(), kChunk.size());
    EXPECT_TRUE(results[0].converged);
    EXPECT_NE(err.find("sched_setaffinity"), std::string::npos)
        << "blocked sched_setaffinity must be loud; got: " << err;
  });
}

// T2.10: set_mempolicy blocked (no CAP_SYS_NICE). The independent affinity
// half must still run, the failure must be loud, and nothing may leak.
TEST(PinningInvariants, BlockedSetMempolicyStillAppliesAffinityHalf) {
  if (!shimCountersPresent())
    GTEST_SKIP() << "affinity-counter shim not preloaded";
  if (!get_mempolicy_usable())
    GTEST_SKIP() << "get_mempolicy blocked (container seccomp?); the guard "
                    "then skips the mempolicy half entirely and never prints "
                    "\"set_mempolicy failed\"";
  run_on_worker([] {
    auto d = makePinnedLut();
    auto before = thread_placement::capture();
    setenv("CUDAQX_SHIM_FAIL_SET_MEMPOLICY", "1", 1);
    cudaqx_affinity_syscall_reset();
    testing::internal::CaptureStderr();
    auto results = d->decode_batch(kChunk);
    std::string err = testing::internal::GetCapturedStderr();
    unsetenv("CUDAQX_SHIM_FAIL_SET_MEMPOLICY");
    EXPECT_GT(readShimCounts().setaff, 0)
        << "affinity half must still run when set_mempolicy is blocked";
    auto after = thread_placement::capture();
    EXPECT_EQ(before, after)
        << "blocked set_mempolicy leaked a placement change: "
        << before.describe_difference(after);
    ASSERT_EQ(results.size(), kChunk.size());
    EXPECT_TRUE(results[0].converged);
    EXPECT_NE(err.find("set_mempolicy failed"), std::string::npos)
        << "blocked set_mempolicy must be loud; got: " << err;
  });
}

// ---- Binding lifecycle negative tests
// -----------------------------------------------

// T3.13: re-binding on a second thread migrates the guard-skip: the new owner
// decodes guard-free, the previous owner is guarded again. promise/future
// handshakes keep both threads alive through every measured decode (no thread
// id can be recycled) and serialize the decodes (the shim counters are global).
TEST(PinningBenchmark, RebindMigratesGuardSkipToNewOwner) {
  if (!cudaqx_affinity_syscall_count)
    GTEST_SKIP() << "affinity-counter shim not preloaded";
  auto d = makePinnedLut();
  std::promise<void> a_bound_p, b_done_p;
  auto a_bound_f = a_bound_p.get_future();
  auto b_done_f = b_done_p.get_future();
  long a_first = -1, b_count = -1, a_second = -1;
  std::thread ta([&] {
    d->bind_current_thread();
    cudaqx_affinity_syscall_reset();
    d->decode_batch(kChunk);
    a_first = cudaqx_affinity_syscall_count();
    a_bound_p.set_value();
    b_done_f.wait(); // B has re-bound and decoded
    cudaqx_affinity_syscall_reset();
    d->decode_batch(kChunk);
    a_second = cudaqx_affinity_syscall_count();
  });
  std::thread tb([&] {
    a_bound_f.wait();         // A is the current owner
    d->bind_current_thread(); // rebind: ownership migrates to B
    cudaqx_affinity_syscall_reset();
    d->decode_batch(kChunk);
    b_count = cudaqx_affinity_syscall_count();
    b_done_p.set_value();
  });
  ta.join();
  tb.join();
  EXPECT_EQ(a_first, 0) << "owner A must decode guard-free before the rebind";
  EXPECT_EQ(b_count, 0) << "new owner B must decode guard-free after rebinding";
  EXPECT_GT(a_second, 0)
      << "previous owner A must be guarded again after B re-bound";
}

// T3.13b: binding twice on the same thread is idempotent -- no throw, and the
// thread still decodes guard-free.
TEST(PinningBenchmark, DoubleBindIsIdempotent) {
  if (!cudaqx_affinity_syscall_count)
    GTEST_SKIP() << "affinity-counter shim not preloaded";
  run_on_worker([] {
    auto d = makePinnedLut();
    d->bind_current_thread();
    EXPECT_NO_THROW(d->bind_current_thread());
    cudaqx_affinity_syscall_reset();
    d->decode_batch(kChunk);
    EXPECT_EQ(cudaqx_affinity_syscall_count(), 0);
  });
}

// T3.14: unbind_thread() called from a DIFFERENT thread clears the owner; the
// still-alive previously-bound thread must be guarded again. The worker stays
// alive across the unbind via a promise/future handshake (no sleeps).
TEST(PinningBenchmark, UnbindFromAnotherThreadRestoresGuarding) {
  if (!cudaqx_affinity_syscall_count)
    GTEST_SKIP() << "affinity-counter shim not preloaded";
  auto d = makePinnedLut();
  std::promise<void> bound_p, unbound_p;
  auto bound_f = bound_p.get_future();
  auto unbound_f = unbound_p.get_future();
  long count = -1;
  std::thread w([&] {
    d->bind_current_thread();
    bound_p.set_value();
    unbound_f.wait(); // main thread has called unbind_thread()
    cudaqx_affinity_syscall_reset();
    d->decode_batch(kChunk);
    count = cudaqx_affinity_syscall_count();
  });
  bound_f.wait();
  d->unbind_thread(); // from the main thread, not the owner
  unbound_p.set_value();
  w.join();
  EXPECT_GT(count, 0)
      << "after unbind_thread() from another thread, the previously bound "
         "thread must be guarded again";
}

// T3.16: two never-bound threads decode concurrently; each thread's placement
// must be fully restored and each decode correct. No shim-counter asserts:
// the counters are global and would race here.
TEST(PinningInvariants, ConcurrentUnboundDecodesRestoreBothThreads) {
  auto d = makePinnedLut();
  std::atomic<bool> go{false};
  auto body = [&] {
    while (!go.load(std::memory_order_acquire))
      std::this_thread::yield();
    auto before = thread_placement::capture();
    auto results = d->decode_batch(kChunk);
    auto after = thread_placement::capture();
    EXPECT_EQ(before, after) << "concurrent unbound decode leaked placement: "
                             << before.describe_difference(after);
    ASSERT_EQ(results.size(), kChunk.size());
    EXPECT_TRUE(results[0].converged);
  };
  std::thread t1(body), t2(body);
  go.store(true, std::memory_order_release);
  t1.join();
  t2.join();
}

// T3.17: N threads race bind_current_thread() on one decoder. The race itself
// must not throw, and afterwards a FRESH never-bound thread must still be
// guarded and decode correctly. The racers are held alive (promise/shared
// handshake) until the fresh thread finishes so its thread id cannot be a
// recycled racer id that would spuriously skip the guard. No attempt to
// identify the winner.
TEST(PinningBenchmark, BindRaceLeavesFreshThreadsGuarded) {
  if (!cudaqx_affinity_syscall_count)
    GTEST_SKIP() << "affinity-counter shim not preloaded";
  auto d = makePinnedLut();
  constexpr int kRacers = 4;
  std::atomic<bool> go{false};
  std::atomic<int> raced{0};
  std::promise<void> done_p;
  std::shared_future<void> done_f(done_p.get_future());
  std::vector<std::thread> racers;
  for (int i = 0; i < kRacers; ++i)
    racers.emplace_back([&] {
      while (!go.load(std::memory_order_acquire))
        std::this_thread::yield();
      EXPECT_NO_THROW(d->bind_current_thread());
      raced.fetch_add(1, std::memory_order_acq_rel);
      done_f.wait(); // stay alive: our thread id must not be recycled yet
    });
  go.store(true, std::memory_order_release);
  while (raced.load(std::memory_order_acquire) < kRacers)
    std::this_thread::yield();
  std::thread fresh([&] {
    cudaqx_affinity_syscall_reset();
    auto results = d->decode_batch(kChunk);
    EXPECT_GT(cudaqx_affinity_syscall_count(), 0)
        << "a fresh thread that never bound must be guarded after the race";
    ASSERT_EQ(results.size(), kChunk.size());
    EXPECT_TRUE(results[0].converged);
  });
  fresh.join();
  done_p.set_value();
  for (auto &t : racers)
    t.join();
}

// ---- syscall / sysfs budget gates -------------------------------------------

// Perf-regression gate: the per-call guard must stay within a fixed syscall
// budget. If you legitimately add a syscall, change the constant in the same
// PR and say why in the commit message.
TEST(PinningBudget, UnboundDecodeStaysWithinSyscallBudget) {
  if (!shimCountersPresent())
    GTEST_SKIP() << "affinity-counter shim not preloaded";
  run_on_worker([] {
    auto d = makePinnedLut();
    (void)d->decode_batch(kChunk); // warm one-time paths
    constexpr int kCalls = 10;
    cudaqx_affinity_syscall_reset();
    for (int i = 0; i < kCalls; ++i)
      (void)d->decode_batch(kChunk);
    auto c = readShimCounts();
    long total = c.setaff + c.getaff + c.setmem + c.getmem;
    std::printf(
        "[pinning budget] %ld placement syscalls / %d unbound decodes "
        "= %.1f per call (setaff=%ld getaff=%ld setmem=%ld getmem=%ld)\n",
        total, kCalls, static_cast<double>(total) / kCalls, c.setaff, c.getaff,
        c.setmem, c.getmem);
    // Measured on a mempolicy-capable host: 7 syscalls per unbound decode
    // (2 sched_getaffinity + 2 sched_setaffinity + 1 get_mempolicy +
    // 2 set_mempolicy). Budget = measured + one call of headroom.
    constexpr long kMeasuredPerCall = 7;
    EXPECT_LE(total, kMeasuredPerCall * kCalls + kMeasuredPerCall)
        << "guard syscall budget exceeded: " << total << " for " << kCalls
        << " calls";
    EXPECT_GT(total, 0);
  });
}

// Sysfs budget gate: each guarded (unbound) decode currently re-reads the
// node's cpulist -- 1 open of /sys/devices/system/node/... per call. Cpuset
// caching should reduce this to amortized 0 -- tighten when it lands.
TEST(PinningBudget, UnboundDecodeStaysWithinSysfsOpenBudget) {
  if (!shimCountersPresent() || !cudaqx_node_sysfs_open_count)
    GTEST_SKIP() << "affinity-counter shim (with sysfs open counter) not "
                    "preloaded";
  run_on_worker([] {
    auto d = makePinnedLut();
    (void)d->decode_batch(kChunk); // warm one-time paths
    constexpr int kCalls = 10;
    cudaqx_affinity_syscall_reset();
    for (int i = 0; i < kCalls; ++i)
      (void)d->decode_batch(kChunk);
    long opens = cudaqx_node_sysfs_open_count();
    std::printf("[pinning budget] %ld node-sysfs opens / %d unbound decodes "
                "= %.1f per call\n",
                opens, kCalls, static_cast<double>(opens) / kCalls);
    EXPECT_LE(opens, kCalls) << "node-sysfs open budget exceeded: " << opens
                             << " for " << kCalls << " calls";
    EXPECT_GT(opens, 0) << "expected the unbound guard to read the node "
                           "cpulist (counter wired?)";
  });
}
