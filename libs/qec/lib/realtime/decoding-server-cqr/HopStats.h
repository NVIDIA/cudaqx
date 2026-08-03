/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// Env-gated latency probes for the decoding-server HOST_CALL path.
///
/// A served RPC crosses three thread boundaries, each a condvar/promise
/// futex wakeup; these probes measure every hop of every request:
///
///   hop1 (dispatcher->recv):    inject() inbox push -> recv() wake
///   hop2 (recv->worker):        try_enqueue -> worker_loop wake
///   hop3 (worker->dispatcher):  send() done -> inject() fut.wait wake
///
/// plus the stage durations between hops and the producer-side notify cost.
/// handler_total (inject entry -> exit) brackets the whole HOST_CALL handler
/// as the CUDAQ dispatcher sees it.
///
/// Off by default; when disabled every probe is a single predicted branch.
/// Gates (each read once, on first use):
///   QEC_DECODING_SERVER_HOP_STATS      unset/"" = off; "total" = only the
///                                      handler-total stamps (probe-distortion
///                                      A/B); any other value = full probes.
///   QEC_DECODING_SERVER_HOP_STATS_CSV  per-sample CSV dump path, written at
///                                      report time.
///   QEC_PIN_DISPATCHER / QEC_PIN_RECV / QEC_PIN_WORKER
///                                      pin that thread to a cpu id. Intended
///                                      for single-decoder diagnosis runs:
///                                      with multiple decoders every worker
///                                      would share the one QEC_PIN_WORKER
///                                      core.  QEC_PIN_RECV only applies to
///                                      transports without a dispatch sink
///                                      (direct-dispatch transports run no
///                                      recv thread).
///
/// The spin-then-block wait policy (QEC_DECODING_SERVER_SPIN_US) is a
/// production knob owned by SpinPolicy.h, not a diagnostic; the report's
/// spin_us field echoes its effective budget.  Under direct dispatch
/// (ITransceiver::install_dispatch_sink) hop1 no longer exists: both of its
/// probe endpoints stamp back-to-back on the dispatcher thread, so hop1
/// reports as warm ~0 with cold count 0 and notify1 is absent.
///
/// Correlation: a global slot array keyed by request_id & 0xFFFF (in-flight
/// requests are bounded by the transport ring, so wraparound reuse is
/// sequential). Timestamps are relaxed atomics -- the mutex/promise that
/// already order each handoff also order the stamps; the atomics only keep
/// the cross-thread reads well-defined. A slot is armed by inject() and
/// consumed exactly once (worker_loop for fire-and-forget enqueues,
/// inject()'s tail for blocking RPCs); the rid+armed check drops stale or
/// foreign slots. Diagnostic-quality instrumentation: report/CSV formatting
/// happens only at shutdown, never on the hot path.

#include "SpinPolicy.h"
#include "cudaq/qec/realtime/decoder_rpc_wire_format.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <sched.h>
#include <vector>

namespace cudaq::qec::decoding_server::hopstats {

// ---------------------------------------------------------------------------
// Gates and small utilities
// ---------------------------------------------------------------------------

enum class Mode : int { off = 0, full = 1, total_only = 2 };

inline Mode mode() {
  static const Mode m = [] {
    const char *e = std::getenv("QEC_DECODING_SERVER_HOP_STATS");
    if (!e || !e[0])
      return Mode::off;
    if (std::strcmp(e, "total") == 0)
      return Mode::total_only;
    return Mode::full;
  }();
  return m;
}

inline bool enabled() { return mode() != Mode::off; }
inline bool full() { return mode() == Mode::full; }

inline uint64_t now_ns() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

// ---------------------------------------------------------------------------
// Per-request slot (written by 3 threads at disjoint phases; the handoff
// mutex/promise provides the ordering, relaxed atomics keep reads defined)
// ---------------------------------------------------------------------------

struct alignas(128) HopSlot {
  std::atomic<uint64_t> t_entry{0};       // inject() entry (dispatcher)
  std::atomic<uint64_t> t_inbox_push{0};  // hop1 start (dispatcher, under mtx_)
  std::atomic<uint64_t> t_notify1_ret{0}; // after cv_.notify_one (dispatcher)
  std::atomic<uint64_t> t_recv_wake{0};   // hop1 end (recv thread)
  std::atomic<uint64_t> t_queue_push{0};  // hop2 start (recv thread)
  std::atomic<uint64_t> t_notify2_ret{0}; // after queue_cv.notify_one (recv)
  std::atomic<uint64_t> t_worker_wake{0}; // hop2 end (worker)
  std::atomic<uint64_t> t_wait_begin{0};  // before fut.wait (dispatcher)
  std::atomic<uint64_t> t_send_done{0};   // hop3 start (worker)
  std::atomic<uint32_t> rid{0};
  std::atomic<uint32_t> function_id{0};
  std::atomic<uint8_t> armed{0};
  std::atomic<uint8_t> cold1{0}; // recv() inbox was empty -> real wakeup
  std::atomic<uint8_t> cold2{0}; // worker queue was empty -> real wakeup
};
static_assert(sizeof(HopSlot) == 128, "keep HopSlot one cache-line pair");

constexpr std::size_t kSlotCount = 65536; // 8 MB BSS, paged in only if used
inline HopSlot g_slots[kSlotCount];

inline HopSlot &slot_for(uint32_t rid) {
  return g_slots[rid & (kSlotCount - 1)];
}

inline bool armed_for(HopSlot &s, uint32_t rid) {
  return s.armed.load(std::memory_order_relaxed) != 0 &&
         s.rid.load(std::memory_order_relaxed) == rid;
}

// ---------------------------------------------------------------------------
// Completed samples (one POD per request, appended lock-free)
// ---------------------------------------------------------------------------

constexpr int32_t kMissing = INT32_MIN; // stamp absent (mode/kind/race)

struct Sample {
  uint8_t kind;  // 0 enqueue, 1 get_corrections, 2 reset, 3 other
  uint8_t cold1; // hop1 waiter actually slept/spun
  uint8_t cold2; // hop2 waiter actually slept/spun
  uint8_t cold3; // hop3: send() happened after fut.wait began
  uint32_t rid;
  int32_t hop1, hop2, hop3, total;
  int32_t stage_build, stage_dispatch, stage_worker;
  int32_t notify1, notify2;
};

constexpr std::size_t kMaxSamples = 262144; // ~12 MB; recording stops when full
inline Sample g_samples[kMaxSamples];
inline std::atomic<uint64_t> g_sample_count{0};

inline uint8_t kind_of(uint32_t function_id) {
  using namespace cudaq::qec::decoding::rpc;
  if (function_id == kEnqueueSyndromesFunctionId)
    return 0;
  if (function_id == kGetCorrectionsFunctionId)
    return 1;
  if (function_id == kResetDecoderFunctionId)
    return 2;
  return 3;
}

// b - a in ns, kMissing when either stamp is absent, clamped to int32.
inline int32_t delta_ns(uint64_t a, uint64_t b) {
  if (!a || !b)
    return kMissing;
  const int64_t d = static_cast<int64_t>(b) - static_cast<int64_t>(a);
  if (d > INT32_MAX)
    return INT32_MAX;
  if (d < INT32_MIN + 1)
    return INT32_MIN + 1;
  return static_cast<int32_t>(d);
}

inline void append_sample(const Sample &smp) {
  const uint64_t idx = g_sample_count.fetch_add(1, std::memory_order_relaxed);
  if (idx < kMaxSamples)
    g_samples[idx] = smp;
}

// ---------------------------------------------------------------------------
// Probes (call sites in CqrTransceiver.h / DecodingSession.cpp)
// ---------------------------------------------------------------------------

inline uint64_t entry_stamp() { return enabled() ? now_ns() : 0; }

// Arm the slot for a new request; t_entry was taken at inject() entry,
// before the rid was parsed.
inline void begin_request(uint32_t rid, uint32_t function_id,
                          uint64_t t_entry) {
  if (!enabled())
    return;
  auto &s = slot_for(rid);
  s.armed.store(0, std::memory_order_relaxed);
  s.t_inbox_push.store(0, std::memory_order_relaxed);
  s.t_notify1_ret.store(0, std::memory_order_relaxed);
  s.t_recv_wake.store(0, std::memory_order_relaxed);
  s.t_queue_push.store(0, std::memory_order_relaxed);
  s.t_notify2_ret.store(0, std::memory_order_relaxed);
  s.t_worker_wake.store(0, std::memory_order_relaxed);
  s.t_wait_begin.store(0, std::memory_order_relaxed);
  s.t_send_done.store(0, std::memory_order_relaxed);
  s.cold1.store(0, std::memory_order_relaxed);
  s.cold2.store(0, std::memory_order_relaxed);
  s.rid.store(rid, std::memory_order_relaxed);
  s.function_id.store(function_id, std::memory_order_relaxed);
  s.t_entry.store(t_entry, std::memory_order_relaxed);
  s.armed.store(1, std::memory_order_relaxed);
}

inline void stamp_inbox_push(uint32_t rid) {
  if (!full())
    return;
  auto &s = slot_for(rid);
  if (armed_for(s, rid))
    s.t_inbox_push.store(now_ns(), std::memory_order_relaxed);
}

inline void stamp_notify1_ret(uint32_t rid) {
  if (!full())
    return;
  auto &s = slot_for(rid);
  if (armed_for(s, rid))
    s.t_notify1_ret.store(now_ns(), std::memory_order_relaxed);
}

// After recv() pops a frame; reads the rid from the frame's RPCHeader.
inline void stamp_recv_wake(const void *frame_data, std::size_t frame_len,
                            bool was_empty) {
  if (!full())
    return;
  if (!frame_data || frame_len < sizeof(cudaq::realtime::RPCHeader))
    return;
  const auto *hdr = static_cast<const cudaq::realtime::RPCHeader *>(frame_data);
  auto &s = slot_for(hdr->request_id);
  if (!armed_for(s, hdr->request_id))
    return;
  s.t_recv_wake.store(now_ns(), std::memory_order_relaxed);
  s.cold1.store(was_empty ? 1 : 0, std::memory_order_relaxed);
}

inline void stamp_queue_push(uint32_t rid) {
  if (!full())
    return;
  auto &s = slot_for(rid);
  if (armed_for(s, rid))
    s.t_queue_push.store(now_ns(), std::memory_order_relaxed);
}

inline void stamp_notify2_ret(uint32_t rid) {
  if (!full())
    return;
  auto &s = slot_for(rid);
  if (armed_for(s, rid))
    s.t_notify2_ret.store(now_ns(), std::memory_order_relaxed);
}

inline void stamp_worker_wake(uint32_t rid, bool was_empty) {
  if (!full())
    return;
  auto &s = slot_for(rid);
  if (!armed_for(s, rid))
    return;
  s.t_worker_wake.store(now_ns(), std::memory_order_relaxed);
  s.cold2.store(was_empty ? 1 : 0, std::memory_order_relaxed);
}

inline void stamp_wait_begin(uint32_t rid) {
  if (!full())
    return;
  auto &s = slot_for(rid);
  if (armed_for(s, rid))
    s.t_wait_begin.store(now_ns(), std::memory_order_relaxed);
}

// Worker side, just before promise.set_value(): hop3 start.
inline void stamp_send_done(uint32_t rid) {
  if (!full())
    return;
  auto &s = slot_for(rid);
  if (armed_for(s, rid))
    s.t_send_done.store(now_ns(), std::memory_order_relaxed);
}

// Shutdown-drain completions are not latency samples.
inline void invalidate(uint32_t rid) {
  if (!enabled())
    return;
  auto &s = slot_for(rid);
  if (s.rid.load(std::memory_order_relaxed) == rid)
    s.armed.store(0, std::memory_order_relaxed);
}

// Blocking RPC completes: dispatcher thread, after fut.wait() returned.
inline void finish_blocking(uint32_t rid) {
  if (!enabled())
    return;
  auto &s = slot_for(rid);
  if (!armed_for(s, rid))
    return;
  const uint64_t t_exit = now_ns();
  s.armed.store(0, std::memory_order_relaxed);

  Sample smp{};
  smp.kind = kind_of(s.function_id.load(std::memory_order_relaxed));
  smp.rid = rid;
  const uint64_t e = s.t_entry.load(std::memory_order_relaxed);
  smp.total = delta_ns(e, t_exit);
  smp.hop1 = smp.hop2 = smp.hop3 = kMissing;
  smp.stage_build = smp.stage_dispatch = smp.stage_worker = kMissing;
  smp.notify1 = smp.notify2 = kMissing;
  if (full()) {
    const uint64_t ip = s.t_inbox_push.load(std::memory_order_relaxed);
    const uint64_t n1 = s.t_notify1_ret.load(std::memory_order_relaxed);
    const uint64_t rw = s.t_recv_wake.load(std::memory_order_relaxed);
    const uint64_t qp = s.t_queue_push.load(std::memory_order_relaxed);
    const uint64_t n2 = s.t_notify2_ret.load(std::memory_order_relaxed);
    const uint64_t ww = s.t_worker_wake.load(std::memory_order_relaxed);
    const uint64_t wb = s.t_wait_begin.load(std::memory_order_relaxed);
    const uint64_t sd = s.t_send_done.load(std::memory_order_relaxed);
    smp.stage_build = delta_ns(e, ip);
    smp.hop1 = delta_ns(ip, rw);
    smp.notify1 = delta_ns(ip, n1);
    smp.stage_dispatch = delta_ns(rw, qp);
    smp.hop2 = delta_ns(qp, ww);
    smp.notify2 = delta_ns(qp, n2);
    smp.stage_worker = delta_ns(ww, sd);
    smp.hop3 = delta_ns(sd, t_exit);
    smp.cold1 = s.cold1.load(std::memory_order_relaxed);
    smp.cold2 = s.cold2.load(std::memory_order_relaxed);
    smp.cold3 = (sd && wb && sd > wb) ? 1 : 0;
  }
  append_sample(smp);
}

// Fire-and-forget enqueue completes: worker thread, after the handler ran.
// total here = inject entry -> handler done (the enqueue pipeline latency).
inline void finish_enqueue(uint32_t rid) {
  if (!full())
    return;
  auto &s = slot_for(rid);
  if (!armed_for(s, rid))
    return;
  const uint64_t t_done = now_ns();
  s.armed.store(0, std::memory_order_relaxed);

  Sample smp{};
  smp.kind = kind_of(s.function_id.load(std::memory_order_relaxed));
  smp.rid = rid;
  const uint64_t e = s.t_entry.load(std::memory_order_relaxed);
  const uint64_t ip = s.t_inbox_push.load(std::memory_order_relaxed);
  const uint64_t n1 = s.t_notify1_ret.load(std::memory_order_relaxed);
  const uint64_t rw = s.t_recv_wake.load(std::memory_order_relaxed);
  const uint64_t qp = s.t_queue_push.load(std::memory_order_relaxed);
  const uint64_t n2 = s.t_notify2_ret.load(std::memory_order_relaxed);
  const uint64_t ww = s.t_worker_wake.load(std::memory_order_relaxed);
  smp.total = delta_ns(e, t_done);
  smp.stage_build = delta_ns(e, ip);
  smp.hop1 = delta_ns(ip, rw);
  smp.notify1 = delta_ns(ip, n1);
  smp.stage_dispatch = delta_ns(rw, qp);
  smp.hop2 = delta_ns(qp, ww);
  smp.notify2 = delta_ns(qp, n2);
  smp.stage_worker = delta_ns(ww, t_done);
  smp.hop3 = kMissing;
  smp.cold1 = s.cold1.load(std::memory_order_relaxed);
  smp.cold2 = s.cold2.load(std::memory_order_relaxed);
  smp.cold3 = 0;
  append_sample(smp);
}

// ---------------------------------------------------------------------------
// Thread naming + optional pinning (named threads make ftrace readable)
// ---------------------------------------------------------------------------

inline void name_and_pin(const char *name, const char *pin_env) {
  pthread_setname_np(pthread_self(), name);
  const char *e = std::getenv(pin_env);
  if (!e || !e[0])
    return;
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(std::atoi(e), &set);
  if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0)
    std::fprintf(stderr, "hopstats: %s=%s pin failed for %s\n", pin_env, e,
                 name);
}

inline void on_dispatcher_thread() {
  thread_local bool done = false;
  if (!done) {
    done = true;
    name_and_pin("cqr-dispatch", "QEC_PIN_DISPATCHER");
  }
}

inline void on_recv_thread() {
  thread_local bool done = false;
  if (!done) {
    done = true;
    name_and_pin("qec-recv", "QEC_PIN_RECV");
  }
}

inline void on_worker_thread() {
  thread_local bool done = false;
  if (!done) {
    done = true;
    name_and_pin("qec-worker", "QEC_PIN_WORKER");
  }
}

// ---------------------------------------------------------------------------
// Shutdown report (formatting cost paid only here)
// ---------------------------------------------------------------------------

inline double percentile_sorted(const std::vector<int32_t> &sorted, double p) {
  if (sorted.empty())
    return 0.0;
  const double pos = p * static_cast<double>(sorted.size() - 1);
  const std::size_t lo = static_cast<std::size_t>(pos);
  const std::size_t hi = std::min(lo + 1, sorted.size() - 1);
  const double frac = pos - static_cast<double>(lo);
  return static_cast<double>(sorted[lo]) +
         (static_cast<double>(sorted[hi]) - static_cast<double>(sorted[lo])) *
             frac;
}

inline void report() {
  if (!enabled())
    return;
  static std::atomic<bool> reported{false};
  if (reported.exchange(true))
    return;

  const uint64_t total = g_sample_count.load(std::memory_order_relaxed);
  const std::size_t n =
      static_cast<std::size_t>(std::min<uint64_t>(total, kMaxSamples));
  std::printf("QEC_HOP_STATS mode=%s recorded=%zu dropped=%llu spin_us=%lld\n",
              full() ? "full" : "total", n,
              static_cast<unsigned long long>(total - n),
              static_cast<long long>(
                  spin_budget_ns() < 0 ? -1 : spin_budget_ns() / 1000));

  struct MetricDesc {
    const char *name;
    int32_t Sample::*field;
    uint8_t Sample::*cold; // nullptr = no cold/warm split
  };
  const MetricDesc metrics[] = {
      {"hop1", &Sample::hop1, &Sample::cold1},
      {"hop2", &Sample::hop2, &Sample::cold2},
      {"hop3", &Sample::hop3, &Sample::cold3},
      {"total", &Sample::total, nullptr},
      {"stage_build", &Sample::stage_build, nullptr},
      {"stage_dispatch", &Sample::stage_dispatch, nullptr},
      {"stage_worker", &Sample::stage_worker, nullptr},
      {"notify1", &Sample::notify1, nullptr},
      {"notify2", &Sample::notify2, nullptr},
  };
  const char *kind_names[] = {"enqueue_syndromes", "get_corrections",
                              "reset_decoder", "other"};

  std::vector<int32_t> vals;
  for (int kind = 0; kind < 4; ++kind) {
    for (const auto &m : metrics) {
      const int subsets = m.cold ? 2 : 1; // cold, warm | all
      for (int subset = 0; subset < subsets; ++subset) {
        vals.clear();
        for (std::size_t i = 0; i < n; ++i) {
          const Sample &smp = g_samples[i];
          if (smp.kind != kind)
            continue;
          const int32_t v = smp.*(m.field);
          if (v == kMissing)
            continue;
          if (m.cold && ((smp.*(m.cold) != 0) != (subset == 0)))
            continue;
          vals.push_back(v);
        }
        if (vals.empty())
          continue;
        std::sort(vals.begin(), vals.end());
        int64_t sum = 0;
        for (int32_t v : vals)
          sum += v;
        const double avg = static_cast<double>(sum) /
                           static_cast<double>(vals.size()) / 1000.0;
        std::printf(
            "QEC_HOP_STATS kind=%s metric=%s subset=%s count=%zu "
            "min_us=%.2f avg_us=%.2f p50_us=%.2f p90_us=%.2f p99_us=%.2f "
            "max_us=%.2f\n",
            kind_names[kind], m.name,
            m.cold ? (subset == 0 ? "cold" : "warm") : "all", vals.size(),
            vals.front() / 1000.0, avg, percentile_sorted(vals, 0.50) / 1000.0,
            percentile_sorted(vals, 0.90) / 1000.0,
            percentile_sorted(vals, 0.99) / 1000.0, vals.back() / 1000.0);
      }
    }
  }

  if (const char *path = std::getenv("QEC_DECODING_SERVER_HOP_STATS_CSV");
      path && path[0]) {
    if (FILE *f = std::fopen(path, "w")) {
      std::fprintf(f, "kind,rid,cold1,cold2,cold3,hop1_ns,hop2_ns,hop3_ns,"
                      "total_ns,stage_build_ns,stage_dispatch_ns,"
                      "stage_worker_ns,notify1_ns,notify2_ns\n");
      for (std::size_t i = 0; i < n; ++i) {
        const Sample &s = g_samples[i];
        std::fprintf(f, "%s,%u,%u,%u,%u,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
                     kind_names[s.kind], s.rid, s.cold1, s.cold2, s.cold3,
                     s.hop1, s.hop2, s.hop3, s.total, s.stage_build,
                     s.stage_dispatch, s.stage_worker, s.notify1, s.notify2);
      }
      std::fclose(f);
      std::printf("QEC_HOP_STATS csv=%s rows=%zu\n", path, n);
    } else {
      std::fprintf(stderr, "hopstats: cannot open csv path %s\n", path);
    }
  }
  std::fflush(stdout);
}

} // namespace cudaq::qec::decoding_server::hopstats
