/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file emulator.h
/// @brief Top-level entry points: parse() -> plan() -> run() 
/// Callable identically from the CLI tool and the Python binding.

#include "cudaq/qec/playback/session.h"
#include "cudaq/qec/playback/syndrome_source.h"
#include "cudaq/qec/playback/types.h"

#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <ostream>
#include <unordered_map>

namespace cudaq::qec::playback {

/// Parse a line-oriented playback description directly into a
/// `schedule`. `known_decoder_ids` is the set of decoder_ids the config
/// declares; a decoder_id on a schedule line that is absent from this set is
/// a parse error. `tick_ns` resolves each line's `<tick>` to
/// `deadline_ns = tick * tick_ns`. Throws std::invalid_argument 
/// naming the offending line on any parse error
schedule parse(std::string_view text,
               const std::vector<std::uint64_t> &known_decoder_ids,
               std::uint64_t tick_ns);

/// Shared dispatch policy, applied once above every
/// session -- not duplicated into each backend.
struct dispatch_policy {
  /// Default: fail loudly on NOT_READY (a schedule too tight for the
  /// decoder should surface). Set true to retry to `not_ready_deadline_ns`.
  bool retry_not_ready = false;
  std::uint64_t not_ready_deadline_ns = 5'000'000; // 5 ms, only if retrying
};

struct run_params {
  std::uint64_t lead_in_ns = 20'000'000; // 20 ms before t0
  /// 0 = auto-calibrate from clock_nanosleep overshoot at startup
  std::uint64_t spin_slack_ns = 0;
  dispatch_policy dispatch;
};

/// Per-event plan-time state: where its pre-built frame lives (direct ops
/// whose bits are known before t0). stream_until, and any enqueue whose
/// bits are only known at dispatch time, leave `has_frame` false and build
/// their frame at run time instead.
struct event_plan {
  bool has_frame = false;
  // For pre-built RPC frames; offset in the frame arena.
  std::uint32_t frame_offset = 0;
  std::uint32_t frame_len = 0;
};

/// Pre-planned, immediately-runnable schedule: frames
/// pre-serialized into one contiguous buffer, capabilities validated,
/// records and log arenas pre-faulted at final size. Built by plan(),
/// consumed by run().
struct run_plan {
  schedule sched;
  std::vector<std::uint8_t> frame_arena;
  std::vector<event_plan> event_plans;
  std::unordered_map<std::uint64_t, session *> router;
  std::unordered_map<std::uint32_t, syndrome_source *> sources;
  run_params params;
};

/// Validate `sched` against `router`'s session capabilities and pre-build
/// everything run()'s timing loop must not do on the hot path. Throws
/// std::invalid_argument on any gap. `router` maps decoder_id -> session;
/// `sources` maps a schedule's `source_id` (event::source_id /
/// stream_params::source_id) to the syndrome_source instance it reads from.
/// Ownership of both the sessions and the sources stays with the caller for
/// the lifetime of the returned run_plan.
std::shared_ptr<run_plan>
plan(const schedule &sched, const std::unordered_map<std::uint64_t, session *> &router,
     const std::unordered_map<std::uint32_t, syndrome_source *> &sources,
     const run_params &params = {});

/// Run the plan: for each event, wait_until(t0 + deadline), dispatch,
/// record. Single timing thread, blocking. Returns when every event has
/// been dispatched and every record finalized.
run_result run(std::shared_ptr<run_plan> plan);

/// Downstream analysis writes CSV.One row per record: identity, timings, derived lateness/latency, status,
/// rounds streamed, and the syndrome/correction bits resolved from the logs
/// (hex-encoded, MSB-first-nibble, one column each).
void write_csv(const run_result &result, std::ostream &out);
std::string write_csv(const run_result &result);

} // namespace cudaq::qec::playback
