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

#include "session.h"
#include "syndrome_source.h"
#include "types.h"

#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <ostream>
#include <unordered_map>

namespace cudaq::qec::playback {

/// Parse a line-oriented playback description directly into a `schedule`.
/// A line is `<trigger> <op> [key=value...]`
///
/// The trigger reads three ways. An integer is a deadline, resolved here to
/// `deadline_ns = tick * tick_ns`. An integer with a `+` before it, i.e. `+N` is relative instead:
/// resolved at run time to `N` ticks after the previous line's actual
/// completion (its `return_ns`). Finally, a `-` means execute as fast as possible,
/// equivalent to `+0`. 
///
/// `session=N` picks which decoder the line talks to, defaulting to 0.
/// `known_decoder_ids` is the set the config declares, and a `session=`
/// outside it is a parse error. Throws std::invalid_argument naming the
/// offending line on any parse error.
schedule parse(std::string_view text,
               const std::vector<std::uint64_t> &known_decoder_ids,
               std::uint64_t tick_ns);

struct run_params {
  std::uint64_t lead_in_ns = 20'000'000; // 20 ms before t0
};

/// One pre-serialized round: where its frame sits in the run_plan's frame
/// arena, and where the syndrome bits that frame carries sit in the
/// schedule's. `bits_count` is 0 for the ops that send no syndromes.
struct round_plan {
  std::uint32_t frame_offset = 0;
  std::uint32_t frame_len = 0;
  std::uint32_t bits_offset = 0;
  std::uint32_t bits_count = 0;
};

/// Per-event plan-time state: every frame whose bytes are known before t0.
/// `reset` and `get_corrections` have exactly one. A `stream` has one per
/// round when its round count is fixed and its source could be drawn from
/// ahead of time; otherwise it has none and builds its frames as it goes,
/// which is also what a source-streamed or `until=` stream always does.
struct event_plan {
  std::vector<round_plan> rounds;
};

/// Pre-planned, immediately-runnable schedule: frames
/// pre-serialized into one contiguous buffer, frame sizes validated,
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

/// Validate `sched` against `router`'s session frame limits and pre-build
/// everything run()'s timing loop must not do on the hot path. Throws
/// std::invalid_argument on any gap. `router` maps decoder_id -> session;
/// `sources` maps a schedule's `event::source_id` to the syndrome_source
/// instance it reads from. 
/// Ownership of both the sessions and the sources stays with the caller for
/// the lifetime of the returned run_plan.
std::shared_ptr<run_plan>
plan(const schedule &sched, const std::unordered_map<std::uint64_t, session *> &router,
     const std::unordered_map<std::uint32_t, syndrome_source *> &sources,
     const run_params &params = {});

/// Run the plan on one timing thread: wait_until(t0 + deadline), dispatch,
/// record, in schedule order, with nothing done between deadlines but wait.
/// A `reset` or `get_corrections` carrying `signal=` submits its request and
/// returns; the answer is collected on the routed session's own completion
/// thread, which fills in the record and raises the signal
/// the concurrency model: one clock, one completion thread per session.
/// A hard error aborts the run, but `result.records` is never truncated:
/// every event gets a slot, and `record::dispatched` distinguishes what ran
/// from what the abort pre-empted.
run_result run(std::shared_ptr<run_plan> plan);

/// Downstream analysis writes CSV. One row per record: identity, timings, derived lateness/latency, status,
/// rounds streamed, and the syndrome/correction bits resolved from the logs
/// (hex-encoded, MSB-first-nibble, one column each).
void write_csv(const run_result &result, std::ostream &out);
std::string write_csv(const run_result &result);

} // namespace cudaq::qec::playback
