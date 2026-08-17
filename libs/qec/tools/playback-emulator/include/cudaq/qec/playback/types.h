/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file types.h
/// @brief Playback emulator data model: `schedule` (parsed input), `record`
/// (per-event output), and `run_result` (what `run()` returns). These three
/// types carry everything, are flat, and are allocation-free on the hot
/// path.

#include <cstdint>
#include <string>
#include <vector>

namespace cudaq::qec::playback {

/// Sentinel: no syndrome source attached to this event.
inline constexpr std::uint32_t kNoSource = ~std::uint32_t(0);

/// The five operations a schedule line can name. `stream_until` has no wire
/// RPC of its own -- it is client-side logic built from `enqueue` and
/// `get_corrections`. `enqueue_data` is wire-identical to
/// `enqueue` (same RPC, same frame shape) but marks a shot boundary: it pulls
/// a source's terminal data-qubit readout (`syndrome_source::read_data()`)
/// instead of another stabilizer round (`next_round()`). Requires a
/// `source=N` operand.
enum class operation : std::uint8_t {
  reset,
  enqueue,
  enqueue_data,
  get_corrections,
  stream_until,
};

inline const char *to_string(operation op) {
  switch (op) {
  case operation::reset:
    return "reset";
  case operation::enqueue:
    return "enqueue";
  case operation::enqueue_data:
    return "enqueue_data";
  case operation::get_corrections:
    return "get_corrections";
  case operation::stream_until:
    return "stream_until";
  }
  return "unknown";
}

/// Final outcome of one record. For `reset`/`enqueue`/`get_corrections` this
/// is the wire `RpcStatus` (see decoder_rpc_wire_format.h) cast into this
/// field. For `stream_until` it is instead one of four termination
/// reasons -- the two enumerations are disjoint (RpcStatus
/// occupies 0..7) so a reader can tell which one a given record used from
/// `record::op` alone.
enum class stream_terminate : std::int32_t {
  READY = 100,
  SOURCE_EXHAUSTED = 101,
  EXHAUSTED_ROUNDS = 102,
  ERROR = 103,
};

inline const char *to_string(stream_terminate t) {
  switch (t) {
  case stream_terminate::READY:
    return "READY";
  case stream_terminate::SOURCE_EXHAUSTED:
    return "SOURCE_EXHAUSTED";
  case stream_terminate::EXHAUSTED_ROUNDS:
    return "EXHAUSTED_ROUNDS";
  case stream_terminate::ERROR:
    return "ERROR";
  }
  return "unknown";
}

/// One line of the parsed playback schedule. Data and timing only -- no
/// blocking/retry/result-size semantics, which are properties of the
/// operation's wire mapping (op_traits), derived once, not stored per event.
struct event {
  std::uint64_t deadline_ns = 0; // absolute offset from t0, resolved at parse
  std::uint64_t decoder_id = 0;  // routing key -- required on every event
  operation op = operation::reset;
  // syndrome source for enqueue/enqueue_data/stream_until (one op per
  // event, so one source_id suffices for all of them); kNoSource otherwise.
  std::uint32_t source_id = kNoSource;
  std::uint32_t syndrome_offset = 0; // into schedule::syndrome_arena
  std::uint32_t syndrome_count = 0;
  std::uint32_t expected_offset = 0; // into schedule::expected_arena
  std::uint32_t expected_count = 0;

  // -- stream_until only; meaningless (left default) for every other op --
  /// Pacing, in ticks. 0 means unpaced (rounds fire as fast as the decoder
  /// answers); the parser defaults this to 1 tick (paced) when `every=` is
  /// omitted, and only sets it to 0 on an explicit `every=0`
  std::uint64_t stream_every_ticks = 1;
  /// Bounded, always: maximum number of streamed rounds before aborting
  std::uint32_t stream_max_rounds = 1000;
};

/// The parsed input. Bits live in shared arenas; events hold
/// (offset, count) into them.
struct schedule {
  std::vector<event> events;              // sorted by deadline_ns
  std::vector<std::uint8_t> syndrome_arena;  // one byte per bit (0x00/0x01)
  std::vector<std::uint8_t> expected_arena;  // one byte per bit
  std::vector<std::uint64_t> decoders;       // known decoder_ids
  std::uint64_t tick_ns = 1000; // wall-clock duration of one tick
};

/// One record per event, in event order, preallocated before t0
struct record {
  // -- identity --
  std::uint32_t event_index = 0;
  std::uint64_t decoder_id = 0;
  operation op = operation::reset;

  // -- timing (ns, relative to t0) --
  std::uint64_t deadline_ns = 0; // where it was supposed to fire
  std::uint64_t call_ns = 0;     // when the dispatch actually began
  std::uint64_t return_ns = 0;   // when it completed

  // -- outcome --
  std::int32_t status = 0;       // RpcStatus, or stream_terminate (see above)
  std::uint32_t rounds_streamed = 0; // stream_until only
  bool read_completed = false;   // a correction was actually consumed

  // -- data (offsets into the run's arenas) --
  std::uint32_t syndrome_offset = 0, syndrome_count = 0;   // what was SENT
  std::uint32_t correction_offset = 0, correction_count = 0; // what came BACK
  bool correction_mismatch = false; // vs. the event's expected bits

  // -- cross-referencing --
  std::uint32_t first_request_id = 0; // for correlating with server logs
};

/// Run-level metadata: everything a report needs to interpret the records
/// without re-running.
struct run_metadata {
  std::uint64_t t0_ns = 0;      // CLOCK_MONOTONIC value run() aligned to
  std::uint64_t tick_ns = 0;
  std::string backend;           // "null" | "inproc" | "udp"
  std::uint64_t spin_slack_ns = 0; // calibrated wait_until() slack
  std::uint64_t config_hash = 0;
};

/// What `run()` returns. 
struct run_result {
  std::vector<record> records;
  std::vector<std::uint8_t> syndrome_log;   // arena the records index into
  std::vector<std::uint8_t> correction_log;
  std::vector<std::string> warnings; 
  run_metadata meta;
};

} // namespace cudaq::qec::playback
