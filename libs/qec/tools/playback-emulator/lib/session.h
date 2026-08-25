/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file session.h
/// @brief The `session` interface and the factories for the backends
/// that implement it. A session is anything that can carry an RPC frame
/// (cudaq::realtime::RPCHeader followed by the operation's payload, followed
/// for `enqueue` by bit-packed syndrome bytes) to a decoder and bring a reply
/// back. The concrete session classes are private to their .cpp files; only
/// the factories at the bottom of this header are public.

#include "cudaq/qec/realtime/decoder_rpc_wire_format.h"
#include "cudaq/qec/realtime/decoding_config.h"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace cudaq::qec::playback {

using cudaq::qec::decoding::rpc::RpcStatus;

inline const char *to_string(RpcStatus status) {
  switch (status) {
  case RpcStatus::OK:
    return "OK";
  case RpcStatus::INVALID_DECODER:
    return "INVALID_DECODER";
  case RpcStatus::BAD_REQUEST:
    return "BAD_REQUEST";
  case RpcStatus::INTERNAL_ERROR:
    return "INTERNAL_ERROR";
  case RpcStatus::NOT_READY:
    return "NOT_READY";
  case RpcStatus::BUSY:
    return "BUSY";
  case RpcStatus::SYNDROMES_DROPPED:
    return "SYNDROMES_DROPPED";
  }
  return "unknown";
}

/// A pre-serialized RPC request: RPCHeader + payload (+ trailing bit-packed
/// syndrome bytes for `enqueue`). Non-owning -- the bytes live in the
/// run_plan's frame arena, built once before t0.
struct frame {
  const std::uint8_t *bytes = nullptr;
  std::size_t size = 0;
};

/// Anything that can carry an RPC frame to a decoder and bring a reply back.
/// Both backends (in-process, UDP) implement this; `null` does too.
/// This interface is deliberately ignorant of which RPC a
/// frame holds --  only "fire-and-forget" vs. "blocking with a reply".
///
/// SINGLE PUBLISHER. `send_async`/`submit` must be called from one thread (or
/// be externally serialized); `await` is the only method safe to call
/// concurrently, and only for distinct request_ids. Given that, a session
/// delivers frames in the order the caller returned from `send_async`/
/// `submit` -- the publisher's own program order is wire order, so no backend
/// needs a lock to establish it -- and `await` neither influences that order
/// nor delays a later submission. The emulator upholds this: one timing
/// thread publishes, and at most one reader thread per session awaits.
/// Two publishers would not corrupt anything, they would just get arbitrary
/// wire order -- a silent ordering bug in a latency tool, which is why the
/// constraint is written here rather than left to be inferred.
///
/// LIFETIME. No thread may be inside any method when the session is
/// destroyed. The emulator upholds this too: `run()` finishes dispatching and
/// joins every reader before the router that owns the sessions goes away.
class session {
public:
  virtual ~session() = default;

  /// Fire-and-forget. Returns once the frame is published. Used for RPCs
  /// with no reply on the wire. Call from the publishing thread only.
  virtual void send_async(const frame &f) = 0;

  /// Publish a frame whose reply is collected later, and return the
  /// request_id it will come back under. Returns as soon as the frame is
  /// published: never waits on the decoder. Call from the publishing thread
  /// only.
  virtual std::uint32_t submit(const frame &f) = 0;

  /// Collect `request_id`'s reply, blocking the caller. Returns the
  /// RpcStatus; on OK with a result body, copies the (still bit-packed) reply
  /// into `reply`. Safe to call from a thread other than the publisher, and
  /// from two threads at once so long as they name different request_ids.
  virtual RpcStatus await(std::uint32_t request_id, std::span<std::uint8_t> reply,
                          std::size_t &reply_len) = 0;

  /// The largest frame this session can carry, or 0 for unbounded. Set once
  /// at construction and validated against the schedule before t0, so a
  /// session that cannot carry what the schedule asks for is a startup error
  /// rather than a runtime surprise.
  std::uint32_t max_frame_bytes = 0;
};

// -- Backend factories. The concrete session classes live in anonymous
// namespaces in their own .cpp files

/// Discards everything, but still builds and serializes a full frame per
/// event and checksums it so the compiler cannot elide the work.
std::unique_ptr<session> make_null_session();

/// One `make_null_session()` per decoder_id -- each decoder dispatches on
/// its own thread, so sharing one instance across decoder_ids is never
/// safe. Same shape as make_inproc_sessions()/make_udp_sessions() below.
std::vector<std::pair<std::uint64_t, std::unique_ptr<session>>>
make_null_sessions(const std::vector<std::uint64_t> &decoder_ids);

/// Realizes the decoders named in `config` in this process, one session per
/// decoder, each dispatching directly to that decoder's own DecodingSession
std::vector<std::pair<std::uint64_t, std::unique_ptr<session>>>
make_inproc_sessions(
    const cudaq::qec::decoding::config::multi_decoder_config &config);

/// Connected UDP client session(s) to a decoding server speaking the wire
/// format in decoder_rpc_wire_format.h. `endpoints` maps decoder_id ->
/// "host:port"; one session per decoder_id
/// `timeout_ms` bounds how long await() waits on a silent socket.
std::vector<std::pair<std::uint64_t, std::unique_ptr<session>>>
make_udp_sessions(const std::unordered_map<std::uint64_t, std::string> &endpoints,
                   std::uint32_t timeout_ms = 200);

/// Points `router[id]` at each session's owning pointer, for a
/// make_*_sessions() result the caller is keeping alive elsewhere. The
/// shared last step of adopting any backend's sessions into a run.
void route_sessions(
    const std::vector<std::pair<std::uint64_t, std::unique_ptr<session>>> &sessions,
    std::unordered_map<std::uint64_t, session *> &router);

} // namespace cudaq::qec::playback
