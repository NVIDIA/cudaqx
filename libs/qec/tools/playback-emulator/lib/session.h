/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file session.h
/// @brief The `session` interface and `capabilities`. A session is
/// anything that can carry an RPC frame (cudaq::realtime::RPCHeader followed by the operation's payload,
/// followed for `enqueue` by bit-packed syndrome bytes) to a decoder and
/// bring a reply back. 

#include "cudaq/qec/realtime/decoder_rpc_wire_format.h"

#include <cstdint>
#include <span>

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

/// Every session declares what it can do, and the emulator validates the
/// schedule against this before t0. A capability gap must
/// be a startup error, never a runtime surprise.
struct capabilities {
  bool reports_not_ready = false; // can get_corrections answer NOT_READY?
  std::uint32_t max_frame_bytes = 0; // 0 = unbounded
  std::uint32_t observables = 0; // this session's one decoder's `get_corrections` reply size
};

/// Anything that can carry an RPC frame to a decoder and bring a reply back.
/// Both backends (in-process, UDP) implement this; `null` does too.
/// This interface is deliberately ignorant of which RPC a
/// frame holds --  only "fire-and-forget" vs. "blocking with a reply". 
class session {
public:
  virtual ~session() = default;

  /// Fire-and-forget. Returns once the frame is published. Used for RPCs
  /// with no reply on the wire.
  virtual void send_async(const frame &f) = 0;

  /// Blocking. Returns the RpcStatus; on OK with a result body, copies the
  /// (still bit-packed) reply into `reply`. Must NOT retry on NOT_READY --
  /// that is the caller's (dispatch layer's) policy decision.
  virtual RpcStatus send_sync(const frame &f, std::span<std::uint8_t> reply,
                               std::size_t &reply_len) = 0;

  virtual capabilities caps() const = 0;

  /// Runs before t0 so first-call costs (lazy page mapping, cold branch
  /// predictors, connection setup) do not land on event 0.
  virtual void warm_up() {}
};

} // namespace cudaq::qec::playback
