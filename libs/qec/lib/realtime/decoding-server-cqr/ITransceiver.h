/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace cudaq::qec::decoder_server {

/// Peer identity — the address to which the server sends a response.
struct PeerId {
  std::array<uint8_t, 16> addr; ///< GID / IPv6 (16 bytes)
  uint16_t port;

  bool operator==(const PeerId &) const = default;
};

/// A received frame: owns the wire bytes (RPCHeader + payload) plus transport
/// metadata needed for syndrome scatter and response routing.
/// Ownership of buf is transferred to WorkItem on enqueue.
///
/// release_fn: when non-null, must be called exactly once after the frame
/// payload is no longer needed (i.e. after the GPU scheduler has accepted the
/// data).  For host-copy transports (CQR, Loopback, CPU ring buffer path) this
/// is always null — the copy itself constitutes "release."  For GPU ring buffer
/// transports (full RelayBP path), release_fn returns the slot to Hololink
/// once GPU decode completes; it must NOT be called before that.
struct RxFrame {
  std::vector<uint8_t> buf; ///< RPCHeader + payload (owned copy)
  uint32_t vp_id;
  PeerId peer;
  std::function<void()> release_fn; ///< null except on GPU ring-buffer path
};

/// Transport abstraction used by DecoderServer and DecoderSession.
struct ITransceiver {
  /// Block until a frame is available and return it (buf is owned by caller).
  /// After shutdown() this may return a frame with an EMPTY buf -- the
  /// sentinel that unblocks the receive loop so it can observe the shutdown
  /// flag and exit.
  virtual RxFrame recv() = 0;

  /// Unblock any thread waiting in recv() (which then returns an empty
  /// sentinel frame). Called by DecoderServer::stop().
  virtual void shutdown() {}

  /// Send a response to \p peer.  Thread-safe: called from session worker
  /// threads, which may be concurrent.
  virtual void send(const PeerId &peer, const uint8_t *data, size_t len) = 0;

  virtual ~ITransceiver() = default;
};

} // namespace cudaq::qec::decoder_server
