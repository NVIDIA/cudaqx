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
#include <span>

namespace cudaq::qec::decoder_server {

/// Peer identity — the address to which the server sends a response.
struct PeerId {
  std::array<uint8_t, 16> addr; ///< GID / IPv6 (16 bytes)
  uint16_t port;

  bool operator==(const PeerId &) const = default;
};

/// A received frame: full wire bytes (RPCHeader + payload) plus transport
/// metadata needed for syndrome scatter and response routing.
struct RxFrame {
  std::span<const uint8_t> bytes; ///< RPCHeader + payload
  uint32_t vp_id;                 ///< transport QP / VP index
  PeerId peer;                    ///< destination for the response
};

/// Transport abstraction used by DecoderServer and DecoderSession.
struct ITransceiver {
  /// Block until a frame is available and return it.
  virtual RxFrame recv() = 0;

  /// Send a response to \p peer.  Thread-safe: called from session worker
  /// threads, which may be concurrent.
  virtual void send(const PeerId &peer, const uint8_t *buf, size_t len) = 0;

  /// Release ring-buffer slot; no-op for copy-based impls.
  virtual void release(RxFrame /*frame*/) {}

  virtual ~ITransceiver() = default;
};

} // namespace cudaq::qec::decoder_server
