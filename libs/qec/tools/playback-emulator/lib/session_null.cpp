/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// The `null` backend. It discards every frame, but touches every byte of it
/// through an atomic checksum so the optimizer cannot prove the frame is dead
/// and elide the serialization work the real backends would also have to pay
/// for.

#include "cudaq/qec/playback/backends.h"

#include <algorithm>
#include <atomic>

namespace cudaq::qec::playback {

namespace {

class null_session : public session {
public:
  void send_async(const frame &f) override { checksum(f); }

  RpcStatus send_sync(const frame &f, std::span<std::uint8_t> reply,
                       std::size_t &reply_len) override {
    checksum(f);
    std::fill(reply.begin(), reply.end(), std::uint8_t{0});
    reply_len = 0;
    return RpcStatus::OK;
  }

  capabilities caps() const override {
    capabilities c;
    c.reports_not_ready = true;
    c.max_frame_bytes = 0; // unbounded
    return c;
  }

private:
  // Folds every byte of the frame into an atomic accumulator so the compiler
  // cannot elide the build/serialize step even though the result is unused.
  void checksum(const frame &f) {
    std::uint64_t acc = 0;
    for (std::size_t i = 0; i < f.size; ++i)
      acc ^= (std::uint64_t(f.bytes[i]) << (8 * (i & 7)));
    checksum_.fetch_xor(acc, std::memory_order_relaxed);
  }

  std::atomic<std::uint64_t> checksum_{0};
};

} // namespace

std::unique_ptr<session> make_null_session() {
  return std::make_unique<null_session>();
}

} // namespace cudaq::qec::playback
