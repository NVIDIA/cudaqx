/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 ******************************************************************************/

#pragma once

/// @file test_session_fakes.h
/// @brief `blocking_session`: test-double base for a backend whose answer is
/// already at hand -- implement `send_sync` and submit/await come for free.
/// Lives in the test tree on purpose: a real backend's reply genuinely
/// arrives later, so it must implement submit/await itself to avoid blocking
/// the timing thread inside submit -- no shipping backend is built on this.

#include "session.h"

#include <algorithm>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace cudaq::qec::playback {

class blocking_session : public session {
public:
  virtual RpcStatus send_sync(const frame &f, std::span<std::uint8_t> reply,
                              std::size_t &reply_len) = 0;

  void send_async(const frame &f) override {
    std::size_t ignored = 0;
    send_sync(f, {}, ignored);
  }

  std::uint32_t submit(const frame &f) override {
    auto held = std::make_shared<held_reply>();
    held->bytes.resize(kHeldReplyBytes);
    std::size_t len = 0;
    held->status = send_sync(f, held->bytes, len);
    held->bytes.resize(len);
    std::lock_guard<std::mutex> lock(mu_);
    const std::uint32_t id = next_request_id_++;
    held_[id] = std::move(held);
    return id;
  }

  RpcStatus await(std::uint32_t request_id, std::span<std::uint8_t> reply,
                  std::size_t &reply_len) override {
    reply_len = 0;
    std::shared_ptr<held_reply> held;
    {
      std::lock_guard<std::mutex> lock(mu_);
      auto it = held_.find(request_id);
      if (it == held_.end())
        return RpcStatus::INTERNAL_ERROR; // never submitted, or already taken
      held = std::move(it->second);
      held_.erase(it);
    }
    reply_len = std::min(held->bytes.size(), reply.size());
    std::copy_n(held->bytes.begin(), reply_len, reply.begin());
    return held->status;
  }

private:
  /// Nothing here knows which RPC a frame holds (that is the point of the
  /// session interface), so a submitted reply lands in a buffer big enough
  /// for any of them and is trimmed to what came back.
  static constexpr std::size_t kHeldReplyBytes = 4096;

  struct held_reply {
    std::vector<std::uint8_t> bytes;
    RpcStatus status = RpcStatus::OK;
  };

  std::mutex mu_;
  std::uint32_t next_request_id_ = 1;
  std::unordered_map<std::uint32_t, std::shared_ptr<held_reply>> held_;
};

} // namespace cudaq::qec::playback
