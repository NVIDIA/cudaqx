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
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace cudaq::qec::playback {

class blocking_session : public session {
public:
  virtual RpcStatus send_sync(const frame &f, std::span<std::uint8_t> reply,
                              std::size_t &reply_len) = 0;

  // Echoes the frame's own request_id, same as every real backend, so a
  // reader can key completions by id consistently.
  std::uint32_t submit(const frame &f) override {
    if (f.size < sizeof(cudaq::realtime::RPCHeader))
      throw std::invalid_argument(
          "blocking_session: frame is smaller than RPCHeader");
    const std::uint32_t id =
        reinterpret_cast<const cudaq::realtime::RPCHeader *>(f.bytes)
            ->request_id;
    auto held = std::make_shared<held_reply>();
    held->bytes.resize(kHeldReplyBytes);
    std::size_t len = 0;
    held->status = send_sync(f, held->bytes, len);
    held->bytes.resize(len);
    {
      std::lock_guard<std::mutex> lock(mu_);
      held_[id] = std::move(held);
      completed_.push_back(id);
    }
    completed_cv_.notify_one();
    return id;
  }

  bool wait_next_completion(std::uint32_t &request_id,
                            std::chrono::milliseconds timeout) override {
    std::unique_lock<std::mutex> lock(mu_);
    if (!completed_cv_.wait_for(lock, timeout,
                                [&] { return !completed_.empty(); }))
      return false;
    request_id = completed_.front();
    completed_.pop_front();
    return true;
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
  std::condition_variable completed_cv_;
  std::deque<std::uint32_t> completed_;
  std::unordered_map<std::uint32_t, std::shared_ptr<held_reply>> held_;
};

} // namespace cudaq::qec::playback
