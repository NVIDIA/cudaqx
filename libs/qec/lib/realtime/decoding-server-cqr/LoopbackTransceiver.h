/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "ITransceiver.h"

#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <vector>

namespace cudaq::qec::decoder_server {

/// Paired in-process transceivers for development and testing.
///
/// LoopbackTransceiver::make() returns two endpoints A and B.
/// Data written with A.send() is readable via B.recv(), and vice versa.
/// All payload bytes are copied on enqueue so each RxFrame owns its buffer.
class LoopbackTransceiver final : public ITransceiver {
public:
  static std::pair<std::unique_ptr<LoopbackTransceiver>,
                   std::unique_ptr<LoopbackTransceiver>>
  make();

  RxFrame recv() override;

  void send(const PeerId &peer, const uint8_t *data, size_t len) override;

  void shutdown() override;

private:
  explicit LoopbackTransceiver(
      std::shared_ptr<std::deque<std::vector<uint8_t>>> inbox,
      std::shared_ptr<std::deque<std::vector<uint8_t>>> outbox,
      std::shared_ptr<std::mutex> mtx,
      std::shared_ptr<std::condition_variable> cv)
      : inbox_(std::move(inbox)), outbox_(std::move(outbox)),
        mtx_(std::move(mtx)), cv_(std::move(cv)),
        stopped_(std::make_shared<bool>(false)) {}

  std::shared_ptr<std::deque<std::vector<uint8_t>>> inbox_;
  std::shared_ptr<std::deque<std::vector<uint8_t>>> outbox_;
  std::shared_ptr<std::mutex> mtx_;
  std::shared_ptr<std::condition_variable> cv_;
  std::shared_ptr<bool> stopped_;
};

inline std::pair<std::unique_ptr<LoopbackTransceiver>,
                 std::unique_ptr<LoopbackTransceiver>>
LoopbackTransceiver::make() {
  auto q_ab = std::make_shared<std::deque<std::vector<uint8_t>>>();
  auto q_ba = std::make_shared<std::deque<std::vector<uint8_t>>>();
  auto mtx = std::make_shared<std::mutex>();
  auto cv = std::make_shared<std::condition_variable>();

  auto a = std::unique_ptr<LoopbackTransceiver>(
      new LoopbackTransceiver(q_ba, q_ab, mtx, cv));
  auto b = std::unique_ptr<LoopbackTransceiver>(
      new LoopbackTransceiver(q_ab, q_ba, mtx, cv));
  return {std::move(a), std::move(b)};
}

inline RxFrame LoopbackTransceiver::recv() {
  std::unique_lock<std::mutex> lk(*mtx_);
  cv_->wait(lk, [this] { return !inbox_->empty() || *stopped_; });
  if (inbox_->empty())
    return {}; // shutdown sentinel (empty buf)
  RxFrame frame;
  frame.buf = std::move(inbox_->front());
  inbox_->pop_front();
  return frame;
}

inline void LoopbackTransceiver::shutdown() {
  {
    std::lock_guard<std::mutex> lk(*mtx_);
    *stopped_ = true;
  }
  cv_->notify_all();
}

inline void LoopbackTransceiver::send(const PeerId & /*peer*/,
                                      const uint8_t *data, size_t len) {
  {
    std::lock_guard<std::mutex> lk(*mtx_);
    outbox_->emplace_back(data, data + len);
  }
  cv_->notify_all();
}

} // namespace cudaq::qec::decoder_server
