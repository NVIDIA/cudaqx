/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "logger_forwarder.h"
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <memory>
#include <mutex>
#include <thread>

/// @file
/// @brief Internal forwarder implementation for QEC logger.
/// @details
/// Implements a bounded lock-free ring buffer (multi-producer, single-consumer)
/// used when asynchronous forwarding is enabled. Producers enqueue fixed-size
/// records without heap allocation; a dedicated worker thread converts them to
/// user-facing records and invokes the configured callback.

namespace cudaq::qec::detail::forwarder_internal {
namespace {

class AsyncForwarder {
public:
  AsyncForwarder() = default;
  ~AsyncForwarder() { clear(); }

  void set(ForwarderConfig config) {
    clear();
    mConfig = std::move(config);
    if (!mConfig.callback)
      return;
    if (mConfig.queueCapacity == 0)
      mConfig.queueCapacity = 1;
    if (mConfig.messageCapacity == 0)
      mConfig.messageCapacity = 1;
    mConfig.messageCapacity =
        std::min(mConfig.messageCapacity, kRealtimeForwarderMaxMessageCapacity);
    mMessageCapacity.store(mConfig.messageCapacity, std::memory_order_relaxed);

    mCapacity = roundUpToPow2(mConfig.queueCapacity);
    mRing = std::make_unique<RingSlot[]>(mCapacity);
    for (std::size_t i = 0; i < mCapacity; ++i)
      mRing[i].sequence.store(i, std::memory_order_relaxed);

    mMask = mCapacity - 1;
    mEnqueuePos.store(0, std::memory_order_relaxed);
    mDequeuePos.store(0, std::memory_order_relaxed);
    mStop.store(false, std::memory_order_relaxed);
    mEnabled.store(true, std::memory_order_release);
    mWorker = std::thread([this] { runWorker(); });
  }

  // Stop producer admission first, then wait for in-flight producers before
  // tearing down the ring storage to avoid use-after-free races.
  void clear() {
    mEnabled.store(false, std::memory_order_release);
    while (mActiveProducers.load(std::memory_order_acquire) != 0)
      std::this_thread::yield();
    mStop.store(true, std::memory_order_release);
    mCv.notify_all();
    if (mWorker.joinable())
      mWorker.join();
    mRing.reset();
    mCapacity = 0;
    mMask = 0;
    mEnqueuePos.store(0, std::memory_order_relaxed);
    mDequeuePos.store(0, std::memory_order_relaxed);
    mStop.store(false, std::memory_order_relaxed);
    mMessageCapacity.store(kRealtimeForwarderDefaultMessageCapacity,
                           std::memory_order_relaxed);
    mConfig = {};
  }

  bool isEnabled() const { return mEnabled.load(std::memory_order_relaxed); }

  ForwarderStats stats() const {
    ForwarderStats stats;
    stats.enqueuedRecords = mEnqueuedRecords.load(std::memory_order_relaxed);
    stats.droppedRecords = mDroppedRecords.load(std::memory_order_relaxed);
    stats.truncatedRecords = mTruncatedRecords.load(std::memory_order_relaxed);
    stats.forwardFailures = mForwardFailures.load(std::memory_order_relaxed);
    return stats;
  }

  void resetStats() {
    mEnqueuedRecords.store(0, std::memory_order_relaxed);
    mDroppedRecords.store(0, std::memory_order_relaxed);
    mTruncatedRecords.store(0, std::memory_order_relaxed);
    mForwardFailures.store(0, std::memory_order_relaxed);
    mDropWarningPending.store(false, std::memory_order_relaxed);
    mDropWarningEmitted.store(false, std::memory_order_relaxed);
  }

  std::size_t messageCapacity() const {
    return mMessageCapacity.load(std::memory_order_relaxed);
  }

  void noteTruncation() {
    mTruncatedRecords.fetch_add(1, std::memory_order_relaxed);
  }

  void flush() {
    std::unique_lock<std::mutex> lock(mWaitMutex);
    mCv.wait(lock, [&] {
      return ((!mEnabled.load()) ||
              (isQueueEmpty() &&
               mInFlightCallbacks.load(std::memory_order_acquire) == 0));
    });
  }

  void enqueue(QueuedLogRecord record) {
    if (!isEnabled())
      return;
    ProducerGuard producerGuard(mActiveProducers);
    if (!isEnabled() || !mConfig.callback || !mRing)
      return;

    while (true) {
      if (tryEnqueueOne(record)) {
        mEnqueuedRecords.fetch_add(1, std::memory_order_relaxed);
        mCv.notify_one();
        return;
      }

      // The queue is full. dropOldest evicts the oldest record to admit the
      // incoming one; the evicted record is the genuinely lost message, so it
      // is the only thing counted as a drop here. If the eviction fails because
      // a consumer already drained the slot, the queue is no longer full, so we
      // simply retry the enqueue without counting a drop (the incoming record
      // is not lost). This keeps droppedRecords equal to the number of records
      // actually discarded, and ensures the dropNewest fallthrough below counts
      // only the incoming record it drops.
      if (mConfig.dropPolicy == ForwardDropPolicy::dropOldest) {
        QueuedLogRecord ignored;
        if (tryDequeueOne(ignored))
          notifyDrop();
        continue;
      }

      // dropNewest: the incoming record is the one we lose.
      notifyDrop();
      return;
    }
  }

private:
  struct RingSlot {
    std::atomic<std::size_t> sequence{0};
    QueuedLogRecord record;
  };

  struct ProducerGuard {
    explicit ProducerGuard(std::atomic<std::uint64_t> &counterRef)
        : counter(counterRef) {
      counter.fetch_add(1, std::memory_order_acq_rel);
    }
    ~ProducerGuard() { counter.fetch_sub(1, std::memory_order_acq_rel); }
    std::atomic<std::uint64_t> &counter;
  };

  static std::size_t roundUpToPow2(std::size_t value) {
    // Ring indexing uses `pos & (capacity - 1)`, so capacity must remain a
    // power-of-two for correct slot mapping.
    std::size_t rounded = 2;
    while (rounded < value)
      rounded <<= 1;
    return rounded;
  }

  bool tryEnqueueOne(QueuedLogRecord &record) {
    std::size_t pos = mEnqueuePos.load(std::memory_order_relaxed);
    while (true) {
      RingSlot &slot = mRing[pos & mMask];
      const std::size_t sequence =
          slot.sequence.load(std::memory_order_acquire);
      const std::intptr_t dif = static_cast<std::intptr_t>(sequence) -
                                static_cast<std::intptr_t>(pos);

      if (dif == 0) {
        if (mEnqueuePos.compare_exchange_weak(pos, pos + 1,
                                              std::memory_order_relaxed))
          break;
        continue;
      }
      if (dif < 0)
        return false;
      pos = mEnqueuePos.load(std::memory_order_relaxed);
    }

    RingSlot &slot = mRing[pos & mMask];
    slot.record = std::move(record);
    slot.sequence.store(pos + 1, std::memory_order_release);
    return true;
  }

  bool tryDequeueOne(QueuedLogRecord &record) {
    std::size_t pos = mDequeuePos.load(std::memory_order_relaxed);
    while (true) {
      RingSlot &slot = mRing[pos & mMask];
      const std::size_t sequence =
          slot.sequence.load(std::memory_order_acquire);
      const std::intptr_t dif = static_cast<std::intptr_t>(sequence) -
                                static_cast<std::intptr_t>(pos + 1);

      if (dif == 0) {
        if (mDequeuePos.compare_exchange_weak(pos, pos + 1,
                                              std::memory_order_relaxed))
          break;
        continue;
      }
      if (dif < 0)
        return false;
      pos = mDequeuePos.load(std::memory_order_relaxed);
    }

    RingSlot &slot = mRing[pos & mMask];
    record = std::move(slot.record);
    slot.sequence.store(pos + mCapacity, std::memory_order_release);
    return true;
  }

  bool isQueueEmpty() const {
    return mEnqueuePos.load(std::memory_order_acquire) ==
           mDequeuePos.load(std::memory_order_acquire);
  }

  void notifyDrop() {
    mDroppedRecords.fetch_add(1, std::memory_order_relaxed);
    mDropWarningPending.store(true, std::memory_order_relaxed);
    mCv.notify_one();
  }

  void emitDropWarningIfPending() {
    if (!mDropWarningPending.exchange(false, std::memory_order_relaxed))
      return;
    bool expected = false;
    if (!mDropWarningEmitted.compare_exchange_strong(expected, true,
                                                     std::memory_order_relaxed))
      return;
    std::fputs("[cudaq::qec::logger] forwarder dropped log records "
               "(queue full); increase ForwarderConfig::queueCapacity or "
               "reduce callback latency.\n",
               stderr);
  }

  // Single consumer loop that drains the ring and calls the user callback.
  // Drop warnings are emitted from this worker to keep producer hot paths free
  // of blocking I/O syscalls.
  void runWorker() {
    while (true) {
      emitDropWarningIfPending();
      // Mark callback slot in-flight before dequeue attempt so flush() cannot
      // observe (queue empty && inFlight == 0) in the tiny window between a
      // successful dequeue and callback start.
      mInFlightCallbacks.fetch_add(1, std::memory_order_acq_rel);
      QueuedLogRecord queuedRecord;
      if (!tryDequeueOne(queuedRecord)) {
        mInFlightCallbacks.fetch_sub(1, std::memory_order_acq_rel);
        mCv.notify_all();
        std::unique_lock<std::mutex> lock(mWaitMutex);
        mCv.wait(lock, [&] {
          return mStop.load(std::memory_order_acquire) || !isQueueEmpty();
        });
        if (mStop.load(std::memory_order_acquire) && isQueueEmpty())
          return;
        continue;
      }

      ForwardedLogRecord record;
      record.level = queuedRecord.level;
      record.timestampNs = queuedRecord.timestampNs;
      record.fileName.assign(queuedRecord.fileName.data(),
                             queuedRecord.fileNameLen);
      record.lineNo = queuedRecord.lineNo;
      record.message.assign(queuedRecord.message.data(),
                            queuedRecord.messageLen);
      record.threadId = queuedRecord.threadId;

      try {
        if (mConfig.callback)
          mConfig.callback(record);
      } catch (...) {
        mForwardFailures.fetch_add(1, std::memory_order_relaxed);
      }
      mInFlightCallbacks.fetch_sub(1, std::memory_order_acq_rel);

      emitDropWarningIfPending();
      mCv.notify_all();
    }
  }

  std::condition_variable_any mCv;
  mutable std::mutex mWaitMutex;
  ForwarderConfig mConfig;
  std::unique_ptr<RingSlot[]> mRing;
  std::size_t mCapacity = 0;
  std::size_t mMask = 0;
  std::atomic<std::size_t> mEnqueuePos{0};
  std::atomic<std::size_t> mDequeuePos{0};
  std::atomic<bool> mStop{false};
  std::thread mWorker;
  std::atomic<bool> mEnabled{false};
  std::atomic<std::uint64_t> mEnqueuedRecords{0};
  std::atomic<std::uint64_t> mDroppedRecords{0};
  std::atomic<std::uint64_t> mTruncatedRecords{0};
  std::atomic<std::uint64_t> mForwardFailures{0};
  std::atomic<std::uint64_t> mInFlightCallbacks{0};
  std::atomic<std::uint64_t> mActiveProducers{0};
  std::atomic<bool> mDropWarningPending{false};
  std::atomic<bool> mDropWarningEmitted{false};
  std::atomic<std::size_t> mMessageCapacity{
      kRealtimeForwarderDefaultMessageCapacity};
};

AsyncForwarder &forwarder() {
  static AsyncForwarder instance;
  return instance;
}

} // namespace

void set(ForwarderConfig config) { forwarder().set(std::move(config)); }
void clear() { forwarder().clear(); }
bool isEnabled() { return forwarder().isEnabled(); }
void resetStats() { forwarder().resetStats(); }
ForwarderStats stats() { return forwarder().stats(); }
void flush() { forwarder().flush(); }
void enqueue(QueuedLogRecord record) { forwarder().enqueue(std::move(record)); }
std::size_t messageCapacity() { return forwarder().messageCapacity(); }
void noteTruncation() { forwarder().noteTruncation(); }

} // namespace cudaq::qec::detail::forwarder_internal
