/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "ITransceiver.h"
#include "RoundAccumulator.h"
#include "cudaq/qec/decoder.h"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

namespace cudaq::qec::decoding_server {

/// A unit of work dispatched from the RpcDispatcher to a DecodingSession worker
/// thread.  The payload is an owned copy of the full frame bytes so that the
/// dispatcher can return the transport ring slot immediately after dispatch.
///
/// release_fn: propagated from RxFrame::release_fn.  It fires when the
/// WorkItem is destroyed — after the worker has processed it, or on any drop
/// path (queue full, validation failure).  On CPU/CQR/loopback paths this is
/// always null.
struct WorkItem {
  uint32_t function_id;
  std::vector<uint8_t> frame_buf; ///< RPCHeader + payload (moved from RxFrame)
  PeerId peer;                    ///< response destination
  uint32_t request_id;            ///< echoed from RPCHeader
  uint64_t ptp_timestamp;
  uint32_t vp_id;
  ITransceiver *response_transport; ///< transport the request arrived on
  ReleaseFn release_fn;             ///< null except on GPU ring-buffer path
};

/// RAII wrapper: calls decoder::release_decode_graph() on destruction.
struct GraphResourcesDeleter {
  cudaq::qec::decoder *owner = nullptr;
  void operator()(void *p) const noexcept {
    if (p && owner)
      owner->release_decode_graph(p);
  }
};
using GraphResourcesPtr = std::unique_ptr<void, GraphResourcesDeleter>;

inline constexpr size_t kDefaultQueueDepth = 64;

/// Owns one decoder instance plus a dedicated FIFO worker thread; decoder calls
/// are sequenced through the worker.
struct DecodingSession {
  enum class ShotState { collecting, result_ready, failed };

  // -- Decoder and GPU resources --
  std::unique_ptr<cudaq::qec::decoder> dec;
  GraphResourcesPtr graph_resources;
  SyndromeMappingTable mapping_table;

  // -- Round assembly --
  RoundAccumulator accumulator;

  // -- Worker thread --
  std::thread worker;
  std::queue<WorkItem> work_queue;
  std::mutex queue_mutex;
  std::condition_variable queue_cv;
  std::atomic<bool> shutdown{false};
  size_t queue_depth{kDefaultQueueDepth};

  // Latched when enqueue_syndromes is dropped (queue full). The shot remains
  // poisoned until reset_decoder clears all mutable state.
  std::atomic<bool> syndromes_dropped{false};

  // Worker-owned state for the current shot. result_ready means a decode call
  // completed; it is deliberately independent of decoder_result::converged.
  ShotState shot_state = ShotState::collecting;
  size_t accepted_syndromes = 0;

  // Per-session metrics (updated atomically by the worker thread).
  std::atomic<uint64_t> enqueue_count{0};
  std::atomic<uint64_t> get_corrections_count{0};
  std::atomic<uint64_t> reset_count{0};
  std::atomic<uint64_t> error_count{0};
  std::atomic<uint64_t> busy_count{0};
  std::atomic<uint64_t> syndromes_dropped_count{0};

  DecodingSession() = default;
  DecodingSession(const DecodingSession &) = delete;
  DecodingSession &operator=(const DecodingSession &) = delete;
  DecodingSession(DecodingSession &&) = delete;
  DecodingSession &operator=(DecodingSession &&) = delete;
  ~DecodingSession();

  /// Construct a session around an already configured decoder and capture graph
  /// resources if supported.
  static std::unique_ptr<DecodingSession>
  create(std::unique_ptr<cudaq::qec::decoder> decoder,
         SyndromeMappingTable mapping_table);

  /// Start the FIFO worker thread.  Must be called after create().
  void start_worker();

  /// Signal shutdown and join the worker (drains any queued items first).
  /// Idempotent; also called from the destructor.  DecodingServer calls this
  /// before its transports are destroyed because queued items reply through
  /// raw ITransceiver pointers.
  void stop_worker();

  /// Non-blocking enqueue.  Returns false if the work queue is full.
  bool try_enqueue(WorkItem item);

  /// Latch the syndromes_dropped flag (called by dispatcher on queue-full
  /// enqueue_syndromes; no response is sent to the client).
  void latch_syndromes_dropped();

  // -- Worker-thread-only methods --
  void on_enqueue(const WorkItem &item);
  void on_get_corrections(const WorkItem &item);
  void on_reset(const WorkItem &item);
  void worker_loop();
};

/// High-water mark of simultaneously-busy DecodingSession workers across all
/// sessions in this process (concurrency evidence for multi-logical-qubit
/// tests and server stats).
uint64_t max_concurrent_busy_sessions();

} // namespace cudaq::qec::decoding_server
