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
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

namespace cudaq::qec::decoder_server {

/// Byte vector of observable correction bits returned by get_corrections.
using CorrectionBits = std::vector<uint8_t>;

/// A unit of work dispatched from the RpcDispatcher to a DecoderSession worker
/// thread.  The payload is an owned copy of the full frame bytes so that the
/// dispatcher can return the transport ring slot immediately after dispatch.
/// Zero-copy (holding a span into the ring buffer) is a future optimization;
/// if adopted, release(frame) must move to after the worker consumes the
/// payload.
struct WorkItem {
  uint32_t function_id;
  std::vector<uint8_t> frame_buf; ///< RPCHeader + payload (moved from RxFrame)
  PeerId peer;                    ///< response destination
  uint32_t request_id;            ///< echoed from RPCHeader
  uint64_t ptp_timestamp;
  uint32_t vp_id;
  ITransceiver *response_transport; ///< transport the request arrived on
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
struct DecoderSession {
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

  // Latched when enqueue_syndromes is dropped (queue full).
  // Cleared by the worker after surfacing the error in get_corrections.
  std::atomic<bool> syndromes_dropped{false};

  // Sticky deferred-execution error: a decoder exception during on_enqueue
  // is latched here (enqueue is fire-and-forget, so there is no response to
  // carry it) and surfaced as INTERNAL_ERROR at this session's next
  // get_corrections. Cleared by on_reset. Worker-thread-only.
  bool decoder_error = false;

  // Per-session metrics (updated atomically by the worker thread).
  std::atomic<uint64_t> enqueue_count{0};
  std::atomic<uint64_t> get_corrections_count{0};
  std::atomic<uint64_t> reset_count{0};
  std::atomic<uint64_t> error_count{0};
  std::atomic<uint64_t> busy_count{0};
  std::atomic<uint64_t> syndromes_dropped_count{0};

  DecoderSession() = default;
  DecoderSession(const DecoderSession &) = delete;
  DecoderSession &operator=(const DecoderSession &) = delete;
  DecoderSession(DecoderSession &&) = delete;
  DecoderSession &operator=(DecoderSession &&) = delete;
  ~DecoderSession();

  /// Construct a session on the heap: create the decoder, capture graph
  /// resources if supported.
  static std::unique_ptr<DecoderSession>
  create(const std::string &decoder_name, const cudaq::qec::decoder_init &init,
         const cudaqx::heterogeneous_map &params,
         SyndromeMappingTable mapping_table);

  /// Start the FIFO worker thread.  Must be called after create().
  void start_worker();

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

/// High-water mark of simultaneously-busy DecoderSession workers across all
/// sessions in this process (concurrency evidence for multi-logical-qubit
/// tests and daemon stats).
uint64_t max_concurrent_busy_sessions();

} // namespace cudaq::qec::decoder_server
