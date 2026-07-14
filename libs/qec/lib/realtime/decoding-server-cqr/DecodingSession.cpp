/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "DecodingSession.h"
#include "DecodingServer.h"
#include "RpcWireFormat.h"
#include "../../hardware_guards.h"
#include "cudaq/qec/logger.h"

#include <chrono>
#include <cstring>
#include <future>
#include <stdexcept>
#include <vector>

namespace cudaq::qec::decoding_server {

namespace {

void set_graph_capture_device(const cudaq::qec::decoder &decoder) {
  const int device = resolve_decode_device(decoder.get_cuda_device_id());
  cudaq::qec::detail_affinity::set_cuda_device_for_decode(device);
  if (device >= 0)
    CUDA_QEC_INFO(
        "DecodingSession::create: set CUDA device {} before graph capture",
        device);
}

} // namespace

// Busy high-water mark across all sessions (worker threads increment while
// executing an item).
static std::atomic<uint64_t> g_busy_sessions{0};
static std::atomic<uint64_t> g_max_busy_sessions{0};

uint64_t max_concurrent_busy_sessions() {
  return g_max_busy_sessions.load(std::memory_order_relaxed);
}

DecodingSession::~DecodingSession() { stop_worker(); }

void DecodingSession::stop_worker() {
  {
    // The store must happen under queue_mutex: worker_loop's untimed wait
    // checks the flag under the same lock, so this serializes against the
    // predicate-check-then-block window and the notify cannot be lost.
    std::lock_guard<std::mutex> lk(queue_mutex);
    shutdown.store(true, std::memory_order_release);
  }
  queue_cv.notify_one();
  if (worker.joinable())
    worker.join();
}

std::unique_ptr<DecodingSession>
DecodingSession::create(std::unique_ptr<cudaq::qec::decoder> decoder,
                        SyndromeMappingTable mapping_table_arg) {
  if (!decoder)
    throw std::invalid_argument("DecodingSession requires a decoder");

  auto s = std::make_unique<DecodingSession>();
  s->dec = std::move(decoder);

  if (s->dec->supports_graph_dispatch()) {
    set_graph_capture_device(*s->dec);
    void *gr = s->dec->capture_decode_graph();
    s->graph_resources =
        GraphResourcesPtr(gr, GraphResourcesDeleter{s->dec.get()});
  }

  s->mapping_table = std::move(mapping_table_arg);
  return s;
}

void DecodingSession::start_worker() {
  // The pin must happen ON the worker thread (CUDA device selection is
  // thread-local), but a failure is a startup error that belongs to the
  // caller: hand it back through a promise so load_from_config aborts the
  // server instead of a worker silently decoding on the wrong device.
  std::promise<void> pinned;
  auto pin_result = pinned.get_future();
  worker = std::thread([this, &pinned] {
    try {
      cudaq::qec::detail_affinity::set_cuda_device_for_decode(
          dec->get_cuda_device_id());
      pinned.set_value();
    } catch (...) {
      pinned.set_exception(std::current_exception());
      return; // never serve work from a mispinned thread
    }
    worker_loop();
  });
  try {
    pin_result.get();
  } catch (...) {
    if (worker.joinable())
      worker.join();
    throw;
  }
}

bool DecodingSession::try_enqueue(WorkItem item) {
  std::lock_guard<std::mutex> lk(queue_mutex);
  if (work_queue.size() >= queue_depth) {
    ++busy_count;
    return false;
  }
  work_queue.push(std::move(item));
  queue_cv.notify_one();
  return true;
}

void DecodingSession::latch_syndromes_dropped() {
  syndromes_dropped.store(true, std::memory_order_release);
  ++syndromes_dropped_count;
}

static void send_response(ITransceiver &transport, const PeerId &peer,
                          uint32_t request_id, uint64_t ptp_timestamp,
                          RpcStatus status,
                          const uint8_t *result_data = nullptr,
                          size_t result_len = 0) {
  std::vector<uint8_t> buf(sizeof(RPCResponse) + result_len);
  auto *hdr = reinterpret_cast<RPCResponse *>(buf.data());
  hdr->magic = kRPCResponseMagic;
  hdr->status = static_cast<int32_t>(status);
  hdr->result_len = static_cast<uint32_t>(result_len);
  hdr->request_id = request_id;
  hdr->ptp_timestamp = ptp_timestamp;
  if (result_data && result_len)
    std::memcpy(buf.data() + sizeof(RPCResponse), result_data, result_len);
  transport.send(peer, buf.data(), buf.size());
}

// Uses item.response_transport so split-transport sessions reply on the correct
// transport.
void DecodingSession::on_enqueue(const WorkItem &item) {
  ++enqueue_count;
  // No manual release_fn handling: WorkItem::release_fn is a ReleaseFn that
  // fires when the item is destroyed at the end of the worker-loop iteration,
  // covering every early return and the exception path.

  // Once an enqueue has been dropped or processing has failed, accepting more
  // fragments would make the shot's measurement history unknowable. Only a
  // full reset can establish a clean epoch again.
  if (syndromes_dropped.load(std::memory_order_acquire) ||
      shot_state == ShotState::failed)
    return;

  const size_t min_size = sizeof(RPCHeader) + sizeof(EnqueuePayload);
  if (item.frame_buf.size() < min_size) {
    ++error_count;
    shot_state = ShotState::failed;
    return; // enqueue_syndromes never sends a response
  }

  const auto *req = reinterpret_cast<const EnqueuePayload *>(
      item.frame_buf.data() + sizeof(RPCHeader));

  // enqueue_syndromes is fire-and-forget: the caller already received the
  // transport-level ACK, so a response here would be unsolicited — silently
  // dropped on CQR (no pending_ entry) and protocol-desynchronizing on
  // in-order transports.  Latch the failure; it surfaces as INTERNAL_ERROR
  // at this decoder's next get_corrections.
  if (req->num_syndromes <= 0 ||
      static_cast<uint64_t>(req->num_syndromes) > kMaxSyndromeBits) {
    ++error_count;
    shot_state = ShotState::failed;
    return;
  }
  const size_t syndrome_bytes =
      bit_packed_bytes(static_cast<size_t>(req->num_syndromes));
  if (item.frame_buf.size() < min_size + syndrome_bytes) {
    ++error_count;
    shot_state = ShotState::failed;
    return;
  }

  const uint8_t *bit_data =
      item.frame_buf.data() + sizeof(RPCHeader) + sizeof(EnqueuePayload);

  // TODO: add byte-packed compat path once compiler lowering PR lands.
  // Unpack bit-packed syndromes to byte-per-bit for the decoder.
  std::vector<uint8_t> unpacked(static_cast<size_t>(req->num_syndromes));
  for (int64_t i = 0; i < req->num_syndromes; ++i)
    unpacked[i] = (bit_data[i / 8] >> (i % 8)) & 1u;

  RoundKey key{
      .decoder_id = static_cast<uint64_t>(req->decoder_id),
      .counter = static_cast<uint64_t>(req->counter),
      .syndrome_mapping_id = static_cast<uint64_t>(req->syndrome_mapping_id),
  };

  try {
    // Any accepted input after a completed decode starts a new volume; the old
    // correction vector must not be reported as the result of that volume.
    shot_state = ShotState::collecting;
    auto completed = accumulator.ingest(key, item.vp_id, unpacked.data(),
                                        unpacked.size(), mapping_table);
    if (!completed)
      return;

    const size_t expected_syndromes = dec->get_num_msyn_per_decode();
    if (accepted_syndromes > expected_syndromes ||
        completed->bits.size() > expected_syndromes - accepted_syndromes)
      throw std::invalid_argument(
          "Syndrome volume exceeds decoder measurement capacity");

    accepted_syndromes += completed->bits.size();
    // Host-decoder path (CQR / Loopback transports).  On the gpu_roce path,
    // the CUDAQ device-graph scheduler (cudaq_create_dispatch_graph_regular)
    // handles RX→dispatch→decode→TX entirely on the GPU; this worker thread
    // is never reached for GPU RoCE sessions.
    const bool did_decode =
        dec->enqueue_syndrome(completed->bits.data(), completed->bits.size());

    if (did_decode) {
      accepted_syndromes = 0;
      shot_state = ShotState::result_ready;
    }
  } catch (const std::exception &e) {
    cudaq::qec::error("DecodingSession::on_enqueue: {}", e.what());
    ++error_count;
    // Fire-and-forget: no response carries this failure, so latch it and
    // surface it until the client establishes a clean epoch with reset.
    shot_state = ShotState::failed;
  }
}

void DecodingSession::on_get_corrections(const WorkItem &item) {
  ++get_corrections_count;

  if (item.frame_buf.size() <
      sizeof(RPCHeader) + sizeof(GetCorrectionsPayload)) {
    ++error_count;
    send_response(*item.response_transport, item.peer, item.request_id,
                  item.ptp_timestamp, RpcStatus::BAD_REQUEST);
    return;
  }

  const auto *req = reinterpret_cast<const GetCorrectionsPayload *>(
      item.frame_buf.data() + sizeof(RPCHeader));

  // Spec validation: return_size (the OUT std::vector<bool> length) must be
  // positive.
  if (req->return_size <= 0) {
    ++error_count;
    send_response(*item.response_transport, item.peer, item.request_id,
                  item.ptp_timestamp, RpcStatus::BAD_REQUEST);
    return;
  }

  if (syndromes_dropped.load(std::memory_order_acquire)) {
    send_response(*item.response_transport, item.peer, item.request_id,
                  item.ptp_timestamp, RpcStatus::SYNDROMES_DROPPED);
    return;
  }

  // Surface a sticky deferred enqueue failure from this shot. Reporting it
  // does not make partially accumulated decoder state safe to reuse.
  if (shot_state == ShotState::failed) {
    send_response(*item.response_transport, item.peer, item.request_id,
                  item.ptp_timestamp, RpcStatus::INTERNAL_ERROR);
    return;
  }

  try {
    const auto return_size = static_cast<size_t>(req->return_size);
    if (return_size != dec->get_num_observables()) {
      ++error_count;
      send_response(*item.response_transport, item.peer, item.request_id,
                    item.ptp_timestamp, RpcStatus::BAD_REQUEST);
      return;
    }
    if (shot_state != ShotState::result_ready) {
      send_response(*item.response_transport, item.peer, item.request_id,
                    item.ptp_timestamp, RpcStatus::NOT_READY);
      return;
    }
    const uint8_t *corrections = dec->get_obs_corrections();
    if (!corrections) {
      shot_state = ShotState::failed;
      send_response(*item.response_transport, item.peer, item.request_id,
                    item.ptp_timestamp, RpcStatus::INTERNAL_ERROR);
      return;
    }
    // result_len = ceil(R/8) exactly per decoder_server_runtime.md spec.
    // The spec forbids trailing padding in the wire result_len; if a transport
    // layer needs 8-byte alignment, it must add padding in its own framing.
    const size_t result_len = bit_packed_bytes(return_size);
    // get_obs_corrections() returns byte-per-bit; pack into the wire format.
    std::vector<uint8_t> packed(result_len, 0);
    for (size_t i = 0; i < return_size; ++i) {
      if (corrections[i] & 1u)
        packed[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
    }
    if (req->reset) {
      // clear_corrections (not a full reset_decoder): matches the host-path
      // semantics of get_corrections(reset=true).  Runs BEFORE the OK is
      // sent: `packed` already owns a copy of the correction bits, and a
      // throw here must produce the single INTERNAL_ERROR response below,
      // not a second response after an already-delivered OK.
      dec->clear_corrections();
      shot_state = ShotState::collecting;
    }
    send_response(*item.response_transport, item.peer, item.request_id,
                  item.ptp_timestamp, RpcStatus::OK, packed.data(), result_len);
  } catch (const std::exception &e) {
    cudaq::qec::error("DecodingSession::on_get_corrections: {}", e.what());
    ++error_count;
    shot_state = ShotState::failed;
    send_response(*item.response_transport, item.peer, item.request_id,
                  item.ptp_timestamp, RpcStatus::INTERNAL_ERROR);
  }
}

void DecodingSession::on_reset(const WorkItem &item) {
  ++reset_count;
  try {
    dec->reset_decoder();
    accumulator.clear();
    syndromes_dropped.store(false, std::memory_order_release);
    accepted_syndromes = 0;
    shot_state = ShotState::collecting;
    send_response(*item.response_transport, item.peer, item.request_id,
                  item.ptp_timestamp, RpcStatus::OK);
  } catch (const std::exception &e) {
    cudaq::qec::error("DecodingSession::on_reset: {}", e.what());
    ++error_count;
    shot_state = ShotState::failed;
    send_response(*item.response_transport, item.peer, item.request_id,
                  item.ptp_timestamp, RpcStatus::INTERNAL_ERROR);
  }
}

void DecodingSession::worker_loop() {
  while (true) {
    WorkItem item;
    {
      std::unique_lock<std::mutex> lk(queue_mutex);
      // Untimed wait: stop_worker() stores the shutdown flag under
      // queue_mutex before notifying, so the wakeup cannot be lost and no
      // 100 ms poll is needed.
      queue_cv.wait(lk, [this] {
        return !work_queue.empty() || shutdown.load(std::memory_order_acquire);
      });

      if (work_queue.empty())
        break; // woken by stop_worker() with nothing left to drain

      item = std::move(work_queue.front());
      work_queue.pop();
    }

    const uint64_t busy =
        g_busy_sessions.fetch_add(1, std::memory_order_relaxed) + 1;
    uint64_t observed = g_max_busy_sessions.load(std::memory_order_relaxed);
    while (busy > observed && !g_max_busy_sessions.compare_exchange_weak(
                                  observed, busy, std::memory_order_relaxed))
      ;

    // Last-resort containment: an exception escaping the worker thread would
    // std::terminate the whole process.  The handlers catch std::exception
    // internally, but allocations outside their try blocks (e.g. the unpacked
    // syndrome vector) and non-std exceptions from decoder plugins would
    // otherwise escape.
    try {
      if (item.function_id == kEnqueueSyndromesFunctionId)
        on_enqueue(item);
      else if (item.function_id == kGetCorrectionsFunctionId)
        on_get_corrections(item);
      else if (item.function_id == kResetDecoderFunctionId)
        on_reset(item);
    } catch (const std::exception &e) {
      cudaq::qec::error("DecodingSession worker: unhandled exception: {}",
                        e.what());
      ++error_count;
      shot_state = ShotState::failed;
    } catch (...) {
      cudaq::qec::error("DecodingSession worker: unhandled non-std exception");
      ++error_count;
      shot_state = ShotState::failed;
    }

    g_busy_sessions.fetch_sub(1, std::memory_order_relaxed);
  }
}

} // namespace cudaq::qec::decoding_server
