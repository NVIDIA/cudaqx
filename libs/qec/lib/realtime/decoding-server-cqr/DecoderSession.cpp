/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "DecoderSession.h"
#include "RpcWireFormat.h"
#include "cudaq/qec/logger.h"

#include <chrono>
#include <cstring>
#include <stdexcept>

namespace cudaq::qec::decoder_server {

// Busy high-water mark across all sessions (worker threads increment while
// executing an item).
static std::atomic<uint64_t> g_busy_sessions{0};
static std::atomic<uint64_t> g_max_busy_sessions{0};

uint64_t max_concurrent_busy_sessions() {
  return g_max_busy_sessions.load(std::memory_order_relaxed);
}

DecoderSession::~DecoderSession() {
  shutdown.store(true, std::memory_order_release);
  queue_cv.notify_one();
  if (worker.joinable())
    worker.join();
}

std::unique_ptr<DecoderSession>
DecoderSession::create(const std::string &decoder_name,
                       const cudaq::qec::decoder_init &init,
                       const cudaqx::heterogeneous_map &params,
                       SyndromeMappingTable mapping_table_arg) {
  auto s = std::make_unique<DecoderSession>();
  s->dec = cudaq::qec::decoder::get(decoder_name, init, params);
  if (!s->dec)
    throw std::runtime_error("Failed to create decoder: " + decoder_name);

  if (s->dec->supports_graph_dispatch()) {
    void *gr = s->dec->capture_decode_graph();
    s->graph_resources =
        GraphResourcesPtr(gr, GraphResourcesDeleter{s->dec.get()});
  }

  s->mapping_table = std::move(mapping_table_arg);
  return s;
}

void DecoderSession::start_worker() {
  worker = std::thread([this] { worker_loop(); });
}

bool DecoderSession::try_enqueue(WorkItem item) {
  std::lock_guard<std::mutex> lk(queue_mutex);
  if (work_queue.size() >= queue_depth) {
    ++busy_count;
    return false;
  }
  work_queue.push(std::move(item));
  queue_cv.notify_one();
  return true;
}

void DecoderSession::latch_syndromes_dropped() {
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
void DecoderSession::on_enqueue(const WorkItem &item) {
  ++enqueue_count;

  const size_t min_size = sizeof(RPCHeader) + sizeof(EnqueuePayload);
  if (item.frame_buf.size() < min_size) {
    ++error_count;
    return; // enqueue_syndromes never sends a response
  }

  const auto *req = reinterpret_cast<const EnqueuePayload *>(
      item.frame_buf.data() + sizeof(RPCHeader));

  if (req->num_syndromes <= 0) {
    ++error_count;
    // enqueue_syndromes is fire-and-forget: the caller already received an ACK
    // and send_response() would be silently dropped on CQR (no pending_ entry).
    // Latch decoder_error so the next get_corrections surfaces the failure.
    decoder_error = true;
    return;
  }
  const size_t syndrome_bytes =
      bit_packed_bytes(static_cast<size_t>(req->num_syndromes));
  if (item.frame_buf.size() < min_size + syndrome_bytes) {
    ++error_count;
    decoder_error = true;
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
    auto completed = accumulator.ingest(key, item.vp_id, unpacked.data(),
                                        unpacked.size(), mapping_table);
    if (completed)
      dec->enqueue_syndrome(completed->bits.data(), completed->bits.size());
  } catch (const std::exception &e) {
    CUDA_QEC_ERROR("DecoderSession::on_enqueue: {}", e.what());
    ++error_count;
    // Fire-and-forget: no response carries this failure, so latch it and
    // surface it at this session's next get_corrections.
    decoder_error = true;
  }
}

void DecoderSession::on_get_corrections(const WorkItem &item) {
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

  if (req->return_size <= 0) {
    ++error_count;
    send_response(*item.response_transport, item.peer, item.request_id,
                  item.ptp_timestamp, RpcStatus::BAD_REQUEST);
    return;
  }

  if (syndromes_dropped.exchange(false, std::memory_order_acq_rel)) {
    send_response(*item.response_transport, item.peer, item.request_id,
                  item.ptp_timestamp, RpcStatus::SYNDROMES_DROPPED);
    return;
  }

  // Surface a sticky deferred enqueue failure from this shot.
  if (decoder_error) {
    decoder_error = false;
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
    const uint8_t *corrections = dec->get_obs_corrections();
    if (!corrections) {
      send_response(*item.response_transport, item.peer, item.request_id,
                    item.ptp_timestamp, RpcStatus::NOT_READY);
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
    send_response(*item.response_transport, item.peer, item.request_id,
                  item.ptp_timestamp, RpcStatus::OK, packed.data(), result_len);

    if (req->reset)
      dec->clear_corrections();
  } catch (const std::exception &e) {
    CUDA_QEC_ERROR("DecoderSession::on_get_corrections: {}", e.what());
    ++error_count;
    send_response(*item.response_transport, item.peer, item.request_id,
                  item.ptp_timestamp, RpcStatus::INTERNAL_ERROR);
  }
}

void DecoderSession::on_reset(const WorkItem &item) {
  ++reset_count;
  try {
    dec->reset_decoder();
    accumulator.clear();
    syndromes_dropped.store(false, std::memory_order_release);
    decoder_error = false;
    send_response(*item.response_transport, item.peer, item.request_id,
                  item.ptp_timestamp, RpcStatus::OK);
  } catch (const std::exception &e) {
    CUDA_QEC_ERROR("DecoderSession::on_reset: {}", e.what());
    ++error_count;
    send_response(*item.response_transport, item.peer, item.request_id,
                  item.ptp_timestamp, RpcStatus::INTERNAL_ERROR);
  }
}

void DecoderSession::worker_loop() {
  while (true) {
    WorkItem item;
    {
      std::unique_lock<std::mutex> lk(queue_mutex);
      // Timed wait so the shutdown flag is observed even if no WorkItem
      // arrives to trigger a notify (avoids a separate stop() notification).
      queue_cv.wait_for(lk, std::chrono::milliseconds(100), [this] {
        return !work_queue.empty() || shutdown.load(std::memory_order_acquire);
      });

      if (work_queue.empty()) {
        if (shutdown.load(std::memory_order_acquire))
          break;
        continue;
      }

      item = std::move(work_queue.front());
      work_queue.pop();
    }

    const uint64_t busy =
        g_busy_sessions.fetch_add(1, std::memory_order_relaxed) + 1;
    uint64_t observed = g_max_busy_sessions.load(std::memory_order_relaxed);
    while (busy > observed && !g_max_busy_sessions.compare_exchange_weak(
                                  observed, busy, std::memory_order_relaxed))
      ;

    if (item.function_id == kEnqueueSyndromesFunctionId)
      on_enqueue(item);
    else if (item.function_id == kGetCorrectionsFunctionId)
      on_get_corrections(item);
    else if (item.function_id == kResetDecoderFunctionId)
      on_reset(item);

    g_busy_sessions.fetch_sub(1, std::memory_order_relaxed);
  }
}

} // namespace cudaq::qec::decoder_server
