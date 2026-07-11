/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "ITransceiver.h"
#include "cudaq/qec/realtime/decoder_rpc_wire_format.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#include <condition_variable>
#include <cstring>
#include <deque>
#include <future>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace cudaq::qec::decoding_server {

using cudaq::qec::decoding::rpc::bit_packed_bytes;
using cudaq::qec::decoding::rpc::EnqueueRequestPayload;
using cudaq::qec::decoding::rpc::kEnqueueSyndromesFunctionId;
using cudaq::qec::decoding::rpc::kMaxSyndromeBits;
using cudaq::qec::decoding::rpc::RpcStatus;
using cudaq::realtime::RPCHeader;
using cudaq::realtime::RPCResponse;

namespace detail {

/// Validated view of the cudaq-realtime enqueue_syndromes request format.
struct CqrEnqueueFrameView {
  const cudaq::realtime::RPCHeader *header = nullptr;
  uint64_t decoder_id = 0;
  uint64_t counter = 0;
  uint64_t syndrome_mapping_id = 0;
  uint64_t num_syndromes = 0;
  uint64_t byte_count = 0;
  const uint8_t *packed_bits = nullptr;
};

/// Parse an enqueue request only after proving the advertised payload is
/// physically present in the supplied slot.
inline bool parse_cqr_enqueue_frame(const void *rx_slot, std::size_t slot_size,
                                    CqrEnqueueFrameView &out) {
  if (!rx_slot || slot_size < sizeof(cudaq::realtime::RPCHeader))
    return false;

  const auto *header = static_cast<const cudaq::realtime::RPCHeader *>(rx_slot);
  if (header->magic != cudaq::realtime::RPC_MAGIC_REQUEST ||
      header->function_id != kEnqueueSyndromesFunctionId)
    return false;

  const std::size_t physical_payload =
      slot_size - sizeof(cudaq::realtime::RPCHeader);
  const std::size_t arg_len = header->arg_len;
  if (arg_len > physical_payload)
    return false;

  const auto *payload = static_cast<const uint8_t *>(rx_slot) +
                        sizeof(cudaq::realtime::RPCHeader);
  std::size_t offset = 0;
  auto read_u64 = [&](uint64_t &value) {
    if (offset > arg_len || sizeof(uint64_t) > arg_len - offset)
      return false;
    std::memcpy(&value, payload + offset, sizeof(uint64_t));
    offset += sizeof(uint64_t);
    return true;
  };

  CqrEnqueueFrameView parsed;
  parsed.header = header;
  // arg3 is a std::vector<bool> (CUDAQ_TYPE_BIT_PACKED): the realtime
  // device_call ABI serialises the 4th u64 as the stdvec array-length prefix,
  // i.e. the # of logical elements = # of bits = num_syndromes. The byte count
  // is derived (ceil(bits/8)), not carried on the wire.
  if (!read_u64(parsed.decoder_id) || !read_u64(parsed.counter) ||
      !read_u64(parsed.syndrome_mapping_id) || !read_u64(parsed.num_syndromes))
    return false;

  parsed.byte_count =
      bit_packed_bytes(static_cast<std::size_t>(parsed.num_syndromes));
  if (parsed.num_syndromes == 0 || parsed.num_syndromes > kMaxSyndromeBits ||
      offset > arg_len || parsed.byte_count > arg_len - offset)
    return false;

  parsed.packed_bits = payload + offset;
  out = parsed;
  return true;
}

} // namespace detail

/// Bridges CUDAQ_REALTIME DeviceCallService handler callbacks to ITransceiver.
///
/// CUDAQ calls handler functions synchronously with (rx_slot, tx_slot,
/// slot_size), all on the SINGLE transport dispatcher thread.
///
/// Response-bearing calls (get_corrections, reset_decoder): inject() copies
/// rx_slot bytes into an RxFrame, stores the tx_slot pointer keyed by
/// request_id, and blocks until the DecodingSession worker calls send() with
/// the response — at which point send() copies the bytes to tx_slot and
/// unblocks the handler thread so CUDAQ can return.
///
/// Fire-and-forget calls (enqueue_syndromes): inject() enqueues the frame
/// and writes an immediate ACCEPTED response into tx_slot WITHOUT blocking —
/// blocking the lone dispatcher thread on decoder execution would serialize
/// every decoder's stream behind every other's, defeating the per-session
/// worker parallelism (and nothing would ever unblock it: on_enqueue sends
/// no response). A deferred decoder error is reported at that decoder's
/// next get_corrections.
///
/// Format translation: the wire frame follows decoder_server_runtime.md
/// (bit-packed syndromes with an explicit uint64 byte-count prefix on the
/// ARRAY_UINT8 argument); inject() validates it and re-frames to the
/// internal EnqueueRequestPayload layout (same fields, no byte-count prefix).
class CqrTransceiver final : public ITransceiver {
public:
  /// Called from CUDAQ handler threads for each incoming RPC.
  /// Translates the CUDAQ-format payload to our wire format, enqueues an
  /// RxFrame, then blocks until DecodingServer sends the response.
  void inject(const void *rx_slot, void *tx_slot, std::size_t slot_size,
              uint32_t function_id);

  RxFrame recv() override;
  void send(const PeerId &peer, const uint8_t *data, std::size_t len) override;
  void shutdown() override;

private:
  bool stopped_ = false;

  // Write an immediate RPCResponse (no result payload) into the CUDAQ
  // tx_slot: OK acks fire-and-forget calls; error statuses complete blocking
  // calls that will never be dispatched (rejects after shutdown).
  static void write_ack(void *tx_slot, uint32_t request_id,
                        uint64_t ptp_timestamp,
                        RpcStatus status = RpcStatus::OK);

  struct PendingTx {
    void *tx_slot;
    std::size_t slot_size;
    std::promise<void> done;
  };

  std::mutex mtx_;
  std::condition_variable cv_;
  std::deque<RxFrame> inbox_;
  std::unordered_map<uint32_t, PendingTx> pending_; // keyed by request_id

  // Translate CUDAQ enqueue_syndromes payload (stdvec<i1> format) to our
  // RPCHeader + EnqueueRequestPayload + bit-packed bytes.
  static bool build_enqueue_frame(const void *rx_slot, std::size_t slot_size,
                                  RxFrame &out);

  // For get_corrections and reset_decoder the field layouts are compatible;
  // copy rx_slot verbatim after swapping to our magic/RPCHeader type.
  static bool build_passthrough_frame(const void *rx_slot,
                                      std::size_t slot_size, uint32_t fn_id,
                                      RxFrame &out);
};

// ---------------------------------------------------------------------------
// Inline implementation
// ---------------------------------------------------------------------------

inline void CqrTransceiver::inject(const void *rx_slot, void *tx_slot,
                                   std::size_t slot_size,
                                   uint32_t function_id) {
  if (!rx_slot || !tx_slot || slot_size < sizeof(RPCHeader))
    return;

  RxFrame frame;
  bool ok =
      (function_id == kEnqueueSyndromesFunctionId)
          ? build_enqueue_frame(rx_slot, slot_size, frame)
          : build_passthrough_frame(rx_slot, slot_size, function_id, frame);
  if (!ok) {
    // For blocking calls (get_corrections, reset_decoder) an unwritten tx_slot
    // would stall the CUDAQ dispatcher indefinitely.  Write BAD_REQUEST so it
    // gets a valid magic word regardless of call type.
    const auto *cqr = static_cast<const cudaq::realtime::RPCHeader *>(rx_slot);
    auto *resp = static_cast<cudaq::realtime::RPCResponse *>(tx_slot);
    resp->status = static_cast<int32_t>(RpcStatus::BAD_REQUEST);
    resp->result_len = 0;
    resp->request_id = cqr->request_id;
    resp->ptp_timestamp = cqr->ptp_timestamp;
    __atomic_store_n(reinterpret_cast<uint32_t *>(tx_slot),
                     cudaq::realtime::RPC_MAGIC_RESPONSE, __ATOMIC_RELEASE);
    return;
  }

  const auto *hdr = reinterpret_cast<const RPCHeader *>(frame.buf.data());
  const uint32_t rid = hdr->request_id;
  const uint64_t ptp = hdr->ptp_timestamp; // save before frame is moved

  if (function_id == kEnqueueSyndromesFunctionId) {
    // Fire-and-forget: hand the frame to the server and ACK immediately
    // (status OK = ACCEPTED) -- the dispatcher thread must not park on
    // decoder execution, and per the spec the dispatcher still emits an
    // RPCResponse into the tx_slot (the transport needs it to complete the
    // slot; the caller drops it). A deferred decoder error is reported at
    // this decoder's next get_corrections.
    {
      std::lock_guard<std::mutex> lk(mtx_);
      inbox_.push_back(std::move(frame));
    }
    cv_.notify_one();
    write_ack(tx_slot, rid, ptp);
    return;
  }

  std::future<void> fut;
  {
    std::lock_guard<std::mutex> lk(mtx_);
    // Reject new blocking RPCs after shutdown: the recv loop is exiting and
    // will never dispatch this frame, so parking on the promise would hang
    // the CUDAQ dispatcher thread forever.  Complete the slot immediately.
    if (stopped_) {
      write_ack(tx_slot, rid, ptp, RpcStatus::BAD_REQUEST);
      return;
    }
    auto &p = pending_[rid];
    p.tx_slot = tx_slot;
    p.slot_size = slot_size;
    fut = p.done.get_future();
    inbox_.push_back(std::move(frame));
  }
  cv_.notify_one();

  // Block until the DecodingSession worker calls send() with the response.
  fut.wait();
}

inline RxFrame CqrTransceiver::recv() {
  std::unique_lock<std::mutex> lk(mtx_);
  cv_.wait(lk, [this] { return !inbox_.empty() || stopped_; });
  if (inbox_.empty())
    return {}; // shutdown sentinel (empty buf)
  RxFrame frame = std::move(inbox_.front());
  inbox_.pop_front();
  return frame;
}

inline void CqrTransceiver::shutdown() {
  // Move out all in-flight pending entries under the lock, then complete
  // them outside it.  The recv loop exits without draining inbox_, so a
  // frame that inject() already queued would otherwise leave its handler
  // thread parked in fut.wait() forever.  Write BAD_REQUEST into each
  // tx_slot (the CUDAQ transport needs a valid response to complete the
  // slot) and fulfill the promise to unblock the waiter.
  std::unordered_map<uint32_t, PendingTx> drained;
  {
    std::lock_guard<std::mutex> lk(mtx_);
    stopped_ = true;
    drained = std::move(pending_);
    pending_.clear();
  }
  cv_.notify_all();
  for (auto &[rid, p] : drained) {
    write_ack(p.tx_slot, rid, /*ptp_timestamp=*/0, RpcStatus::BAD_REQUEST);
    p.done.set_value();
  }
}

inline void CqrTransceiver::write_ack(void *tx_slot, uint32_t request_id,
                                      uint64_t ptp_timestamp,
                                      RpcStatus status) {
  auto *resp = static_cast<cudaq::realtime::RPCResponse *>(tx_slot);
  resp->status = static_cast<int32_t>(status);
  resp->result_len = 0;
  resp->request_id = request_id;
  resp->ptp_timestamp = ptp_timestamp;
  // Publish the magic last (release store) so the CUDAQ runtime sees a
  // complete response before observing the magic word.
  __atomic_store_n(reinterpret_cast<uint32_t *>(tx_slot),
                   cudaq::realtime::RPC_MAGIC_RESPONSE, __ATOMIC_RELEASE);
}

inline void CqrTransceiver::send(const PeerId & /*peer*/, const uint8_t *data,
                                 std::size_t len) {
  if (!data || len < sizeof(RPCResponse))
    return;

  const auto *resp = reinterpret_cast<const RPCResponse *>(data);
  const uint32_t rid = resp->request_id;

  std::lock_guard<std::mutex> lk(mtx_);
  auto it = pending_.find(rid);
  if (it == pending_.end())
    return;

  auto &p = it->second;
  if (len > p.slot_size) {
    // Truncating would leave result_len advertising bytes that were never
    // written, so the client would read stale slot memory as correction
    // bits.  Fail the RPC explicitly instead (the pre-decoding-server code
    // returned result-buffer-too-small here).
    write_ack(p.tx_slot, rid, resp->ptp_timestamp, RpcStatus::INTERNAL_ERROR);
    p.done.set_value();
    pending_.erase(it);
    return;
  }

  // Write our RPCResponse into the CUDAQ tx_slot (layouts are compatible).
  std::memcpy(p.tx_slot, data, len);
  // Publish the magic last (release store) so the CUDAQ runtime sees a
  // complete response before observing the magic word.
  __atomic_store_n(reinterpret_cast<uint32_t *>(p.tx_slot),
                   cudaq::realtime::RPC_MAGIC_RESPONSE, __ATOMIC_RELEASE);

  p.done.set_value();
  pending_.erase(it);
}

inline bool CqrTransceiver::build_enqueue_frame(const void *rx_slot,
                                                std::size_t slot_size,
                                                RxFrame &out) {
  // Spec 5-arg wire format (decoder_server_runtime.md):
  //   [u64 decoder_id][u64 counter][u64 syndrome_mapping_id]
  //   [u64 num_syndromes][u64 array_len][u8 x ceil(bits/8) (bit-packed)]
  // syndrome_bits is a std::vector<bool> (CUDAQ_TYPE_BIT_PACKED); CUDAQ
  // serialises it as [u64 array_len = # logical bits][packed bytes].
  detail::CqrEnqueueFrameView request;
  if (!detail::parse_cqr_enqueue_frame(rx_slot, slot_size, request))
    return false;

  // Re-frame to RPCHeader + EnqueueRequestPayload + bit-packed bytes (the
  // internal layout drops the byte-count prefix; the bits stay packed as-is).
  out.buf.resize(sizeof(RPCHeader) + sizeof(EnqueueRequestPayload) +
                 request.byte_count);

  auto *hdr = reinterpret_cast<RPCHeader *>(out.buf.data());
  hdr->magic = cudaq::realtime::RPC_MAGIC_REQUEST;
  hdr->function_id = kEnqueueSyndromesFunctionId;
  hdr->arg_len =
      static_cast<uint32_t>(sizeof(EnqueueRequestPayload) + request.byte_count);
  hdr->request_id = request.header->request_id;
  hdr->ptp_timestamp = request.header->ptp_timestamp;

  auto *req = reinterpret_cast<EnqueueRequestPayload *>(out.buf.data() +
                                                        sizeof(RPCHeader));
  req->decoder_id = static_cast<int64_t>(request.decoder_id);
  req->counter = static_cast<int64_t>(request.counter);
  req->syndrome_mapping_id = static_cast<int64_t>(request.syndrome_mapping_id);
  req->num_syndromes = static_cast<int64_t>(request.num_syndromes);

  uint8_t *dst =
      out.buf.data() + sizeof(RPCHeader) + sizeof(EnqueueRequestPayload);
  std::memcpy(dst, request.packed_bits, request.byte_count);

  out.vp_id = 0;
  return true;
}

inline bool CqrTransceiver::build_passthrough_frame(const void *rx_slot,
                                                    std::size_t slot_size,
                                                    uint32_t fn_id,
                                                    RxFrame &out) {
  if (slot_size < sizeof(cudaq::realtime::RPCHeader))
    return false;

  const auto *cqr_hdr =
      static_cast<const cudaq::realtime::RPCHeader *>(rx_slot);
  if (cqr_hdr->magic != cudaq::realtime::RPC_MAGIC_REQUEST)
    return false;

  // get_corrections and reset_decoder payloads already match
  // GetCorrectionsRequestPayload / ResetRequestPayload; copy verbatim and
  // rewrite the header.
  const std::size_t total =
      sizeof(cudaq::realtime::RPCHeader) + cqr_hdr->arg_len;
  if (total > slot_size)
    return false;

  out.buf.resize(sizeof(RPCHeader) + cqr_hdr->arg_len);
  auto *hdr = reinterpret_cast<RPCHeader *>(out.buf.data());
  hdr->magic = cudaq::realtime::RPC_MAGIC_REQUEST;
  hdr->function_id = fn_id;
  hdr->arg_len = cqr_hdr->arg_len;
  hdr->request_id = cqr_hdr->request_id;
  hdr->ptp_timestamp = cqr_hdr->ptp_timestamp;

  if (cqr_hdr->arg_len > 0)
    std::memcpy(out.buf.data() + sizeof(RPCHeader),
                static_cast<const uint8_t *>(rx_slot) +
                    sizeof(cudaq::realtime::RPCHeader),
                cqr_hdr->arg_len);
  out.vp_id = 0;
  return true;
}

} // namespace cudaq::qec::decoding_server
