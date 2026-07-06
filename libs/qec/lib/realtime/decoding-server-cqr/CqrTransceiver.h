/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "ITransceiver.h"
#include "RpcWireFormat.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"

#include <condition_variable>
#include <cstring>
#include <deque>
#include <future>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace cudaq::qec::decoder_server {

/// Bridges CUDAQ_REALTIME DeviceCallService handler callbacks to ITransceiver.
///
/// CUDAQ calls handler functions synchronously with (rx_slot, tx_slot,
/// slot_size). inject() copies rx_slot bytes into an RxFrame, stores the
/// tx_slot pointer keyed by request_id, and blocks until DecoderServer calls
/// send() with the response — at which point send() copies the bytes to
/// tx_slot and unblocks the handler thread so CUDAQ can return.
///
/// Format translation: CUDAQ serializes stdvec<i1> as a uint64 length followed
/// by one byte per bit; inject() repacks this into the bit-packed
/// EnqueuePayload layout that DecoderServer expects.
class CqrTransceiver final : public ITransceiver {
public:
  /// Called from CUDAQ handler threads for each incoming RPC.
  /// Translates the CUDAQ-format payload to our wire format, enqueues an
  /// RxFrame, then blocks until DecoderServer sends the response.
  void inject(const void *rx_slot, void *tx_slot, std::size_t slot_size,
              uint32_t function_id);

  RxFrame recv() override;
  void send(const PeerId &peer, const uint8_t *data, std::size_t len) override;

private:
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
  // RPCHeader + EnqueuePayload + bit-packed bytes.
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
  if (!ok)
    return;

  const auto *hdr = reinterpret_cast<const RPCHeader *>(frame.buf.data());
  const uint32_t rid = hdr->request_id;

  std::future<void> fut;
  {
    std::lock_guard<std::mutex> lk(mtx_);
    auto &p = pending_[rid];
    p.tx_slot = tx_slot;
    p.slot_size = slot_size;
    fut = p.done.get_future();
    inbox_.push_back(std::move(frame));
  }
  cv_.notify_one();

  // Block until DecoderServer worker calls send() with the response.
  fut.wait();
}

inline RxFrame CqrTransceiver::recv() {
  std::unique_lock<std::mutex> lk(mtx_);
  cv_.wait(lk, [this] { return !inbox_.empty(); });
  RxFrame frame = std::move(inbox_.front());
  inbox_.pop_front();
  return frame;
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

  // Write our RPCResponse into the CUDAQ tx_slot (layouts are compatible).
  auto &p = it->second;
  const std::size_t copy_len = std::min(len, p.slot_size);
  std::memcpy(p.tx_slot, data, copy_len);
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
  if (slot_size < sizeof(cudaq::realtime::RPCHeader))
    return false;

  const auto *cqr_hdr =
      static_cast<const cudaq::realtime::RPCHeader *>(rx_slot);
  if (cqr_hdr->magic != cudaq::realtime::RPC_MAGIC_REQUEST)
    return false;

  const std::size_t arg_len = cqr_hdr->arg_len;
  const auto *payload = static_cast<const uint8_t *>(rx_slot) +
                        sizeof(cudaq::realtime::RPCHeader);

  // CUDAQ enqueue_syndromes payload:
  //   [u64 decoder_id][u64 num_syndromes][u8 bit0]...[u8 bitN-1][u64 tag]
  std::size_t off = 0;
  auto read_u64 = [&](uint64_t &v) -> bool {
    if (off + sizeof(uint64_t) > arg_len)
      return false;
    std::memcpy(&v, payload + off, sizeof(uint64_t));
    off += sizeof(uint64_t);
    return true;
  };

  uint64_t decoder_id = 0, num_syndromes = 0, tag = 0;
  if (!read_u64(decoder_id) || !read_u64(num_syndromes))
    return false;
  if (num_syndromes > arg_len - off)
    return false;
  const uint8_t *bits_src = payload + off;
  off += static_cast<std::size_t>(num_syndromes);
  if (!read_u64(tag))
    return false;

  // Build RPCHeader + EnqueuePayload + bit-packed bytes.
  const std::size_t packed_bytes = bit_packed_bytes(num_syndromes);
  out.buf.resize(sizeof(RPCHeader) + sizeof(EnqueuePayload) + packed_bytes, 0);

  auto *hdr = reinterpret_cast<RPCHeader *>(out.buf.data());
  hdr->magic = kRPCRequestMagic;
  hdr->function_id = kEnqueueSyndromesFunctionId;
  hdr->arg_len = static_cast<uint32_t>(sizeof(EnqueuePayload) + packed_bytes);
  hdr->request_id = cqr_hdr->request_id;
  hdr->ptp_timestamp = cqr_hdr->ptp_timestamp;

  auto *req =
      reinterpret_cast<EnqueuePayload *>(out.buf.data() + sizeof(RPCHeader));
  req->decoder_id = static_cast<int64_t>(decoder_id);
  req->counter = static_cast<int64_t>(tag);
  req->syndrome_mapping_id = 0; // not present in CQR path; use default
  req->num_syndromes = static_cast<int64_t>(num_syndromes);

  // Repack stdvec<i1> (one byte per bit) → bit-packed.
  uint8_t *dst = out.buf.data() + sizeof(RPCHeader) + sizeof(EnqueuePayload);
  for (uint64_t i = 0; i < num_syndromes; ++i)
    dst[i / 8] |= static_cast<uint8_t>((bits_src[i] & 1u) << (i % 8));

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

  // get_corrections and reset_decoder payloads are field-compatible with our
  // GetCorrectionsPayload / ResetPayload; copy verbatim and rewrite the header.
  const std::size_t total =
      sizeof(cudaq::realtime::RPCHeader) + cqr_hdr->arg_len;
  if (total > slot_size)
    return false;

  out.buf.resize(sizeof(RPCHeader) + cqr_hdr->arg_len);
  auto *hdr = reinterpret_cast<RPCHeader *>(out.buf.data());
  hdr->magic = kRPCRequestMagic;
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

} // namespace cudaq::qec::decoder_server
