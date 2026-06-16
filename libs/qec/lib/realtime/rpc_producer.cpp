/*******************************************************************************
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifdef CUDAQ_REALTIME_ROOT

#include "rpc_producer.h"

#include "qec_realtime_session.h"
#include "cudaq/qec/realtime/decoder_rpc_ids.h"

#include <atomic>
#include <cstring>
#include <sstream>
#include <unistd.h>
#include <vector>

namespace cudaq::qec::decoding::rpc_producer {
namespace {

std::atomic<std::uint32_t> g_request_id_counter{1};
std::atomic<std::uint32_t> g_next_slot_hint{0};

// Producer-owned "busy" sentinel written into tx_flags while a slot is held.
// The host loop runs with skip_tx_markers=1 and never reads/writes tx_flags, so
// the producer repurposes it as an ownership token (any non-zero value works).
constexpr std::uint64_t kSlotBusyMarker = ~std::uint64_t{0};

std::uint32_t next_request_id() {
  return g_request_id_counter.fetch_add(1, std::memory_order_relaxed);
}

std::uint32_t acquire_slot(cudaq::qec::realtime::qec_realtime_session &session,
                           int timeout_ms) {
  volatile std::uint64_t *rx = session.rx_flags_host();
  volatile std::uint64_t *tx = session.tx_flags_host();
  for (int waited = 0; waited < timeout_ms; ++waited) {
    const std::uint32_t start =
        g_next_slot_hint.load(std::memory_order_relaxed) %
        static_cast<std::uint32_t>(session.num_slots());
    for (std::uint32_t s = 0; s < session.num_slots(); ++s) {
      const std::uint32_t slot =
          static_cast<std::uint32_t>((start + s) % session.num_slots());
      // A slot is free only when no request is in flight (rx == 0).  Claim it
      // by atomically flipping tx_flags 0 -> BUSY: this reserves the TX
      // response buffer for the entire acquire..release_slot window rather than
      // only until the host loop clears rx_flags, and the CAS makes the claim
      // safe against concurrent producers.  tx == 0 only happens after
      // release_slot (which runs after the host loop consumed the request and
      // cleared rx), so rx is already 0 at a successful claim.
      if (rx[slot] != 0)
        continue;
      std::uint64_t expected = 0;
      if (__atomic_compare_exchange_n(&tx[slot], &expected, kSlotBusyMarker,
                                      false, __ATOMIC_ACQ_REL,
                                      __ATOMIC_RELAXED)) {
        g_next_slot_hint.store(
            static_cast<std::uint32_t>((slot + 1) % session.num_slots()),
            std::memory_order_relaxed);
        return slot;
      }
    }
    usleep(1000);
  }
  return UINT32_MAX;
}

void require_initialized(cudaq::qec::realtime::qec_realtime_session &session,
                         const char *fn) {
  if (!session.initialized()) {
    std::ostringstream os;
    os << "rpc_producer::" << fn
       << ": session is not initialized; call configure_decoders first";
    throw std::runtime_error(os.str());
  }
}

void write_and_signal(cudaq::qec::realtime::qec_realtime_session &session,
                      std::uint32_t slot, std::uint32_t function_id,
                      std::uint32_t request_id, const void *payload,
                      std::size_t payload_len) {
  std::uint8_t *rx_slot_host =
      session.rx_data_host() + slot * session.slot_size();
  std::memset(rx_slot_host, 0, session.slot_size());
  auto *header = reinterpret_cast<cudaq::realtime::RPCHeader *>(rx_slot_host);
  header->magic = cudaq::realtime::RPC_MAGIC_REQUEST;
  header->function_id = function_id;
  header->arg_len = static_cast<std::uint32_t>(payload_len);
  header->request_id = request_id;
  header->ptp_timestamp = 0;
  std::memcpy(rx_slot_host + sizeof(cudaq::realtime::RPCHeader), payload,
              payload_len);
  __sync_synchronize();
  session.rx_flags_host()[slot] = reinterpret_cast<std::uint64_t>(rx_slot_host);
}

bool wait_for_response(cudaq::qec::realtime::qec_realtime_session &session,
                       std::uint32_t slot, int timeout_ms) {
  std::uint8_t *tx_slot_host =
      session.tx_data_host() + slot * session.slot_size();
  auto *resp =
      reinterpret_cast<const cudaq::realtime::RPCResponse *>(tx_slot_host);
  for (int waited = 0; waited < timeout_ms; ++waited) {
    std::uint32_t magic = 0;
    __atomic_load(&resp->magic, &magic, __ATOMIC_ACQUIRE);
    if (magic == cudaq::realtime::RPC_MAGIC_RESPONSE)
      return true;
    usleep(200);
  }
  return false;
}

void release_slot(cudaq::qec::realtime::qec_realtime_session &session,
                  std::uint32_t slot) {
  // rx_flags[slot] is cleared by cudaq_host_dispatcher_loop after it consumes
  // the request.  acquire_slot set tx_flags[slot] to the BUSY token to reserve
  // the response buffer; clear it here (after reading the response) to return
  // the slot to the pool.  Order the TX data wipe before the token release so a
  // re-acquiring producer never observes stale response bytes.
  std::memset(session.tx_data_host() + slot * session.slot_size(), 0,
              session.slot_size());
  __sync_synchronize();
  session.tx_flags_host()[slot] = 0;
}

const cudaq::realtime::RPCResponse *
checked_response(cudaq::qec::realtime::qec_realtime_session &session,
                 std::uint32_t slot, std::uint32_t request_id, const char *fn) {
  auto *resp = reinterpret_cast<const cudaq::realtime::RPCResponse *>(
      session.tx_data_host() + slot * session.slot_size());
  if (resp->request_id != request_id) {
    std::ostringstream os;
    os << "rpc_producer::" << fn << ": response request_id mismatch (got "
       << resp->request_id << ", expected " << request_id << ")";
    throw std::runtime_error(os.str());
  }
  if (resp->status != 0) {
    std::ostringstream os;
    os << "rpc_producer::" << fn << ": RPC status " << resp->status;
    throw std::runtime_error(os.str());
  }
  return resp;
}

} // namespace

void enqueue_syndromes(cudaq::qec::realtime::qec_realtime_session &session,
                       std::size_t decoder_id, const std::uint8_t *syndromes,
                       std::uint64_t num_syndromes, std::uint64_t counter,
                       std::uint64_t syndrome_mapping_id) {
  namespace rpc = cudaq::qec::decoding::rpc;
  require_initialized(session, "enqueue_syndromes");
  if (syndromes == nullptr && num_syndromes != 0)
    throw std::runtime_error("rpc_producer::enqueue_syndromes: null syndromes");

  const std::size_t bit_bytes = rpc::bit_packed_bytes(num_syndromes);
  const std::size_t body_bytes =
      rpc::align_to_8(sizeof(rpc::EnqueueRequestPayload) + bit_bytes);
  std::vector<std::uint8_t> payload(body_bytes, 0);
  auto *body = reinterpret_cast<rpc::EnqueueRequestPayload *>(payload.data());
  body->decoder_id = static_cast<std::int64_t>(decoder_id);
  body->counter = static_cast<std::int64_t>(counter);
  body->syndrome_mapping_id = static_cast<std::int64_t>(syndrome_mapping_id);
  body->num_syndromes = static_cast<std::int64_t>(num_syndromes);
  auto *bits = payload.data() + sizeof(rpc::EnqueueRequestPayload);
  for (std::uint64_t i = 0; i < num_syndromes; ++i) {
    if (syndromes[i] & 0x1u)
      bits[i >> 3] |= static_cast<std::uint8_t>(1u << (i & 7));
  }

  const std::uint32_t request_id = next_request_id();
  const std::uint32_t slot = acquire_slot(session, kAcquireSlotTimeoutMs);
  if (slot == UINT32_MAX)
    throw dispatcher_unresponsive_error(
        "rpc_producer::enqueue_syndromes: timed out acquiring a free slot");

  write_and_signal(session, slot, rpc::kEnqueueSyndromesFunctionId, request_id,
                   payload.data(), payload.size());
  if (!wait_for_response(session, slot, kResponseTimeoutMs)) {
    release_slot(session, slot);
    throw dispatcher_unresponsive_error(
        "rpc_producer::enqueue_syndromes: timed out waiting for response");
  }
  const cudaq::realtime::RPCResponse *resp = nullptr;
  try {
    resp = checked_response(session, slot, request_id, "enqueue_syndromes");
  } catch (...) {
    release_slot(session, slot);
    throw;
  }
  if (resp->result_len != 0) {
    release_slot(session, slot);
    throw std::runtime_error(
        "rpc_producer::enqueue_syndromes: expected empty ACK response");
  }
  release_slot(session, slot);
}

void get_corrections(cudaq::qec::realtime::qec_realtime_session &session,
                     std::size_t decoder_id, std::uint8_t *corrections,
                     std::uint64_t correction_length, std::uint64_t reset) {
  namespace rpc = cudaq::qec::decoding::rpc;
  require_initialized(session, "get_corrections");
  if (!corrections && correction_length != 0)
    throw std::runtime_error("rpc_producer::get_corrections: null corrections");

  rpc::GetCorrectionsRequestPayload payload{};
  payload.decoder_id = static_cast<std::int64_t>(decoder_id);
  payload.return_size = static_cast<std::int64_t>(correction_length);
  payload.reset = reset ? 1 : 0;

  const std::uint32_t request_id = next_request_id();
  const std::uint32_t slot = acquire_slot(session, kAcquireSlotTimeoutMs);
  if (slot == UINT32_MAX)
    throw dispatcher_unresponsive_error(
        "rpc_producer::get_corrections: timed out acquiring a free slot");

  write_and_signal(session, slot, rpc::kGetCorrectionsFunctionId, request_id,
                   &payload, sizeof(payload));
  if (!wait_for_response(session, slot, kResponseTimeoutMs)) {
    release_slot(session, slot);
    throw dispatcher_unresponsive_error(
        "rpc_producer::get_corrections: timed out waiting for response");
  }
  const cudaq::realtime::RPCResponse *resp = nullptr;
  try {
    resp = checked_response(session, slot, request_id, "get_corrections");
  } catch (...) {
    release_slot(session, slot);
    throw;
  }
  const std::size_t expected_len =
      rpc::align_to_8(rpc::bit_packed_bytes(correction_length));
  if (resp->result_len != expected_len) {
    release_slot(session, slot);
    throw std::runtime_error(
        "rpc_producer::get_corrections: malformed result_len");
  }

  const std::uint8_t *bits = session.tx_data_host() +
                             slot * session.slot_size() +
                             sizeof(cudaq::realtime::RPCResponse);
  for (std::uint64_t i = 0; i < correction_length; ++i)
    corrections[i] = (bits[i >> 3] >> (i & 7)) & 0x1u;
  release_slot(session, slot);
}

void reset_decoder(cudaq::qec::realtime::qec_realtime_session &session,
                   std::size_t decoder_id) {
  namespace rpc = cudaq::qec::decoding::rpc;
  require_initialized(session, "reset_decoder");
  rpc::ResetRequestPayload payload{};
  payload.decoder_id = static_cast<std::int64_t>(decoder_id);

  const std::uint32_t request_id = next_request_id();
  const std::uint32_t slot = acquire_slot(session, kAcquireSlotTimeoutMs);
  if (slot == UINT32_MAX)
    throw dispatcher_unresponsive_error(
        "rpc_producer::reset_decoder: timed out acquiring a free slot");

  write_and_signal(session, slot, rpc::kResetDecoderFunctionId, request_id,
                   &payload, sizeof(payload));
  if (!wait_for_response(session, slot, kResponseTimeoutMs)) {
    release_slot(session, slot);
    throw dispatcher_unresponsive_error(
        "rpc_producer::reset_decoder: timed out waiting for response");
  }
  const cudaq::realtime::RPCResponse *resp = nullptr;
  try {
    resp = checked_response(session, slot, request_id, "reset_decoder");
  } catch (...) {
    release_slot(session, slot);
    throw;
  }
  if (resp->result_len != 0) {
    release_slot(session, slot);
    throw std::runtime_error(
        "rpc_producer::reset_decoder: expected empty ACK response");
  }
  release_slot(session, slot);
}

} // namespace cudaq::qec::decoding::rpc_producer

#endif // CUDAQ_REALTIME_ROOT
