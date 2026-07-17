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
#include "cudaq/qec/realtime/decoder_rpc_wire_format.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h> // for usleep
#include <vector>

namespace cudaq::qec::decoding::rpc_producer {

namespace {

// Process-wide monotonic counter for `RPCHeader::request_id`.  The wire
// protocol echoes request_id in `RPCResponse::request_id`; today we don't
// match-by-id (we wait on the magic of the slot we wrote to), but the
// handlers still validate it's nonzero, and DEVICE_LOOP stats keys off it.
std::atomic<std::uint32_t> g_request_id_counter{1};

std::uint32_t next_request_id() {
  return g_request_id_counter.fetch_add(1, std::memory_order_relaxed);
}

// Enforces the single-producer contract documented in rpc_producer.h.  The
// producer path is single-producer by design -- the sole caller is the single-
// threaded QEC decode loop.  acquire_slot() selects a free slot without an
// atomic reservation, so two concurrent producers could pick the same slot and
// corrupt each other's RPC.  Rather than pay for full multi-producer support
// (CAS slot-claim / per-producer arena), we DETECT a contract violation and
// fail loudly.  This is a real throw, NOT assert(): release builds compile
// assert() out, so an assert would enforce nothing in production.
std::atomic<bool> g_producer_active{false};

struct single_producer_guard {
  single_producer_guard() {
    bool expected = false;
    if (!g_producer_active.compare_exchange_strong(expected, true,
                                                   std::memory_order_acquire))
      throw std::runtime_error(
          "rpc_producer: concurrent producer detected. This RPC path is "
          "single-producer (the single-threaded QEC decode loop); serialize "
          "calls or add multi-producer support (CAS slot-claim / per-producer "
          "arena).");
  }
  ~single_producer_guard() {
    g_producer_active.store(false, std::memory_order_release);
  }
  single_producer_guard(const single_producer_guard &) = delete;
  single_producer_guard &operator=(const single_producer_guard &) = delete;
};

// Bounded spin for the NEXT slot in monotonic ring order.  Returns
// UINT32_MAX on timeout.
//
// Ring discipline: the producer walks slots monotonically
// (session.producer_cursor(), advanced mod num_slots each acquire) rather
// than picking the lowest free slot.  This keeps it in lockstep with the
// strict-FIFO consumer (the device-graph scheduler / host loop, both with
// shared-ring scanning OFF), which waits at exactly its own monotonically-
// advancing cursor.  The cursor is reset to 0 by
// qec_realtime_session::initialize(), matching the consumer's reset, so both
// start at slot 0.  Back-pressure is preserved: we wait until the chosen slot
// is free (rx_flags[s]==0 AND tx_flags[s]==0 -- request consumed by the
// dispatcher and response consumed by release_slot()).
//
// THREAD-SAFETY ASSUMPTION (single producer):
// We read-modify-write producer_cursor() non-atomically, which is correct
// under the single-producer invariant documented in rpc_producer.h (the QEC
// main loop is the only producer, single-threaded by construction).  A future
// multi-producer design would make the cursor advance an atomic fetch-add
// (the natural multi-producer ring head) and add a CAS claim of the slot;
// this is deferred to a follow-up MR.
std::uint32_t acquire_slot(cudaq::qec::realtime::qec_realtime_session &session,
                           int timeout_ms) {
  volatile std::uint64_t *rx = session.rx_flags_host();
  volatile std::uint64_t *tx = session.tx_flags_host();
  const std::size_t n = session.num_slots();
  if (rx == nullptr || tx == nullptr || n == 0)
    return UINT32_MAX;
  const std::uint32_t s =
      static_cast<std::uint32_t>(session.producer_cursor() % n);
  for (int waited = 0; waited < timeout_ms; ++waited) {
    if (rx[s] == 0 && tx[s] == 0) {
      session.set_producer_cursor((s + 1u) % n);
      return s;
    }
    // 1 ms granularity matches the test's spin cadence.  The shared ring
    // is host-pinned + UVA-mapped, so the producer's view of rx/tx flags
    // is coherent with the GPU consumer's writes after a __sync_-
    // synchronize on the consumer side; usleep here keeps the busy-wait
    // off the critical path.
    usleep(1000);
  }
  return UINT32_MAX;
}

// Write an RPC request into `slot` and publish it by writing the device-
// visible slot address into rx_flags[slot].  Mirrors WriteAndSignal in the
// test; the slot is pre-acquired so the caller owns it from acquire_slot()
// through release_slot().
void write_and_signal(cudaq::qec::realtime::qec_realtime_session &session,
                      std::uint32_t slot, std::uint32_t function_id,
                      std::uint32_t request_id, const void *payload,
                      std::size_t payload_len) {
  // Two-ring wire format: requests go into the RX backing, responses
  // come back via the TX backing.  We zero just the RX slot here; the
  // TX slot is cleared lazily on release_slot() so a stale response
  // body from a previous round doesn't confound the next reader.
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
  // Address-as-flag publish: the DEVICE-visible RX slot address is what
  // the dispatcher polls for on rx_flags.  Same UVA on host and device
  // because the RX data backing is pinned+mapped.
  session.rx_flags_host()[slot] = reinterpret_cast<std::uint64_t>(
      session.rx_data_dev() + slot * session.slot_size());
}

// Bounded spin for response publication.  Returns false on timeout.  The
// response is considered complete only after the writer has produced an
// RPCResponse header and published the matching tx_flags entry.
bool wait_for_response(cudaq::qec::realtime::qec_realtime_session &session,
                       std::uint32_t slot, int timeout_ms) {
  // Two-ring wire format: the response lives in the TX slot.  The
  // dispatcher's writer (the captured graph for GRAPH_LAUNCH; the
  // DEVICE_LOOP kernel for DEVICE_CALL) writes RPCResponse and then signals
  // tx_flags[slot].  Wait for both before reading or releasing the slot.
  std::uint8_t *tx_slot_host =
      session.tx_data_host() + slot * session.slot_size();
  auto *resp =
      reinterpret_cast<const cudaq::realtime::RPCResponse *>(tx_slot_host);
  for (int waited = 0; waited < timeout_ms; ++waited) {
    __sync_synchronize();
    if (resp->magic == cudaq::realtime::RPC_MAGIC_RESPONSE &&
        session.tx_flags_host()[slot] != 0)
      return true;
    // 200us granularity matches the test.  Shorter than acquire_slot's
    // sleep because get_corrections / reset round-trips are sub-ms on
    // typical GPUs and a 1ms cadence would dominate the round-trip
    // budget for small per-shot payloads.
    usleep(200);
  }
  return false;
}

// Release a slot after the caller has finished consuming the response.
// Two-ring wire format: clears the TX backing (so the next reader of
// this slot won't see a stale `RPC_MAGIC_RESPONSE`).  The RX backing
// for this slot was already overwritten by the dispatcher (the captured
// graph) when it parsed the request; we zero it again defensively in
// write_and_signal() before the next request, so we don't need to wipe
// it here.  Clears tx_flags[slot] to unblock acquire_slot for the next
// caller.  rx_flags is cleared by the dispatcher (see
// host_dispatcher.cu::finish_slot_and_advance), not by the producer.
void release_slot(cudaq::qec::realtime::qec_realtime_session &session,
                  std::uint32_t slot) {
  std::uint8_t *tx_slot_host =
      session.tx_data_host() + slot * session.slot_size();
  std::memset(tx_slot_host, 0, session.slot_size());
  __sync_synchronize();
  session.tx_flags_host()[slot] = 0;
}

// Common pre-flight: session must be initialized + ring must be live.
// Pulled into a helper so each public function's pre-flight error is
// uniform (and so the runtime error includes the function name).
void require_initialized(cudaq::qec::realtime::qec_realtime_session &session,
                         const char *fn) {
  if (!session.initialized()) {
    std::ostringstream os;
    os << "cudaq::qec::decoding::rpc_producer::" << fn
       << ": session is not initialized().  Call qec_realtime_session::"
          "initialize() before sending RPCs.";
    throw std::runtime_error(os.str());
  }
  if (session.rx_flags_host() == nullptr ||
      session.tx_flags_host() == nullptr || session.rx_data_host() == nullptr ||
      session.rx_data_dev() == nullptr || session.tx_data_host() == nullptr ||
      session.tx_data_dev() == nullptr) {
    std::ostringstream os;
    os << "cudaq::qec::decoding::rpc_producer::" << fn
       << ": session ring buffer pointers are null even though "
          "initialized()==true.  This indicates a teardown race or a "
          "double-finalize bug.";
    throw std::runtime_error(os.str());
  }
}

} // namespace

void enqueue_syndromes(cudaq::qec::realtime::qec_realtime_session &session,
                       std::size_t decoder_id, const std::uint8_t *syndromes,
                       std::uint64_t num_syndromes, std::uint64_t tag) {
  single_producer_guard producer_guard;
  require_initialized(session, "enqueue_syndromes");

  if (syndromes == nullptr && num_syndromes > 0)
    throw std::runtime_error(
        "rpc_producer::enqueue_syndromes: syndromes == nullptr but "
        "num_syndromes > 0");

  // Build the wire payload per decoder_server_runtime.md#enqueue_syndromes:
  //   32-byte EnqueueRequestPayload (decoder_id, counter,
  //   syndrome_mapping_id, num_syndromes; all INT64)
  //   + ceil(num_syndromes/8) bit-packed syndrome bytes (LSB-first), no pad.
  const std::size_t bp_bytes =
      cudaq::qec::decoding::rpc::bit_packed_bytes(num_syndromes);
  const std::size_t body_bytes =
      sizeof(cudaq::qec::decoding::rpc::EnqueueRequestPayload) + bp_bytes;
  std::vector<std::uint8_t> payload(body_bytes, 0);
  auto *p =
      reinterpret_cast<cudaq::qec::decoding::rpc::EnqueueRequestPayload *>(
          payload.data());
  p->decoder_id = static_cast<std::int64_t>(decoder_id);
  p->counter = static_cast<std::int64_t>(tag);
  p->syndrome_mapping_id = 0;
  p->num_syndromes = static_cast<std::int64_t>(num_syndromes);
  if (num_syndromes > 0) {
    std::uint8_t *bits =
        payload.data() +
        sizeof(cudaq::qec::decoding::rpc::EnqueueRequestPayload);
    for (std::uint64_t i = 0; i < num_syndromes; ++i) {
      // Source format: one bit per byte (low bit significant), matching
      // the existing plugin / test contract.  Bit i lands at bit (i mod
      // 8) of bits[i / 8] -- LSB-first, per the spec.
      if (syndromes[i] & 0x1u)
        bits[i / 8] |=
            static_cast<std::uint8_t>(1u << static_cast<unsigned>(i % 8));
    }
  }

  std::uint32_t slot = acquire_slot(session, kAcquireSlotTimeoutMs);
  if (slot == UINT32_MAX)
    throw dispatcher_unresponsive_error(
        "rpc_producer::enqueue_syndromes: timed out acquiring a free slot");

  // request_id is for correlation. The full-width application tag travels as
  // payload arg1 (`counter`); request_id is a 32-bit window so use the low 32
  // bits of tag. When `tag = (shot << 16) | round` (the production
  // convention), this yields a unique-per-shot request_id within a 65k-shot
  // window, more than enough for in-flight correlation in DEVICE_LOOP stats /
  // debugging.
  const std::uint32_t request_id = static_cast<std::uint32_t>(tag);
  write_and_signal(session, slot,
                   cudaq::qec::decoding::rpc::kEnqueueSyndromesFunctionId,
                   request_id, payload.data(), payload.size());
  if (!wait_for_response(session, slot, kResponseTimeoutMs)) {
    release_slot(session, slot);
    std::ostringstream os;
    os << "rpc_producer::enqueue_syndromes: timed out waiting for ACK "
          "(decoder_id="
       << decoder_id << ", tag=0x" << std::hex << tag << ")";
    throw dispatcher_unresponsive_error(os.str());
  }

  std::uint8_t *tx_slot_host =
      session.tx_data_host() + slot * session.slot_size();
  const auto *resp =
      reinterpret_cast<const cudaq::realtime::RPCResponse *>(tx_slot_host);

  if (resp->status != 0) {
    const std::int32_t status = resp->status;
    release_slot(session, slot);
    std::ostringstream os;
    os << "rpc_producer::enqueue_syndromes: non-zero status (" << status
       << ") for decoder_id=" << decoder_id << ", tag=0x" << std::hex << tag;
    throw std::runtime_error(os.str());
  }
  // Per decoder_server_runtime.md the dispatcher always emits a 24-byte
  // RPCResponse, even for fire-and-forget calls; the body is empty
  // (result_len == 0).  Drop the ACK and release the slot.
  if (resp->result_len != 0) {
    const std::uint32_t got = resp->result_len;
    release_slot(session, slot);
    std::ostringstream os;
    os << "rpc_producer::enqueue_syndromes: unexpected non-empty ACK "
          "(decoder_id="
       << decoder_id << "), expected result_len=0, got " << got;
    throw std::runtime_error(os.str());
  }

  release_slot(session, slot);
}

void get_corrections(cudaq::qec::realtime::qec_realtime_session &session,
                     std::size_t decoder_id, std::uint8_t *corrections,
                     std::uint64_t correction_length, std::uint64_t reset) {
  single_producer_guard producer_guard;
  require_initialized(session, "get_corrections");

  if (corrections == nullptr && correction_length > 0)
    throw std::runtime_error(
        "rpc_producer::get_corrections: corrections == nullptr but "
        "correction_length > 0");

  // Build the wire payload per decoder_server_runtime.md#get_corrections:
  //   17 bytes total: decoder_id (INT64) + return_size (INT64, the OUT
  //   std::vector<bool> length) + reset (UINT8, trailing bool, no pad).  The
  //   struct is laid out exactly this way.
  cudaq::qec::decoding::rpc::GetCorrectionsRequestPayload payload{};
  payload.decoder_id = static_cast<std::int64_t>(decoder_id);
  payload.return_size = static_cast<std::int64_t>(correction_length);
  payload.reset = reset ? std::uint8_t{1} : std::uint8_t{0};

  std::uint32_t slot = acquire_slot(session, kAcquireSlotTimeoutMs);
  if (slot == UINT32_MAX)
    throw dispatcher_unresponsive_error(
        "rpc_producer::get_corrections: timed out acquiring a free slot");

  const std::uint32_t request_id = next_request_id();
  write_and_signal(session, slot,
                   cudaq::qec::decoding::rpc::kGetCorrectionsFunctionId,
                   request_id, &payload, sizeof(payload));

  if (!wait_for_response(session, slot, kResponseTimeoutMs)) {
    release_slot(session, slot);
    std::ostringstream os;
    os << "rpc_producer::get_corrections: timed out waiting for response "
          "(decoder_id="
       << decoder_id << ")";
    throw dispatcher_unresponsive_error(os.str());
  }

  std::uint8_t *tx_slot_host =
      session.tx_data_host() + slot * session.slot_size();
  const auto *resp =
      reinterpret_cast<const cudaq::realtime::RPCResponse *>(tx_slot_host);

  if (resp->status != 0) {
    const std::int32_t status = resp->status;
    release_slot(session, slot);
    std::ostringstream os;
    os << "rpc_producer::get_corrections: non-zero status (" << status
       << ") for decoder_id=" << decoder_id;
    throw std::runtime_error(os.str());
  }
  // Per spec result_len = ceil(R/8) exactly (no trailing pad).
  const std::size_t expected_bp =
      cudaq::qec::decoding::rpc::bit_packed_bytes(correction_length);
  if (resp->result_len != static_cast<std::uint32_t>(expected_bp)) {
    const std::uint32_t got = resp->result_len;
    release_slot(session, slot);
    std::ostringstream os;
    os << "rpc_producer::get_corrections: result_len mismatch (decoder_id="
       << decoder_id << "), expected " << expected_bp
       << " (ceil(R/8) for R=" << correction_length << "), got " << got;
    throw std::runtime_error(os.str());
  }

  if (correction_length > 0) {
    // Unpack the bit-packed result (LSB-first) into the caller's
    // byte-per-bit output buffer to preserve the API surface used by
    // realtime_decoding.cpp / the ABI seen by test code.  Bit i of the
    // correction vector lives at bit (i mod 8) of bits[i/8].
    const std::uint8_t *bits =
        tx_slot_host + sizeof(cudaq::realtime::RPCResponse);
    for (std::uint64_t i = 0; i < correction_length; ++i) {
      corrections[i] = static_cast<std::uint8_t>(
          (bits[i / 8] >> static_cast<unsigned>(i % 8)) & 0x1u);
    }
  }

  release_slot(session, slot);
}

void reset_decoder(cudaq::qec::realtime::qec_realtime_session &session,
                   std::size_t decoder_id) {
  single_producer_guard producer_guard;
  require_initialized(session, "reset_decoder");

  cudaq::qec::decoding::rpc::ResetRequestPayload payload{};
  payload.decoder_id = static_cast<std::int64_t>(decoder_id);

  std::uint32_t slot = acquire_slot(session, kAcquireSlotTimeoutMs);
  if (slot == UINT32_MAX)
    throw dispatcher_unresponsive_error(
        "rpc_producer::reset_decoder: timed out acquiring a free slot");

  const std::uint32_t request_id = next_request_id();
  write_and_signal(session, slot,
                   cudaq::qec::decoding::rpc::kResetDecoderFunctionId,
                   request_id, &payload, sizeof(payload));

  if (!wait_for_response(session, slot, kResponseTimeoutMs)) {
    release_slot(session, slot);
    std::ostringstream os;
    os << "rpc_producer::reset_decoder: timed out waiting for response "
          "(decoder_id="
       << decoder_id << ")";
    throw dispatcher_unresponsive_error(os.str());
  }

  std::uint8_t *tx_slot_host =
      session.tx_data_host() + slot * session.slot_size();
  const auto *resp =
      reinterpret_cast<const cudaq::realtime::RPCResponse *>(tx_slot_host);

  if (resp->status != 0) {
    const std::int32_t status = resp->status;
    release_slot(session, slot);
    std::ostringstream os;
    os << "rpc_producer::reset_decoder: non-zero status (" << status
       << ") for decoder_id=" << decoder_id;
    throw std::runtime_error(os.str());
  }
  // Per spec the dispatcher always emits an empty 24-byte RPCResponse for
  // fire-and-forget reset; result_len must be 0.
  if (resp->result_len != 0) {
    const std::uint32_t got = resp->result_len;
    release_slot(session, slot);
    std::ostringstream os;
    os << "rpc_producer::reset_decoder: unexpected non-empty ACK "
          "(decoder_id="
       << decoder_id << "), expected result_len=0, got " << got;
    throw std::runtime_error(os.str());
  }

  release_slot(session, slot);
}

} // namespace cudaq::qec::decoding::rpc_producer

#endif // CUDAQ_REALTIME_ROOT
