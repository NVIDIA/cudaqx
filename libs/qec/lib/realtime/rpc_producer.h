/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace cudaq::qec::realtime {
class qec_realtime_session;
} // namespace cudaq::qec::realtime

namespace cudaq::qec::decoding::rpc_producer {

/// @brief Thrown when an RPC round-trip cannot make progress because the
/// dispatcher is unresponsive -- either no free slot appears within
/// `kAcquireSlotTimeoutMs`, or no `RPCResponse` appears within
/// `kResponseTimeoutMs`.  Distinct from `std::runtime_error` so the host-side
/// caller (realtime_decoding.cpp) can recognize it as fatal-to-the-session
/// (the device kernel / host monitor has stopped servicing the ring) and tear
/// the session down rather than retry into a slow ghost-slot leak.  All other
/// producer errors (bad args, non-zero RPC status, malformed result_len)
/// stay plain `std::runtime_error`.
struct dispatcher_unresponsive_error : std::runtime_error {
  using std::runtime_error::runtime_error;
};

//==============================================================================
// Host-side RPC producer for the inproc_rpc realtime decoding path.
//
// These three functions are the host-process counterparts to the three RPC
// handlers wired up by qec_realtime_session + decoder_rpc_dispatch.cu, all
// conformant with proposals/decoder_server_runtime.md:
//   - enqueue_syndromes  GRAPH_LAUNCH; function_id ==
//   fnv1a("enqueue_syndromes")
//                        == kEnqueueSyndromesFunctionId; one captured graph
//                        per decoder; host monitor sub-routes by
//                        (function_id, routing_key=decoder_id).
//   - get_corrections    DEVICE_CALL; function_id == fnv1a("get_corrections")
//                        == kGetCorrectionsFunctionId.
//   - reset_decoder      DEVICE_CALL; function_id == fnv1a("reset_decoder")
//                        == kResetDecoderFunctionId.
//
// All three RPCs are request/response on the wire: the dispatcher always
// emits a 24-byte RPCResponse (status=0, result_len=0 for the two fire-
// and-forget calls; bit-packed correction bytes + 0..7 pad for get_-
// corrections).  Producers wait for the ACK and drop the response body
// for enqueue/reset.
//
// Each function:
//   1. Resolves the per-process session via the `session` argument (no global
//      lookup) so test code can drive a private session without touching
//      `g_realtime_session`.
//   2. Acquires a free RX slot from the ring buffer.
//   3. Writes RPCHeader + the matching wire-format payload from
//      decoder_rpc_wire_format.h.  Payload scalars are INT64; bools are
//      UINT8+pad; bit-packed arrays are LSB-first; whole payload is padded to
//      an 8-byte multiple.
//   4. Publishes the slot by writing the DEVICE-visible RX slot address
//      into rx_flags[slot] (the "address-as-flag" convention of the shared
//      ring; both the HOST_LOOP and DEVICE_LOOP dispatchers expect this).
//   5. Spins on RPCResponse::magic, checks status, and (for
//      get_corrections) reads back the bit-packed result.
//   6. Releases the slot (clears rx_flags + tx_flags + the slot's first
//      bytes so the response magic doesn't get misread on slot reuse).
//
// On any error (no slot within timeout_ms, response timeout, non-zero
// RPC status, malformed result_len), the corresponding function throws
// `std::runtime_error` with a contextual message.  The host-side caller in
// the production path (realtime_decoding.cpp::enqueue_syndromes /
// get_corrections / reset) is also a free function in
// `cudaq::qec::decoding::host`, so it gets to choose how to surface the
// error -- e.g. by logging then rethrowing.
//
// THREAD-SAFETY:
// All three functions are SINGLE-PRODUCER -- this is a hard contract, not
// merely an assumption.  The production caller in realtime_decoding.cpp::
// enqueue_syndromes / get_corrections / reset_decoder is invoked from the QEC
// main loop, which is single-threaded by construction.  The contract is now
// ENFORCED at runtime: each function holds a single_producer_guard that throws
// if a second producer is active concurrently (an always-on check -- a real
// throw, not assert(), so it stays active in release builds).  Full multi-
// producer support (CAS slot-claim or a per-producer arena) remains a
// deliberate follow-up.
//
// acquire_slot() in rpc_producer.cpp scans for a free slot (rx_flags ==
// tx_flags == 0) but does NOT atomically claim it on return.  That's
// fine while there's a single producer (the chosen slot is written
// before the next call to acquire_slot()), but it would race if a
// second producer thread were ever introduced.  Multi-producer support
// is a deliberate follow-up: it requires either (a) a CAS-based atomic
// claim on the rx_flags slot ("0 -> in-progress") inside acquire_slot,
// or (b) a per-producer arena of slots in the ring.  See
// `acquire_slot()` for the inline assumption comment.
//
// As a separate constraint, the *handlers* themselves are not
// re-entrant per-decoder (the plugin's GpuDecoderState is shared
// across in-flight enqueue rounds for that decoder_id), so the
// wire-level contract is "one in-flight RPC per (decoder_id,
// function_id) on the wire at a time" regardless of producer count.
//==============================================================================

/// @brief Send a per-round enqueue RPC for `decoder_id` carrying
/// `num_syndromes` syndrome bits, bit-packed LSB-first into the wire payload.
///
/// The session holds N GRAPH_LAUNCH entries that all share the canonical
/// `kEnqueueSyndromesFunctionId`; the host monitor disambiguates them by
/// `routing_key = decoder_id` (see proposals/cudaq_realtime_host_api.bs#
/// host-path-graph-routing-key).  The dispatcher always emits a 24-byte
/// `RPCResponse` with `status=0, result_len=0` (no body); this producer
/// waits for the ACK and drops it.
///
/// @param session       Realtime session.  Must be `initialized()`.
/// @param decoder_id    Routing key.  Written into payload arg0 AND
///                      matched against the function table's
///                      `routing_key` field.
/// @param syndromes     Pointer to `syndrome_length` raw syndrome bytes,
///                      one bit-per-byte at the source (each input byte
///                      contributes one bit to the wire-side bit-packed
///                      array).
/// @param num_syndromes Number of syndrome BITS for this round
///                      (== `syndrome_length`).  Written into payload
///                      arg3.
/// @param tag           Application-level breadcrumb written into
///                      `RPCHeader::request_id` (low 32 bits) and payload
///                      arg1 (`counter`, full 64 bits).  This first pass
///                      always emits payload arg2 (`syndrome_mapping_id`) as
///                      0 for contiguous identity mapping.  Production callers
///                      typically pack `(shot << 16) | round` into `tag`.
__attribute__((visibility("default"))) void
enqueue_syndromes(cudaq::qec::realtime::qec_realtime_session &session,
                  std::size_t decoder_id, const std::uint8_t *syndromes,
                  std::uint64_t num_syndromes, std::uint64_t tag);

/// @brief Fetch `correction_length` correction bytes for `decoder_id` and
/// optionally reset the device-side accumulated correction buffer.
///
/// Handler is the shared DEVICE_CALL `get_corrections_ui64`; routing happens
/// in `decoder_rpc_dispatch.cu` based on `decoder_id` in the payload.
///
/// @param session           Realtime session.  Must be `initialized()`.
/// @param decoder_id        Decoder index for state lookup.
/// @param corrections       Output buffer (caller-owned, at least
///                          `correction_length` bytes).
/// @param correction_length # of correction bytes to read.  Must match the
///                          decoder's declared num_observables.
/// @param reset             1 to zero the device-side correction buffer
///                          after the read, 0 to leave it accumulated.
__attribute__((visibility("default"))) void
get_corrections(cudaq::qec::realtime::qec_realtime_session &session,
                std::size_t decoder_id, std::uint8_t *corrections,
                std::uint64_t correction_length, std::uint64_t reset);

/// @brief Send a reset RPC, clearing per-decoder device-side state
/// (correction buffer + the plugin's BP context).
///
/// Handler is the shared DEVICE_CALL `reset_decoder_ui64`.
///
/// @param session     Realtime session.  Must be `initialized()`.
/// @param decoder_id  Decoder index for state lookup.
__attribute__((visibility("default"))) void
reset_decoder(cudaq::qec::realtime::qec_realtime_session &session,
              std::size_t decoder_id);

//==============================================================================
// Spin/timeout knobs (visible so tests can shorten timeouts for negative
// fixtures without touching the production defaults).
//==============================================================================

/// @brief Max time (ms) `AcquireSlot` will spin waiting for a free slot
/// before throwing.
constexpr int kAcquireSlotTimeoutMs = 5000;

/// @brief Max time (ms) `WaitForResponse` will spin waiting for
/// `RPCResponse::magic` before throwing.
constexpr int kResponseTimeoutMs = 5000;

} // namespace cudaq::qec::decoding::rpc_producer
