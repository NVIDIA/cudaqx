/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstddef>
#include <cstdint>

namespace cudaq::qec::decoding_server {

// Function IDs for the three decoder RPCs (FNV1a-32 of function names).
// Values match the static_asserts in decoder_rpc_ids.h; duplicated here to
// avoid a dependency on CUDAQ_REALTIME headers.
inline constexpr uint32_t kEnqueueSyndromesFunctionId = 0x7ED8BE82u;
inline constexpr uint32_t kGetCorrectionsFunctionId = 0x882D5BA1u;
inline constexpr uint32_t kResetDecoderFunctionId = 0x977A59CFu;

// Hard cap for one enqueue_syndromes request. Enforced at both transport and
// session boundaries so alternate transports cannot bypass allocation and
// packed-length validation.
inline constexpr uint64_t kMaxSyndromeBits = 1u << 20; // 1 M bits

// Wire magic bytes (from cudaq-realtime spec).
inline constexpr uint32_t kRPCRequestMagic = 0x43555152u;  // 'CUQR'
inline constexpr uint32_t kRPCResponseMagic = 0x43555153u; // 'CUQS'

// Status codes carried in RPCResponse::status.
enum class RpcStatus : int32_t {
  OK = 0,
  INVALID_DECODER = 1,
  BAD_REQUEST = 2,
  INTERNAL_ERROR = 3,
  NOT_READY = 4,
  BUSY = 5,
  SYNDROMES_DROPPED = 6,
};

// Request header — 24 bytes, packed, little-endian, no padding.
// Layout matches cudaq-realtime RPCHeader exactly.
struct __attribute__((packed)) RPCHeader {
  uint32_t magic;         ///< kRPCRequestMagic
  uint32_t function_id;   ///< FNV1a-32 of the callee name
  uint32_t arg_len;       ///< bytes of payload following this header
  uint32_t request_id;    ///< caller-assigned; echoed in response
  uint64_t ptp_timestamp; ///< PTP send timestamp in ns (0 if unused)
};
static_assert(sizeof(RPCHeader) == 24, "RPCHeader must be 24 bytes");

// Payload structs for each of the three RPCs.
// Layouts mirror decoder_rpc_ids.h without requiring CUDAQ_REALTIME headers.

struct __attribute__((packed)) EnqueuePayload {
  int64_t decoder_id;          ///< arg0
  int64_t counter;             ///< arg1
  int64_t syndrome_mapping_id; ///< arg2
  int64_t num_syndromes;       ///< arg3 (# syndrome bits)
  // Trailing: ceil(num_syndromes/8) bit-packed bytes (LSB-first), no pad
};
static_assert(sizeof(EnqueuePayload) == 32, "EnqueuePayload must be 32 bytes");

// Layout per decoder_server_runtime.md: two 8-byte scalars in schema order plus
// a trailing 1-byte bool, NO trailing padding (arg_len = 17 exactly).
struct __attribute__((packed)) GetCorrectionsPayload {
  int64_t decoder_id;  ///< arg0
  int64_t return_size; ///< arg1 (# correction bits to fetch; the cc.device_call
                       ///<       lowering serializes the OUT std::vector<bool>
                       ///<       length here)
  uint8_t reset;       ///< arg2 (1 = reset decoder after read; trailing bool,
                       ///<       no padding)
};
static_assert(sizeof(GetCorrectionsPayload) == 17,
              "GetCorrectionsPayload must be 17 bytes");

struct __attribute__((packed)) ResetPayload {
  int64_t decoder_id; ///< arg0
};
static_assert(sizeof(ResetPayload) == 8, "ResetPayload must be 8 bytes");

// Response header — 24 bytes, packed, little-endian, no padding.
// Layout matches cudaq-realtime RPCResponse exactly.
struct __attribute__((packed)) RPCResponse {
  uint32_t magic;         ///< kRPCResponseMagic
  int32_t status;         ///< RpcStatus cast to int32_t; 0 = success
  uint32_t result_len;    ///< bytes of result payload following (0 on error)
  uint32_t request_id;    ///< echoed from RPCHeader::request_id
  uint64_t ptp_timestamp; ///< echoed from RPCHeader::ptp_timestamp
};
static_assert(sizeof(RPCResponse) == 24, "RPCResponse must be 24 bytes");

// Utility: bytes required to bit-pack N bits.
constexpr size_t bit_packed_bytes(size_t num_bits) noexcept {
  return (num_bits + 7) / 8;
}

} // namespace cudaq::qec::decoding_server
