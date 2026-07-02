/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq_internal/device_call/DeviceCallService.h"
#include "../realtime_decoding.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {

using cudaq_internal::device_call::DeviceCallHostDispatchTable;
using cudaq_internal::device_call::DeviceCallService;
using cudaq_internal::device_call::DeviceCallServicePluginInfo;

// Realtime function ids are fnv1a_32 of the kernel-facing callee name, matching
// the generic device_call targets emitted by the
// cudaq-qec-realtime-decoding-simulation-cqr device wrappers (and the names
// defined in decoder_server_runtime.md). All three are extern "C", so no name
// mangling is involved.
constexpr std::uint32_t kEnqueueSyndromesFnId =
    cudaq::realtime::fnv1a_hash("enqueue_syndromes");
constexpr std::uint32_t kGetCorrectionsFnId =
    cudaq::realtime::fnv1a_hash("get_corrections");
constexpr std::uint32_t kResetDecoderFnId =
    cudaq::realtime::fnv1a_hash("reset_decoder");

constexpr std::int32_t kStatusSuccess = 0;
constexpr std::int32_t kStatusInvalidRequest = -1;
constexpr std::int32_t kStatusHandlerException = -2;
constexpr std::int32_t kStatusPayloadTooLarge = -3;
constexpr std::int32_t kStatusResultBufferTooSmall = -5;

constexpr std::uint32_t kHostDispatchDeviceId = 0;
constexpr std::uint32_t kScalarElementCount = 1;

constexpr std::uint8_t kNoResults = 0;
constexpr std::uint8_t kSingleResult = 1;

constexpr std::uint8_t kEnqueueDecoderIdArg = 0;
constexpr std::uint8_t kEnqueueSyndromesArg = 1;
constexpr std::uint8_t kEnqueueTagArg = 2;
constexpr std::uint8_t kEnqueueArgCount = 3;

constexpr std::uint8_t kGetCorrectionsDecoderIdArg = 0;
constexpr std::uint8_t kGetCorrectionsLengthArg = 1;
constexpr std::uint8_t kGetCorrectionsResetArg = 2;
constexpr std::uint8_t kGetCorrectionsArgCount = 3;

constexpr std::uint8_t kResetDecoderIdArg = 0;
constexpr std::uint8_t kResetDecoderArgCount = 1;

constexpr std::uint8_t kCorrectionsResult = 0;

constexpr std::uint8_t kScalarU8Size = sizeof(std::uint8_t);
constexpr std::uint8_t kScalarU64Size = sizeof(std::uint64_t);
constexpr std::size_t kMaxRealtimeVectorLength = 64;

struct ByteSpan {
  const std::uint8_t *data = nullptr;
  std::uint64_t size = 0;
};

struct EnqueueSyndromesRequest {
  std::uint64_t decoder_id = 0;
  ByteSpan syndromes;
  std::uint64_t tag = 0;
};

struct GetCorrectionsRequest {
  std::uint64_t decoder_id = 0;
  std::uint64_t correction_length = 0;
  bool reset = false;
};

enum DeviceCallEntryIndex : std::size_t {
  kEnqueueSyndromesEntry,
  kGetCorrectionsEntry,
  kResetDecoderEntry,
  kDeviceCallEntryCount
};

bool align_offset(std::size_t &offset, std::size_t alignment,
                  std::size_t arg_len) {
  if (alignment <= 1)
    return offset <= arg_len;
  const auto addend = alignment - 1;
  if (offset > std::numeric_limits<std::size_t>::max() - addend)
    return false;
  offset = (offset + addend) & ~addend;
  return offset <= arg_len;
}

template <typename T>
bool read_scalar(const std::uint8_t *payload, std::size_t arg_len,
                 std::size_t &offset, T &value) {
  if (!align_offset(offset, alignof(T), arg_len) ||
      sizeof(T) > arg_len - offset)
    return false;
  std::memcpy(&value, payload + offset, sizeof(T));
  offset += sizeof(T);
  return true;
}

bool read_stdvec_i1(const std::uint8_t *payload, std::size_t arg_len,
                    std::size_t &offset, ByteSpan &span) {
  // CUDA-Q realtime lowering serializes stdvec<i1> as a uint64 length followed
  // by one byte per bool element.
  std::uint64_t length = 0;
  if (!read_scalar(payload, arg_len, offset, length))
    return false;
  if (length > static_cast<std::uint64_t>(arg_len - offset))
    return false;
  span = {payload + offset, length};
  offset += static_cast<std::size_t>(length);
  return true;
}

// Counts validated requests this host-dispatch service has handled. Lets a test
// confirm the device_call actually traversed the host-dispatch ring to the
// service (HOP1) rather than resolving to a direct host trampoline. Exposed via
// cudaqx_qec_device_call_dispatch_count() below.
std::atomic<std::uint64_t> g_service_dispatch_count{0};

bool read_request_payload(const void *rx_slot, std::size_t slot_size,
                          const cudaq::realtime::RPCHeader *&request,
                          const std::uint8_t *&payload, std::size_t &arg_len) {
  if (!rx_slot || slot_size < sizeof(cudaq::realtime::RPCHeader))
    return false;

  request = static_cast<const cudaq::realtime::RPCHeader *>(rx_slot);
  if (request->magic != cudaq::realtime::RPC_MAGIC_REQUEST)
    return false;

  arg_len = request->arg_len;
  if (arg_len > slot_size - sizeof(cudaq::realtime::RPCHeader))
    return false;

  payload = static_cast<const std::uint8_t *>(rx_slot) +
            sizeof(cudaq::realtime::RPCHeader);
  g_service_dispatch_count.fetch_add(1, std::memory_order_relaxed);
  return true;
}

bool read_enqueue_syndromes_request(const void *rx_slot, std::size_t slot_size,
                                    EnqueueSyndromesRequest &out) {
  const cudaq::realtime::RPCHeader *request = nullptr;
  const std::uint8_t *payload = nullptr;
  std::size_t arg_len = 0;
  if (!read_request_payload(rx_slot, slot_size, request, payload, arg_len))
    return false;

  std::size_t offset = 0;
  return read_scalar(payload, arg_len, offset, out.decoder_id) &&
         read_stdvec_i1(payload, arg_len, offset, out.syndromes) &&
         read_scalar(payload, arg_len, offset, out.tag) && offset == arg_len;
}

bool read_get_corrections_request(const void *rx_slot, std::size_t slot_size,
                                  GetCorrectionsRequest &out) {
  const cudaq::realtime::RPCHeader *request = nullptr;
  const std::uint8_t *payload = nullptr;
  std::size_t arg_len = 0;
  if (!read_request_payload(rx_slot, slot_size, request, payload, arg_len))
    return false;

  std::size_t offset = 0;
  std::uint8_t reset = 0;
  if (!read_scalar(payload, arg_len, offset, out.decoder_id) ||
      !read_scalar(payload, arg_len, offset, out.correction_length) ||
      !read_scalar(payload, arg_len, offset, reset) || offset != arg_len)
    return false;

  out.reset = reset != 0;
  return true;
}

bool read_reset_decoder_request(const void *rx_slot, std::size_t slot_size,
                                std::uint64_t &decoder_id) {
  const cudaq::realtime::RPCHeader *request = nullptr;
  const std::uint8_t *payload = nullptr;
  std::size_t arg_len = 0;
  if (!read_request_payload(rx_slot, slot_size, request, payload, arg_len))
    return false;

  std::size_t offset = 0;
  return read_scalar(payload, arg_len, offset, decoder_id) && offset == arg_len;
}

void write_response(void *tx_slot, const void *rx_slot, std::int32_t status,
                    std::uint32_t result_len = 0) {
  const auto *request =
      static_cast<const cudaq::realtime::RPCHeader *>(rx_slot);
  auto *response = static_cast<cudaq::realtime::RPCResponse *>(tx_slot);
  response->status = status;
  response->result_len = result_len;
  response->request_id = request ? request->request_id : 0;
  response->ptp_timestamp = request ? request->ptp_timestamp : 0;
  __atomic_store_n(&response->magic, cudaq::realtime::RPC_MAGIC_RESPONSE,
                   __ATOMIC_RELEASE);
}

void enqueue_syndromes_host(const void *rx_slot, void *tx_slot,
                            std::size_t slot_size) {
  try {
    EnqueueSyndromesRequest request;
    if (!tx_slot ||
        !read_enqueue_syndromes_request(rx_slot, slot_size, request)) {
      if (tx_slot && rx_slot)
        write_response(tx_slot, rx_slot, kStatusInvalidRequest);
      return;
    }

    if (request.syndromes.size > kMaxRealtimeVectorLength) {
      write_response(tx_slot, rx_slot, kStatusPayloadTooLarge);
      return;
    }

    std::array<std::uint8_t, kMaxRealtimeVectorLength> syndrome{};
    if (request.syndromes.size != 0)
      std::memcpy(syndrome.data(), request.syndromes.data,
                  request.syndromes.size);
    cudaq::qec::decoding::host::enqueue_syndromes(
        static_cast<std::size_t>(request.decoder_id), syndrome.data(),
        static_cast<std::size_t>(request.syndromes.size), request.tag);
    write_response(tx_slot, rx_slot, kStatusSuccess);
  } catch (...) {
    if (tx_slot && rx_slot)
      write_response(tx_slot, rx_slot, kStatusHandlerException);
  }
}

void get_corrections_host(const void *rx_slot, void *tx_slot,
                          std::size_t slot_size) {
  try {
    GetCorrectionsRequest request;
    if (!tx_slot ||
        !read_get_corrections_request(rx_slot, slot_size, request)) {
      if (tx_slot && rx_slot)
        write_response(tx_slot, rx_slot, kStatusInvalidRequest);
      return;
    }

    if (request.correction_length > kMaxRealtimeVectorLength) {
      write_response(tx_slot, rx_slot, kStatusPayloadTooLarge);
      return;
    }

    const auto result_len =
        static_cast<std::uint32_t>(request.correction_length);
    if (sizeof(cudaq::realtime::RPCResponse) + result_len > slot_size) {
      write_response(tx_slot, rx_slot, kStatusResultBufferTooSmall);
      return;
    }

    std::array<std::uint8_t, kMaxRealtimeVectorLength> corrections{};
    cudaq::qec::decoding::host::get_corrections(
        static_cast<std::size_t>(request.decoder_id), corrections.data(),
        static_cast<std::size_t>(request.correction_length), request.reset);

    auto *result = static_cast<std::uint8_t *>(tx_slot) +
                   sizeof(cudaq::realtime::RPCResponse);
    if (result_len != 0)
      std::memcpy(result, corrections.data(), result_len);
    write_response(tx_slot, rx_slot, kStatusSuccess, result_len);
  } catch (...) {
    if (tx_slot && rx_slot)
      write_response(tx_slot, rx_slot, kStatusHandlerException);
  }
}

void reset_decoder_host(const void *rx_slot, void *tx_slot,
                        std::size_t slot_size) {
  try {
    std::uint64_t decoder_id = 0;
    if (!tx_slot ||
        !read_reset_decoder_request(rx_slot, slot_size, decoder_id)) {
      if (tx_slot && rx_slot)
        write_response(tx_slot, rx_slot, kStatusInvalidRequest);
      return;
    }

    cudaq::qec::decoding::host::reset_decoder(
        static_cast<std::size_t>(decoder_id));
    write_response(tx_slot, rx_slot, kStatusSuccess);
  } catch (...) {
    if (tx_slot && rx_slot)
      write_response(tx_slot, rx_slot, kStatusHandlerException);
  }
}

void set_scalar(cudaq_type_desc_t &desc, std::uint8_t type_id,
                std::uint32_t size_bytes) {
  desc = {};
  desc.type_id = type_id;
  desc.size_bytes = size_bytes;
  desc.num_elements = kScalarElementCount;
}

void set_u64(cudaq_type_desc_t &desc) {
  set_scalar(desc, CUDAQ_TYPE_INT64, kScalarU64Size);
}

void set_u8(cudaq_type_desc_t &desc) {
  set_scalar(desc, CUDAQ_TYPE_UINT8, kScalarU8Size);
}

void set_array_u8(cudaq_type_desc_t &desc) {
  desc = {};
  desc.type_id = CUDAQ_TYPE_ARRAY_UINT8;
}

void configure_entry(cudaq_function_entry_t &entry, std::uint32_t function_id,
                     cudaq_host_rpc_fn_t handler, std::uint8_t num_args,
                     std::uint8_t num_results) {
  entry = {};
  entry.handler.host_fn = handler;
  entry.function_id = function_id;
  entry.dispatch_mode = CUDAQ_DISPATCH_HOST_CALL;
  entry.schema.num_args = num_args;
  entry.schema.num_results = num_results;
}

// Registers the fixed default-route RPC handlers (enqueue_syndromes /
// get_corrections / reset_decoder), all as CUDAQ_DISPATCH_HOST_CALL entries
// that decode on the CPU via the host API.
//
// TODO(decoding-server): a more complete decoding server should build the
// function table per configured decoder rather than as this fixed CPU set. Each
// decoder instance would get its own configure_entry establishing the correct
// processing mechanism -- CUDAQ_DISPATCH_HOST_CALL for CPU/host decoders, or
// CUDAQ_DISPATCH_GRAPH_LAUNCH for GPU decoders driven by a captured CUDA graph
// (see decoder_server_runtime.md "per-decoder alternative dispatch units").
// That is what would let the device_call host-dispatch path (this service)
// dispatch straight to the right per-decoder mechanism, superseding the
// separate qec_realtime_session decode ring entirely.
std::array<cudaq_function_entry_t, kDeviceCallEntryCount> make_entries() {
  std::array<cudaq_function_entry_t, kDeviceCallEntryCount> entries{};

  auto &enqueue_entry = entries[kEnqueueSyndromesEntry];
  configure_entry(enqueue_entry, kEnqueueSyndromesFnId, enqueue_syndromes_host,
                  kEnqueueArgCount, kNoResults);
  set_u64(enqueue_entry.schema.args[kEnqueueDecoderIdArg]);
  set_array_u8(enqueue_entry.schema.args[kEnqueueSyndromesArg]);
  set_u64(enqueue_entry.schema.args[kEnqueueTagArg]);

  auto &get_entry = entries[kGetCorrectionsEntry];
  configure_entry(get_entry, kGetCorrectionsFnId, get_corrections_host,
                  kGetCorrectionsArgCount, kSingleResult);
  set_u64(get_entry.schema.args[kGetCorrectionsDecoderIdArg]);
  set_u64(get_entry.schema.args[kGetCorrectionsLengthArg]);
  set_u8(get_entry.schema.args[kGetCorrectionsResetArg]);
  set_array_u8(get_entry.schema.results[kCorrectionsResult]);

  auto &reset_entry = entries[kResetDecoderEntry];
  configure_entry(reset_entry, kResetDecoderFnId, reset_decoder_host,
                  kResetDecoderArgCount, kNoResults);
  set_u64(reset_entry.schema.args[kResetDecoderIdArg]);

  return entries;
}

class QecDeviceCallService : public DeviceCallService {
public:
  int getHostDispatchTable(DeviceCallHostDispatchTable &table) override {
    static auto entries = make_entries();
    table.entries = entries.data();
    table.count = entries.size();
    table.deviceId = kHostDispatchDeviceId;
    table.mailbox = nullptr;
    return kStatusSuccess;
  }
};

QecDeviceCallService g_service;

DeviceCallService *get_service() { return &g_service; }

} // namespace

extern "C" __attribute__((visibility("default"))) void
cudaqx_qec_realtime_device_call_service_force_link() {}

// Test hook: number of requests this service has dispatched. Non-zero only if
// device_calls were routed through the host-dispatch ring to this service.
extern "C" __attribute__((visibility("default"))) std::uint64_t
cudaqx_qec_device_call_dispatch_count() {
  return g_service_dispatch_count.load(std::memory_order_relaxed);
}

extern "C" __attribute__((visibility("default"))) DeviceCallServicePluginInfo
cudaqGetDeviceCallServicePluginInfo() {
  return {"cudaq-qec-realtime-device-call", &get_service};
}
