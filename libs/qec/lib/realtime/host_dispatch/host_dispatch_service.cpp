/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "../realtime_decoding.h"

#include "cudaq_internal/device_call/DeviceCallError.h"
#include "cudaq_internal/device_call/DeviceCallService.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

using cudaq::qec::decoding::host::enqueue_syndromes;
using cudaq::qec::decoding::host::get_corrections;
using cudaq::qec::decoding::host::reset_decoder;

constexpr std::uint32_t kEnqueueSyndromesUi64FunctionId =
    cudaq::realtime::fnv1a_hash("qec_enqueue_syndromes_ui64");
constexpr std::uint32_t kGetCorrectionsUi64FunctionId =
    cudaq::realtime::fnv1a_hash("qec_get_corrections_ui64");
constexpr std::uint32_t kResetDecoderUi64FunctionId =
    cudaq::realtime::fnv1a_hash("qec_reset_decoder_ui64");
constexpr std::int32_t kInvalidArgumentStatus =
    cudaq_internal::device_call::toAbiStatus(
        cudaq_internal::device_call::DeviceCallStatus::InvalidArgument);
constexpr std::int32_t kRemoteErrorStatus =
    cudaq_internal::device_call::toAbiStatus(
        cudaq_internal::device_call::DeviceCallStatus::RemoteError);

struct EnqueueSyndromesUi64Payload {
  std::uint64_t decoder_id;
  std::uint64_t syndrome_size;
  std::uint64_t syndrome;
  std::uint64_t tag;
};

struct GetCorrectionsUi64Payload {
  std::uint64_t decoder_id;
  std::uint64_t return_size;
  std::uint64_t reset;
};

struct ResetDecoderUi64Payload {
  std::uint64_t decoder_id;
};

static_assert(sizeof(EnqueueSyndromesUi64Payload) == 32);
static_assert(sizeof(GetCorrectionsUi64Payload) == 24);
static_assert(sizeof(ResetDecoderUi64Payload) == 8);

cudaq::realtime::RPCResponse *
start_response(const cudaq::realtime::RPCHeader *request, void *tx_slot) {
  auto *response = static_cast<cudaq::realtime::RPCResponse *>(tx_slot);
  std::memset(response, 0, sizeof(cudaq::realtime::RPCResponse));
  response->request_id = request ? request->request_id : 0;
  response->ptp_timestamp = request ? request->ptp_timestamp : 0;
  return response;
}

void finish_response(cudaq::realtime::RPCResponse *response,
                     std::int32_t status, std::uint32_t result_len = 0) {
  response->status = status;
  response->result_len = status == 0 ? result_len : 0;
  __atomic_store_n(&response->magic, cudaq::realtime::RPC_MAGIC_RESPONSE,
                   __ATOMIC_RELEASE);
}

void write_uint64_result(void *tx_slot, std::uint64_t result) {
  std::memcpy(static_cast<std::uint8_t *>(tx_slot) +
                  sizeof(cudaq::realtime::RPCResponse),
              &result, sizeof(result));
}

bool has_uint64_result_capacity(std::size_t slot_size) {
  return slot_size >=
         sizeof(cudaq::realtime::RPCResponse) + sizeof(std::uint64_t);
}

template <typename Payload>
const Payload *get_payload(const cudaq::realtime::RPCHeader *request) {
  if (!request || request->magic != cudaq::realtime::RPC_MAGIC_REQUEST ||
      request->arg_len != sizeof(Payload))
    return nullptr;
  return reinterpret_cast<const Payload *>(request + 1);
}

void qec_enqueue_syndromes_ui64_host(const void *rx_slot, void *tx_slot,
                                     std::size_t slot_size) {
  const auto *request =
      static_cast<const cudaq::realtime::RPCHeader *>(rx_slot);
  auto *response = start_response(request, tx_slot);
  const auto *payload = get_payload<EnqueueSyndromesUi64Payload>(request);
  if (!payload || payload->syndrome_size > 64 ||
      !has_uint64_result_capacity(slot_size)) {
    finish_response(response, kInvalidArgumentStatus);
    return;
  }

  try {
    std::vector<std::uint8_t> syndromes(payload->syndrome_size);
    for (std::uint64_t bit = 0; bit < payload->syndrome_size; ++bit)
      syndromes[bit] = (payload->syndrome >> bit) & 1u;
    enqueue_syndromes(static_cast<std::size_t>(payload->decoder_id),
                      syndromes.data(), payload->syndrome_size, payload->tag);
    write_uint64_result(tx_slot, 0);
    finish_response(response, 0, sizeof(std::uint64_t));
  } catch (...) {
    finish_response(response, kRemoteErrorStatus);
  }
}

void qec_get_corrections_ui64_host(const void *rx_slot, void *tx_slot,
                                   std::size_t slot_size) {
  const auto *request =
      static_cast<const cudaq::realtime::RPCHeader *>(rx_slot);
  auto *response = start_response(request, tx_slot);
  const auto *payload = get_payload<GetCorrectionsUi64Payload>(request);
  if (!payload || payload->return_size > 64 ||
      !has_uint64_result_capacity(slot_size)) {
    finish_response(response, kInvalidArgumentStatus);
    return;
  }

  try {
    std::vector<std::uint8_t> corrections(payload->return_size);
    get_corrections(static_cast<std::size_t>(payload->decoder_id),
                    corrections.data(), payload->return_size,
                    payload->reset != 0);

    std::uint64_t packed = 0;
    for (std::uint64_t bit = 0; bit < payload->return_size; ++bit)
      if (corrections[bit] & 1u)
        packed |= std::uint64_t{1} << bit;
    write_uint64_result(tx_slot, packed);
    finish_response(response, 0, sizeof(packed));
  } catch (...) {
    finish_response(response, kRemoteErrorStatus);
  }
}

void qec_reset_decoder_ui64_host(const void *rx_slot, void *tx_slot,
                                 std::size_t slot_size) {
  const auto *request =
      static_cast<const cudaq::realtime::RPCHeader *>(rx_slot);
  auto *response = start_response(request, tx_slot);
  const auto *payload = get_payload<ResetDecoderUi64Payload>(request);
  if (!payload || !has_uint64_result_capacity(slot_size)) {
    finish_response(response, kInvalidArgumentStatus);
    return;
  }

  try {
    reset_decoder(static_cast<std::size_t>(payload->decoder_id));
    write_uint64_result(tx_slot, 0);
    finish_response(response, 0, sizeof(std::uint64_t));
  } catch (...) {
    finish_response(response, kRemoteErrorStatus);
  }
}

void fill_int64_arg(cudaq_type_desc_t &arg) {
  arg.type_id = CUDAQ_TYPE_INT64;
  arg.size_bytes = sizeof(std::uint64_t);
  arg.num_elements = 1;
}

void fill_entry(cudaq_function_entry_t &entry, cudaq_host_rpc_fn_t handler,
                std::uint32_t function_id, std::uint8_t num_args,
                std::uint8_t num_results) {
  entry = {};
  entry.handler.host_fn = handler;
  entry.function_id = function_id;
  entry.dispatch_mode = CUDAQ_DISPATCH_HOST_CALL;
  entry.schema.num_args = num_args;
  entry.schema.num_results = num_results;
  for (std::uint8_t i = 0; i < num_args; ++i)
    fill_int64_arg(entry.schema.args[i]);
  for (std::uint8_t i = 0; i < num_results; ++i)
    fill_int64_arg(entry.schema.results[i]);
}

class QecHostDispatchService
    : public cudaq_internal::device_call::DeviceCallService {
public:
  int getHostDispatchTable(
      cudaq_internal::device_call::DeviceCallHostDispatchTable &table)
      override {
    fill_entry(entries[0], qec_enqueue_syndromes_ui64_host,
               kEnqueueSyndromesUi64FunctionId, /*num_args=*/4,
               /*num_results=*/1);
    fill_entry(entries[1], qec_get_corrections_ui64_host,
               kGetCorrectionsUi64FunctionId, /*num_args=*/3,
               /*num_results=*/1);
    fill_entry(entries[2], qec_reset_decoder_ui64_host,
               kResetDecoderUi64FunctionId, /*num_args=*/1,
               /*num_results=*/1);
    table.entries = entries.data();
    table.count = static_cast<std::uint32_t>(entries.size());
    table.deviceId = 0;
    table.mailbox = nullptr;
    return 0;
  }

private:
  std::array<cudaq_function_entry_t, 3> entries{};
};

cudaq_internal::device_call::DeviceCallService *
get_qec_host_dispatch_service() {
  static QecHostDispatchService service;
  return &service;
}

} // namespace

extern "C" __attribute__((visibility("default")))
cudaq_internal::device_call::DeviceCallServicePluginInfo
cudaqGetDeviceCallServicePluginInfo() {
  return {"cudaq-qec-host-dispatch", &get_qec_host_dispatch_service};
}
