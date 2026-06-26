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
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

using cudaq_internal::device_call::DeviceCallHostDispatchTable;
using cudaq_internal::device_call::DeviceCallService;
using cudaq_internal::device_call::DeviceCallServicePluginInfo;

constexpr std::uint32_t kEnqueueSyndromesUi64FnId =
    cudaq::realtime::fnv1a_hash("enqueue_syndromes_ui64");
constexpr std::uint32_t kGetCorrectionsUi64FnId =
    cudaq::realtime::fnv1a_hash("get_corrections_ui64");
constexpr std::uint32_t kResetDecoderUi64FnId =
    cudaq::realtime::fnv1a_hash("reset_decoder_ui64");

template <std::size_t NumArgs>
bool read_u64_args(const void *rx_slot, std::size_t slot_size,
                   std::array<std::uint64_t, NumArgs> &args) {
  if (!rx_slot || slot_size < sizeof(cudaq::realtime::RPCHeader))
    return false;

  const auto *request =
      static_cast<const cudaq::realtime::RPCHeader *>(rx_slot);
  const std::size_t arg_len = NumArgs * sizeof(std::uint64_t);
  if (request->magic != cudaq::realtime::RPC_MAGIC_REQUEST ||
      request->arg_len != arg_len ||
      sizeof(cudaq::realtime::RPCHeader) + arg_len > slot_size)
    return false;

  const auto *body = static_cast<const std::uint8_t *>(rx_slot) +
                     sizeof(cudaq::realtime::RPCHeader);
  std::memcpy(args.data(), body, arg_len);
  return true;
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

void enqueue_syndromes_ui64_host(const void *rx_slot, void *tx_slot,
                                 std::size_t slot_size) {
  try {
    std::array<std::uint64_t, 4> args{};
    if (!tx_slot || !read_u64_args(rx_slot, slot_size, args)) {
      if (tx_slot && rx_slot)
        write_response(tx_slot, rx_slot, -1);
      return;
    }

    const auto decoder_id = static_cast<std::size_t>(args[0]);
    const auto syndrome_size = args[1];
    if (syndrome_size > 64) {
      write_response(tx_slot, rx_slot, -3);
      return;
    }

    std::vector<std::uint8_t> syndrome(syndrome_size);
    for (std::uint64_t i = 0; i < syndrome_size; ++i)
      syndrome[i] = (args[2] >> i) & 1;

    cudaq::qec::decoding::host::enqueue_syndromes(decoder_id, syndrome.data(),
                                                  syndrome.size(), args[3]);
    write_response(tx_slot, rx_slot, 0);
  } catch (...) {
    if (tx_slot && rx_slot)
      write_response(tx_slot, rx_slot, -2);
  }
}

void get_corrections_ui64_host(const void *rx_slot, void *tx_slot,
                               std::size_t slot_size) {
  try {
    std::array<std::uint64_t, 3> args{};
    if (!tx_slot || !read_u64_args(rx_slot, slot_size, args)) {
      if (tx_slot && rx_slot)
        write_response(tx_slot, rx_slot, -1);
      return;
    }

    constexpr std::uint32_t result_len = sizeof(std::uint64_t);
    const auto return_size = args[1];
    if (return_size > 64) {
      write_response(tx_slot, rx_slot, -3);
      return;
    }

    if (sizeof(cudaq::realtime::RPCResponse) + result_len > slot_size) {
      write_response(tx_slot, rx_slot, -5);
      return;
    }

    const auto decoder_id = static_cast<std::size_t>(args[0]);
    std::vector<std::uint8_t> corrections(return_size);
    cudaq::qec::decoding::host::get_corrections(decoder_id, corrections.data(),
                                                corrections.size(),
                                                static_cast<bool>(args[2]));

    std::uint64_t correction = 0;
    for (std::uint64_t i = 0; i < return_size; ++i)
      correction |= static_cast<std::uint64_t>(corrections[i]) << i;
    auto *result = static_cast<std::uint8_t *>(tx_slot) +
                   sizeof(cudaq::realtime::RPCResponse);
    std::memcpy(result, &correction, result_len);
    write_response(tx_slot, rx_slot, 0, result_len);
  } catch (...) {
    if (tx_slot && rx_slot)
      write_response(tx_slot, rx_slot, -2);
  }
}

void reset_decoder_ui64_host(const void *rx_slot, void *tx_slot,
                             std::size_t slot_size) {
  try {
    std::array<std::uint64_t, 1> args{};
    if (!tx_slot || !read_u64_args(rx_slot, slot_size, args)) {
      if (tx_slot && rx_slot)
        write_response(tx_slot, rx_slot, -1);
      return;
    }

    cudaq::qec::decoding::host::reset_decoder(
        static_cast<std::size_t>(args[0]));
    write_response(tx_slot, rx_slot, 0);
  } catch (...) {
    if (tx_slot && rx_slot)
      write_response(tx_slot, rx_slot, -2);
  }
}

void set_u64(cudaq_type_desc_t &desc) {
  desc = {};
  desc.type_id = CUDAQ_TYPE_INT64;
  desc.size_bytes = sizeof(std::uint64_t);
  desc.num_elements = 1;
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
  for (std::uint8_t i = 0; i < num_args; ++i)
    set_u64(entry.schema.args[i]);
  for (std::uint8_t i = 0; i < num_results; ++i)
    set_u64(entry.schema.results[i]);
}

std::array<cudaq_function_entry_t, 3> make_entries() {
  std::array<cudaq_function_entry_t, 3> entries{};
  configure_entry(entries[0], kEnqueueSyndromesUi64FnId,
                  enqueue_syndromes_ui64_host, 4, 0);
  configure_entry(entries[1], kGetCorrectionsUi64FnId,
                  get_corrections_ui64_host, 3, 1);
  configure_entry(entries[2], kResetDecoderUi64FnId, reset_decoder_ui64_host, 1,
                  0);
  return entries;
}

class QecDeviceCallService : public DeviceCallService {
public:
  int getHostDispatchTable(DeviceCallHostDispatchTable &table) override {
    static auto entries = make_entries();
    table.entries = entries.data();
    table.count = entries.size();
    table.deviceId = 0;
    table.mailbox = nullptr;
    return 0;
  }
};

QecDeviceCallService g_service;

DeviceCallService *get_service() { return &g_service; }

} // namespace

extern "C" __attribute__((visibility("default"))) void
cudaqx_qec_realtime_device_call_service_force_link() {}

extern "C" __attribute__((visibility("default"))) DeviceCallServicePluginInfo
cudaqGetDeviceCallServicePluginInfo() {
  return {"cudaq-qec-realtime-device-call", &get_service};
}
