/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CqrTransceiver.h"
#include "DecoderServer.h"
#include "RpcWireFormat.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"
#include "cudaq/realtime/device_call_service.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

using cudaq::qec::decoder_server::CqrTransceiver;
using cudaq::qec::decoder_server::DecoderServer;
using cudaq::qec::decoder_server::kEnqueueSyndromesFunctionId;
using cudaq::qec::decoder_server::kGetCorrectionsFunctionId;
using cudaq::qec::decoder_server::kResetDecoderFunctionId;
using cudaq::realtime::DeviceCallDispatchMode;
using cudaq::realtime::DeviceCallDispatchTable;
using cudaq::realtime::DeviceCallService;
using cudaq::realtime::DeviceCallServicePluginInfo;
using cudaq::realtime::DeviceCallServiceSession;

// Set CUDAQ_QEC_DECODER_CONFIG=/path/to/config.yaml before starting the daemon.
static std::string g_config_yaml;

static CqrTransceiver *g_transceiver = nullptr;
static std::unique_ptr<DecoderServer> g_server;
static std::thread g_server_thread;
static std::once_flag g_init_flag;

// Counts requests dispatched through this service (test hook).
static std::atomic<uint64_t> g_service_dispatch_count{0};

static void init_server() {
  const char *cfg = std::getenv("CUDAQ_QEC_DECODER_CONFIG");
  if (!cfg || cfg[0] == '\0')
    throw std::runtime_error("CUDAQ_QEC_DECODER_CONFIG env var not set; "
                             "point it to a multi_decoder_config YAML file");
  g_config_yaml = cfg;

  auto t = std::make_unique<CqrTransceiver>();
  g_transceiver = t.get();
  g_server = std::make_unique<DecoderServer>(std::move(t), g_config_yaml);
  g_server_thread = std::thread([] { g_server->run(); });
}

// ---------------------------------------------------------------------------
// CUDAQ handler functions — thin delegates to CqrTransceiver::inject()
// ---------------------------------------------------------------------------

void enqueue_syndromes_host(const void *rx_slot, void *tx_slot,
                            std::size_t slot_size) {
  g_service_dispatch_count.fetch_add(1, std::memory_order_relaxed);
  g_transceiver->inject(rx_slot, tx_slot, slot_size,
                        kEnqueueSyndromesFunctionId);
}

void get_corrections_host(const void *rx_slot, void *tx_slot,
                          std::size_t slot_size) {
  g_service_dispatch_count.fetch_add(1, std::memory_order_relaxed);
  g_transceiver->inject(rx_slot, tx_slot, slot_size, kGetCorrectionsFunctionId);
}

void reset_decoder_host(const void *rx_slot, void *tx_slot,
                        std::size_t slot_size) {
  g_service_dispatch_count.fetch_add(1, std::memory_order_relaxed);
  g_transceiver->inject(rx_slot, tx_slot, slot_size, kResetDecoderFunctionId);
}

// ---------------------------------------------------------------------------
// DeviceCallService plugin
// ---------------------------------------------------------------------------

constexpr uint32_t kEnqueueSyndromesFnId =
    cudaq::realtime::fnv1a_hash("enqueue_syndromes");
constexpr uint32_t kGetCorrectionsFnId =
    cudaq::realtime::fnv1a_hash("get_corrections");
constexpr uint32_t kResetDecoderFnId =
    cudaq::realtime::fnv1a_hash("reset_decoder");

constexpr int32_t kHostDispatchDeviceId = 0;
constexpr uint8_t kNoResults = 0;
constexpr uint8_t kSingleResult = 1;
constexpr uint8_t kScalarU8Size = sizeof(uint8_t);
constexpr uint8_t kScalarU64Size = sizeof(uint64_t);

enum DeviceCallEntryIndex : std::size_t {
  kEnqueueSyndromesEntry,
  kGetCorrectionsEntry,
  kResetDecoderEntry,
  kDeviceCallEntryCount
};

static void set_u64(cudaq_type_desc_t &d) {
  d = {};
  d.type_id = CUDAQ_TYPE_INT64;
  d.size_bytes = kScalarU64Size;
  d.num_elements = 1;
}

static void set_u8(cudaq_type_desc_t &d) {
  d = {};
  d.type_id = CUDAQ_TYPE_UINT8;
  d.size_bytes = kScalarU8Size;
  d.num_elements = 1;
}

static void set_array_u8(cudaq_type_desc_t &d) {
  d = {};
  d.type_id = CUDAQ_TYPE_ARRAY_UINT8;
}

static void configure_entry(cudaq_function_entry_t &e, uint32_t fn_id,
                            cudaq_host_rpc_fn_t handler, uint8_t num_args,
                            uint8_t num_results) {
  e = {};
  e.handler.host_fn = handler;
  e.function_id = fn_id;
  e.dispatch_mode = CUDAQ_DISPATCH_HOST_CALL;
  e.schema.num_args = num_args;
  e.schema.num_results = num_results;
}

static std::array<cudaq_function_entry_t, kDeviceCallEntryCount>
make_entries() {
  std::array<cudaq_function_entry_t, kDeviceCallEntryCount> entries{};

  auto &eq = entries[kEnqueueSyndromesEntry];
  configure_entry(eq, kEnqueueSyndromesFnId, enqueue_syndromes_host, 3,
                  kNoResults);
  set_u64(eq.schema.args[0]);
  set_array_u8(eq.schema.args[1]);
  set_u64(eq.schema.args[2]);

  auto &gc = entries[kGetCorrectionsEntry];
  configure_entry(gc, kGetCorrectionsFnId, get_corrections_host, 3,
                  kSingleResult);
  set_u64(gc.schema.args[0]);
  set_u64(gc.schema.args[1]);
  set_u8(gc.schema.args[2]);
  set_array_u8(gc.schema.results[0]);

  auto &rd = entries[kResetDecoderEntry];
  configure_entry(rd, kResetDecoderFnId, reset_decoder_host, 1, kNoResults);
  set_u64(rd.schema.args[0]);

  return entries;
}

class QecDeviceCallSession : public DeviceCallServiceSession {
public:
  QecDeviceCallSession() {
    table_.mode = DeviceCallDispatchMode::Host;
    table_.entries = entries_.data();
    table_.count = entries_.size();
    table_.deviceId = kHostDispatchDeviceId;
    table_.mailbox = nullptr;
  }

  const DeviceCallDispatchTable &dispatchTable() const noexcept override {
    return table_;
  }

private:
  std::array<cudaq_function_entry_t, kDeviceCallEntryCount> entries_ =
      make_entries();
  DeviceCallDispatchTable table_;
};

class QecDeviceCallService : public DeviceCallService {
public:
  std::unique_ptr<DeviceCallServiceSession>
  createDispatchSession(DeviceCallDispatchMode mode) override {
    if (mode != DeviceCallDispatchMode::Host)
      return nullptr;
    std::call_once(g_init_flag, init_server);
    return std::make_unique<QecDeviceCallSession>();
  }
};

QecDeviceCallService g_service;
DeviceCallService *get_service() { return &g_service; }

} // namespace

extern "C" __attribute__((visibility("default"))) void
cudaqx_qec_realtime_device_call_service_force_link() {}

extern "C" __attribute__((visibility("default"))) uint64_t
cudaqx_qec_device_call_dispatch_count() {
  return g_service_dispatch_count.load(std::memory_order_relaxed);
}

extern "C" __attribute__((visibility("default"))) DeviceCallServicePluginInfo
cudaqGetDeviceCallServicePluginInfo() {
  return {"cudaq-qec-realtime-device-call", &get_service};
}
