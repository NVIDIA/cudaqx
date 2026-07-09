/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// cudaq-realtime (cqr) DeviceCallService plugin for the decoding server.
///
/// The plugin registers the three default-route RPCs (enqueue_syndromes /
/// get_corrections / reset_decoder) as CUDAQ_DISPATCH_HOST_CALL entries whose
/// handlers are thin delegates into CqrTransceiver::inject(); the actual
/// decoding runs in DecoderServer (one DecoderSession worker thread per
/// configured decoder, so multiple decoders decode concurrently).
///
/// The decoder configuration comes from, in priority order:
///   1. the CUDAQ_QEC_DECODER_CONFIG env var (path to a multi_decoder_config
///      YAML) -- the standalone-daemon path;
///   2. the last multi_decoder_config passed to
///      cudaq::qec::decoding::config::configure_decoders() in this process --
///      the in-process (host_dispatch) application path.

#include "CqrTransceiver.h"
#include "DecoderServer.h"
#include "RpcWireFormat.h"
#include "cudaq/qec/logger.h"
#include "cudaq/qec/realtime/decoding_config.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"
#include "cudaq/realtime/device_call_service.h"

#include "../realtime_decoding.h"

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" void cudaqx_qec_decoder_server_shutdown();
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

static CqrTransceiver *g_transceiver = nullptr;
static std::unique_ptr<DecoderServer> g_server;
static std::thread g_server_thread;
static std::once_flag g_init_flag;

// Counts requests dispatched through this service (test hook).
static std::atomic<uint64_t> g_service_dispatch_count{0};

static void init_server() {
  auto t = std::make_unique<CqrTransceiver>();
  g_transceiver = t.get();

  if (const char *cfg = std::getenv("CUDAQ_QEC_DECODER_CONFIG");
      cfg && cfg[0] != '\0') {
    g_server = std::make_unique<DecoderServer>(std::move(t), std::string(cfg));
  } else if (const auto *config = cudaq::qec::decoding::config::
                 last_configured_multi_decoder_config()) {
    g_server = std::make_unique<DecoderServer>(std::move(t), *config);
  } else {
    g_transceiver = nullptr;
    throw std::runtime_error(
        "decoder-server config not found: set CUDAQ_QEC_DECODER_CONFIG to a "
        "multi_decoder_config YAML path, or call "
        "cudaq::qec::decoding::config::configure_decoders() before realtime "
        "initialization");
  }
  g_server_thread = std::thread([] { g_server->run(); });
  // In-process applications never call the explicit shutdown hook the daemon
  // uses; stop the server at exit() so the static g_server_thread is joined
  // before static destruction (a still-joinable thread would
  // std::terminate, aborting the process and losing buffered stdout).
  std::atexit([] { cudaqx_qec_decoder_server_shutdown(); });
}

// ---------------------------------------------------------------------------
// CUDAQ handler functions — thin delegates to CqrTransceiver::inject()
// ---------------------------------------------------------------------------

// Write an error RPCResponse into tx_slot (handler-level failures must not
// propagate into the transport dispatcher loop).
constexpr int32_t kStatusHandlerException = 3; // RpcStatus::INTERNAL_ERROR

static void write_error_response(const void *rx_slot, void *tx_slot,
                                 std::size_t slot_size, int32_t status) {
  if (!tx_slot || !rx_slot || slot_size < sizeof(cudaq::realtime::RPCHeader))
    return;
  const auto *req = static_cast<const cudaq::realtime::RPCHeader *>(rx_slot);
  auto *resp = static_cast<cudaq::realtime::RPCResponse *>(tx_slot);
  resp->status = status;
  resp->result_len = 0;
  resp->request_id = req->request_id;
  resp->ptp_timestamp = req->ptp_timestamp;
  __atomic_store_n(reinterpret_cast<uint32_t *>(tx_slot),
                   cudaq::realtime::RPC_MAGIC_RESPONSE, __ATOMIC_RELEASE);
}

// --save_syndrome support: the served path bypasses host::enqueue_syndromes
// (where capture used to hook), so replicate its capture here -- unpack the
// wire's LSB-first bits and repack MSB-first, byte-identical to the host
// path's saved-syndrome format.
static void capture_enqueue_syndromes(const void *rx_slot,
                                      std::size_t slot_size) {
  auto callback = cudaq::qec::decoding::host::_get_syndrome_capture_callback();
  if (!callback)
    return;
  if (!rx_slot || slot_size < sizeof(cudaq::realtime::RPCHeader))
    return;
  const auto *hdr = static_cast<const cudaq::realtime::RPCHeader *>(rx_slot);
  const auto *payload = static_cast<const uint8_t *>(rx_slot) +
                        sizeof(cudaq::realtime::RPCHeader);
  const std::size_t arg_len = hdr->arg_len;
  // [decoder_id][counter][mapping_id][num_syndromes][byte_count][bits]
  if (arg_len < 5 * sizeof(uint64_t))
    return;
  uint64_t num_syndromes = 0, byte_count = 0;
  std::memcpy(&num_syndromes, payload + 3 * sizeof(uint64_t), sizeof(uint64_t));
  std::memcpy(&byte_count, payload + 4 * sizeof(uint64_t), sizeof(uint64_t));
  if (byte_count != (num_syndromes + 7) / 8 ||
      byte_count > arg_len - 5 * sizeof(uint64_t))
    return;
  const uint8_t *bits = payload + 5 * sizeof(uint64_t);
  std::vector<uint8_t> packed(byte_count, 0);
  for (uint64_t i = 0; i < num_syndromes; ++i)
    if ((bits[i / 8] >> (i % 8)) & 1u)
      packed[i / 8] |= static_cast<uint8_t>(1u << (7 - (i % 8)));
  callback(packed.data(), packed.size());
}

// The server is constructed lazily on the first RPC (the in-process
// application path configures decoders AFTER the realtime channel — and
// with it this dispatch session — is created); the daemon path instead
// initializes eagerly at session creation via CUDAQ_QEC_DECODER_CONFIG so
// slow decoder construction happens before its READY line.
static void dispatch_rpc(const void *rx_slot, void *tx_slot,
                         std::size_t slot_size, uint32_t function_id) {
  g_service_dispatch_count.fetch_add(1, std::memory_order_relaxed);
  try {
    std::call_once(g_init_flag, init_server);
    // g_transceiver is null if init_server failed or after shutdown().
    // g_init_flag is not resettable, so call_once won't retry after shutdown.
    if (!g_transceiver) {
      write_error_response(rx_slot, tx_slot, slot_size,
                           kStatusHandlerException);
      return;
    }
    if (function_id == kEnqueueSyndromesFunctionId)
      capture_enqueue_syndromes(rx_slot, slot_size);
    g_transceiver->inject(rx_slot, tx_slot, slot_size, function_id);
  } catch (const std::exception &e) {
    CUDA_QEC_ERROR("decoder-server RPC failed: {}", e.what());
    write_error_response(rx_slot, tx_slot, slot_size, kStatusHandlerException);
  } catch (...) {
    write_error_response(rx_slot, tx_slot, slot_size, kStatusHandlerException);
  }
}

void enqueue_syndromes_host(const void *rx_slot, void *tx_slot,
                            std::size_t slot_size) {
  dispatch_rpc(rx_slot, tx_slot, slot_size, kEnqueueSyndromesFunctionId);
}

void get_corrections_host(const void *rx_slot, void *tx_slot,
                          std::size_t slot_size) {
  dispatch_rpc(rx_slot, tx_slot, slot_size, kGetCorrectionsFunctionId);
}

void reset_decoder_host(const void *rx_slot, void *tx_slot,
                        std::size_t slot_size) {
  dispatch_rpc(rx_slot, tx_slot, slot_size, kResetDecoderFunctionId);
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

// Wire argument order per decoder_server_runtime.md: fixed-size scalars
// first, the variable-length bit-packed byte array last.
constexpr std::uint8_t kEnqueueDecoderIdArg = 0;
constexpr std::uint8_t kEnqueueCounterArg = 1;
constexpr std::uint8_t kEnqueueMappingIdArg = 2;
constexpr std::uint8_t kEnqueueNumSyndromesArg = 3;
constexpr std::uint8_t kEnqueueSyndromeBitsArg = 4;
constexpr std::uint8_t kEnqueueArgCount = 5;

constexpr std::uint8_t kGetCorrectionsDecoderIdArg = 0;
constexpr std::uint8_t kGetCorrectionsReturnSizeArg = 1;
constexpr std::uint8_t kGetCorrectionsBytesArg = 2;
constexpr std::uint8_t kGetCorrectionsResetArg = 3;
constexpr std::uint8_t kGetCorrectionsArgCount = 4;

constexpr std::uint8_t kResetDecoderIdArg = 0;
constexpr std::uint8_t kResetDecoderArgCount = 1;

constexpr std::uint8_t kCorrectionsResult = 0;

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

  // enqueue_syndromes: 5-arg spec format per decoder_server_runtime.md.
  // decoder_id, counter, syndrome_mapping_id, num_syndromes (scalars) +
  // packed_bytes (array_u8: [u64 byte_count][u8 x byte_count], bit-packed
  // LSB-first).
  auto &eq = entries[kEnqueueSyndromesEntry];
  configure_entry(eq, kEnqueueSyndromesFnId, enqueue_syndromes_host,
                  kEnqueueArgCount, kNoResults);
  set_u64(eq.schema.args[kEnqueueDecoderIdArg]);
  set_u64(eq.schema.args[kEnqueueCounterArg]);
  set_u64(eq.schema.args[kEnqueueMappingIdArg]);
  set_u64(eq.schema.args[kEnqueueNumSyndromesArg]);
  set_array_u8(eq.schema.args[kEnqueueSyndromeBitsArg]);

  // get_corrections: 4-arg spec format per decoder_server_runtime.md.
  auto &gc = entries[kGetCorrectionsEntry];
  configure_entry(gc, kGetCorrectionsFnId, get_corrections_host,
                  kGetCorrectionsArgCount, kSingleResult);
  set_u64(gc.schema.args[kGetCorrectionsDecoderIdArg]);
  set_u64(gc.schema.args[kGetCorrectionsReturnSizeArg]);
  set_u64(gc.schema.args[kGetCorrectionsBytesArg]);
  set_u8(gc.schema.args[kGetCorrectionsResetArg]);
  set_array_u8(gc.schema.results[kCorrectionsResult]);

  auto &rd = entries[kResetDecoderEntry];
  configure_entry(rd, kResetDecoderFnId, reset_decoder_host,
                  kResetDecoderArgCount, kNoResults);
  set_u64(rd.schema.args[kResetDecoderIdArg]);

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
    // Daemon path: the config path is in the environment, so build the
    // decoder sessions NOW (before the daemon's READY line). The in-process
    // application path has not called configure_decoders yet at this point;
    // it initializes lazily on the first RPC (see dispatch_rpc).
    if (const char *cfg = std::getenv("CUDAQ_QEC_DECODER_CONFIG");
        cfg && cfg[0] != '\0')
      std::call_once(g_init_flag, init_server);
    return std::make_unique<QecDeviceCallSession>();
  }
};

QecDeviceCallService g_service;
DeviceCallService *get_service() { return &g_service; }

} // namespace

extern "C" __attribute__((visibility("default")))
cudaq::realtime::DeviceCallServicePluginInfo
cudaqGetDeviceCallServicePluginInfo() {
  return {"cudaq-qec-realtime-device-call", &get_service};
}

extern "C" __attribute__((visibility("default"))) void
cudaqx_qec_realtime_device_call_service_force_link() {}

extern "C" __attribute__((visibility("default"))) uint64_t
cudaqx_qec_device_call_dispatch_count() {
  return g_service_dispatch_count.load(std::memory_order_relaxed);
}

/// High-water mark of simultaneously-busy DecoderSession workers -- the
/// daemon's concurrency evidence for multi-logical-qubit tests.
extern "C" __attribute__((visibility("default"))) uint64_t
cudaqx_qec_decoder_server_max_concurrent() {
  return cudaq::qec::decoder_server::max_concurrent_busy_sessions();
}

/// Stop the DecoderServer receive loop and join its thread. The daemon calls
/// this before exiting; without it the static g_server_thread would still be
/// joinable at static destruction (std::terminate).
extern "C" __attribute__((visibility("default"))) void
cudaqx_qec_decoder_server_shutdown() {
  if (g_server) {
    g_server->stop();
    if (g_server_thread.joinable())
      g_server_thread.join();
    g_server.reset();
    g_transceiver = nullptr;
  }
}
