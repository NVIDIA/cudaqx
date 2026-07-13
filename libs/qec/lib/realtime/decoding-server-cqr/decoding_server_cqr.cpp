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
/// decoding runs in DecodingServer (one DecodingSession worker thread per
/// configured decoder, so multiple decoders decode concurrently).
///
/// The decoder configuration comes from, in priority order:
///   1. the CUDAQ_QEC_DECODER_CONFIG env var (path to a multi_decoder_config
///      YAML) -- the standalone-server path;
///   2. the last multi_decoder_config passed to
///      cudaq::qec::decoding::config::configure_decoders() in this process --
///      the in-process (host_dispatch) application path.

#include "CqrTransceiver.h"
#include "DecodingServer.h"
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

extern "C" void cudaqx_qec_decoding_server_shutdown();
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

using cudaq::qec::decoding_server::CqrTransceiver;
using cudaq::qec::decoding_server::DecodingServer;
using cudaq::qec::decoding_server::kEnqueueSyndromesFunctionId;
using cudaq::qec::decoding_server::kGetCorrectionsFunctionId;
using cudaq::qec::decoding_server::kResetDecoderFunctionId;
using cudaq::realtime::DeviceCallDispatchMode;
using cudaq::realtime::DeviceCallDispatchTable;
using cudaq::realtime::DeviceCallService;
using cudaq::realtime::DeviceCallServicePluginInfo;
using cudaq::realtime::DeviceCallServiceSession;

static CqrTransceiver *g_transceiver = nullptr;
static std::unique_ptr<DecodingServer> g_server;
static std::thread g_server_thread;
static std::once_flag g_init_flag;

// Counts requests dispatched through this service (test hook).
static std::atomic<uint64_t> g_service_dispatch_count{0};

static void init_server() {
  auto t = std::make_unique<CqrTransceiver>();
  CqrTransceiver *raw = t.get();

  if (const char *cfg = std::getenv("CUDAQ_QEC_DECODER_CONFIG");
      cfg && cfg[0] != '\0') {
    g_server = std::make_unique<DecodingServer>(std::move(t), std::string(cfg));
  } else if (const auto config = cudaq::qec::decoding::config::
                 last_configured_multi_decoder_config()) {
    g_server = std::make_unique<DecodingServer>(std::move(t), *config);
  } else {
    throw std::runtime_error(
        "decoding-server config not found: set CUDAQ_QEC_DECODER_CONFIG to a "
        "multi_decoder_config YAML path, or call "
        "cudaq::qec::decoding::config::configure_decoders() before realtime "
        "initialization");
  }
  // Publish the transceiver only after the server is fully constructed: a
  // throwing DecodingServer constructor has already freed the transceiver, and
  // dispatch_rpc treats a null g_transceiver as "not serving".
  g_transceiver = raw;
  g_server_thread = std::thread([] { g_server->run(); });
  // In-process applications never call the explicit shutdown hook the server
  // uses; stop the server at exit() so the static g_server_thread is joined
  // before static destruction (a still-joinable thread would
  // std::terminate, aborting the process and losing buffered stdout).
  std::atexit([] { cudaqx_qec_decoding_server_shutdown(); });
}

// ---------------------------------------------------------------------------
// CUDAQ handler functions — thin delegates to CqrTransceiver::inject()
// ---------------------------------------------------------------------------

// Write an error RPCResponse into tx_slot (handler-level failures must not
// propagate into the transport dispatcher loop).
constexpr int32_t kStatusHandlerException = static_cast<int32_t>(
    cudaq::qec::decoding_server::RpcStatus::INTERNAL_ERROR);

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
  cudaq::qec::decoding_server::detail::CqrEnqueueFrameView request;
  if (!cudaq::qec::decoding_server::detail::parse_cqr_enqueue_frame(
          rx_slot, slot_size, request))
    return;
  std::vector<uint8_t> packed(request.byte_count, 0);
  for (uint64_t i = 0; i < request.num_syndromes; ++i)
    if ((request.packed_bits[i / 8] >> (i % 8)) & 1u)
      packed[i / 8] |= static_cast<uint8_t>(1u << (7 - (i % 8)));
  callback(packed.data(), packed.size());
}

// The server is constructed lazily on the first RPC (the in-process
// application path configures decoders AFTER the realtime channel — and
// with it this dispatch session — is created); the server path instead
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
    // Log via the non-throwing cudaq::qec::error() free function, NOT the
    // CUDA_QEC_ERROR macro: the macro throws, and an exception escaping this
    // handler into the transport dispatcher loop would terminate the process
    // instead of returning the error response written below.
    cudaq::qec::error("decoding-server RPC failed: {}", e.what());
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

// The schema entries below register under the SAME function IDs the handlers
// and CqrTransceiver route on (the kXFunctionId constants from
// RpcWireFormat.h); these asserts pin them to the fnv1a hashes of the RPC
// names so a rename cannot silently desynchronize registration from routing.
static_assert(kEnqueueSyndromesFunctionId ==
              cudaq::realtime::fnv1a_hash("enqueue_syndromes"));
static_assert(kGetCorrectionsFunctionId ==
              cudaq::realtime::fnv1a_hash("get_corrections"));
static_assert(kResetDecoderFunctionId ==
              cudaq::realtime::fnv1a_hash("reset_decoder"));

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
constexpr std::uint8_t kEnqueueSyndromeBitsArg = 3;
constexpr std::uint8_t kEnqueueArgCount = 4;

constexpr std::uint8_t kGetCorrectionsDecoderIdArg = 0;
constexpr std::uint8_t kGetCorrectionsReturnSizeArg = 1;
constexpr std::uint8_t kGetCorrectionsResetArg = 2;
constexpr std::uint8_t kGetCorrectionsArgCount = 3;

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

// Syndrome/correction bits cross the wire bit-packed (LSB-first), so the
// argument type is CUDAQ_TYPE_BIT_PACKED -- matching the realtime device_call
// lowering for std::vector<bool> (cudaq PR 4816) -- rather than the old
// CUDAQ_TYPE_ARRAY_UINT8 stand-in used before that lowering existed.
static void set_bit_packed(cudaq_type_desc_t &d) {
  d = {};
  d.type_id = CUDAQ_TYPE_BIT_PACKED;
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

  // enqueue_syndromes: 4-arg spec format per decoder_server_runtime.md.
  // decoder_id, counter, syndrome_mapping_id (scalars) + syndrome_bits
  // (bit_packed: element-count prefix == num_syndromes, then LSB-first bits).
  auto &eq = entries[kEnqueueSyndromesEntry];
  configure_entry(eq, kEnqueueSyndromesFunctionId, enqueue_syndromes_host,
                  kEnqueueArgCount, kNoResults);
  set_u64(eq.schema.args[kEnqueueDecoderIdArg]);
  set_u64(eq.schema.args[kEnqueueCounterArg]);
  set_u64(eq.schema.args[kEnqueueMappingIdArg]);
  set_bit_packed(eq.schema.args[kEnqueueSyndromeBitsArg]);

  // get_corrections: 3-arg spec format per decoder_server_runtime.md.
  // decoder_id (scalar) + corrections (OUT std::vector<bool>: the request
  // carries its length as return_size) + reset (scalar).
  auto &gc = entries[kGetCorrectionsEntry];
  configure_entry(gc, kGetCorrectionsFunctionId, get_corrections_host,
                  kGetCorrectionsArgCount, kSingleResult);
  set_u64(gc.schema.args[kGetCorrectionsDecoderIdArg]);
  set_u64(gc.schema.args[kGetCorrectionsReturnSizeArg]);
  set_u8(gc.schema.args[kGetCorrectionsResetArg]);
  set_bit_packed(gc.schema.results[kCorrectionsResult]);

  auto &rd = entries[kResetDecoderEntry];
  configure_entry(rd, kResetDecoderFunctionId, reset_decoder_host,
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
    // Server path: the config path is in the environment, so build the
    // decoder sessions NOW (before the server's READY line). The in-process
    // application path has not called configure_decoders yet at this point;
    // it initializes lazily on the first RPC (see dispatch_rpc).
    if (const char *cfg = std::getenv("CUDAQ_QEC_DECODER_CONFIG");
        cfg && cfg[0] != '\0') {
      try {
        std::call_once(g_init_flag, init_server);
      } catch (const std::exception &e) {
        // CUDAQ core does not expect plugin session creation to throw; a
        // propagating exception would escape the channel-setup path and
        // terminate. Report the config/decoder failure and decline the
        // session instead.
        cudaq::qec::error(
            "decoding-server init failed (CUDAQ_QEC_DECODER_CONFIG={}): {}",
            cfg, e.what());
        return nullptr;
      }
    }
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

/// High-water mark of simultaneously-busy DecodingSession workers -- the
/// server's concurrency evidence for multi-logical-qubit tests.
extern "C" __attribute__((visibility("default"))) uint64_t
cudaqx_qec_decoding_server_max_concurrent() {
  return cudaq::qec::decoding_server::max_concurrent_busy_sessions();
}

/// Stop the DecodingServer receive loop and join its thread. The server calls
/// this before exiting; without it the static g_server_thread would still be
/// joinable at static destruction (std::terminate).
extern "C" __attribute__((visibility("default"))) void
cudaqx_qec_decoding_server_shutdown() {
  if (g_server) {
    g_server->stop();
    if (g_server_thread.joinable())
      g_server_thread.join();
    g_server.reset();
    g_transceiver = nullptr;
  }
}
