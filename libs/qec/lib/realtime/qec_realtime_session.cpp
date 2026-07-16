/*******************************************************************************
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifdef CUDAQ_REALTIME_ROOT

#include "qec_realtime_session.h"

#include "cudaq/qec/logger.h"
#include "cudaq/qec/realtime/decoder_rpc_ids.h"
#include "cudaq/qec/realtime/graph_resources.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <stdexcept>
#include <string>

namespace cudaq::qec::realtime {

namespace {

using Decoders = std::vector<std::unique_ptr<cudaq::qec::decoder>>;

//==============================================================================
// DEVICE-mode helpers
//==============================================================================

// Resolves a host-side C-ABI shim `cudaqx_qec_realtime_dispatch_populate_*`
// at runtime via dlsym(RTLD_DEFAULT, ...).  These shims are defined in
// libcudaq-qec-realtime-cudevice.a and only enter the process when the final
// executable absorbs that static archive (typically via the
// `qec_realtime_app_link_options()` CMake helper).  Resolving by name rather
// than by direct symbol reference keeps libcudaq-qec-realtime-decoding.so free
// of unresolved C-ABI symbols, so it can be safely dlopen'd from consumers that
// do NOT link the cudevice archive (notably the Python extension).  Any such
// consumer that tries to actually USE the device dispatch path lands here, does
// not find the symbol, and surfaces a clean runtime_error with actionable
// linker guidance.
using populate_device_entry_fn = void (*)(void *);
populate_device_entry_fn resolve_populate_shim(const char *symbol_name) {
  void *sym = ::dlsym(RTLD_DEFAULT, symbol_name);
  return reinterpret_cast<populate_device_entry_fn>(sym);
}

// Pinned mapped flags + pinned mapped data, with the device pointer obtained
// via UVA so the GPU dispatcher can read the same backing.
bool allocate_pinned_mapped(std::size_t bytes, void **host_out,
                            void **device_out) {
  void *h = nullptr;
  if (cudaHostAlloc(&h, bytes, cudaHostAllocMapped) != cudaSuccess)
    return false;
  void *d = nullptr;
  if (cudaHostGetDevicePointer(&d, h, 0) != cudaSuccess) {
    cudaFreeHost(h);
    return false;
  }
  std::memset(h, 0, bytes);
  *host_out = h;
  *device_out = d;
  return true;
}

//==============================================================================
// HOST-mode helpers (two-ring CUDAQ_DISPATCH_HOST_CALL handlers)
//==============================================================================

// The HOST_CALL handlers below are plain C function pointers with no
// user-context argument, so the active decoder table must be reachable from a
// process-global.  It is published/cleared by a HOST-mode initialize()/
// finalize() and read concurrently by the host dispatcher thread, hence the
// atomic.  Only one HOST-mode session may be live at a time (enforced in
// initialize()).
std::atomic<Decoders *> g_active_decoders{nullptr};

cudaq::qec::decoder *get_decoder_or_throw(std::int64_t decoder_id) {
  Decoders *decoders = g_active_decoders.load(std::memory_order_acquire);
  if (!decoders || decoder_id < 0 ||
      static_cast<std::size_t>(decoder_id) >= decoders->size() ||
      !(*decoders)[static_cast<std::size_t>(decoder_id)])
    throw std::runtime_error("invalid decoder_id " +
                             std::to_string(decoder_id));
  return (*decoders)[static_cast<std::size_t>(decoder_id)].get();
}

// Two-ring response writer: the request stays in `rx_slot` (read-only); the
// response is written into the distinct `tx_slot`.  The preserved header fields
// (request_id, ptp_timestamp) must be echoed explicitly from rx to tx.  The
// caller (cudaq_host_dispatcher_loop::handle_host_call) publishes tx_flags
// AFTER the handler returns; the handler only needs to write the response body
// + header and release-store the magic.
void write_response(void *tx_slot, const void *rx_slot, std::int32_t status,
                    std::uint32_t result_len = 0) {
  const auto *request =
      static_cast<const cudaq::realtime::RPCHeader *>(rx_slot);
  auto *response = static_cast<cudaq::realtime::RPCResponse *>(tx_slot);
  response->status = status;
  response->result_len = result_len;
  response->request_id = request->request_id;
  response->ptp_timestamp = request->ptp_timestamp;
  __atomic_store_n(&response->magic, cudaq::realtime::RPC_MAGIC_RESPONSE,
                   __ATOMIC_RELEASE);
}

std::uint8_t *response_body(void *tx_slot) {
  return static_cast<std::uint8_t *>(tx_slot) +
         sizeof(cudaq::realtime::RPCResponse);
}

void enqueue_syndromes_host(const void *rx_slot, void *tx_slot,
                            std::size_t slot_size) {
  namespace rpc = cudaq::qec::decoding::rpc;
  try {
    const auto *header =
        static_cast<const cudaq::realtime::RPCHeader *>(rx_slot);
    if (header->arg_len < sizeof(rpc::EnqueueRequestPayload)) {
      write_response(tx_slot, rx_slot, -1);
      return;
    }
    const auto *body = reinterpret_cast<const rpc::EnqueueRequestPayload *>(
        static_cast<const std::uint8_t *>(rx_slot) +
        sizeof(cudaq::realtime::RPCHeader));
    if (body->num_syndromes < 0 || body->syndrome_mapping_id != 0) {
      write_response(tx_slot, rx_slot, -4);
      return;
    }
    const auto num_syndromes = static_cast<std::uint64_t>(body->num_syndromes);
    const std::size_t expected_arg_len =
        rpc::align_to_8(sizeof(rpc::EnqueueRequestPayload) +
                        rpc::bit_packed_bytes(num_syndromes));
    if (header->arg_len != expected_arg_len ||
        sizeof(cudaq::realtime::RPCHeader) + expected_arg_len > slot_size) {
      write_response(tx_slot, rx_slot, -4);
      return;
    }

    auto *decoder = get_decoder_or_throw(body->decoder_id);
    // Reject requests larger than this decoder's per-decode window.  The slot
    // is sized for the largest decoder in the session, so an oversized request
    // for a smaller decoder can still fit the slot; without this guard it would
    // overflow the decoder's accumulation buffer and be silently dropped by
    // enqueue_syndrome (which returns false) while we ACK success.
    if (num_syndromes > decoder->get_num_msyn_per_decode()) {
      write_response(tx_slot, rx_slot, -4);
      return;
    }
    const std::uint8_t *bits = reinterpret_cast<const std::uint8_t *>(body + 1);
    std::vector<std::uint8_t> syndromes(num_syndromes, 0);
    for (std::uint64_t bit = 0; bit < num_syndromes; ++bit)
      syndromes[bit] = (bits[bit >> 3] >> (bit & 7)) & 0x1u;

    // enqueue_syndrome's bool return means "a decode was triggered", not
    // success, so it is intentionally not treated as an error here; the
    // oversize guard above is what rejects malformed lengths.
    (void)decoder->enqueue_syndrome(syndromes.data(), syndromes.size());
    write_response(tx_slot, rx_slot, 0);
  } catch (...) {
    write_response(tx_slot, rx_slot, -2);
  }
}

void get_corrections_host(const void *rx_slot, void *tx_slot,
                          std::size_t slot_size) {
  namespace rpc = cudaq::qec::decoding::rpc;
  try {
    const auto *header =
        static_cast<const cudaq::realtime::RPCHeader *>(rx_slot);
    if (header->arg_len != sizeof(rpc::GetCorrectionsRequestPayload)) {
      write_response(tx_slot, rx_slot, -1);
      return;
    }
    const auto *body =
        reinterpret_cast<const rpc::GetCorrectionsRequestPayload *>(
            static_cast<const std::uint8_t *>(rx_slot) +
            sizeof(cudaq::realtime::RPCHeader));
    if (body->return_size < 0) {
      write_response(tx_slot, rx_slot, -4);
      return;
    }

    auto *decoder = get_decoder_or_throw(body->decoder_id);
    const auto return_size = static_cast<std::uint64_t>(body->return_size);
    if (return_size > decoder->get_num_observables()) {
      write_response(tx_slot, rx_slot, -4);
      return;
    }
    const std::size_t result_len =
        rpc::align_to_8(rpc::bit_packed_bytes(return_size));
    if (sizeof(cudaq::realtime::RPCResponse) + result_len > slot_size) {
      write_response(tx_slot, rx_slot, -5);
      return;
    }

    std::uint8_t *out = response_body(tx_slot);
    std::memset(out, 0, result_len);
    const std::uint8_t *corrections = decoder->get_obs_corrections();
    for (std::uint64_t i = 0; i < return_size; ++i) {
      if (corrections[i] & 0x1u)
        out[i >> 3] |= static_cast<std::uint8_t>(1u << (i & 7));
    }
    if (body->reset != 0)
      decoder->clear_corrections();
    write_response(tx_slot, rx_slot, 0, static_cast<std::uint32_t>(result_len));
  } catch (...) {
    write_response(tx_slot, rx_slot, -2);
  }
}

void reset_decoder_host(const void *rx_slot, void *tx_slot, std::size_t) {
  namespace rpc = cudaq::qec::decoding::rpc;
  try {
    const auto *header =
        static_cast<const cudaq::realtime::RPCHeader *>(rx_slot);
    if (header->arg_len != sizeof(rpc::ResetRequestPayload)) {
      write_response(tx_slot, rx_slot, -1);
      return;
    }
    const auto *body = reinterpret_cast<const rpc::ResetRequestPayload *>(
        static_cast<const std::uint8_t *>(rx_slot) +
        sizeof(cudaq::realtime::RPCHeader));
    get_decoder_or_throw(body->decoder_id)->reset_decoder();
    write_response(tx_slot, rx_slot, 0);
  } catch (...) {
    write_response(tx_slot, rx_slot, -2);
  }
}

} // namespace

//==============================================================================
// ctor / dtor
//==============================================================================

qec_realtime_session::qec_realtime_session(
    std::vector<std::unique_ptr<cudaq::qec::decoder>> &decoders,
    cudaq_dispatch_launch_fn_t device_launch_fn)
    : decoders_(decoders), device_launch_fn_(device_launch_fn) {}

qec_realtime_session::~qec_realtime_session() {
  // Best-effort teardown.  finalize() is null-safe at every step (each resource
  // has its own guard), so calling it from a never-fully-initialized or
  // already-finalized session is a no-op beyond the trace message.
  finalize();
}

//==============================================================================
// classify_mode()
//==============================================================================

void qec_realtime_session::classify_mode() {
  bool any_graph = false;
  bool any_host = false;
  std::size_t non_null = 0;
  for (auto &decoder : decoders_) {
    if (!decoder)
      continue;
    ++non_null;
    if (decoder->supports_graph_dispatch())
      any_graph = true;
    else
      any_host = true;
  }

  if (non_null == 0)
    throw std::runtime_error(
        "qec_realtime_session::initialize: no (non-null) decoders to serve");

  if (any_graph && any_host)
    throw std::runtime_error(
        "qec_realtime_session::initialize: mixed decoder set -- some decoders "
        "support graph dispatch (DEVICE mode) and some do not (HOST mode).  A "
        "single session must be homogeneous: the host loop resolves a slot to "
        "a function table entry by function_id alone, so a GRAPH_LAUNCH and a "
        "HOST_CALL enqueue sharing kEnqueueSyndromesFunctionId would collide.  "
        "Use one decoder per session (or a homogeneous decoder set).");

  device_mode_ = any_graph;

  if (device_mode_ && !device_launch_fn_)
    throw std::runtime_error(
        "qec_realtime_session::initialize: DEVICE mode requires a non-null "
        "device_launch_fn (typically &cudaq_launch_dispatch_kernel_regular "
        "from libcudaq-realtime-dispatch), but the session was constructed "
        "without one.");
}

//==============================================================================
// initialize()
//==============================================================================

void qec_realtime_session::initialize() {
  if (initialized_)
    return;

  classify_mode();

  if (device_mode_) {
    // Be tolerant of being called before any CUDA setup.  The pinned
    // allocations below require cudaDeviceMapHost on the active device.
    cudaError_t flags_err = cudaSetDeviceFlags(cudaDeviceMapHost);
    if (flags_err != cudaSuccess && flags_err != cudaErrorSetOnActiveProcess)
      throw std::runtime_error(
          std::string("qec_realtime_session::initialize: "
                      "cudaSetDeviceFlags(cudaDeviceMapHost) failed: ") +
          cudaGetErrorString(flags_err));
  } else {
    // HOST mode: claim the process-global decoder table up front.  With a
    // single global, a second HOST-mode initialize() would hijack decoder-id
    // resolution for the first session's host loop.
    Decoders *expected = nullptr;
    if (!g_active_decoders.compare_exchange_strong(expected, &decoders_,
                                                   std::memory_order_acq_rel))
      throw std::runtime_error(
          "qec_realtime_session::initialize: another HOST_CALL session is "
          "already active; concurrent HOST-mode sessions are not supported");
  }

  // Everything below acquires resources, so it must be transactional.
  // finalize() is null-safe at every step, so we can roll a half-built session
  // back from any throw.
  try {
    if (device_mode_)
      capture_decoder_graphs();
    allocate_ring_buffer();
    populate_function_table();
    if (device_mode_)
      start_device_loop();
    start_host_loop();
    initialized_ = true;
  } catch (...) {
    CUDA_QEC_WARN("qec_realtime_session::initialize: rolling back partial "
                  "initialization after exception");
    finalize();
    throw;
  }

  if (device_mode_)
    CUDA_QEC_INFO("qec_realtime_session: initialized DEVICE mode "
                  "(num_decoders_with_graph={}, num_slots={}, slot_size={})",
                  num_decoders_with_graph_, num_slots_, slot_size_);
  else
    CUDA_QEC_INFO("qec_realtime_session: initialized HOST mode "
                  "(num_slots={}, slot_size={})",
                  num_slots_, slot_size_);
}

//==============================================================================
// finalize()
//==============================================================================

void qec_realtime_session::finalize() {
  const bool was_initialized = initialized_;
  if (was_initialized)
    initialized_ = false;
  // Note: we intentionally do NOT early-return on !was_initialized; the
  // initialize() rollback path depends on running through the cleanup below.

  stop_loops();

  if (device_mode_) {
    // After stop_loops() the host monitor thread is joined and the persistent
    // device kernel has been signalled to exit, but in-flight worker-stream
    // graph launches submitted before the join can still be running.
    // cudaDeviceSynchronize() drains all outstanding work, so the subsequent
    // release_decode_graph() can't free buffers a still-running enqueue graph
    // dereferences.  finalize() is cold-path so the sync cost is irrelevant.
    cudaDeviceSynchronize();

    for (std::size_t i = 0; i < captured_graphs_.size(); ++i) {
      if (captured_graphs_[i] && i < decoders_.size() && decoders_[i])
        decoders_[i]->release_decode_graph(captured_graphs_[i]);
    }
  }
  captured_graphs_.clear();
  num_decoders_with_graph_ = 0;

  // Release the HOST-mode global if this session owns it (no-op otherwise).
  Decoders *self = &decoders_;
  g_active_decoders.compare_exchange_strong(self, nullptr,
                                            std::memory_order_acq_rel);

  // Free the function table with the allocator that matches the mode it was
  // built with.
  if (function_table_host_) {
    if (device_mode_)
      cudaFreeHost(function_table_host_);
    else
      std::free(function_table_host_);
    function_table_host_ = nullptr;
    function_table_dev_ = nullptr;
  }
  function_table_count_ = 0;
  get_corrections_fn_id_ = 0;
  reset_decoder_fn_id_ = 0;

  if (device_stats_dev_) {
    cudaFree(device_stats_dev_);
    device_stats_dev_ = nullptr;
  }

  // Free the ring.  DEVICE mode allocated pinned-mapped backings; HOST mode
  // allocated plain host memory with _dev aliasing _host (free _host only).
  auto free_ring_u64 = [&](volatile std::uint64_t *&host,
                           volatile std::uint64_t *&dev) {
    if (host) {
      if (device_mode_)
        cudaFreeHost(const_cast<std::uint64_t *>(host));
      else
        std::free(const_cast<std::uint64_t *>(host));
    }
    host = nullptr;
    dev = nullptr;
  };
  auto free_ring_u8 = [&](std::uint8_t *&host, std::uint8_t *&dev) {
    if (host) {
      if (device_mode_)
        cudaFreeHost(host);
      else
        std::free(host);
    }
    host = nullptr;
    dev = nullptr;
  };
  free_ring_u64(tx_flags_host_, tx_flags_dev_);
  free_ring_u64(rx_flags_host_, rx_flags_dev_);
  free_ring_u8(tx_data_host_, tx_data_dev_);
  free_ring_u8(rx_data_host_, rx_data_dev_);

  if (shutdown_flag_host_) {
    cudaFreeHost(shutdown_flag_host_);
    shutdown_flag_host_ = nullptr;
    shutdown_flag_dev_ = nullptr;
  }

  std::memset(&ringbuffer_, 0, sizeof(ringbuffer_));
  std::memset(&host_ctx_, 0, sizeof(host_ctx_));

  if (was_initialized)
    CUDA_QEC_INFO("qec_realtime_session: finalized");
}

//==============================================================================
// capture_decoder_graphs()  [DEVICE mode]
//==============================================================================

void qec_realtime_session::capture_decoder_graphs() {
  captured_graphs_.assign(decoders_.size(), nullptr);
  num_decoders_with_graph_ = 0;

  // kMaxDispatchedDecoders sizes the device-side g_decoder_state_table[].
  if (decoders_.size() > cudaq::qec::decoding::rpc::kMaxDispatchedDecoders)
    throw std::runtime_error(
        "qec_realtime_session::initialize: requested " +
        std::to_string(decoders_.size()) +
        " decoders but the realtime dispatch supports at most " +
        std::to_string(cudaq::qec::decoding::rpc::kMaxDispatchedDecoders) +
        " (kMaxDispatchedDecoders).");

  for (std::size_t i = 0; i < decoders_.size(); ++i) {
    auto *dec = decoders_[i].get();
    if (!dec)
      continue;
    if (!dec->supports_graph_dispatch())
      throw std::runtime_error(
          "qec_realtime_session::initialize: decoder " + std::to_string(i) +
          " does not support graph dispatch in DEVICE mode.");

    // reserved_sms = 0 is intentional for the inproc_rpc desktop / CI path.
    void *raw = dec->capture_decode_graph(/*reserved_sms=*/0);
    if (!raw)
      throw std::runtime_error("qec_realtime_session::initialize: decoder " +
                               std::to_string(i) +
                               " returned null from capture_decode_graph()");
    captured_graphs_[i] = raw;

    auto *gres = static_cast<cudaq::qec::realtime::graph_resources *>(raw);
    if (!gres->graph_exec || !gres->function_id)
      throw std::runtime_error(
          "qec_realtime_session::initialize: decoder " + std::to_string(i) +
          " produced incomplete graph_resources (graph_exec / function_id)");

    // All N enqueue_syndromes graphs share a single canonical function_id; the
    // host monitor disambiguates per-decoder via routing_key == decoder_id.
    if (gres->function_id !=
        cudaq::qec::decoding::rpc::kEnqueueSyndromesFunctionId)
      throw std::runtime_error(
          "qec_realtime_session::initialize: decoder " + std::to_string(i) +
          " published a non-canonical enqueue function_id");

    ++num_decoders_with_graph_;
  }

  if (num_decoders_with_graph_ == 0)
    throw std::runtime_error(
        "qec_realtime_session::initialize: no decoders to capture graphs for");
}

//==============================================================================
// allocate_ring_buffer()  [branches on device_mode_]
//==============================================================================

void qec_realtime_session::allocate_ring_buffer() {
  namespace rpc = cudaq::qec::decoding::rpc;
  using cudaq::realtime::RPCHeader;
  using cudaq::realtime::RPCResponse;

  // Slot size: largest body across the RPC trio, over all served decoders.
  std::size_t max_measurements = 0;
  std::size_t max_observables = 0;
  for (std::size_t i = 0; i < decoders_.size(); ++i) {
    auto *dec = decoders_[i].get();
    if (!dec)
      continue;
    // DEVICE mode only sizes for decoders that captured a graph.
    if (device_mode_ && !captured_graphs_[i])
      continue;
    max_measurements =
        std::max<std::size_t>(max_measurements, dec->get_num_msyn_per_decode());
    max_observables =
        std::max<std::size_t>(max_observables, dec->get_num_observables());
  }

  const std::size_t enqueue_req =
      sizeof(RPCHeader) +
      rpc::align_to_8(sizeof(rpc::EnqueueRequestPayload) +
                      rpc::bit_packed_bytes(max_measurements));
  const std::size_t get_req =
      sizeof(RPCHeader) + sizeof(rpc::GetCorrectionsRequestPayload);
  const std::size_t reset_req =
      sizeof(RPCHeader) + sizeof(rpc::ResetRequestPayload);
  const std::size_t enqueue_resp = sizeof(RPCResponse);
  const std::size_t get_resp =
      sizeof(RPCResponse) +
      rpc::align_to_8(rpc::bit_packed_bytes(max_observables));
  const std::size_t reset_resp = sizeof(RPCResponse);

  slot_size_ = std::max({enqueue_req, get_req, reset_req, enqueue_resp,
                         get_resp, reset_resp, std::size_t{64}});

  if (device_mode_) {
    // Round up to 256-byte alignment (keeps slot stride deterministic).
    constexpr std::size_t kSlotAlignment = 256;
    slot_size_ = (slot_size_ + (kSlotAlignment - 1)) & ~(kSlotAlignment - 1);

    auto alloc_u64 = [&](volatile std::uint64_t *&host,
                         volatile std::uint64_t *&dev, const char *what) {
      void *h = nullptr;
      void *d = nullptr;
      if (!allocate_pinned_mapped(num_slots_ * sizeof(std::uint64_t), &h, &d))
        throw std::runtime_error(
            std::string(
                "qec_realtime_session::initialize: failed to allocate ") +
            what);
      host = static_cast<volatile std::uint64_t *>(h);
      dev = static_cast<volatile std::uint64_t *>(d);
    };
    auto alloc_u8 = [&](std::uint8_t *&host, std::uint8_t *&dev,
                        const char *what) {
      void *h = nullptr;
      void *d = nullptr;
      if (!allocate_pinned_mapped(num_slots_ * slot_size_, &h, &d))
        throw std::runtime_error(
            std::string(
                "qec_realtime_session::initialize: failed to allocate ") +
            what);
      host = static_cast<std::uint8_t *>(h);
      dev = static_cast<std::uint8_t *>(d);
    };
    alloc_u64(rx_flags_host_, rx_flags_dev_, "rx_flags");
    alloc_u64(tx_flags_host_, tx_flags_dev_, "tx_flags");
    alloc_u8(rx_data_host_, rx_data_dev_, "RX ring data");
    alloc_u8(tx_data_host_, tx_data_dev_, "TX ring data");

    {
      void *h = nullptr;
      void *d = nullptr;
      if (!allocate_pinned_mapped(sizeof(int), &h, &d))
        throw std::runtime_error("qec_realtime_session::initialize: failed to "
                                 "allocate shutdown flag");
      shutdown_flag_host_ = static_cast<int *>(h);
      *shutdown_flag_host_ = 0;
      shutdown_flag_dev_ = static_cast<int *>(d);
    }
  } else {
    // HOST mode: plain host memory; the device-visible pointers alias the host
    // backings (no GPU required at runtime).  The host loop reads only the
    // *_host views; the producer's address-as-flag publish uses rx_data_dev()
    // (== rx_data_host_ here), which the host loop dereferences as host memory.
    auto alloc_u64 = [&](volatile std::uint64_t *&host,
                         volatile std::uint64_t *&dev, const char *what) {
      void *p = std::calloc(num_slots_, sizeof(std::uint64_t));
      if (!p)
        throw std::runtime_error(
            std::string(
                "qec_realtime_session::initialize: failed to allocate ") +
            what);
      host = static_cast<volatile std::uint64_t *>(p);
      dev = host;
    };
    auto alloc_u8 = [&](std::uint8_t *&host, std::uint8_t *&dev,
                        const char *what) {
      void *p = std::calloc(num_slots_, slot_size_);
      if (!p)
        throw std::runtime_error(
            std::string(
                "qec_realtime_session::initialize: failed to allocate ") +
            what);
      host = static_cast<std::uint8_t *>(p);
      dev = host;
    };
    alloc_u64(rx_flags_host_, rx_flags_dev_, "rx_flags");
    alloc_u64(tx_flags_host_, tx_flags_dev_, "tx_flags");
    alloc_u8(rx_data_host_, rx_data_dev_, "RX ring data");
    alloc_u8(tx_data_host_, tx_data_dev_, "TX ring data");
  }

  std::memset(&ringbuffer_, 0, sizeof(ringbuffer_));
  ringbuffer_.rx_flags = rx_flags_dev_;
  ringbuffer_.tx_flags = tx_flags_dev_;
  ringbuffer_.rx_data = rx_data_dev_;
  ringbuffer_.tx_data = tx_data_dev_;
  ringbuffer_.rx_stride_sz = slot_size_;
  ringbuffer_.tx_stride_sz = slot_size_;
  ringbuffer_.rx_flags_host = rx_flags_host_;
  ringbuffer_.tx_flags_host = tx_flags_host_;
  ringbuffer_.rx_data_host = rx_data_host_;
  ringbuffer_.tx_data_host = tx_data_host_;
}

//==============================================================================
// populate_function_table()  [branches on device_mode_]
//==============================================================================

void qec_realtime_session::populate_function_table() {
  namespace rpc = cudaq::qec::decoding::rpc;

  if (!device_mode_) {
    // HOST mode: 3 HOST_CALL entries (enqueue, get_corrections, reset).  Plain
    // host allocation -- host_fn pointers are host code addresses; _dev aliases
    // _host.  decoder_id routing happens inside each handler via the payload.
    function_table_count_ = 3;
    void *p =
        std::calloc(function_table_count_, sizeof(cudaq_function_entry_t));
    if (!p)
      throw std::runtime_error("qec_realtime_session::initialize: failed to "
                               "allocate function table");
    function_table_host_ = static_cast<cudaq_function_entry_t *>(p);
    function_table_dev_ = function_table_host_;

    function_table_host_[0].handler.host_fn = enqueue_syndromes_host;
    function_table_host_[0].function_id = rpc::kEnqueueSyndromesFunctionId;
    function_table_host_[0].dispatch_mode = CUDAQ_DISPATCH_HOST_CALL;

    function_table_host_[1].handler.host_fn = get_corrections_host;
    function_table_host_[1].function_id = rpc::kGetCorrectionsFunctionId;
    function_table_host_[1].dispatch_mode = CUDAQ_DISPATCH_HOST_CALL;

    function_table_host_[2].handler.host_fn = reset_decoder_host;
    function_table_host_[2].function_id = rpc::kResetDecoderFunctionId;
    function_table_host_[2].dispatch_mode = CUDAQ_DISPATCH_HOST_CALL;

    get_corrections_fn_id_ = rpc::kGetCorrectionsFunctionId;
    reset_decoder_fn_id_ = rpc::kResetDecoderFunctionId;
    return;
  }

  // DEVICE mode: N GRAPH_LAUNCH entries (one per captured decoder) + 2
  // DEVICE_CALL entries (get_corrections, reset_decoder).  Pinned-mapped so the
  // device dispatcher kernel reads the same backing.
  function_table_count_ = num_decoders_with_graph_ + 2;

  void *h = nullptr;
  void *d = nullptr;
  if (!allocate_pinned_mapped(
          function_table_count_ * sizeof(cudaq_function_entry_t), &h, &d))
    throw std::runtime_error(
        "qec_realtime_session::initialize: failed to allocate function table");
  function_table_host_ = static_cast<cudaq_function_entry_t *>(h);
  function_table_dev_ = static_cast<cudaq_function_entry_t *>(d);

  // [0..N-1] GRAPH_LAUNCH per-decoder enqueue.  All share
  // kEnqueueSyndromesFunctionId; the host monitor disambiguates via
  // routing_key = source decoder_id (matched against arg0 of the request).
  std::size_t slot = 0;
  for (std::size_t i = 0; i < decoders_.size(); ++i) {
    if (!captured_graphs_[i])
      continue;
    auto *gres = static_cast<cudaq::qec::realtime::graph_resources *>(
        captured_graphs_[i]);
    auto &entry = function_table_host_[slot++];
    entry.handler.graph_exec = gres->graph_exec;
    entry.function_id = rpc::kEnqueueSyndromesFunctionId;
    entry.dispatch_mode = CUDAQ_DISPATCH_GRAPH_LAUNCH;
    entry.routing_key = static_cast<std::uint64_t>(i);
  }

  // [N] DEVICE_CALL get_corrections.  Resolved by name from the cudevice
  // archive so consumers that did not link it can still load this .so.
  get_corrections_fn_id_ = rpc::kGetCorrectionsFunctionId;
  auto populate_get_corrections = resolve_populate_shim(
      "cudaqx_qec_realtime_dispatch_populate_get_corrections_device_entry");
  if (!populate_get_corrections)
    throw std::runtime_error(
        "qec_realtime_session::initialize: "
        "cudaqx_qec_realtime_dispatch_populate_get_corrections_device_entry "
        "not found via dlsym(RTLD_DEFAULT, ...).  The final binary must link "
        "libcudaq-qec-realtime-cudevice.a (or the static parts of "
        "decoder_rpc_dispatch.cu via qec_realtime_app_link_options()).");
  populate_get_corrections(&function_table_host_[slot]);
  function_table_host_[slot].function_id = get_corrections_fn_id_;
  function_table_host_[slot].routing_key = 0;
  if (function_table_host_[slot].dispatch_mode != CUDAQ_DISPATCH_DEVICE_CALL ||
      !function_table_host_[slot].handler.device_fn_ptr)
    throw std::runtime_error(
        "qec_realtime_session::initialize: "
        "populate_get_corrections_device_entry did not produce a valid "
        "DEVICE_CALL entry (plugin bug)");
  ++slot;

  // [N+1] DEVICE_CALL reset_decoder.  Same dlsym contract as above.
  reset_decoder_fn_id_ = rpc::kResetDecoderFunctionId;
  auto populate_reset_decoder = resolve_populate_shim(
      "cudaqx_qec_realtime_dispatch_populate_reset_decoder_device_entry");
  if (!populate_reset_decoder)
    throw std::runtime_error(
        "qec_realtime_session::initialize: "
        "cudaqx_qec_realtime_dispatch_populate_reset_decoder_device_entry "
        "not found via dlsym(RTLD_DEFAULT, ...).  The final binary must link "
        "libcudaq-qec-realtime-cudevice.a (or the static parts of "
        "decoder_rpc_dispatch.cu via qec_realtime_app_link_options()).");
  populate_reset_decoder(&function_table_host_[slot]);
  function_table_host_[slot].function_id = reset_decoder_fn_id_;
  function_table_host_[slot].routing_key = 0;
  if (function_table_host_[slot].dispatch_mode != CUDAQ_DISPATCH_DEVICE_CALL ||
      !function_table_host_[slot].handler.device_fn_ptr)
    throw std::runtime_error(
        "qec_realtime_session::initialize: "
        "populate_reset_decoder_device_entry did not produce a valid "
        "DEVICE_CALL entry (plugin bug)");
  ++slot;
}

//==============================================================================
// start_device_loop()  [DEVICE mode]
//==============================================================================

void qec_realtime_session::start_device_loop() {
  if (cudaq_dispatch_manager_create(&device_manager_) != CUDAQ_OK)
    throw std::runtime_error(
        "qec_realtime_session::initialize: cudaq_dispatch_manager_create "
        "failed");

  cudaq_dispatcher_config_t dev_config{};
  dev_config.device_id = 0;
  dev_config.num_blocks = 1;
  dev_config.threads_per_block = 64;
  dev_config.num_slots = static_cast<std::uint32_t>(num_slots_);
  dev_config.slot_size = static_cast<std::uint32_t>(slot_size_);
  dev_config.vp_id = 0;
  dev_config.kernel_type = CUDAQ_KERNEL_REGULAR;
  dev_config.dispatch_mode = CUDAQ_DISPATCH_DEVICE_CALL;
  dev_config.dispatch_path = CUDAQ_DISPATCH_PATH_DEVICE;
  dev_config.shared_ring_mode = 1;
  dev_config.skip_tx_markers = 1;

  if (cudaq_dispatcher_create(device_manager_, &dev_config,
                              &device_dispatcher_) != CUDAQ_OK)
    throw std::runtime_error(
        "qec_realtime_session::initialize: cudaq_dispatcher_create (DEVICE) "
        "failed");

  if (cudaq_dispatcher_set_ringbuffer(device_dispatcher_, &ringbuffer_) !=
      CUDAQ_OK)
    throw std::runtime_error(
        "qec_realtime_session::initialize: cudaq_dispatcher_set_ringbuffer "
        "(DEVICE) failed");

  cudaq_function_table_t shared_table{};
  shared_table.entries = function_table_dev_;
  shared_table.count = static_cast<std::uint32_t>(function_table_count_);
  if (cudaq_dispatcher_set_function_table(device_dispatcher_, &shared_table) !=
      CUDAQ_OK)
    throw std::runtime_error(
        "qec_realtime_session::initialize: "
        "cudaq_dispatcher_set_function_table (DEVICE) failed");

  if (cudaMalloc(&device_stats_dev_, sizeof(std::uint64_t)) != cudaSuccess ||
      cudaMemset(device_stats_dev_, 0, sizeof(std::uint64_t)) != cudaSuccess)
    throw std::runtime_error(
        "qec_realtime_session::initialize: device_stats_dev allocation failed");

  if (cudaq_dispatcher_set_control(device_dispatcher_, shutdown_flag_dev_,
                                   device_stats_dev_) != CUDAQ_OK)
    throw std::runtime_error(
        "qec_realtime_session::initialize: cudaq_dispatcher_set_control "
        "failed");

  if (cudaq_dispatcher_set_launch_fn(device_dispatcher_, device_launch_fn_) !=
      CUDAQ_OK)
    throw std::runtime_error(
        "qec_realtime_session::initialize: cudaq_dispatcher_set_launch_fn "
        "failed");

  if (cudaq_dispatcher_start(device_dispatcher_) != CUDAQ_OK)
    throw std::runtime_error(
        "qec_realtime_session::initialize: cudaq_dispatcher_start (DEVICE) "
        "failed");
}

//==============================================================================
// start_host_loop()  [branches on device_mode_]
//==============================================================================

void qec_realtime_session::start_host_loop() {
  if (!device_mode_) {
    // HOST mode: inline HOST_CALL handlers, no graph worker pool.
    std::memset(&host_ctx_, 0, sizeof(host_ctx_));
    host_ctx_.ringbuffer = ringbuffer_;
    host_ctx_.config.num_slots = static_cast<std::uint32_t>(num_slots_);
    host_ctx_.config.slot_size = static_cast<std::uint32_t>(slot_size_);
    host_ctx_.config.dispatch_path = CUDAQ_DISPATCH_PATH_HOST;
    host_ctx_.config.dispatch_mode = CUDAQ_DISPATCH_HOST_CALL;
    host_ctx_.config.skip_tx_markers = 1;
    // shared_ring_mode lets the loop scan the whole ring for a non-zero
    // rx_flag instead of sitting on a single advancing cursor.  The producer's
    // acquire_slot() reuses the first free slot (slot 0 for serialized
    // round-trips), so without ring-scanning the loop's cursor would advance
    // past the reused slot and deadlock.  There is no second dispatcher here;
    // we only want the scan behavior.
    host_ctx_.config.shared_ring_mode = 1;
    host_ctx_.function_table.entries = function_table_host_;
    host_ctx_.function_table.count =
        static_cast<std::uint32_t>(function_table_count_);
    host_ctx_.shutdown_flag = &shutdown_flag_;
    host_ctx_.stats_counter = &host_stats_counter_;
    host_ctx_.skip_stream_sweep = true;
    shutdown_flag_ = 0;

    host_loop_thread_ =
        std::thread([this]() { cudaq_host_dispatcher_loop(&host_ctx_); });
    return;
  }

  // DEVICE mode: one graph worker per captured decoder.
  host_workers_.assign(num_decoders_with_graph_,
                       cudaq_host_dispatch_worker_t{});
  // host_worker_streams_ is indexed by *source* decoder_id (sparse), not packed
  // worker slot, because callers pass decoder_id.
  host_worker_streams_.assign(decoders_.size(),
                              static_cast<cudaStream_t>(nullptr));

  // The host_dispatcher.h public ctx exposes a SINGLE mailbox bank per ctx.
  // Demo 1's scope is one decoder per session, so the single-bank limitation
  // is fine -- the lone worker dispatches every enqueue RPC.
  if (num_decoders_with_graph_ > 1)
    throw std::runtime_error(
        "qec_realtime_session::initialize: multi-decoder host dispatch is not "
        "yet supported (num_decoders_with_graph=" +
        std::to_string(num_decoders_with_graph_) +
        ").  libcudaq-realtime's host dispatcher exposes a single h_mailbox_"
        "bank per loop ctx; Demo 1 is scoped to one decoder.");

  void **mailbox_bank = nullptr;

  std::size_t slot = 0;
  for (std::size_t i = 0; i < decoders_.size(); ++i) {
    if (!captured_graphs_[i])
      continue;
    auto *gres = static_cast<cudaq::qec::realtime::graph_resources *>(
        captured_graphs_[i]);

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess)
      throw std::runtime_error(
          "qec_realtime_session::initialize: cudaStreamCreate for HOST_LOOP "
          "worker " +
          std::to_string(slot) + " (decoder_id=" + std::to_string(i) +
          ") failed");
    host_worker_streams_[i] = stream;

    auto &w = host_workers_[slot];
    w.graph_exec = gres->graph_exec;
    w.stream = stream;
    w.function_id = cudaq::qec::decoding::rpc::kEnqueueSyndromesFunctionId;
    w.routing_key = static_cast<std::uint64_t>(i);
    w.pre_launch_fn = nullptr;
    w.pre_launch_data = nullptr;
    w.post_launch_fn = nullptr;
    w.post_launch_data = nullptr;

    if (slot == 0)
      mailbox_bank = gres->h_mailbox;
    ++slot;
  }

  host_idle_mask_storage_ = new std::uint64_t(
      num_decoders_with_graph_ < 64
          ? ((std::uint64_t{1} << num_decoders_with_graph_) - 1)
          : ~std::uint64_t{0});
  host_live_dispatched_storage_ = new std::uint64_t(0);
  host_inflight_slot_tags_ = new int[num_decoders_with_graph_];
  for (std::size_t i = 0; i < num_decoders_with_graph_; ++i)
    host_inflight_slot_tags_[i] = -1;

  // Per-worker GraphIOContext array (pinned-mapped so both CPU monitor and GPU
  // graph see the same backing).
  {
    void *h = nullptr;
    void *d = nullptr;
    const std::size_t bytes =
        num_decoders_with_graph_ * sizeof(cudaq::realtime::GraphIOContext);
    if (!allocate_pinned_mapped(bytes, &h, &d))
      throw std::runtime_error("qec_realtime_session::start_host_loop: failed "
                               "to allocate per-worker "
                               "GraphIOContext array");
    std::memset(h, 0, bytes);
    io_ctxs_host_ = static_cast<cudaq::realtime::GraphIOContext *>(h);
    io_ctxs_dev_ = static_cast<cudaq::realtime::GraphIOContext *>(d);
  }

  std::memset(&host_ctx_, 0, sizeof(host_ctx_));
  host_ctx_.ringbuffer = ringbuffer_;
  host_ctx_.config.num_slots = static_cast<std::uint32_t>(num_slots_);
  host_ctx_.config.slot_size = static_cast<std::uint32_t>(slot_size_);
  host_ctx_.config.shared_ring_mode = 1;
  host_ctx_.config.skip_tx_markers = 1;
  // The host loop dereferences these entries on the CPU, so it must use the
  // host view of the (pinned-mapped) table.  Under UVA the host and device
  // addresses are equal, but the host pointer is the correct/portable choice;
  // the device dispatcher separately receives function_table_dev_ via
  // cudaq_dispatcher_set_function_table().
  host_ctx_.function_table.entries = function_table_host_;
  host_ctx_.function_table.count =
      static_cast<std::uint32_t>(function_table_count_);
  host_ctx_.workers = host_workers_.data();
  host_ctx_.num_workers = host_workers_.size();
  host_ctx_.h_mailbox_bank = mailbox_bank;
  host_ctx_.shutdown_flag = shutdown_flag_host_;
  host_ctx_.stats_counter = &host_stats_counter_;
  host_ctx_.live_dispatched = host_live_dispatched_storage_;
  host_ctx_.idle_mask = host_idle_mask_storage_;
  host_ctx_.inflight_slot_tags = host_inflight_slot_tags_;
  host_ctx_.io_ctxs_host = io_ctxs_host_;
  host_ctx_.io_ctxs_dev = io_ctxs_dev_;
  host_ctx_.skip_stream_sweep = false;

  host_loop_thread_ =
      std::thread([this]() { cudaq_host_dispatcher_loop(&host_ctx_); });
}

//==============================================================================
// stop_loops()
//==============================================================================

void qec_realtime_session::stop_loops() {
  // Signal shutdown to whichever flag the active host loop polls (and, in
  // DEVICE mode, the persistent device kernel which shares the pinned flag).
  if (device_mode_) {
    if (shutdown_flag_host_) {
      __atomic_store_n(shutdown_flag_host_, 1, __ATOMIC_RELEASE);
      __sync_synchronize();
    }
  } else {
    __atomic_store_n(&shutdown_flag_, 1, __ATOMIC_RELEASE);
    __sync_synchronize();
  }

  if (host_loop_thread_.joinable())
    host_loop_thread_.join();

  if (device_dispatcher_) {
    cudaq_dispatcher_stop(device_dispatcher_);
    cudaq_dispatcher_destroy(device_dispatcher_);
    device_dispatcher_ = nullptr;
  }
  if (device_manager_) {
    cudaq_dispatch_manager_destroy(device_manager_);
    device_manager_ = nullptr;
  }

  for (auto s : host_worker_streams_) {
    if (s)
      cudaStreamDestroy(s);
  }
  host_worker_streams_.clear();
  host_workers_.clear();

  delete host_idle_mask_storage_;
  host_idle_mask_storage_ = nullptr;
  delete host_live_dispatched_storage_;
  host_live_dispatched_storage_ = nullptr;
  delete[] host_inflight_slot_tags_;
  host_inflight_slot_tags_ = nullptr;

  if (io_ctxs_host_) {
    cudaFreeHost(io_ctxs_host_);
    io_ctxs_host_ = nullptr;
    io_ctxs_dev_ = nullptr;
  }
}

} // namespace cudaq::qec::realtime

#endif // CUDAQ_REALTIME_ROOT
