/*******************************************************************************
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifdef CUDAQ_REALTIME_ROOT

#include "qec_realtime_session.h"

#include "../hardware_guards.h"

#include "cudaq/qec/logger.h"
#include "cudaq/qec/realtime/decoder_rpc_wire_format.h"
#include "cudaq/qec/realtime/graph_resources.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime_api.h>
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

// The device-graph dispatch API (cudaq_create_dispatch_graph_regular /
// cudaq_launch_dispatch_graph / cudaq_destroy_dispatch_graph) lives in
// libcudaq-realtime-dispatch.a -- relocatable device code the host executable
// absorbs and device-links (see qec_realtime_app_link_options()).  It must be
// resolved from the SAME image as the populate shims above: the dispatch
// kernel invokes the decoder's __device__ function pointers captured by those
// shims, and a device function pointer is only valid inside the CUDA module
// (device-link unit) that produced it.  If this .so carried its own copy of
// the archive, its dispatch kernel would live in a different CUDA module than
// the executable's decoder handlers and the DEVICE_CALL would trap with
// cudaErrorIllegalInstruction.  Resolving via dlsym(RTLD_DEFAULT, ...) binds
// to the executable's copy (exported via --export-dynamic +
// CUDAQ_REALTIME_DISPATCH_API default visibility), keeping kernel and handlers
// in one module -- and keeps this .so free of undefined C-ABI symbols for
// consumers that never touch the device path.
using create_dispatch_graph_fn_t = cudaError_t (*)(
    volatile std::uint64_t *, volatile std::uint64_t *, std::uint8_t *,
    std::uint8_t *, std::size_t, std::size_t, cudaq_function_entry_t *,
    std::size_t, void *, volatile int *, std::uint64_t *, std::size_t,
    std::uint32_t, std::uint32_t, cudaGraphExec_t, cudaStream_t,
    cudaq_dispatch_graph_context **);
using launch_dispatch_graph_fn_t =
    cudaError_t (*)(cudaq_dispatch_graph_context *, cudaStream_t);

template <typename FnT>
FnT resolve_dispatch_graph_api(const char *symbol_name) {
  void *sym = ::dlsym(RTLD_DEFAULT, symbol_name);
  if (!sym)
    throw std::runtime_error(
        std::string("qec_realtime_session::initialize: ") + symbol_name +
        " not found via dlsym(RTLD_DEFAULT, ...).  The host executable must "
        "absorb libcudaq-realtime-dispatch.a and link with --export-dynamic "
        "(see qec_realtime_app_link_options()).");
  return reinterpret_cast<FnT>(sym);
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

namespace rpc = cudaq::qec::decoding::rpc;

// The HOST_CALL handlers below are plain C function pointers with no
// user-context argument, so the active decoder table must be reachable from a
// process-global.  It is published/cleared by a HOST-mode initialize()/
// finalize() and read concurrently by the host dispatcher thread, hence the
// atomic.  Only one HOST-mode session may be live at a time (enforced in
// initialize()).
std::atomic<Decoders *> g_active_decoders{nullptr};

// Throws std::out_of_range for an unknown decoder_id so the handlers below can
// map it to RpcStatus::INVALID_DECODER, mirroring SessionRegistry::get() /
// RpcDispatcher on the network-transport path.
cudaq::qec::decoder *get_decoder_or_throw(std::int64_t decoder_id) {
  Decoders *decoders = g_active_decoders.load(std::memory_order_acquire);
  if (!decoders || decoder_id < 0 ||
      static_cast<std::size_t>(decoder_id) >= decoders->size() ||
      !(*decoders)[static_cast<std::size_t>(decoder_id)])
    throw std::out_of_range("invalid decoder_id " + std::to_string(decoder_id));
  return (*decoders)[static_cast<std::size_t>(decoder_id)].get();
}

// Point the calling thread at the decoder's pinned CUDA device before work
// that allocates or launches on it. Set-and-leave (no restore): one
// dispatcher thread serves all decoders, so the thread simply converges to
// the device of the decoder it is currently serving; cudaSetDevice on an
// already-current device is a cheap no-op.
static void apply_decoder_cuda_device(cudaq::qec::decoder *dec) {
  if (!dec)
    return;
  // Throws on failure (fail fast): host dispatch surfaces it as an error
  // response and graph initialization aborts, rather than continuing on
  // whichever device happened to be current.
  cudaq::qec::detail_affinity::set_cuda_device_for_decode(
      dec->get_cuda_device_id());
}

// Two-ring response writer: the request stays in `rx_slot` (read-only); the
// response is written into the distinct `tx_slot`.  The preserved header fields
// (request_id, ptp_timestamp) must be echoed explicitly from rx to tx.  The
// caller (cudaq_host_dispatcher_loop::handle_host_call) publishes tx_flags
// AFTER the handler returns; the handler only needs to write the response body
// + header and release-store the magic.
void write_response(void *tx_slot, const void *rx_slot, rpc::RpcStatus status,
                    std::uint32_t result_len = 0) {
  const auto *request =
      static_cast<const cudaq::realtime::RPCHeader *>(rx_slot);
  auto *response = static_cast<cudaq::realtime::RPCResponse *>(tx_slot);
  response->status = static_cast<std::int32_t>(status);
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
  try {
    const auto *header =
        static_cast<const cudaq::realtime::RPCHeader *>(rx_slot);
    if (header->arg_len < sizeof(rpc::EnqueueRequestPayload)) {
      write_response(tx_slot, rx_slot, rpc::RpcStatus::BAD_REQUEST);
      return;
    }
    const auto *body = reinterpret_cast<const rpc::EnqueueRequestPayload *>(
        static_cast<const std::uint8_t *>(rx_slot) +
        sizeof(cudaq::realtime::RPCHeader));
    if (body->num_syndromes < 0 || body->syndrome_mapping_id != 0) {
      write_response(tx_slot, rx_slot, rpc::RpcStatus::BAD_REQUEST);
      return;
    }
    const auto num_syndromes = static_cast<std::uint64_t>(body->num_syndromes);
    const std::size_t expected_arg_len = sizeof(rpc::EnqueueRequestPayload) +
                                         rpc::bit_packed_bytes(num_syndromes);
    if (header->arg_len != expected_arg_len ||
        sizeof(cudaq::realtime::RPCHeader) + expected_arg_len > slot_size) {
      write_response(tx_slot, rx_slot, rpc::RpcStatus::BAD_REQUEST);
      return;
    }

    auto *decoder = get_decoder_or_throw(body->decoder_id);
    apply_decoder_cuda_device(decoder);
    // Reject requests larger than this decoder's per-decode window.  The slot
    // is sized for the largest decoder in the session, so an oversized request
    // for a smaller decoder can still fit the slot; without this guard it would
    // overflow the decoder's accumulation buffer and be silently dropped by
    // enqueue_syndrome (which returns false) while we ACK success.
    if (num_syndromes > decoder->get_num_msyn_per_decode()) {
      write_response(tx_slot, rx_slot, rpc::RpcStatus::BAD_REQUEST);
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
    write_response(tx_slot, rx_slot, rpc::RpcStatus::OK);
  } catch (const std::out_of_range &) {
    write_response(tx_slot, rx_slot, rpc::RpcStatus::INVALID_DECODER);
  } catch (...) {
    write_response(tx_slot, rx_slot, rpc::RpcStatus::INTERNAL_ERROR);
  }
}

void get_corrections_host(const void *rx_slot, void *tx_slot,
                          std::size_t slot_size) {
  try {
    const auto *header =
        static_cast<const cudaq::realtime::RPCHeader *>(rx_slot);
    if (header->arg_len != sizeof(rpc::GetCorrectionsRequestPayload)) {
      write_response(tx_slot, rx_slot, rpc::RpcStatus::BAD_REQUEST);
      return;
    }
    const auto *body =
        reinterpret_cast<const rpc::GetCorrectionsRequestPayload *>(
            static_cast<const std::uint8_t *>(rx_slot) +
            sizeof(cudaq::realtime::RPCHeader));
    if (body->return_size < 0) {
      write_response(tx_slot, rx_slot, rpc::RpcStatus::BAD_REQUEST);
      return;
    }

    auto *decoder = get_decoder_or_throw(body->decoder_id);
    const auto return_size = static_cast<std::uint64_t>(body->return_size);
    if (return_size > decoder->get_num_observables()) {
      write_response(tx_slot, rx_slot, rpc::RpcStatus::BAD_REQUEST);
      return;
    }
    // result_len = ceil(R/8) exactly per decoder_server_runtime.md (no pad).
    const std::size_t result_len = rpc::bit_packed_bytes(return_size);
    if (sizeof(cudaq::realtime::RPCResponse) + result_len > slot_size) {
      write_response(tx_slot, rx_slot, rpc::RpcStatus::BAD_REQUEST);
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
    write_response(tx_slot, rx_slot, rpc::RpcStatus::OK,
                   static_cast<std::uint32_t>(result_len));
  } catch (const std::out_of_range &) {
    write_response(tx_slot, rx_slot, rpc::RpcStatus::INVALID_DECODER);
  } catch (...) {
    write_response(tx_slot, rx_slot, rpc::RpcStatus::INTERNAL_ERROR);
  }
}

void reset_decoder_host(const void *rx_slot, void *tx_slot, std::size_t) {
  try {
    const auto *header =
        static_cast<const cudaq::realtime::RPCHeader *>(rx_slot);
    if (header->arg_len != sizeof(rpc::ResetRequestPayload)) {
      write_response(tx_slot, rx_slot, rpc::RpcStatus::BAD_REQUEST);
      return;
    }
    const auto *body = reinterpret_cast<const rpc::ResetRequestPayload *>(
        static_cast<const std::uint8_t *>(rx_slot) +
        sizeof(cudaq::realtime::RPCHeader));
    auto *decoder = get_decoder_or_throw(body->decoder_id);
    apply_decoder_cuda_device(decoder);
    decoder->reset_decoder();
    write_response(tx_slot, rx_slot, rpc::RpcStatus::OK);
  } catch (const std::out_of_range &) {
    write_response(tx_slot, rx_slot, rpc::RpcStatus::INVALID_DECODER);
  } catch (...) {
    write_response(tx_slot, rx_slot, rpc::RpcStatus::INTERNAL_ERROR);
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

  // Reset the monotonic producer cursor so it starts in lockstep with the
  // strict-FIFO consumer (both begin at slot 0).
  producer_cursor_ = 0;

  if (device_mode_) {
    // The device-graph scheduler relies on device-side graph launch, which
    // requires compute capability 9.0+ (Hopper).  Below sm_90 the dispatch
    // kernel's TRIGGER_GRAPH interception is compiled out and the enqueue RPC
    // would surface the raw sentinel (0x12A6E5) as a non-zero status -- fail
    // fast with an actionable message instead.
    {
      int device = 0;
      cudaDeviceProp prop{};
      if (cudaGetDevice(&device) != cudaSuccess ||
          cudaGetDeviceProperties(&prop, device) != cudaSuccess)
        throw std::runtime_error(
            "qec_realtime_session::initialize: unable to query the active "
            "CUDA device for the DEVICE-mode compute-capability check");
      if (prop.major < 9)
        throw std::runtime_error(
            "qec_realtime_session::initialize: DEVICE mode (the device-graph "
            "scheduler) requires device-side graph launch, i.e. compute "
            "capability 9.0+ (Hopper); found " +
            std::to_string(prop.major) + "." + std::to_string(prop.minor) +
            ".  Use a CPU decoder (HOST mode) or a Hopper+ GPU.");
    }

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
      start_device_loop(); // self-relaunching device-graph scheduler
    else
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
  std::memset(&host_table_, 0, sizeof(host_table_));
  std::memset(&host_config_, 0, sizeof(host_config_));
  host_engine_ = nullptr;

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

    apply_decoder_cuda_device(dec);
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

  const std::size_t enqueue_req = sizeof(RPCHeader) +
                                  sizeof(rpc::EnqueueRequestPayload) +
                                  rpc::bit_packed_bytes(max_measurements);
  const std::size_t get_req =
      sizeof(RPCHeader) + sizeof(rpc::GetCorrectionsRequestPayload);
  const std::size_t reset_req =
      sizeof(RPCHeader) + sizeof(rpc::ResetRequestPayload);
  const std::size_t enqueue_resp = sizeof(RPCResponse);
  const std::size_t get_resp =
      sizeof(RPCResponse) + rpc::bit_packed_bytes(max_observables);
  const std::size_t reset_resp = sizeof(RPCResponse);

  slot_size_ = std::max({enqueue_req, get_req, reset_req, enqueue_resp,
                         get_resp, reset_resp, std::size_t{64}});

  // Round the slot stride up to an alignment boundary so every slot -- and thus
  // every RPCHeader/RPCResponse placed at slot offset i*slot_size_ -- starts
  // aligned.  write_response() (and the producer) perform a 4-byte __atomic RMW
  // on the header/response `magic` field at slot offset 0; with an unaligned
  // stride (e.g. slot_size_ == 75 at distance 5) that atomic lands on an
  // unaligned address, which the CPU splits into a bus-locked, cache-line-
  // crossing access -- fatal SIGBUS on hosts with split-lock detection enabled
  // (common on cloud / CI runners, tolerated silently elsewhere).  A distance-3
  // config happens to floor at the aligned 64-byte minimum and so never hit
  // this.  DEVICE mode additionally wants a 256-byte stride for deterministic
  // GPU-visible slot addressing; HOST mode only needs the atomics aligned.
  {
    constexpr std::size_t kDeviceSlotAlignment = 256;
    constexpr std::size_t kHostSlotAlignment = 16;
    const std::size_t alignment =
        device_mode_ ? kDeviceSlotAlignment : kHostSlotAlignment;
    slot_size_ = (slot_size_ + (alignment - 1)) & ~(alignment - 1);
  }

  if (device_mode_) {
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

  // DEVICE mode: 3 DEVICE_CALL entries (enqueue_syndromes accumulate,
  // get_corrections, reset_decoder), all serviced by the self-relaunching
  // device-graph scheduler.  enqueue_syndromes is an accumulate handler (NOT a
  // GRAPH_LAUNCH): it appends the round's syndromes into the registered
  // GpuDecoderState and returns CUDAQ_DISPATCH_STATUS_TRIGGER_GRAPH when a full
  // window has accumulated, which tells the scheduler to fire the per-decoder
  // device-launchable decode graph fire-and-forget.  All three handlers live
  // in the cudevice archive and are resolved by name via dlsym so a .so that
  // did not link that archive can still load.  Pinned-mapped so the scheduler
  // kernel reads the same backing.
  function_table_count_ = 3;

  void *h = nullptr;
  void *d = nullptr;
  if (!allocate_pinned_mapped(
          function_table_count_ * sizeof(cudaq_function_entry_t), &h, &d))
    throw std::runtime_error(
        "qec_realtime_session::initialize: failed to allocate function table");
  function_table_host_ = static_cast<cudaq_function_entry_t *>(h);
  function_table_dev_ = static_cast<cudaq_function_entry_t *>(d);

  // Resolve a DEVICE_CALL populate shim by name, invoke it on the target entry,
  // stamp the function_id/routing_key, and validate it produced a real
  // DEVICE_CALL handler.  routing_key is unused by the scheduler (single
  // decoder) but kept 0 for forward compatibility with per-decoder routing.
  auto populate_device_call = [&](std::size_t slot, const char *symbol,
                                  std::uint32_t function_id) {
    auto shim = resolve_populate_shim(symbol);
    if (!shim)
      throw std::runtime_error(
          std::string("qec_realtime_session::initialize: ") + symbol +
          " not found via dlsym(RTLD_DEFAULT, ...).  The final binary must "
          "link libcudaq-qec-realtime-cudevice.a (or the static parts of "
          "decoder_rpc_dispatch.cu via qec_realtime_app_link_options()).");
    shim(&function_table_host_[slot]);
    function_table_host_[slot].function_id = function_id;
    function_table_host_[slot].routing_key = 0;
    if (function_table_host_[slot].dispatch_mode !=
            CUDAQ_DISPATCH_DEVICE_CALL ||
        !function_table_host_[slot].handler.device_fn_ptr)
      throw std::runtime_error(
          std::string("qec_realtime_session::initialize: ") + symbol +
          " did not produce a valid DEVICE_CALL entry (plugin bug)");
  };

  // [0] enqueue_syndromes accumulate.
  populate_device_call(
      0, "cudaqx_qec_realtime_dispatch_populate_enqueue_syndromes_device_entry",
      rpc::kEnqueueSyndromesFunctionId);
  // [1] get_corrections.
  get_corrections_fn_id_ = rpc::kGetCorrectionsFunctionId;
  populate_device_call(
      1, "cudaqx_qec_realtime_dispatch_populate_get_corrections_device_entry",
      get_corrections_fn_id_);
  // [2] reset_decoder.
  reset_decoder_fn_id_ = rpc::kResetDecoderFunctionId;
  populate_device_call(
      2, "cudaqx_qec_realtime_dispatch_populate_reset_decoder_device_entry",
      reset_decoder_fn_id_);
}

//==============================================================================
// start_device_loop()  [DEVICE mode]
//==============================================================================

void qec_realtime_session::start_device_loop() {
  // The scheduler fires a single per-decoder decode graph fire-and-forget when
  // the enqueue accumulate handler signals a full window.  The public host
  // dispatcher exposes one triggered graph per scheduler, so DEVICE mode is
  // scoped to a single graph-dispatch decoder (matching the prior single-
  // mailbox-bank limitation).  A multi-decoder scheduler would need a
  // function-id/routing-key -> triggered-graph map inside the dispatch kernel.
  if (num_decoders_with_graph_ != 1)
    throw std::runtime_error(
        "qec_realtime_session::initialize: the device-graph scheduler supports "
        "exactly one graph-dispatch decoder per session (got " +
        std::to_string(num_decoders_with_graph_) + ").");

  // The lone captured decoder's device-launchable decode graph.
  cudaGraphExec_t decode_graph_exec = nullptr;
  for (std::size_t i = 0; i < captured_graphs_.size(); ++i) {
    if (!captured_graphs_[i])
      continue;
    auto *gres = static_cast<cudaq::qec::realtime::graph_resources *>(
        captured_graphs_[i]);
    decode_graph_exec = gres->graph_exec;
    break;
  }
  if (!decode_graph_exec)
    throw std::runtime_error(
        "qec_realtime_session::initialize: no captured decode graph for the "
        "scheduler to trigger");

  if (cudaMalloc(&device_stats_dev_, sizeof(std::uint64_t)) != cudaSuccess ||
      cudaMemset(device_stats_dev_, 0, sizeof(std::uint64_t)) != cudaSuccess)
    throw std::runtime_error(
        "qec_realtime_session::initialize: device_stats_dev allocation failed");

  if (cudaStreamCreate(&scheduler_stream_) != cudaSuccess)
    throw std::runtime_error(
        "qec_realtime_session::initialize: cudaStreamCreate for the scheduler "
        "stream failed");

  // Bind the graph dispatch API from the host executable's device-link unit
  // (NOT from any copy this .so might carry) so the dispatch kernel and the
  // decoder's DEVICE_CALL handlers share one CUDA module -- see
  // resolve_dispatch_graph_api().  The destroy fn is stashed for stop_loops().
  auto create_fn = resolve_dispatch_graph_api<create_dispatch_graph_fn_t>(
      "cudaq_create_dispatch_graph_regular");
  auto launch_fn = resolve_dispatch_graph_api<launch_dispatch_graph_fn_t>(
      "cudaq_launch_dispatch_graph");
  destroy_dispatch_graph_fn_ =
      resolve_dispatch_graph_api<destroy_dispatch_graph_fn_t>(
          "cudaq_destroy_dispatch_graph");

  // The scheduler kernel itself is lightweight (poll + parse + DEVICE_CALL +
  // fire-and-forget); the heavy cooperative decode lives in the triggered
  // graph, so a single-block scheduler is sufficient.
  cudaError_t err =
      create_fn(rx_flags_dev_, tx_flags_dev_, rx_data_dev_, tx_data_dev_,
                slot_size_, slot_size_, function_table_dev_,
                static_cast<std::size_t>(function_table_count_),
                /*graph_io_ctx=*/nullptr, shutdown_flag_dev_, device_stats_dev_,
                num_slots_, /*num_blocks=*/1, /*threads_per_block=*/64,
                decode_graph_exec, scheduler_stream_, &scheduler_ctx_);
  if (err != cudaSuccess)
    throw std::runtime_error(
        std::string("qec_realtime_session::initialize: "
                    "cudaq_create_dispatch_graph_regular failed: ") +
        cudaGetErrorString(err));

  err = launch_fn(scheduler_ctx_, scheduler_stream_);
  if (err != cudaSuccess)
    throw std::runtime_error(
        std::string("qec_realtime_session::initialize: "
                    "cudaq_launch_dispatch_graph failed: ") +
        cudaGetErrorString(err));
}

//==============================================================================
// start_host_loop()  [HOST mode only]
//==============================================================================

void qec_realtime_session::start_host_loop() {
  // HOST mode only: inline HOST_CALL handlers, no graph worker pool.  DEVICE
  // mode uses the self-relaunching device-graph scheduler launched in
  // start_device_loop() (see initialize()'s dispatch), so this is reached only
  // when device_mode_ is false.  Every table entry is HOST_CALL, so a
  // GRAPH_LAUNCH engine would build no workers -- we drive the ring loop with a
  // NULL engine (the loop runs the HOST_CALL handlers inline).
  std::memset(&host_config_, 0, sizeof(host_config_));
  host_config_.num_slots = static_cast<std::uint32_t>(num_slots_);
  host_config_.slot_size = static_cast<std::uint32_t>(slot_size_);
  host_config_.dispatch_path = CUDAQ_DISPATCH_PATH_HOST;
  host_config_.dispatch_mode = CUDAQ_DISPATCH_HOST_CALL;
  host_config_.skip_tx_markers = 1;
  // Strict-FIFO: the host loop is the sole consumer and its current_slot
  // already advances monotonically; with shared-ring scanning OFF it simply
  // waits at current_slot for the next frame.  The rpc_producer advances its
  // slot monotonically too (see producer_cursor()), so producer and consumer
  // walk the ring in lockstep -- no scan needed.
  host_config_.shared_ring_mode = 0;
  host_table_.entries = function_table_host_;
  host_table_.count = static_cast<std::uint32_t>(function_table_count_);
  host_engine_ = nullptr;
  shutdown_flag_ = 0;

  host_loop_thread_ = std::thread([this]() {
    cudaq_host_ring_dispatch_loop(&ringbuffer_, &host_table_, &host_config_,
                                  /*engine=*/nullptr, &shutdown_flag_,
                                  &host_stats_counter_);
  });
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

  // Drain and destroy the device-graph scheduler (DEVICE mode).  The shutdown
  // flag was set above; the scheduler observes it, stops tail-self-relaunching,
  // and the in-flight self-relaunch chain drains.  cudaStreamSynchronize waits
  // for that to finish before we destroy the graph context.
  if (scheduler_ctx_) {
    if (scheduler_stream_)
      cudaStreamSynchronize(scheduler_stream_);
    if (destroy_dispatch_graph_fn_)
      destroy_dispatch_graph_fn_(scheduler_ctx_);
    scheduler_ctx_ = nullptr;
  }
  if (scheduler_stream_) {
    cudaStreamDestroy(scheduler_stream_);
    scheduler_stream_ = nullptr;
  }

  // Legacy DEVICE_LOOP teardown (unused by the scheduler; both are null in
  // scheduler mode, so these are harmless no-ops).
  if (device_dispatcher_) {
    cudaq_dispatcher_stop(device_dispatcher_);
    cudaq_dispatcher_destroy(device_dispatcher_);
    device_dispatcher_ = nullptr;
  }
  if (device_manager_) {
    cudaq_dispatch_manager_destroy(device_manager_);
    device_manager_ = nullptr;
  }

  // The engine owns the worker streams, idle mask, inflight-slot tags, and
  // GraphIOContext array; destroy it after the driving loop has stopped.
  if (host_engine_) {
    cudaq_graph_launch_engine_destroy(host_engine_);
    host_engine_ = nullptr;
  }
}

} // namespace cudaq::qec::realtime

#endif // CUDAQ_REALTIME_ROOT
