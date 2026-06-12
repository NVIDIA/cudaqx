/*******************************************************************************
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifdef CUDAQ_REALTIME_ROOT

#include "qec_realtime_session.h"

#include "cudaq/qec/realtime/decoder_rpc_ids.h"
#include "cudaq/qec/realtime/graph_resources.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"
#include "cudaq/runtime/logger/logger.h"

#include <algorithm>
#include <cstring>
#include <dlfcn.h>
#include <stdexcept>
#include <string>
namespace cudaq::qec::realtime {

namespace {

// Resolves a host-side C-ABI shim `cudaqx_qec_realtime_dispatch_populate_*`
// at runtime via dlsym(RTLD_DEFAULT, ...).  These shims are defined in
// libcudaq-qec-realtime-cudevice.a and only enter the process when the final
// executable (or .so consumer) absorbs that static archive -- typically via
// the `qec_realtime_app_link_options()` CMake helper.  Resolving by name
// rather than by direct symbol reference keeps libcudaq-qec-realtime-decoding
// .so free of unresolved C-ABI symbols, so it can be safely dlopen'd from
// consumers that do NOT link the cudevice archive (notably the Python
// extension `_pycudaqx_qec_the_suffix_matters_cudaq_qec.so`).  Any such
// consumer that tries to actually USE the inproc_rpc path will land here,
// not find the symbol, and surface a clean runtime_error with actionable
// linker guidance -- which the caller (maybe_init_realtime_session()) then
// propagates per the spec's fail-fast contract.
using populate_device_entry_fn = void (*)(void *);
populate_device_entry_fn resolve_populate_shim(const char *symbol_name) {
  void *sym = ::dlsym(RTLD_DEFAULT, symbol_name);
  return reinterpret_cast<populate_device_entry_fn>(sym);
}

// Mirrors the test's allocate_ring_buffer() helper -- pinned mapped flags +
// pinned mapped data, with the device pointer obtained via UVA so the GPU
// dispatcher can read the same backing.
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

} // namespace

qec_realtime_session::qec_realtime_session(
    std::vector<std::unique_ptr<cudaq::qec::decoder>> &decoders,
    cudaq_dispatch_launch_fn_t device_launch_fn)
    : decoders_(decoders), device_launch_fn_(device_launch_fn) {}

qec_realtime_session::~qec_realtime_session() {
  // Best-effort teardown.  If a derived consumer forgot to call finalize()
  // explicitly while g_decoders was still alive, we still try to bring the
  // dispatchers down (release_decode_graph may dereference dangling decoder
  // pointers, but we have no safer choice in a destructor).
  //
  // We call finalize() unconditionally rather than gating on initialized_
  // because a throw from initialize() (e.g. a stream-create failure inside
  // start_host_loop after start_device_loop already spun up the persistent
  // DEVICE_LOOP) would otherwise leave `initialized_` false and leak the
  // dispatcher kernel + pinned ring buffers.  finalize() is null-safe at
  // every step (each resource has its own non-null guard), so calling it
  // from a never-fully-initialized or already-finalized session is a no-op
  // beyond the trace message.
  finalize();
}

//==============================================================================
// initialize()
//==============================================================================

void qec_realtime_session::initialize() {
  if (initialized_)
    return;

  // Be tolerant of being called before any CUDA setup -- callers in
  // production may have already done this; the contract test does it
  // before constructing the session.  Either way, the cudaHostAlloc calls
  // below require cudaDeviceMapHost on the active device.
  {
    cudaError_t flags_err = cudaSetDeviceFlags(cudaDeviceMapHost);
    if (flags_err != cudaSuccess && flags_err != cudaErrorSetOnActiveProcess)
      throw std::runtime_error(
          std::string("qec_realtime_session::initialize: "
                      "cudaSetDeviceFlags(cudaDeviceMapHost) failed: ") +
          cudaGetErrorString(flags_err));
  }

  // Everything below acquires resources -- pinned host memory, the
  // persistent DEVICE_LOOP kernel, the HOST_LOOP worker thread, captured
  // CUDA graphs -- so it must be transactional.  If any step throws after
  // we've started the device dispatcher kernel, we must tear it (and any
  // upstream resources) back down before propagating the exception;
  // otherwise the dispatcher kernel runs forever and the pinned ring
  // buffer leaks.  finalize() is null-safe at every step, so we can call
  // it from a half-built session.
  try {
    capture_decoder_graphs();
    allocate_ring_buffer();
    populate_function_table();

    // NOTE: The caller (the final executable -- the contract test, the
    // surface_code-1-local binary, etc.) is required to call
    // `cudaq_dispatch_kernel_set_shared_ring_mode(1)` BEFORE invoking
    // initialize().  We cannot call it from here because that function
    // lives in libcudaq-realtime-dispatch.a (static archive, hidden
    // visibility), which is linked only into the final exe -- not into
    // this shared library.  Calling it from here would cause an
    // unresolved-symbol error at exe link time.  See the constructor's
    // `device_launch_fn` argument for the same architectural reason
    // applied to the dispatch launch function pointer.

    // Ring buffer struct.  Two separate physical backings for RX and TX
    // (the Hololink-aligned `rx_data != tx_data` configuration).  Both
    // backings are still shared between the device dispatcher (DEVICE_CALL
    // get_corrections + reset_decoder) and the host monitor (GRAPH_LAUNCH
    // per-decoder enqueue) via `shared_ring_mode=1`; the device kernel
    // skips RX slots whose function_id targets a GRAPH_LAUNCH entry so
    // the host monitor can pick them up.
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

    start_device_loop();
    start_host_loop();
  } catch (...) {
    // RAII-safe rollback: tear down anything start_device_loop()/
    // start_host_loop()/the allocators may have brought up before the
    // throw.  finalize() ignores its own initialized_ flag so it works
    // from a partial state too.  We rethrow after rollback so the
    // caller still sees the original error.
    CUDAQ_WARN("qec_realtime_session::initialize: rolling back partial "
               "initialization after exception");
    finalize();
    throw;
  }

  initialized_ = true;
  CUDAQ_INFO("qec_realtime_session: initialized "
             "(num_decoders_with_graph={}, num_slots={}, slot_size={})",
             num_decoders_with_graph_, num_slots_, slot_size_);
}

//==============================================================================
// finalize()
//==============================================================================

void qec_realtime_session::finalize() {
  // finalize() is called from three places:
  //   1. realtime_decoding.cpp's finalize_decoders(), after a successful
  //      initialize() -- the steady-state shutdown path.
  //   2. ~qec_realtime_session(), as a best-effort backstop for callers
  //      that forgot (1).
  //   3. initialize() itself, on any throw during the resource-acquisition
  //      phase, to roll partial state back before rethrowing.
  // Cases (1) and (2) imply initialized_ is true; case (3) implies it is
  // not.  Every step below is null-safe (each resource has its own guard),
  // so the only thing the initialized_ flag governs is whether we log the
  // closing INFO message and whether we re-run.  The was_initialized
  // capture lets a second call (e.g. caller's finalize() followed by the
  // destructor's backstop) become a no-op.
  const bool was_initialized = initialized_;
  if (was_initialized)
    initialized_ = false;
  // Note: we intentionally do NOT early-return on !was_initialized; the
  // initialize() rollback path depends on running through all the cleanup
  // steps below.

  stop_loops();

  // NOTE: The caller is responsible for restoring
  // `cudaq_dispatch_kernel_set_shared_ring_mode(0)` if it wants follow-up
  // sessions in the same process to start with shared_ring_mode unset.
  // We cannot call it from here for the same reason explained in
  // initialize() above.

  // After stop_loops() returns the host monitor thread is joined and the
  // persistent device kernel has been signalled to exit, but in-flight
  // worker-stream graph launches submitted before the join can still be
  // running.  stop_loops() destroys those streams via cudaStreamDestroy,
  // which per the CUDA spec does NOT block on pending work -- it returns
  // immediately and reclaims the stream asynchronously once its work
  // completes.  Without an explicit fence here, a still-running enqueue
  // graph (per_round_dispatch.cuh) can dereference the per-decoder
  // mailbox / d_msyn_buffer / bp_decoder_context buffers AFTER the next
  // step (release_decode_graph) has freed them.  cudaDeviceSynchronize()
  // drains all outstanding work on the current device, closing that race.
  // finalize() is cold-path so the sync cost is irrelevant.
  cudaDeviceSynchronize();

  // Release captured graphs in index order.  Safe to call before
  // decoders_.clear() because the production caller
  // (`finalize_decoders()` in realtime_decoding.cpp) is required to invoke
  // session->finalize() BEFORE clearing g_decoders.
  for (std::size_t i = 0; i < captured_graphs_.size(); ++i) {
    if (captured_graphs_[i] && i < decoders_.size() && decoders_[i])
      decoders_[i]->release_decode_graph(captured_graphs_[i]);
  }
  captured_graphs_.clear();
  num_decoders_with_graph_ = 0;

  // Free pinned + device memory.
  if (function_table_host_) {
    cudaFreeHost(function_table_host_);
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

  if (tx_flags_host_) {
    cudaFreeHost(const_cast<std::uint64_t *>(tx_flags_host_));
    tx_flags_host_ = nullptr;
    tx_flags_dev_ = nullptr;
  }
  if (rx_flags_host_) {
    cudaFreeHost(const_cast<std::uint64_t *>(rx_flags_host_));
    rx_flags_host_ = nullptr;
    rx_flags_dev_ = nullptr;
  }
  if (tx_data_host_) {
    cudaFreeHost(tx_data_host_);
    tx_data_host_ = nullptr;
    tx_data_dev_ = nullptr;
  }
  if (rx_data_host_) {
    cudaFreeHost(rx_data_host_);
    rx_data_host_ = nullptr;
    rx_data_dev_ = nullptr;
  }
  if (shutdown_flag_host_) {
    cudaFreeHost(shutdown_flag_host_);
    shutdown_flag_host_ = nullptr;
    shutdown_flag_dev_ = nullptr;
  }

  if (was_initialized)
    CUDAQ_INFO("qec_realtime_session: finalized");
}

//==============================================================================
// capture_decoder_graphs()
//==============================================================================

void qec_realtime_session::capture_decoder_graphs() {
  captured_graphs_.assign(decoders_.size(), nullptr);
  num_decoders_with_graph_ = 0;

  // Surface the dispatch-capacity limit up front with a clear message rather
  // than letting the N-th registration throw deep inside the plugin's
  // capture_decode_graph() / register_decoder_state() (decoder_rpc_dispatch.
  // cu).  kMaxDispatchedDecoders sizes the device-side g_decoder_state_table[]
  // (see decoder_rpc_ids.h).
  if (decoders_.size() > cudaq::qec::decoding::rpc::kMaxDispatchedDecoders)
    throw std::runtime_error(
        "qec_realtime_session::initialize: requested " +
        std::to_string(decoders_.size()) +
        " decoders but the realtime dispatch supports at most " +
        std::to_string(cudaq::qec::decoding::rpc::kMaxDispatchedDecoders) +
        " (kMaxDispatchedDecoders, the device-side decoder state-table size). "
        "Reduce the decoder count, or bump kMaxDispatchedDecoders in "
        "decoder_rpc_ids.h and grow the device table to match.");

  for (std::size_t i = 0; i < decoders_.size(); ++i) {
    auto *dec = decoders_[i].get();
    if (!dec)
      continue;
    if (!dec->supports_graph_dispatch())
      throw std::runtime_error(
          "qec_realtime_session::initialize: decoder " + std::to_string(i) +
          " does not support graph dispatch.  The inproc_rpc realtime mode "
          "requires every realized decoder to provide a per-round capture "
          "(see decoder::supports_graph_dispatch / capture_decode_graph).  "
          "Use the default host mode for this decoder type.");

    // reserved_sms = 0 is intentional and final for the inproc_rpc desktop /
    // CI path.  This path does not run the HSB persistent kernels, so the
    // decode graph need not reserve SMs for them (the dispatch loop and the
    // decode kernel run on separate streams).  The earlier 0->1 bump
    // (bbeee0bc) was the cause of the A100/cu12.6 timeout on test 166
    // (`app_examples.surface_code-1-local-test-distance-3-inproc-rpc`);
    // reverting to 0 cleared it.  Revisit only if this graph is ever run
    // alongside HSB persistent kernels on the same device.
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

    // Per decoder_server_runtime.md all N enqueue_syndromes graphs share a
    // single canonical function_id; the host monitor disambiguates per-
    // decoder via the routing_key sub-filter (routing_key == decoder_id).
    // Cross-check the plugin handed us the spec-mandated fid.
    if (gres->function_id !=
        cudaq::qec::decoding::rpc::kEnqueueSyndromesFunctionId)
      throw std::runtime_error(
          "qec_realtime_session::initialize: decoder " + std::to_string(i) +
          " published a non-canonical enqueue function_id 0x" +
          std::to_string(gres->function_id) +
          " (expected kEnqueueSyndromesFunctionId 0x" +
          std::to_string(
              cudaq::qec::decoding::rpc::kEnqueueSyndromesFunctionId) +
          " per decoder_server_runtime.md).");

    ++num_decoders_with_graph_;
  }

  if (num_decoders_with_graph_ == 0)
    throw std::runtime_error(
        "qec_realtime_session::initialize: no decoders to capture graphs "
        "for (decoders_.size()=" +
        std::to_string(decoders_.size()) + ")");
}

//==============================================================================
// allocate_ring_buffer()
//==============================================================================

void qec_realtime_session::allocate_ring_buffer() {
  // Slot size: largest body across the trio, over all captured decoders,
  // using the wire format from proposals/decoder_server_runtime.md:
  //   enqueue_syndromes request:
  //     RPCHeader(24) + EnqueueRequestPayload(32)
  //                   + bit_packed_bytes(num_measurements)
  //                   + 0..7 zero pad   (whole arg block is 8-byte aligned)
  //   enqueue_syndromes response:
  //     RPCResponse(24); result_len = 0 (no body)
  //   get_corrections request:
  //     RPCHeader(24) + GetCorrectionsRequestPayload(24); already 8-aligned
  //   get_corrections response:
  //     RPCResponse(24) + bit_packed_bytes(num_observables)
  //                     + 0..7 zero pad
  //   reset_decoder request:
  //     RPCHeader(24) + ResetRequestPayload(8); already 8-aligned
  //   reset_decoder response:
  //     RPCResponse(24); result_len = 0
  std::size_t max_measurements_per_round = 0;
  std::size_t max_num_observables = 0;
  for (std::size_t i = 0; i < decoders_.size(); ++i) {
    auto *dec = decoders_[i].get();
    if (!dec || !captured_graphs_[i])
      continue;
    max_measurements_per_round = std::max<std::size_t>(
        max_measurements_per_round, dec->get_num_msyn_per_decode());
    max_num_observables =
        std::max<std::size_t>(max_num_observables, dec->get_num_observables());
  }

  using namespace cudaq::realtime;
  using cudaq::qec::decoding::rpc::align_to_8;
  using cudaq::qec::decoding::rpc::bit_packed_bytes;
  using cudaq::qec::decoding::rpc::EnqueueRequestPayload;
  using cudaq::qec::decoding::rpc::GetCorrectionsRequestPayload;
  using cudaq::qec::decoding::rpc::ResetRequestPayload;

  const std::size_t enqueue_body =
      sizeof(RPCHeader) +
      align_to_8(sizeof(EnqueueRequestPayload) +
                 bit_packed_bytes(max_measurements_per_round));
  const std::size_t get_corr_body =
      sizeof(RPCHeader) + sizeof(GetCorrectionsRequestPayload);
  const std::size_t reset_body =
      sizeof(RPCHeader) + sizeof(ResetRequestPayload);
  // Empty-body ACK still needs 24 bytes for the RPCResponse header.
  const std::size_t enqueue_resp = sizeof(RPCResponse);
  const std::size_t get_corr_resp =
      sizeof(RPCResponse) + align_to_8(bit_packed_bytes(max_num_observables));
  const std::size_t reset_resp = sizeof(RPCResponse);

  slot_size_ = std::max({enqueue_body, get_corr_body, reset_body, enqueue_resp,
                         get_corr_resp, reset_resp});
  // Round up to 256-byte alignment (also keeps slot stride deterministic
  // across decoder geometries).
  constexpr std::size_t kSlotAlignment = 256;
  slot_size_ = (slot_size_ + (kSlotAlignment - 1)) & ~(kSlotAlignment - 1);

  // num_slots_ keeps its default unless an env-var override later wants to
  // bump it.  Eight is enough for the contract test and surface_code-1.

  // Allocate flags + shared data backing.
  {
    void *h = nullptr;
    void *d = nullptr;
    if (!allocate_pinned_mapped(num_slots_ * sizeof(std::uint64_t), &h, &d))
      throw std::runtime_error(
          "qec_realtime_session::initialize: failed to allocate rx_flags");
    rx_flags_host_ = static_cast<volatile std::uint64_t *>(h);
    rx_flags_dev_ = static_cast<volatile std::uint64_t *>(d);
  }
  {
    void *h = nullptr;
    void *d = nullptr;
    if (!allocate_pinned_mapped(num_slots_ * sizeof(std::uint64_t), &h, &d))
      throw std::runtime_error(
          "qec_realtime_session::initialize: failed to allocate tx_flags");
    tx_flags_host_ = static_cast<volatile std::uint64_t *>(h);
    tx_flags_dev_ = static_cast<volatile std::uint64_t *>(d);
  }
  // Two separate pinned-mapped data backings -- one for RX requests, one
  // for TX responses.  Both rings have the same slot stride and slot
  // count, but live at different physical addresses.  Matches the
  // Hololink configuration where the producer/RX-transport writes one
  // ring and the dispatcher/TX-transport reads the other.  Under
  // `shared_ring_mode=1` the device kernel and the host monitor still
  // share both rings (each can see incoming RX slots and write outgoing
  // TX slots), but the rings themselves are now physically distinct.
  {
    void *h = nullptr;
    void *d = nullptr;
    if (!allocate_pinned_mapped(num_slots_ * slot_size_, &h, &d))
      throw std::runtime_error(
          "qec_realtime_session::initialize: failed to allocate RX ring data");
    rx_data_host_ = static_cast<std::uint8_t *>(h);
    rx_data_dev_ = static_cast<std::uint8_t *>(d);
  }
  {
    void *h = nullptr;
    void *d = nullptr;
    if (!allocate_pinned_mapped(num_slots_ * slot_size_, &h, &d))
      throw std::runtime_error(
          "qec_realtime_session::initialize: failed to allocate TX ring data");
    tx_data_host_ = static_cast<std::uint8_t *>(h);
    tx_data_dev_ = static_cast<std::uint8_t *>(d);
  }
  {
    void *h = nullptr;
    void *d = nullptr;
    if (!allocate_pinned_mapped(sizeof(int), &h, &d))
      throw std::runtime_error(
          "qec_realtime_session::initialize: failed to allocate shutdown flag");
    shutdown_flag_host_ = static_cast<int *>(h);
    *shutdown_flag_host_ = 0;
    shutdown_flag_dev_ = static_cast<int *>(d);
  }
}

//==============================================================================
// populate_function_table()
//==============================================================================

void qec_realtime_session::populate_function_table() {
  // N GRAPH_LAUNCH entries (one per captured decoder) + 2 DEVICE_CALL
  // entries (get_corrections, reset_decoder).  The DEVICE_CALL entries
  // share a single fid each across all decoders (decoder_id is in the
  // request payload).
  function_table_count_ = num_decoders_with_graph_ + 2;

  void *h = nullptr;
  void *d = nullptr;
  if (!allocate_pinned_mapped(
          function_table_count_ * sizeof(cudaq_function_entry_t), &h, &d))
    throw std::runtime_error(
        "qec_realtime_session::initialize: failed to allocate function table");
  function_table_host_ = static_cast<cudaq_function_entry_t *>(h);
  function_table_dev_ = static_cast<cudaq_function_entry_t *>(d);

  // [0..N-1] GRAPH_LAUNCH per-decoder enqueue.  Per decoder_server_runtime.md
  // all enqueue entries SHARE function_id == kEnqueueSyndromesFunctionId; the
  // host monitor disambiguates them via routing_key = source decoder_id
  // (matched against arg0 of the request payload).  We pack only the
  // non-null decoders into the front of the table; the index inside the
  // table is not load-bearing (the dispatcher routes by
  // (function_id, routing_key), and routing_key carries the decoder_id).
  std::size_t slot = 0;
  for (std::size_t i = 0; i < decoders_.size(); ++i) {
    if (!captured_graphs_[i])
      continue;
    auto *gres = static_cast<cudaq::qec::realtime::graph_resources *>(
        captured_graphs_[i]);
    auto &entry = function_table_host_[slot++];
    entry.handler.graph_exec = gres->graph_exec;
    entry.function_id = cudaq::qec::decoding::rpc::kEnqueueSyndromesFunctionId;
    entry.dispatch_mode = CUDAQ_DISPATCH_GRAPH_LAUNCH;
    entry.routing_key = static_cast<std::uint64_t>(i);
  }

  // [N] DEVICE_CALL get_corrections.  function_id is unique per DEVICE_CALL
  // handler, so routing_key stays 0 (the spec-aligned default; the host
  // monitor doesn't apply the sub-filter to DEVICE_CALL entries anyway).
  //
  // The populate_* shims live in libcudaq-qec-realtime-cudevice.a and only
  // become visible to dlsym() after the final binary has absorbed that
  // archive (see qec_realtime_app_link_options() in
  // unittests/realtime/app_examples/CMakeLists.txt).  Resolve by name so
  // consumers that did NOT link the archive -- e.g. the Python ext
  // _pycudaqx_qec_the_suffix_matters_cudaq_qec.so -- can still load
  // libcudaq-qec-realtime-decoding.so without an undefined-symbol error;
  // they only hit this code path if they actually opted into inproc_rpc,
  // in which case the runtime_error below is the spec-mandated fail-fast.
  get_corrections_fn_id_ = cudaq::qec::decoding::rpc::kGetCorrectionsFunctionId;
  auto populate_get_corrections = resolve_populate_shim(
      "cudaqx_qec_realtime_dispatch_populate_get_corrections_device_entry");
  if (!populate_get_corrections)
    throw std::runtime_error(
        "qec_realtime_session::initialize: "
        "cudaqx_qec_realtime_dispatch_populate_get_corrections_device_entry "
        "not found via dlsym(RTLD_DEFAULT, ...).  The final binary must "
        "link libcudaq-qec-realtime-cudevice.a (or the static parts of "
        "decoder_rpc_dispatch.cu via qec_realtime_app_link_options()) to "
        "make this DEVICE_CALL handler resolvable at runtime.");
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
  reset_decoder_fn_id_ = cudaq::qec::decoding::rpc::kResetDecoderFunctionId;
  auto populate_reset_decoder = resolve_populate_shim(
      "cudaqx_qec_realtime_dispatch_populate_reset_decoder_device_entry");
  if (!populate_reset_decoder)
    throw std::runtime_error(
        "qec_realtime_session::initialize: "
        "cudaqx_qec_realtime_dispatch_populate_reset_decoder_device_entry "
        "not found via dlsym(RTLD_DEFAULT, ...).  The final binary must "
        "link libcudaq-qec-realtime-cudevice.a (or the static parts of "
        "decoder_rpc_dispatch.cu via qec_realtime_app_link_options()) to "
        "make this DEVICE_CALL handler resolvable at runtime.");
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
// start_device_loop()
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
        "qec_realtime_session::initialize: device_stats_dev allocation "
        "failed");

  // The device API takes `volatile int *`; our plain int* converts
  // implicitly (qualification add) and the device kernel does volatile reads.
  if (cudaq_dispatcher_set_control(device_dispatcher_, shutdown_flag_dev_,
                                   device_stats_dev_) != CUDAQ_OK)
    throw std::runtime_error(
        "qec_realtime_session::initialize: cudaq_dispatcher_set_control "
        "failed");

  if (!device_launch_fn_)
    throw std::runtime_error(
        "qec_realtime_session::initialize: device_launch_fn is null "
        "(constructor required a non-null cudaq_dispatch_launch_fn_t -- "
        "typically &cudaq_launch_dispatch_kernel_regular from libcudaq-"
        "realtime-dispatch)");
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
// start_host_loop()
//==============================================================================

void qec_realtime_session::start_host_loop() {
  // host_workers_ is consumed by libcudaq-realtime's cudaq_host_dispatch_-
  // loop_ctx_t and MUST be packed (one entry per active worker; the host
  // dispatcher iterates 0..num_decoders_with_graph_-1).
  host_workers_.assign(num_decoders_with_graph_,
                       cudaq_host_dispatch_worker_t{});

  // host_worker_streams_ is consumed by rpc_producer::enqueue_syndromes
  // for the post-ACK cudaStreamSynchronize, and it indexes by the
  // *source* decoder_id (i.e. the index into decoders_) -- not by the
  // packed worker slot, because callers pass decoder_id, not slot.  Size
  // it to decoders_.size() with nullptr placeholders for null or non-
  // graph-capable decoders.  Sparse indexing avoids a silent stream-sync
  // skip if/when decoder_ids stop being contiguous from 0.
  host_worker_streams_.assign(decoders_.size(),
                              static_cast<cudaStream_t>(nullptr));

  // h_mailbox_bank for the host dispatcher: the host_dispatcher.h public
  // ctx exposes a SINGLE mailbox bank per ctx, populated by the plugin's
  // graph capture.  Demo 1's scope is N==1 (one nv-qldpc-decoder per
  // multi_decoder_config in surface_code-1-local), so the single-bank
  // limitation is fine -- the lone worker dispatches every enqueue RPC.
  //
  // When the project grows multi-decoder fanout in a follow-up MR, this
  // session will need to either (a) run one cudaq_host_dispatch_loop_ctx_t
  // per decoder (each with its own h_mailbox_bank) on its own thread, or
  // (b) wait for libcudaq-realtime to grow per-worker mailbox storage.
  if (num_decoders_with_graph_ > 1)
    throw std::runtime_error(
        "qec_realtime_session::initialize: multi-decoder host dispatch is "
        "not yet supported (num_decoders_with_graph=" +
        std::to_string(num_decoders_with_graph_) +
        ").  libcudaq-realtime's host dispatcher exposes a single "
        "h_mailbox_bank per loop ctx; Demo 1 is scoped to one decoder.  "
        "See multi-decoder follow-up.");

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
    // host_workers_ is packed (slot index), host_worker_streams_ is
    // sparse (source decoder index).  See start_host_loop()'s leading
    // comment for the rationale.
    host_worker_streams_[i] = stream;

    auto &w = host_workers_[slot];
    w.graph_exec = gres->graph_exec;
    w.stream = stream;
    // Every worker advertises the canonical enqueue function_id; the
    // host monitor distinguishes them by routing_key (= source decoder
    // index `i`) per proposals/cudaq_realtime_host_api.bs
    // #host-path-graph-routing-key.
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

  // idle_mask covers all workers; live_dispatched starts at zero.
  // inflight_slot_tags is one int per worker.
  host_idle_mask_storage_ = new std::uint64_t(
      num_decoders_with_graph_ < 64
          ? ((std::uint64_t{1} << num_decoders_with_graph_) - 1)
          : ~std::uint64_t{0});
  host_live_dispatched_storage_ = new std::uint64_t(0);
  host_inflight_slot_tags_ = new int[num_decoders_with_graph_];
  for (std::size_t i = 0; i < num_decoders_with_graph_; ++i)
    host_inflight_slot_tags_[i] = -1;

  // Allocate the per-worker GraphIOContext array (pinned-mapped so both
  // CPU monitor and GPU graph see the same backing).  Under the two-ring
  // wire format the host monitor populates entry `worker_id` with rx_slot
  // / tx_slot / tx_flag / tx_flag_value before each `cudaGraphLaunch`,
  // and stashes the device address of that entry into
  // `h_mailbox_bank[worker_id]`; the captured graph reads its kernel-arg
  // mailbox to get the GraphIOContext pointer.
  // See [host_api.bs Routing-Key Sub-filter / GraphIOContext sections].
  {
    void *h = nullptr;
    void *d = nullptr;
    const std::size_t bytes =
        num_decoders_with_graph_ * sizeof(cudaq::realtime::GraphIOContext);
    if (!allocate_pinned_mapped(bytes, &h, &d))
      throw std::runtime_error(
          "qec_realtime_session::start_host_loop: failed to allocate "
          "per-worker GraphIOContext array");
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
  host_ctx_.function_table.entries = function_table_dev_;
  host_ctx_.function_table.count =
      static_cast<std::uint32_t>(function_table_count_);
  host_ctx_.workers = host_workers_.data();
  host_ctx_.num_workers = host_workers_.size();
  host_ctx_.h_mailbox_bank = mailbox_bank;
  // host_ctx_.shutdown_flag is void* (the host dispatcher reinterprets it as
  // cuda::std::atomic<int>); a plain int* assigns without a cast.
  host_ctx_.shutdown_flag = shutdown_flag_host_;
  host_ctx_.stats_counter = &host_stats_counter_;
  host_ctx_.live_dispatched = host_live_dispatched_storage_;
  host_ctx_.idle_mask = host_idle_mask_storage_;
  host_ctx_.inflight_slot_tags = host_inflight_slot_tags_;
  // Wire the per-worker GraphIOContext mailbox arrays.  Under the two-ring
  // wire format the host monitor (cudaq-realtime/lib/daemon/dispatcher/
  // host_dispatcher.cu::launch_graph_worker) populates io_ctxs_host_[w]
  // with rx_slot=ringbuffer.rx_data+slot*stride, tx_slot=ringbuffer.
  // tx_data+slot*stride, tx_flag=&ringbuffer.tx_flags[slot],
  // tx_flag_value=(uint64_t)tx_slot, and stashes the device address
  // io_ctxs_dev_ + w*sizeof(GraphIOContext) into h_mailbox_bank[w].
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
  if (shutdown_flag_host_) {
    // Release store paired with the host dispatcher's acquire load
    // (host_dispatcher.cu: as_atomic_int(...)->load(memory_order_acquire)),
    // so the flip can't be hoisted out of its polling loop.  The trailing
    // full fence additionally publishes the write to the persistent DEVICE
    // dispatcher kernel, which polls the same pinned-mapped flag through a
    // volatile int* (atomic<int> and int are layout-compatible).
    __atomic_store_n(shutdown_flag_host_, 1, __ATOMIC_RELEASE);
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
