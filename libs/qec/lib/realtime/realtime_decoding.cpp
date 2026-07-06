/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "realtime_decoding.h"
#include "cudaq/qec/decoder.h"
#include "cudaq/qec/logger.h"
#include "cudaq/qec/pcm_utils.h"
#include "cudaq/qec/realtime/decoding_config.h"
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fmt/core.h>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>

#ifdef CUDAQ_REALTIME_ROOT
#include "qec_realtime_session.h"
#include "rpc_producer.h"
#else
namespace cudaq::qec::realtime {
class qec_realtime_session {};
} // namespace cudaq::qec::realtime
#endif

// Optional syndrome capture callback for --save_syndrome feature
namespace {
using SyndromeCaptureCallback = void (*)(const uint8_t *, size_t);
SyndromeCaptureCallback g_syndrome_capture_callback = nullptr;
} // namespace

std::vector<std::unique_ptr<cudaq::qec::decoder>> g_decoders;
std::unique_ptr<cudaq::qec::realtime::qec_realtime_session> g_realtime_session;

// ---------------------------------------------------------------------------
// Per-decoder execution workers.
//
// The transport dispatcher (cudaq_host_dispatcher_loop) is a single thread;
// executing decoders inline there serializes every logical qubit's stream
// behind every other's. Instead, each decoder owns a worker thread and a
// bounded ring of request slots: the dispatcher VALIDATES and STAGES a
// request (writing the payload directly into the decoder's slot -- no extra
// copy) and returns immediately for the fire-and-forget calls
// (enqueue_syndromes / reset_decoder); decoder execution happens on the
// decoder's own thread, so N logical qubits decode concurrently.
//
// Ordering: the wire protocol requires per-decoder FIFO only. A single
// staging producer (the dispatcher) plus a per-decoder FIFO ring preserves
// exactly that; cross-decoder ordering is deliberately given up -- that is
// the parallelism.
//
// get_corrections is the one response-bearing call: it is staged as a
// sentinel and the caller blocks until the decoder's queue drains and the
// corrections are produced (end-of-shot only, bounded by one decode tail).
//
// Errors: validation failures (bad id / size) still throw synchronously in
// the caller. A decoder exception during deferred execution becomes STICKY
// per-decoder state, delivered (rethrown) at that decoder's next
// get_corrections; reset_decoder clears it.
// ---------------------------------------------------------------------------
namespace {

struct decoder_worker {
  enum class op : std::uint8_t { enqueue, reset, corrections };
  struct item {
    op kind = op::enqueue;
    std::uint64_t length = 0;
    std::uint64_t tag = 0;
    // corrections / rendezvous
    uint8_t *out = nullptr;
    std::uint64_t out_length = 0;
    bool reset_after = false;
    bool done = false;
    std::exception_ptr error;
  };

  static constexpr std::size_t kCapacity = 8192;

  cudaq::qec::decoder *decoder = nullptr;
  std::size_t decoder_id = 0;
  std::size_t payload_stride = 0;
  std::vector<uint8_t> payloads; // kCapacity * payload_stride
  std::vector<item> items;       // kCapacity ring
  std::size_t head = 0;          // next to execute (mod kCapacity)
  std::size_t count = 0;         // committed, not yet executed
  bool staging_open = false;     // a slot handed out, not yet committed
  bool stop = false;
  std::exception_ptr sticky;
  std::mutex mtx;
  std::condition_variable cv_worker, cv_caller;
  std::thread thread;

  decoder_worker(cudaq::qec::decoder *dec, std::size_t id,
                 std::size_t max_payload)
      : decoder(dec), decoder_id(id),
        payload_stride(std::max<std::size_t>(max_payload, 1)),
        payloads(kCapacity * payload_stride), items(kCapacity) {
    thread = std::thread([this] { run(); });
  }

  ~decoder_worker() {
    {
      std::lock_guard<std::mutex> lock(mtx);
      stop = true;
    }
    cv_worker.notify_all();
    if (thread.joinable())
      thread.join();
  }

  uint8_t *slot_payload(std::size_t index) {
    return payloads.data() + (index % kCapacity) * payload_stride;
  }

  // Hand out the next slot's payload buffer (blocks while the ring is
  // full -- natural backpressure onto the transport).
  uint8_t *stage() {
    std::unique_lock<std::mutex> lock(mtx);
    cv_caller.wait(lock,
                   [this] { return count < kCapacity && !staging_open; });
    staging_open = true;
    return slot_payload(head + count);
  }

  // Payload pointer of the currently-staged slot (staging_open must be
  // true; single staging producer).
  uint8_t *staged_payload() {
    std::lock_guard<std::mutex> lock(mtx);
    return slot_payload(head + count);
  }

  void commit(op kind, std::uint64_t length, std::uint64_t tag) {
    {
      std::lock_guard<std::mutex> lock(mtx);
      auto &it = items[(head + count) % kCapacity];
      it = item{};
      it.kind = kind;
      it.length = length;
      it.tag = tag;
      ++count;
      staging_open = false;
    }
    cv_worker.notify_one();
    cv_caller.notify_all();
  }

  // Reserve + commit a payload-less item (reset) in one step.
  void post(op kind) {
    {
      std::unique_lock<std::mutex> lock(mtx);
      cv_caller.wait(lock,
                     [this] { return count < kCapacity && !staging_open; });
      auto &it = items[(head + count) % kCapacity];
      it = item{};
      it.kind = kind;
      ++count;
    }
    cv_worker.notify_one();
  }

  // Stage + wait for a corrections (or reset) rendezvous. `out` may be
  // null for reset.
  void execute_and_wait(op kind, uint8_t *out, std::uint64_t out_length,
                        bool reset_after) {
    item *slot = nullptr;
    {
      std::unique_lock<std::mutex> lock(mtx);
      cv_caller.wait(lock,
                     [this] { return count < kCapacity && !staging_open; });
      slot = &items[(head + count) % kCapacity];
      *slot = item{};
      slot->kind = kind;
      slot->out = out;
      slot->out_length = out_length;
      slot->reset_after = reset_after;
      ++count;
    }
    cv_worker.notify_one();
    std::exception_ptr error;
    {
      std::unique_lock<std::mutex> lock(mtx);
      cv_caller.wait(lock, [slot] { return slot->done; });
      error = slot->error;
    }
    if (error)
      std::rethrow_exception(error);
  }

  void run();
};

std::vector<std::unique_ptr<decoder_worker>> g_decoder_workers;
std::atomic<std::uint64_t> g_busy_decoder_workers{0};
std::atomic<std::uint64_t> g_max_busy_decoder_workers{0};

void decoder_worker::run() {
  for (;;) {
    std::size_t index = 0;
    {
      std::unique_lock<std::mutex> lock(mtx);
      cv_worker.wait(lock, [this] { return count > 0 || stop; });
      if (stop && count == 0)
        return;
      index = head % kCapacity;
    }
    auto &it = items[index];

    const auto busy =
        g_busy_decoder_workers.fetch_add(1, std::memory_order_relaxed) + 1;
    auto observed =
        g_max_busy_decoder_workers.load(std::memory_order_relaxed);
    while (busy > observed &&
           !g_max_busy_decoder_workers.compare_exchange_weak(
               observed, busy, std::memory_order_relaxed))
      ;

    try {
      switch (it.kind) {
      case op::enqueue:
        decoder->enqueue_syndrome(slot_payload(index), it.length);
        break;
      case op::reset:
        decoder->reset_decoder();
        sticky = nullptr; // a fresh shot clears the sticky error
        break;
      case op::corrections: {
        if (sticky) {
          it.error = sticky;
          sticky = nullptr;
          break;
        }
        const auto *ret = decoder->get_obs_corrections();
        for (std::uint64_t i = 0; i < it.out_length; ++i)
          it.out[i] = ret[i];
        if (it.reset_after)
          decoder->clear_corrections();
        break;
      }
      }
    } catch (const std::exception &e) {
      CUDA_QEC_WARN("decoder {} worker: deferred {} failed: {}", decoder_id,
                    it.kind == op::corrections ? "get_corrections"
                    : it.kind == op::reset     ? "reset"
                                               : "enqueue",
                    e.what());
      if (it.kind == op::corrections)
        it.error = std::current_exception();
      else if (!sticky)
        sticky = std::current_exception();
    } catch (...) {
      if (it.kind == op::corrections)
        it.error = std::current_exception();
      else if (!sticky)
        sticky = std::current_exception();
    }

    g_busy_decoder_workers.fetch_sub(1, std::memory_order_relaxed);

    {
      std::lock_guard<std::mutex> lock(mtx);
      it.done = true;
      head = (head + 1) % kCapacity;
      --count;
    }
    cv_caller.notify_all();
  }
}

decoder_worker *worker_for(std::size_t decoder_id) {
  if (decoder_id >= g_decoder_workers.size() ||
      !g_decoder_workers[decoder_id])
    return nullptr;
  return g_decoder_workers[decoder_id].get();
}

} // namespace

namespace {

bool g_realtime_session_owns_shared_ring_mode = false;

#ifdef CUDAQ_REALTIME_ROOT
inline cudaq_dispatch_launch_fn_t resolve_launch_dispatch_kernel_regular() {
  return reinterpret_cast<cudaq_dispatch_launch_fn_t>(
      ::dlsym(RTLD_DEFAULT, "cudaq_launch_dispatch_kernel_regular"));
}

using set_shared_ring_mode_fn_t = cudaError_t (*)(uint32_t);
inline set_shared_ring_mode_fn_t resolve_set_shared_ring_mode() {
  return reinterpret_cast<set_shared_ring_mode_fn_t>(
      ::dlsym(RTLD_DEFAULT, "cudaq_dispatch_kernel_set_shared_ring_mode"));
}
#endif

bool realtime_mode_inproc_rpc_requested() {
  const char *env = std::getenv("CUDAQ_QEC_REALTIME_MODE");
  if (!env || env[0] == '\0')
    return false;
  return std::strcmp(env, "inproc_rpc") == 0;
}

bool any_decoder_supports_graph_dispatch() {
  for (const auto &dec : g_decoders) {
    if (dec && dec->supports_graph_dispatch())
      return true;
  }
  return false;
}

} // namespace

#ifdef CUDAQ_REALTIME_ROOT
namespace {

void maybe_init_realtime_session() {
  if (!realtime_mode_inproc_rpc_requested()) {
    CUDA_QEC_INFO("CUDAQ_QEC_REALTIME_MODE not set to inproc_rpc; using "
                  "legacy direct-call decoding path.");
    return;
  }

  // Pick DEVICE vs HOST dispatch the same way qec_realtime_session does at
  // initialize(): any graph-capable decoder => DEVICE mode (per-round
  // GRAPH_LAUNCH enqueue + DEVICE_CALL get/reset, driven by the device dispatch
  // kernel); otherwise HOST mode -- CPU decoders such as pymatching run all
  // three RPCs inline on the CPU host loop.  A mixed (graph + non-graph) set is
  // rejected by qec_realtime_session::initialize() below.
  const bool device_mode = any_decoder_supports_graph_dispatch();

  cudaq_dispatch_launch_fn_t launch_fn = nullptr;
  if (device_mode) {
    // DEVICE mode needs the dispatch-kernel launch helper and the device-side
    // shared-ring-mode setter, both resolved from libcudaq-realtime-dispatch.a
    // (absorbed into the final executable).  HOST mode uses neither.
    launch_fn = resolve_launch_dispatch_kernel_regular();
    auto set_mode_fn = resolve_set_shared_ring_mode();
    if (!launch_fn || !set_mode_fn)
      throw std::runtime_error(
          "CUDAQ_QEC_REALTIME_MODE=inproc_rpc requested with a graph-capable "
          "decoder but cudaq_launch_dispatch_kernel_regular and/or "
          "cudaq_dispatch_kernel_set_shared_ring_mode could not be resolved "
          "via dlsym(RTLD_DEFAULT, ...). The host executable must absorb "
          "libcudaq-realtime-dispatch.a and link with --export-dynamic.");

    cudaError_t rc = set_mode_fn(1);
    if (rc != cudaSuccess)
      throw std::runtime_error(
          "CUDAQ_QEC_REALTIME_MODE=inproc_rpc requested but "
          "cudaq_dispatch_kernel_set_shared_ring_mode(1) failed with rc=" +
          std::to_string(rc));
    g_realtime_session_owns_shared_ring_mode = true;
  } else {
    CUDA_QEC_INFO("CUDAQ_QEC_REALTIME_MODE=inproc_rpc with CPU (non-graph) "
                  "decoder(s); using HOST dispatch mode (no device kernel / no "
                  "device shared-ring setup).");
  }

  try {
    g_realtime_session =
        std::make_unique<cudaq::qec::realtime::qec_realtime_session>(g_decoders,
                                                                     launch_fn);
    g_realtime_session->initialize();
  } catch (const std::exception &e) {
    const std::string what = e.what();
    g_realtime_session.reset();
    if (g_realtime_session_owns_shared_ring_mode) {
      if (auto set_mode_fn = resolve_set_shared_ring_mode())
        (void)set_mode_fn(0);
      g_realtime_session_owns_shared_ring_mode = false;
    }
    throw std::runtime_error("CUDAQ_QEC_REALTIME_MODE=inproc_rpc requested but "
                             "qec_realtime_session::initialize() threw: " +
                             what);
  }
}

void maybe_finalize_realtime_session() {
  if (g_realtime_session) {
    try {
      g_realtime_session->finalize();
    } catch (const std::exception &e) {
      CUDA_QEC_WARN("qec_realtime_session::finalize threw: {}", e.what());
    }
    g_realtime_session.reset();
  }
  if (g_realtime_session_owns_shared_ring_mode) {
    if (auto set_mode_fn = resolve_set_shared_ring_mode())
      (void)set_mode_fn(0);
  }
  g_realtime_session_owns_shared_ring_mode = false;
}

} // namespace
#else
namespace {
void maybe_init_realtime_session() {}
void maybe_finalize_realtime_session() {}
} // namespace
#endif

// Helper to pack syndrome bits into bytes (8 bits per byte, MSB first for
// readability)
static std::vector<uint8_t> pack_syndrome_bits(const uint8_t *syndromes,
                                               size_t length) {
  size_t num_bytes = (length + 7) / 8; // Round up
  std::vector<uint8_t> packed(num_bytes, 0);

  for (size_t i = 0; i < length; i++) {
    if (syndromes[i]) {
      size_t byte_idx = i / 8;
      size_t bit_idx = 7 - (i % 8); // MSB first
      packed[byte_idx] |= (1 << bit_idx);
    }
  }

  return packed;
}

namespace cudaq::qec::decoding::host {

cudaqx::heterogeneous_map prepare_decoder_params(
    const cudaq::qec::decoding::config::decoder_config &decoder_config) {
  auto params = decoder_config.decoder_custom_args_to_heterogeneous_map();
  if (decoder_config.type != "trt_decoder")
    return params;

  // batch_size > 1 has no effect on the realtime path: enqueue_syndrome decodes
  // one syndrome per call, so the trt_decoder zero-pads the batch and discards
  // all but slot 0. Warn rather than reject -- the result is correct, just
  // wasteful. (Offline decode_batch users set batch_size via a raw params map,
  // not this realtime config path.)
  if (params.contains("batch_size") &&
      params.get<std::size_t>("batch_size") > 1)
    CUDA_QEC_WARN(
        "trt_decoder batch_size > 1 has no effect on the realtime decode path "
        "(one syndrome is decoded per call); the extra batch slots are "
        "zero-padded and discarded. Use batch_size = 1 for realtime.");

  // The trt_decoder plugin attaches a global decoder only when both
  // "global_decoder" and "global_decoder_params" are present. Most config
  // paths materialize defaults for known global decoders, but callers can still
  // provide a hand-built map with only "global_decoder"; synthesize params here
  // before the O_sparse early return so that decoder still attaches.
  const bool has_global_decoder =
      params.contains("global_decoder") &&
      !params.get<std::string>("global_decoder").empty();
  const bool has_pymatching_global =
      has_global_decoder &&
      params.get<std::string>("global_decoder") == "pymatching";
  if (has_global_decoder && !params.contains("global_decoder_params"))
    params.insert("global_decoder_params", cudaqx::heterogeneous_map());

  if (decoder_config.O_sparse.empty())
    return params;

  const auto num_observables = std::count(decoder_config.O_sparse.begin(),
                                          decoder_config.O_sparse.end(), -1);
  if (num_observables == 0)
    return params;

  auto O = cudaq::qec::pcm_from_sparse_vec(
      decoder_config.O_sparse, num_observables, decoder_config.block_size);
  params.insert("O", O);

  // PyMatching consumes the observable matrix through its params; other global
  // decoders receive only the top-level O until they define a matching
  // contract.
  if (has_pymatching_global) {
    auto global_decoder_params =
        params.get<cudaqx::heterogeneous_map>("global_decoder_params");
    global_decoder_params.insert("O", O);
    params.insert("global_decoder_params", global_decoder_params);
  }

  return params;
}

cudaq::qec::realtime::qec_realtime_session *get_realtime_session() {
  return g_realtime_session.get();
}

int configure_decoders(
    cudaq::qec::decoding::config::multi_decoder_config &config) {
  CUDA_QEC_INFO("Initializing decoders...");

  const auto &decoder_configs = config.decoders;

  // First validate that the there are no duplicate decoder IDs.
  std::set<int64_t> decoder_ids;
  auto min_decoder_id = std::numeric_limits<int64_t>::max();
  auto max_decoder_id = std::numeric_limits<int64_t>::min();
  for (auto &decoder_config : decoder_configs) {
    if (decoder_ids.count(decoder_config.id) > 0) {
      CUDA_QEC_WARN("Duplicate decoder ID found: {}", decoder_config.id);
      return 1;
    }
    decoder_ids.insert(decoder_config.id);
    min_decoder_id = std::min(min_decoder_id, decoder_config.id);
    max_decoder_id = std::max(max_decoder_id, decoder_config.id);
  }

  // Then check that the maximum decoder ID is less than the number of decoders.
  if (max_decoder_id >= decoder_configs.size()) {
    CUDA_QEC_WARN(
        "Maximum decoder ID is greater than the number of decoders: {} >= {}",
        max_decoder_id, decoder_configs.size());
    return 2;
  }
  if (min_decoder_id < 0) {
    CUDA_QEC_WARN("Minimum decoder ID is less than 0: {}", min_decoder_id);
    return 3;
  }

#ifdef CUDAQ_REALTIME_ROOT
  // inproc_rpc DEVICE sessions allocate pinned, device-mapped ring buffers
  // (cudaHostAlloc(cudaHostAllocMapped) + cudaHostGetDevicePointer).
  // cudaSetDeviceFlags(cudaDeviceMapHost) only takes effect BEFORE the device's
  // CUDA context is created, and the per-decoder dry-run below
  // (new_decoder->decode(...)) can create that context for GPU decoders -- so
  // set the flag here, before any decoder is realized, rather than (only) later
  // in qec_realtime_session::initialize().  Best-effort: if a context already
  // exists this returns cudaErrorSetOnActiveProcess, which is harmless (mapped
  // host allocation still works via UVA regardless of this device-wide flag),
  // and HOST-mode CPU sessions do not use mapped memory at all.
  if (realtime_mode_inproc_rpc_requested()) {
    cudaError_t flags_err = cudaSetDeviceFlags(cudaDeviceMapHost);
    if (flags_err != cudaSuccess && flags_err != cudaErrorSetOnActiveProcess)
      CUDA_QEC_WARN(
          "cudaSetDeviceFlags(cudaDeviceMapHost) returned '{}' before "
          "decoder init; continuing (mapped alloc works via UVA).",
          cudaGetErrorString(flags_err));
  }
#endif

  // Create the decoders based on the decoder configs.
  try {
    g_decoder_workers.clear(); // workers reference the decoders; drop first
    g_decoders.clear();
    g_decoders.resize(max_decoder_id + 1);
    for (const auto &decoder_config : decoder_configs) {
      // Form the PCM from the sparse vector.
      auto t0 = std::chrono::high_resolution_clock::now();
      CUDA_QEC_INFO("Creating decoder {} of type {}", decoder_config.id,
                    decoder_config.type);
      auto pcm = cudaq::qec::pcm_from_sparse_vec(decoder_config.H_sparse,
                                                 decoder_config.syndrome_size,
                                                 decoder_config.block_size);
      auto new_decoder = cudaq::qec::get_decoder(
          decoder_config.type, pcm, prepare_decoder_params(decoder_config));
      new_decoder->set_decoder_id(decoder_config.id);
      // Count the number of -1's in the O_sparse vector. That is the number of
      // rows (observables) in the observable matrix.
      auto num_observables = std::count(decoder_config.O_sparse.begin(),
                                        decoder_config.O_sparse.end(), -1);
      // Populate the ***real-time*** fields of the decoder.
      auto observable_matrix = cudaq::qec::pcm_from_sparse_vec(
          decoder_config.O_sparse, num_observables, decoder_config.block_size);
      new_decoder->set_O_sparse(decoder_config.O_sparse);
      if (!decoder_config.D_sparse.empty()) {
        new_decoder->set_D_sparse(decoder_config.D_sparse);
      } else {
        throw std::runtime_error(
            "D_sparse must be provided in decoder configuration");
      }

      // Invoke a dummy decoding operation to force the decoder to be
      // initialized.
      auto t1 = std::chrono::high_resolution_clock::now();
      std::vector<cudaq::qec::float_t> syndrome(decoder_config.syndrome_size,
                                                0.0);
      new_decoder->decode(syndrome);
      auto t2 = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> duration1 = t1 - t0;
      std::chrono::duration<double> duration2 = t2 - t1;
      CUDA_QEC_INFO(
          "Done initializing decoder {} in {:.6f} seconds (creation: {:.6f}s, "
          "initial decoding dry run: {:.6f}s)",
          decoder_config.id, duration1.count() + duration2.count(),
          duration1.count(), duration2.count());

      g_decoders[decoder_config.id] = std::move(new_decoder);
    }
  } catch (const std::exception &e) {
    CUDA_QEC_WARN("Error initializing decoders: {}", e.what());
    return 4;
  }

  // Per-decoder execution workers (see the comment block near the top):
  // each configured decoder gets its own thread + request ring so multiple
  // logical qubits' streams decode concurrently.
  g_decoder_workers.clear();
  g_decoder_workers.resize(g_decoders.size());
  for (std::size_t id = 0; id < g_decoders.size(); ++id)
    if (g_decoders[id])
      g_decoder_workers[id] = std::make_unique<decoder_worker>(
          g_decoders[id].get(), id,
          std::max<std::size_t>(g_decoders[id]->get_num_msyn_per_decode(),
                                4096));

  maybe_init_realtime_session();
  return 0;
}

void finalize_decoders() {
  CUDA_QEC_INFO("Finalizing the realtime decoding library.");
  maybe_finalize_realtime_session();
  // Drain and join the per-decoder workers before the decoders they
  // execute on are destroyed.
  g_decoder_workers.clear();
  g_decoders.clear();
}

__attribute__((visibility("default"))) void
set_syndrome_capture_callback(void (*callback)(const uint8_t *, size_t)) {
  g_syndrome_capture_callback = callback;
}

void enqueue_syndromes(std::size_t decoder_id, uint8_t *syndromes,
                       std::uint64_t syndrome_length, std::uint64_t tag) {
  if (decoder_id >= g_decoders.size()) {
    throw std::invalid_argument(
        fmt::format("Decoder {} not found", decoder_id));
  }
  auto *decoder = g_decoders[decoder_id].get();
  if (!decoder) {
    throw std::invalid_argument(
        fmt::format("Decoder {} not found", decoder_id));
  }
  if (syndrome_length == 0) {
    throw std::invalid_argument("syndrome_length must be greater than 0");
  }
  if (!syndromes) {
    throw std::invalid_argument("syndromes buffer is null");
  }
  const auto max_syndromes = decoder->get_num_msyn_per_decode();
  if (max_syndromes == 0) {
    throw std::invalid_argument(
        "Decoder has no measurement syndromes configured");
  }
  if (syndrome_length > max_syndromes) {
    throw std::invalid_argument(
        fmt::format("syndrome_length ({}) exceeds configured measurement count "
                    "({})",
                    syndrome_length, max_syndromes));
  }

  // Invoke syndrome capture callback if registered (for --save_syndrome
  // feature)
  if (g_syndrome_capture_callback) {
    auto packed_syndrome = pack_syndrome_bits(syndromes, syndrome_length);
    g_syndrome_capture_callback(packed_syndrome.data(), packed_syndrome.size());
  }

#ifdef CUDAQ_REALTIME_ROOT
  if (g_realtime_session) {
    try {
      cudaq::qec::decoding::rpc_producer::enqueue_syndromes(
          *g_realtime_session, decoder_id, syndromes, syndrome_length, tag);
    } catch (
        const cudaq::qec::decoding::rpc_producer::dispatcher_unresponsive_error
            &) {
      maybe_finalize_realtime_session();
      throw;
    }
    return;
  }
#endif

  // Stage onto the decoder's worker: the payload lands directly in the
  // worker's ring slot and execution happens on the decoder's own thread.
  auto *worker = worker_for(decoder_id);
  if (!worker) {
    throw std::runtime_error(
        fmt::format("Decoder {} has no execution worker (initialize_decoders "
                    "not run?)",
                    decoder_id));
  }
  auto *slot = worker->stage();
  std::memcpy(slot, syndromes, syndrome_length);
  worker->commit(decoder_worker::op::enqueue, syndrome_length, tag);
  CUDA_QEC_INFO("[decoder={}][tag={}] staged enqueue_syndrome, "
                "syndrome_length={}",
                decoder_id, tag, syndrome_length);
}

// Zero-copy staging entry points for transport handlers: reserve the
// decoder worker's next ring slot, let the caller write the unpacked
// payload straight into it, then commit. Per-decoder FIFO order is that of
// the stage/commit sequence (single staging producer -- the transport
// dispatcher thread).
uint8_t *stage_syndromes(std::size_t decoder_id, std::uint64_t max_length) {
  if (decoder_id >= g_decoders.size() || !g_decoders[decoder_id]) {
    throw std::invalid_argument(
        fmt::format("Decoder {} not found", decoder_id));
  }
  auto *worker = worker_for(decoder_id);
  if (!worker) {
    throw std::runtime_error(
        fmt::format("Decoder {} has no execution worker", decoder_id));
  }
  if (max_length == 0 || max_length > worker->payload_stride) {
    throw std::invalid_argument(
        fmt::format("syndrome staging length {} out of range (max {})",
                    max_length, worker->payload_stride));
  }
  return worker->stage();
}

void commit_syndromes(std::size_t decoder_id, std::uint64_t length,
                      std::uint64_t tag) {
  auto *worker = worker_for(decoder_id);
  if (!worker) {
    throw std::runtime_error(
        fmt::format("Decoder {} has no execution worker", decoder_id));
  }
  const auto max_syndromes =
      g_decoders[decoder_id]->get_num_msyn_per_decode();
  if (length == 0 ||
      (max_syndromes != 0 && length > max_syndromes &&
       length > worker->payload_stride)) {
    throw std::invalid_argument(
        fmt::format("syndrome_length ({}) out of range", length));
  }
  if (g_syndrome_capture_callback) {
    auto packed_syndrome =
        pack_syndrome_bits(worker->staged_payload(), length);
    g_syndrome_capture_callback(packed_syndrome.data(),
                                packed_syndrome.size());
  }
  worker->commit(decoder_worker::op::enqueue, length, tag);
}

std::uint64_t max_concurrent_decoder_workers() {
  return g_max_busy_decoder_workers.load(std::memory_order_relaxed);
}

void get_corrections(std::size_t decoder_id, uint8_t *corrections,
                     std::uint64_t correction_length, bool reset) {
  CUDA_QEC_INFO("Entered get_corrections function decoder_id={}, "
                "correction_length={}, reset={}",
                decoder_id, correction_length, reset);
  if (decoder_id >= g_decoders.size()) {
    throw std::invalid_argument(
        fmt::format("Decoder {} not found", decoder_id));
  }
  auto *decoder = g_decoders[decoder_id].get();
  if (!decoder) {
    throw std::invalid_argument(
        fmt::format("Decoder {} not found", decoder_id));
  }
  const auto num_observables = decoder->get_num_observables();
  if (correction_length == 0) {
    throw std::invalid_argument("correction_length must be greater than 0");
  }
  if (!corrections) {
    throw std::invalid_argument("corrections buffer is null");
  }
  if (correction_length != num_observables) {
    throw std::invalid_argument(
        fmt::format("correction_length ({}) does not match number of "
                    "observables ({})",
                    correction_length, num_observables));
  }

#ifdef CUDAQ_REALTIME_ROOT
  if (g_realtime_session) {
    try {
      cudaq::qec::decoding::rpc_producer::get_corrections(
          *g_realtime_session, decoder_id, corrections, correction_length,
          reset ? 1u : 0u);
    } catch (
        const cudaq::qec::decoding::rpc_producer::dispatcher_unresponsive_error
            &) {
      maybe_finalize_realtime_session();
      throw;
    }
    return;
  }
#endif

  // Rendezvous with the decoder's worker: waits for its queue to drain
  // (per-decoder FIFO), then the worker produces the corrections. A sticky
  // deferred-execution error from this shot's enqueues is rethrown here.
  auto *worker = worker_for(decoder_id);
  if (!worker) {
    throw std::runtime_error(
        fmt::format("Decoder {} has no execution worker", decoder_id));
  }
  worker->execute_and_wait(decoder_worker::op::corrections, corrections,
                           correction_length, reset);
}

void reset_decoder(std::size_t decoder_id) {
  CUDA_QEC_INFO("Entered reset_decoder for decoder_id={}", decoder_id);
  if (decoder_id >= g_decoders.size()) {
    throw std::invalid_argument(
        fmt::format("Decoder {} not found", decoder_id));
  }
  auto *decoder = g_decoders[decoder_id].get();
  if (!decoder) {
    throw std::invalid_argument(
        fmt::format("Decoder {} not found", decoder_id));
  }

#ifdef CUDAQ_REALTIME_ROOT
  if (g_realtime_session) {
    try {
      cudaq::qec::decoding::rpc_producer::reset_decoder(*g_realtime_session,
                                                        decoder_id);
    } catch (
        const cudaq::qec::decoding::rpc_producer::dispatcher_unresponsive_error
            &) {
      maybe_finalize_realtime_session();
      throw;
    }
    return;
  }
#endif

  // Queued behind any in-flight work for this decoder (ordering matters:
  // a reset must not overtake the previous shot's enqueues).
  auto *worker = worker_for(decoder_id);
  if (!worker) {
    throw std::runtime_error(
        fmt::format("Decoder {} has no execution worker", decoder_id));
  }
  worker->post(decoder_worker::op::reset);
}

} // namespace cudaq::qec::decoding::host
