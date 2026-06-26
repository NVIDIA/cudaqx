/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "realtime_decoding.h"
#include "common/FmtCore.h"
#include "cudaq/qec/decoder.h"
#include "cudaq/qec/pcm_utils.h"
#include "cudaq/qec/realtime/decoding_config.h"
#include "cudaq/runtime/logger/logger.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <set>
#include <stdexcept>

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
    CUDAQ_INFO("CUDAQ_QEC_REALTIME_MODE not set to inproc_rpc; using "
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
    CUDAQ_INFO("CUDAQ_QEC_REALTIME_MODE=inproc_rpc with CPU (non-graph) "
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
      CUDAQ_WARN("qec_realtime_session::finalize threw: {}", e.what());
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
    CUDAQ_WARN(
        "trt_decoder batch_size > 1 has no effect on the realtime decode path "
        "(one syndrome is decoded per call); the extra batch slots are "
        "zero-padded and discarded. Use batch_size = 1 for realtime.");

  // The trt_decoder plugin attaches a pymatching global decoder only when both
  // "global_decoder" and "global_decoder_params" are present. Serialization no
  // longer emits an empty params map for the monostate (no-params) case, so
  // synthesize one here -- before the O_sparse early return -- so that a global
  // decoder running on residual detectors without an O matrix still attaches.
  const bool has_pymatching_global =
      params.contains("global_decoder") &&
      params.get<std::string>("global_decoder") == "pymatching";
  if (has_pymatching_global && !params.contains("global_decoder_params"))
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
  CUDAQ_INFO("Initializing decoders...");

  const auto &decoder_configs = config.decoders;

  // First validate that the there are no duplicate decoder IDs.
  std::set<int64_t> decoder_ids;
  auto min_decoder_id = std::numeric_limits<int64_t>::max();
  auto max_decoder_id = std::numeric_limits<int64_t>::min();
  for (auto &decoder_config : decoder_configs) {
    if (decoder_ids.count(decoder_config.id) > 0) {
      CUDAQ_WARN("Duplicate decoder ID found: {}", decoder_config.id);
      return 1;
    }
    decoder_ids.insert(decoder_config.id);
    min_decoder_id = std::min(min_decoder_id, decoder_config.id);
    max_decoder_id = std::max(max_decoder_id, decoder_config.id);
  }

  // Then check that the maximum decoder ID is less than the number of decoders.
  if (max_decoder_id >= decoder_configs.size()) {
    CUDAQ_WARN(
        "Maximum decoder ID is greater than the number of decoders: {} >= {}",
        max_decoder_id, decoder_configs.size());
    return 2;
  }
  if (min_decoder_id < 0) {
    CUDAQ_WARN("Minimum decoder ID is less than 0: {}", min_decoder_id);
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
      CUDAQ_WARN("cudaSetDeviceFlags(cudaDeviceMapHost) returned '{}' before "
                 "decoder init; continuing (mapped alloc works via UVA).",
                 cudaGetErrorString(flags_err));
  }
#endif

  // Create the decoders based on the decoder configs.
  try {
    g_decoders.clear();
    g_decoders.resize(max_decoder_id + 1);
    for (const auto &decoder_config : decoder_configs) {
      // Form the PCM from the sparse vector.
      auto t0 = std::chrono::high_resolution_clock::now();
      CUDAQ_INFO("Creating decoder {} of type {}", decoder_config.id,
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
      CUDAQ_INFO(
          "Done initializing decoder {} in {:.6f} seconds (creation: {:.6f}s, "
          "initial decoding dry run: {:.6f}s)",
          decoder_config.id, duration1.count() + duration2.count(),
          duration1.count(), duration2.count());

      g_decoders[decoder_config.id] = std::move(new_decoder);
    }
  } catch (const std::exception &e) {
    CUDAQ_WARN("Error initializing decoders: {}", e.what());
    return 4;
  }

  maybe_init_realtime_session();
  return 0;
}

void finalize_decoders() {
  CUDAQ_INFO("Finalizing the realtime decoding library.");
  maybe_finalize_realtime_session();
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

  std::vector<uint8_t> syndrome_u8(syndrome_length);
  bool did_decode = false;
  for (std::size_t i = 0; i < syndrome_length; i++) {
    syndrome_u8[i] = syndromes[i];
  }
  std::chrono::duration<double> duration{};
  auto t0 = std::chrono::high_resolution_clock::now();
  did_decode =
      decoder->enqueue_syndrome(syndrome_u8.data(), syndrome_u8.size());
  auto t1 = std::chrono::high_resolution_clock::now();
  duration = t1 - t0;

  // Consider demoting this to a lower log level.
  // Also consider logging the syndrome (at a lower log level).
  CUDAQ_INFO("[decoder={}][tag={}] enqueue_syndrome took {:.3f} us, "
             "syndrome_length={}, did_decode={}",
             decoder_id, tag, duration.count() * 1e6, syndrome_length,
             did_decode ? 'Y' : 'N');
}

void get_corrections(std::size_t decoder_id, uint8_t *corrections,
                     std::uint64_t correction_length, bool reset) {
  CUDAQ_INFO("Entered get_corrections function decoder_id={}, "
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

  auto ret = decoder->get_obs_corrections();
  for (std::size_t i = 0; i < correction_length; ++i) {
    corrections[i] = ret[i];
  }
  if (reset)
    decoder->clear_corrections();
}

void reset_decoder(std::size_t decoder_id) {
  CUDAQ_INFO("Entered reset_decoder for decoder_id={}", decoder_id);
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

  decoder->reset_decoder();
}

} // namespace cudaq::qec::decoding::host
