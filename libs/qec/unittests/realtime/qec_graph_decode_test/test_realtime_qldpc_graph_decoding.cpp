/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025-2026 NVIDIA Corporation & Affiliates.                    *
 * All rights reserved.                                                        *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file test_realtime_qldpc_graph_decoding.cpp
/// @brief CI test for the per-round CUDA-graph relay BP decode trio,
/// exercising the full shared-ring two-dispatcher path
/// (CUDAQ_DISPATCH_PATH_HOST for per-round enqueue + CUDAQ_DISPATCH_PATH_DEVICE
/// for get_corrections and reset_decoder).  Post-Step-8 of the realtime-session
/// refactor, this test consumes `qec_realtime_session` (for ring/dispatcher
/// setup) and `rpc_producer` (for host-side enqueue / get_corrections / reset
/// RPC calls), which are the exact same components surface_code-1-local uses in
/// its CUDAQ_QEC_REALTIME_MODE=inproc_rpc path -- so this test is now a
/// wire-format-and-orchestration-faithful contract test of the production
/// stack rather than a parallel re-implementation.
///
/// Flow:
///   1. Loads the relay BP config YAML and the syndrome fixture
///      (post-ROUND_START markers).
///   2. Creates the nv-qldpc decoder via the .cpp shim around decoder::get()
///      and stores it in a single-element vector owned by the fixture (the
///      session takes a reference to that vector).
///   3. Constructs a `qec_realtime_session` over that vector + the
///      libcudaq-realtime-dispatch.a launch fn (passed in from the exe
///      because the .a is hidden-visibility and the .so can't reference it
///      directly).  `session.initialize()` then:
///         - calls capture_decode_graph() per decoder (each captured graph
///           publishes the canonical kEnqueueSyndromesFunctionId; the
///           session sub-routes them by routing_key = decoder_id per
///           proposals/cudaq_realtime_host_api.bs#host-path-graph-routing-key),
///         - allocates the TWO-RING data backing (rx_data != tx_data) +
///           flags per proposals/decoder_server_runtime.md,
///         - builds the N+2 function table (N GRAPH_LAUNCH sharing
///           function_id + distinct routing_key, 2 DEVICE_CALL),
///         - starts HOST_LOOP (CPU thread) + DEVICE_LOOP (persistent GPU
///           cooperative kernel).
///   4. For each shot: replays per-round measurements via N
///      rpc_producer::enqueue_syndromes calls (each returns an empty
///      24-byte RPCResponse, per the spec's Always-Emitted RPCResponse
///      rule -- no did_decode flag on the wire), then one
///      rpc_producer::get_corrections (asserts first byte matches
///      SyndromeEntry::expected_correction).
///   5. After the loop: one rpc_producer::reset_decoder, then one more
///      rpc_producer::get_corrections, asserting the corrections buffer is
///      now zero.

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <fstream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "cudaq/qec/decoder.h"

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#include "qec_realtime_session.h"
#include "rpc_producer.h"
#include "cudaq/qec/realtime/decoder_rpc_ids.h"

// YAML -> decoder construction lives in a .cpp shim so this .cu file doesn't
// need to include decoding_config.h, which uses C++20 `= default` operator==
// patterns that nvcc 13 ICEs on under -std=c++20 + libstdc++ 13.
#include "qldpc_config_loader.h"

#ifndef TEST_DATA_DIR
#define TEST_DATA_DIR "."
#endif

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err);  \
  } while (0)

using namespace cudaq::qec;
using namespace cudaq::realtime;

//==============================================================================
// Syndrome file loader (unchanged from pre-Step-8 -- this only touches the
// fixture format, not the dispatcher contract).
//==============================================================================

struct SyndromeEntry {
  // Flat per-shot bits (all rounds concatenated).  Kept for backward
  // compatibility with the pre-migration test body.  Total size equals
  // num_measurements when the shot is complete.
  std::vector<uint8_t> measurements;
  // Authoritative per-round slices, populated from ROUND_START markers.
  // Each inner vector is one round's worth of measurement bytes
  // (measurements_per_round bytes for nv-qldpc).  num_rounds = inner size.
  std::vector<std::vector<uint8_t>> per_round_measurements;
  uint8_t expected_correction;
};

static std::vector<SyndromeEntry> load_syndromes(const std::string &path,
                                                 std::size_t num_measurements) {
  std::ifstream file(path);
  if (!file.is_open())
    return {};

  std::vector<SyndromeEntry> entries;
  std::string line;
  bool in_corrections = false;
  std::size_t correction_idx = 0;
  // Per-round slicing state, valid only while not in CORRECTIONS_START block.
  // saw_round_start_in_current_shot lets us assert that every SHOT_START
  // block (post-migration) contains at least one ROUND_START before either
  // the next SHOT_START or CORRECTIONS_START -- per the recorder-round-markers
  // todo "fail-fast (assertion) if any shot lacks ROUND_START markers".
  bool saw_round_start_in_current_shot = false;

  auto seal_current_shot = [&]() {
    if (entries.empty())
      return;
    if (!saw_round_start_in_current_shot) {
      throw std::runtime_error(
          "test_realtime_qldpc_graph_decoding: load_syndromes: shot " +
          std::to_string(entries.size() - 1) +
          " is missing ROUND_START markers.  Re-record the fixture (see "
          "re-record-relay-fixture todo) -- pre-marker fixtures cannot be "
          "consumed by this loader because per-round slicing is mandatory "
          "for the migrated dispatcher-contract test.");
    }
  };

  while (std::getline(file, line)) {
    if (line.empty())
      continue;
    if (line.rfind("NUM_DATA", 0) == 0 || line.rfind("NUM_LOGICAL", 0) == 0)
      continue;
    if (line.rfind("CORRECTIONS_START", 0) == 0) {
      seal_current_shot();
      in_corrections = true;
      correction_idx = 0;
      continue;
    }
    if (line.rfind("CORRECTIONS_END", 0) == 0)
      break;

    if (line.rfind("SHOT_START", 0) == 0) {
      seal_current_shot();
      entries.emplace_back();
      entries.back().measurements.reserve(num_measurements);
      entries.back().expected_correction = 0;
      saw_round_start_in_current_shot = false;
      continue;
    }

    if (line.rfind("ROUND_START", 0) == 0) {
      if (entries.empty()) {
        throw std::runtime_error(
            "test_realtime_qldpc_graph_decoding: load_syndromes: "
            "ROUND_START before any SHOT_START in fixture: " +
            path);
      }
      entries.back().per_round_measurements.emplace_back();
      saw_round_start_in_current_shot = true;
      continue;
    }

    if (in_corrections) {
      if (correction_idx < entries.size())
        entries[correction_idx].expected_correction =
            static_cast<uint8_t>(std::stoi(line));
      correction_idx++;
    } else if (!entries.empty()) {
      uint8_t bit = static_cast<uint8_t>(std::stoi(line));
      entries.back().measurements.push_back(bit);
      if (saw_round_start_in_current_shot)
        entries.back().per_round_measurements.back().push_back(bit);
    }
  }
  if (!in_corrections)
    seal_current_shot();
  return entries;
}

//==============================================================================
// GTest fixture
//==============================================================================

class GraphDecodeTest : public ::testing::Test {
protected:
  // The decoder vector that backs the session.  Single-element today; the
  // session is constructed with a reference to this so the test (and the
  // session) share ownership semantics with the production path's
  // `g_decoders` global.
  std::vector<std::unique_ptr<decoder>> decoders_;

  // The realtime session abstracts: shared ring buffer, function table,
  // HOST_LOOP CPU dispatcher (for per-round GRAPH_LAUNCH enqueue), and
  // DEVICE_LOOP persistent GPU dispatcher (for DEVICE_CALL get_corrections
  // and reset_decoder).  Lifetime managed by SetUp/TearDown so each test
  // method gets a clean dispatcher pair.
  std::unique_ptr<cudaq::qec::realtime::qec_realtime_session> session_;

  std::vector<SyndromeEntry> syndromes_;
  std::size_t num_measurements_ = 0;
  std::size_t num_observables_ = 0;
  std::size_t num_rounds_ = 0;
  std::size_t measurements_per_round_ = 0;
  // decoder_id chosen by the plugin at capture_decode_graph() time; lives
  // in the [0, kNvQldpcMaxDecoders) range.  Cached so we can echo it into
  // every RPC payload (enqueue / get_corrections / reset).  Session
  // initialize() walks decoders_ from index 0; this test creates exactly
  // one decoder, so decoder_id_ is 0.
  std::size_t decoder_id_ = 0;

  void SetUp() override {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0)
      GTEST_SKIP() << "No CUDA devices available";
    cudaError_t flags_err = cudaSetDeviceFlags(cudaDeviceMapHost);
    ASSERT_TRUE(flags_err == cudaSuccess ||
                flags_err == cudaErrorSetOnActiveProcess);

    // ---- Load config + build decoder via the .cpp shim ----
    test_realtime_qldpc::LoadedDecoder loaded;
    try {
      loaded = test_realtime_qldpc::load_decoder_from_yaml(
          std::string(TEST_DATA_DIR) + "/config_nv_qldpc_relay.yml");
    } catch (const std::exception &e) {
      FAIL() << "load_decoder_from_yaml threw: " << e.what();
    }
    num_measurements_ = loaded.num_measurements;
    num_observables_ = loaded.num_observables;
    ASSERT_NE(loaded.decoder, nullptr);
    printf("Config: num_measurements=%zu, num_observables=%zu\n",
           num_measurements_, num_observables_);

    decoders_.clear();
    decoders_.push_back(std::move(loaded.decoder));

    // ---- Load syndromes (post-ROUND_START fixture) ----
    syndromes_ = load_syndromes(std::string(TEST_DATA_DIR) +
                                    "/syndromes_nv_qldpc_relay.txt",
                                num_measurements_);
    printf("Loaded %zu test syndromes\n", syndromes_.size());
    ASSERT_GT(syndromes_.size(), 0u);

    // Per-round shape: take it from the first shot's slicing; assert
    // remaining shots match (otherwise the fixture is internally
    // inconsistent and the per-round dispatch contract has no meaning).
    ASSERT_GT(syndromes_[0].per_round_measurements.size(), 0u);
    num_rounds_ = syndromes_[0].per_round_measurements.size();
    measurements_per_round_ = syndromes_[0].per_round_measurements[0].size();
    ASSERT_GT(measurements_per_round_, 0u);
    for (std::size_t i = 0; i < syndromes_.size(); ++i) {
      ASSERT_EQ(syndromes_[i].per_round_measurements.size(), num_rounds_)
          << "Shot " << i << " has "
          << syndromes_[i].per_round_measurements.size() << " rounds; expected "
          << num_rounds_;
      for (std::size_t r = 0; r < num_rounds_; ++r)
        ASSERT_EQ(syndromes_[i].per_round_measurements[r].size(),
                  measurements_per_round_)
            << "Shot " << i << " round " << r << " has wrong measurement count";
    }
    // Sanity-check that the per-round slicing covers exactly the same
    // measurements the decoder's D matrix expects.
    ASSERT_EQ(num_rounds_ * measurements_per_round_, num_measurements_);
    printf("Per-shot shape: %zu rounds x %zu measurements/round = %zu total\n",
           num_rounds_, measurements_per_round_, num_measurements_);

    // ---- Initialize the realtime session ----
    //
    // The session needs the dispatch-kernel launch function pointer; that
    // symbol lives in libcudaq-realtime-dispatch.a (a static archive with
    // hidden visibility), which this exe -- and ONLY this exe -- links.
    // The session's .so cannot reference it directly, so we hand it in as
    // a constructor parameter.  See qec_realtime_session.h for the
    // rationale (cudaq_dispatch_launch_fn_t docstring).
    //
    // Also -- the session does NOT call cudaq_dispatch_kernel_set_shared_-
    // ring_mode itself, by design: that function lives in the same hidden-
    // visibility .a, and the session shared library can't reach it.  So
    // we set it here (the same way surface_code-1-local will, post-
    // Step-9), restoring 0 in TearDown.
    ASSERT_EQ(cudaq_dispatch_kernel_set_shared_ring_mode(1), cudaSuccess);

    session_ = std::make_unique<cudaq::qec::realtime::qec_realtime_session>(
        decoders_, &cudaq_launch_dispatch_kernel_regular);
    try {
      session_->initialize();
    } catch (const std::exception &e) {
      FAIL() << "qec_realtime_session::initialize threw: " << e.what();
    }

    // Under decoder_server_runtime.md every decoder shares the canonical
    // enqueue_syndromes function_id; per-decoder routing is by routing_key.
    // rpc_producer::enqueue_syndromes writes that canonical fid into the
    // RPCHeader and `decoder_id_` into payload arg0 (which doubles as the
    // routing_key the host monitor matches against the function table).
    printf("Session initialized: enqueue_fn_id=0x%08X (canonical, "
           "decoder_id=%zu), %zu slots\n",
           cudaq::qec::decoding::rpc::kEnqueueSyndromesFunctionId, decoder_id_,
           session_->num_slots());
  }

  void TearDown() override {
    // Session finalize is idempotent + safe to call after a partially-
    // initialized SetUp (some ASSERTs above can return before initialize()
    // succeeds; the unique_ptr might still be null in that case).
    if (session_) {
      session_->finalize();
      session_.reset();
    }

    // Best-effort: restore __constant__ to 0 so we don't affect subsequent
    // tests in the same binary.  The session was constructed expecting
    // shared_ring_mode=1 the whole time; SetUp set it, TearDown clears it.
    (void)cudaq_dispatch_kernel_set_shared_ring_mode(0);

    // Drop the decoder vector AFTER the session releases its captured
    // graphs (session.finalize() above).  Order matters because the
    // session holds non-owning pointers into decoders_ for graph release.
    decoders_.clear();
  }
};

//==============================================================================
// Test: Graph decode of all test syndromes via HOST_LOOP dispatch
//
// Post-Step-8 the test body is reduced to: build a payload, hand it to
// rpc_producer, validate the returned bytes against the fixture.  All slot
// leasing / RPCHeader assembly / spin-on-magic logic now lives in
// rpc_producer.cpp and is shared with surface_code-1-local.
//==============================================================================

TEST_F(GraphDecodeTest, DecodesAllSyndromes) {
  ASSERT_NE(session_, nullptr);
  auto &session = *session_;
  using namespace cudaq::qec::decoding;

  int enqueue_ok = 0;
  int correction_matched = 0;
  int correction_mismatched = 0;

  using clock_t = std::chrono::high_resolution_clock;
  std::vector<double> shot_durations_us;
  shot_durations_us.reserve(syndromes_.size());

  // Re-usable correction output buffer (one shot's worth of observables).
  std::vector<std::uint8_t> corrections(num_observables_, 0);

  for (std::size_t shot = 0; shot < syndromes_.size(); ++shot) {
    auto t_start = clock_t::now();

    // ----------------------------------------------------------------------
    // (a) Per-round enqueue.  Under decoder_server_runtime.md the
    //     dispatcher always emits an empty (result_len == 0) RPCResponse
    //     for enqueue_syndromes -- there is no did_decode flag on the
    //     wire anymore.  Whether a window closed and a decode latched
    //     is observable post-suite through get_corrections(reset=1).
    // ----------------------------------------------------------------------
    for (std::size_t round = 0; round < num_rounds_; ++round) {
      const auto &round_bytes = syndromes_[shot].per_round_measurements[round];
      ASSERT_EQ(round_bytes.size(), measurements_per_round_);

      // `tag` carries (shot, round) for diagnostic / ordering checks.
      // rpc_producer::enqueue_syndromes writes `tag` into BOTH:
      //   - payload arg1 (full 64 bits, per decoder_server_runtime.md
      //     #enqueue_syndromes), and
      //   - RPCHeader::request_id (low 32 bits, for the realtime layer's
      //     id-echo correlation).
      // Either path is sufficient to correlate the response with the
      // request.
      const std::uint64_t tag = (static_cast<std::uint64_t>(shot) << 16) |
                                static_cast<std::uint64_t>(round);

      try {
        rpc_producer::enqueue_syndromes(
            session, decoder_id_, round_bytes.data(),
            static_cast<std::uint64_t>(measurements_per_round_), tag);
      } catch (const std::exception &e) {
        FAIL() << "rpc_producer::enqueue_syndromes threw at shot " << shot
               << " round " << round << ": " << e.what();
      }
      enqueue_ok++;
    }

    // ----------------------------------------------------------------------
    // (b) get_corrections.  reset=1 zeros corrections after fetch so the
    //     next shot starts from cleared state on the device side.  The
    //     plugin's BP iteration warm-start is governed by reset_decoder
    //     (issued at end-of-suite below), not by this reset flag.
    // ----------------------------------------------------------------------
    std::fill(corrections.begin(), corrections.end(),
              static_cast<std::uint8_t>(0xCC)); // poison
    try {
      rpc_producer::get_corrections(
          session, decoder_id_, corrections.data(),
          static_cast<std::uint64_t>(num_observables_),
          /*reset=*/1);
    } catch (const std::exception &e) {
      FAIL() << "rpc_producer::get_corrections threw at shot " << shot << ": "
             << e.what();
    }
    const std::uint8_t got = corrections[0];
    const std::uint8_t expected = syndromes_[shot].expected_correction;
    const bool matches = (got == expected);
    if (matches)
      correction_matched++;
    else
      correction_mismatched++;
    EXPECT_EQ(got, expected)
        << "Correction byte 0 mismatch at shot " << shot
        << " (got=" << static_cast<int>(got)
        << ", expected=" << static_cast<int>(expected) << ")";

    auto t_end = clock_t::now();
    double duration_us =
        std::chrono::duration<double, std::micro>(t_end - t_start).count();
    shot_durations_us.push_back(duration_us);

    printf("Shot %zu: enqueue_rounds=%zu, corr=[", shot, num_rounds_);
    for (std::size_t i = 0; i < num_observables_ && i < 8; i++) {
      printf("%u", corrections[i]);
      if (i + 1 < num_observables_ && i + 1 < 8)
        printf(",");
    }
    printf("] expected=%u %s  (%.1f us shot-total)\n",
           static_cast<unsigned>(expected), matches ? "OK" : "MISMATCH",
           duration_us);
  }

  // --------------------------------------------------------------------------
  // (c) End-of-suite reset_decoder + get_corrections to verify the reset
  //     path actually zeros device-side state.  Without this, a regression
  //     where reset is a no-op would still pass the per-shot checks
  //     (because we already pass reset=1 to get_corrections; that uses a
  //     different code path inside the plugin).
  // --------------------------------------------------------------------------
  try {
    rpc_producer::reset_decoder(*session_, decoder_id_);
  } catch (const std::exception &e) {
    FAIL() << "rpc_producer::reset_decoder threw at end-of-suite: " << e.what();
  }

  std::fill(corrections.begin(), corrections.end(),
            static_cast<std::uint8_t>(0xCC));
  try {
    rpc_producer::get_corrections(*session_, decoder_id_, corrections.data(),
                                  static_cast<std::uint64_t>(num_observables_),
                                  /*reset=*/0);
  } catch (const std::exception &e) {
    FAIL() << "rpc_producer::get_corrections post-reset threw: " << e.what();
  }
  for (std::size_t i = 0; i < num_observables_; ++i)
    EXPECT_EQ(corrections[i], 0)
        << "reset_decoder did not zero corrections[" << i << "]";

  printf("\nCompleted: enqueue OK = %d (expected %zu)\n", enqueue_ok,
         syndromes_.size() * num_rounds_);
  printf("Correction comparison vs fixture: %d matched, %d mismatched "
         "(over %zu shots)\n",
         correction_matched, correction_mismatched, syndromes_.size());

  if (shot_durations_us.size() > 1) {
    auto begin = shot_durations_us.begin() + 1;
    auto end = shot_durations_us.end();
    std::size_t n = std::distance(begin, end);
    double sum = std::accumulate(begin, end, 0.0);
    double avg = sum / n;
    double min_val = *std::min_element(begin, end);
    double max_val = *std::max_element(begin, end);
    printf("\n[GraphDecodeTiming] shots=%zu (excluding warmup shot 0)\n", n);
    printf("[GraphDecodeTiming] min=%.1f us  avg=%.1f us  max=%.1f us\n",
           min_val, avg, max_val);
  }

  EXPECT_EQ(enqueue_ok, static_cast<int>(syndromes_.size() * num_rounds_));
  EXPECT_EQ(correction_matched, static_cast<int>(syndromes_.size()));
  EXPECT_EQ(correction_mismatched, 0);
}
