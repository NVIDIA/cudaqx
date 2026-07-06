/*******************************************************************************
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "qec_realtime_session.h"
#include "realtime_decoding.h"
#include "rpc_producer.h"
#include "cudaq/qec/decoder.h"
#include "cudaq/qec/realtime/decoding_config.h"

#include <gtest/gtest.h>

#include <dlfcn.h>
#include <sched.h>

#include <cstdint>
#include <fstream>
#include <memory>
#include <span>
#include <string>
#include <variant>
#include <vector>

namespace {

using DecoderVec = std::vector<std::unique_ptr<cudaq::qec::decoder>>;

DecoderVec make_pymatching_decoders(const std::vector<std::uint8_t> &h_vec,
                                    std::size_t syndrome_size,
                                    std::size_t block_size) {
  cudaqx::tensor<std::uint8_t> h;
  h.copy(h_vec.data(), {syndrome_size, block_size});

  DecoderVec decoders;
  auto decoder =
      cudaq::qec::decoder::get("pymatching", h, cudaqx::heterogeneous_map{});
  decoder->set_decoder_id(0);
  std::vector<std::vector<std::uint32_t>> d_sparse(syndrome_size);
  for (std::size_t row = 0; row < syndrome_size; ++row)
    d_sparse[row].push_back(static_cast<std::uint32_t>(row));
  decoder->set_D_sparse(d_sparse);
  std::vector<std::vector<std::uint32_t>> o_sparse(block_size);
  for (std::size_t row = 0; row < block_size; ++row)
    o_sparse[row].push_back(static_cast<std::uint32_t>(row));
  decoder->set_O_sparse(o_sparse);
  decoders.push_back(std::move(decoder));
  return decoders;
}

void expect_corrections(cudaq::qec::realtime::qec_realtime_session &session,
                        const std::vector<std::uint8_t> &syndrome,
                        std::span<const std::uint8_t> expected,
                        std::uint64_t counter, bool reset_on_read = true) {
  cudaq::qec::decoding::rpc_producer::enqueue_syndromes(
      session, /*decoder_id=*/0, syndrome.data(), syndrome.size(),
      /*tag=*/counter);

  std::vector<std::uint8_t> corrections(expected.size(), 0xCC);
  cudaq::qec::decoding::rpc_producer::get_corrections(
      session, /*decoder_id=*/0, corrections.data(), corrections.size(),
      reset_on_read ? 1 : 0);
  EXPECT_EQ(corrections,
            std::vector<std::uint8_t>(expected.begin(), expected.end()));
}

std::vector<std::uint8_t>
read_corrections(cudaq::qec::realtime::qec_realtime_session &session,
                 std::size_t num_corrections) {
  std::vector<std::uint8_t> corrections(num_corrections, 0xCC);
  cudaq::qec::decoding::rpc_producer::get_corrections(
      session, /*decoder_id=*/0, corrections.data(), corrections.size(),
      /*reset=*/0);
  return corrections;
}

void run_case(const std::vector<std::uint8_t> &h_vec, std::size_t syndrome_size,
              std::size_t block_size,
              const std::vector<std::pair<std::vector<std::uint8_t>,
                                          std::vector<std::uint8_t>>> &cases) {
  auto decoders = make_pymatching_decoders(h_vec, syndrome_size, block_size);
  cudaq::qec::realtime::qec_realtime_session session(decoders);
  session.initialize();

  std::uint64_t counter = 1;
  for (const auto &[syndrome, expected] : cases) {
    expect_corrections(session, syndrome, expected, counter++);
    EXPECT_EQ(read_corrections(session, block_size),
              std::vector<std::uint8_t>(block_size, 0));
  }

  session.finalize();
}

} // namespace

TEST(PyMatchingRealtime, CheckRegularEdges) {
  run_case(/*H=*/{1, 0, 1, 1, 0, 1}, /*syndrome_size=*/3, /*block_size=*/2,
           {{{1, 1, 0}, {1, 0}}, {{0, 1, 1}, {0, 1}}, {{1, 0, 1}, {1, 1}}});
}

TEST(PyMatchingRealtime, CheckBoundaryEdges) {
  run_case(/*H=*/{1, 0, 0, 0, 1, 0, 0, 0, 1},
           /*syndrome_size=*/3, /*block_size=*/3,
           {{{1, 0, 0}, {1, 0, 0}},
            {{0, 1, 0}, {0, 1, 0}},
            {{0, 0, 1}, {0, 0, 1}},
            {{1, 0, 0}, {1, 0, 0}}});
}

TEST(PyMatchingRealtime, PreservesCallerColumnOrderUnderNonCanonicalOrdering) {
  run_case(/*H=*/{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0},
           /*syndrome_size=*/4, /*block_size=*/4,
           {{{0, 0, 0, 1}, {0, 0, 1, 0}},
            {{0, 0, 1, 0}, {0, 0, 0, 1}},
            {{1, 0, 0, 0}, {1, 0, 0, 0}}});
}

TEST(PyMatchingRealtime, ResetDecoderClearsCorrections) {
  auto decoders =
      make_pymatching_decoders(/*H=*/{1, 0, 1, 1, 0, 1}, /*syndrome_size=*/3,
                               /*block_size=*/2);
  cudaq::qec::realtime::qec_realtime_session session(decoders);
  session.initialize();

  expect_corrections(session, {1, 0, 1}, std::vector<std::uint8_t>{1, 1},
                     /*counter=*/1, /*reset_on_read=*/false);
  EXPECT_EQ(read_corrections(session, 2), (std::vector<std::uint8_t>{1, 1}));
  cudaq::qec::decoding::rpc_producer::reset_decoder(session, /*decoder_id=*/0);
  EXPECT_EQ(read_corrections(session, 2), (std::vector<std::uint8_t>{0, 0}));

  session.finalize();
}

TEST(PyMatchingRealtime, RejectsOversizedSyndromeRequest) {
  // Single decoder per session (the supported configuration).  The ring slot
  // has headroom beyond the decoder's per-decode window -- it is also sized for
  // the response payload and a 64-byte floor -- so an oversized enqueue can
  // still fit the slot, pass the slot-size check, and reach the new per-decoder
  // length guard.  Without that guard enqueue_syndrome would overflow the
  // decoder's accumulation buffer, silently drop the data (it returns false),
  // and the handler would still ACK success.
  auto decoders = make_pymatching_decoders(/*H=*/{1, 0, 0, 0, 1, 0, 0, 0, 1},
                                           /*syndrome_size=*/3,
                                           /*block_size=*/3);
  cudaq::qec::realtime::qec_realtime_session session(decoders);
  session.initialize();

  const std::uint64_t capacity = decoders[0]->get_num_msyn_per_decode();
  const std::uint64_t oversized = capacity + 1;
  std::vector<std::uint8_t> oversized_syndrome(oversized, 0);
  EXPECT_THROW(cudaq::qec::decoding::rpc_producer::enqueue_syndromes(
                   session, /*decoder_id=*/0, oversized_syndrome.data(),
                   oversized_syndrome.size(), /*tag=*/1),
               std::runtime_error);

  // A correctly-sized request to the same decoder is still accepted (confirms
  // the guard rejected only the oversized one and released the slot).
  std::vector<std::uint8_t> ok_syndrome(capacity, 0);
  EXPECT_NO_THROW(cudaq::qec::decoding::rpc_producer::enqueue_syndromes(
      session, /*decoder_id=*/0, ok_syndrome.data(), ok_syndrome.size(),
      /*tag=*/2));

  session.finalize();
}

TEST(PyMatchingRealtime, InitializeToleratesSparseDecoderIds) {
  auto decoders = make_pymatching_decoders(/*h_vec=*/{1, 0, 1, 1, 0, 1},
                                           /*syndrome_size=*/3,
                                           /*block_size=*/2);
  auto other = make_pymatching_decoders(/*h_vec=*/{1, 0, 1, 1, 0, 1},
                                        /*syndrome_size=*/3,
                                        /*block_size=*/2);
  other[0]->set_decoder_id(2);
  // Gap at index 1: decoder ids {0, 2}, mirroring a production config where
  // decoder_config ids are not contiguous.
  decoders.push_back(nullptr);
  decoders.push_back(std::move(other[0]));

  cudaq::qec::realtime::qec_realtime_session session(decoders);
  session.initialize();
  session.finalize();
}

TEST(PyMatchingRealtime, ConfiguresViaRealtimeDecoderConfig) {
  namespace config = cudaq::qec::decoding::config;

  config::decoder_config decoder_config;
  decoder_config.id = 0;
  decoder_config.type = "pymatching";
  decoder_config.block_size = 3;
  decoder_config.syndrome_size = 3;
  decoder_config.H_sparse = {0, -1, 1, -1, 2, -1};
  decoder_config.O_sparse = {0, -1, 1, -1, 2, -1};
  decoder_config.D_sparse = {0, -1, 1, -1, 2, -1};

  decoder_config.decoder_custom_args = config::pymatching_config();
  auto &pymatching_config =
      std::get<config::pymatching_config>(decoder_config.decoder_custom_args);
  pymatching_config.error_rate_vec = std::vector<double>{0.1, 0.1, 0.1};
  pymatching_config.merge_strategy = "smallest_weight";

  config::multi_decoder_config multi_config;
  multi_config.decoders.push_back(decoder_config);

  const auto yaml_str = multi_config.to_yaml_str(200);
  auto multi_config_from_yaml =
      config::multi_decoder_config::from_yaml_str(yaml_str);
  EXPECT_EQ(multi_config_from_yaml, multi_config);

  EXPECT_EQ(config::configure_decoders(multi_config), 0);

  std::vector<std::uint8_t> syndrome{0, 1, 0};
  EXPECT_NO_THROW(cudaq::qec::decoding::host::enqueue_syndromes(
      /*decoder_id=*/0, syndrome.data(), syndrome.size(), /*tag=*/1));

  std::vector<std::uint8_t> corrections(3, 0xCC);
  EXPECT_NO_THROW(cudaq::qec::decoding::host::get_corrections(
      /*decoder_id=*/0, corrections.data(), corrections.size(),
      /*reset=*/true));
  EXPECT_EQ(corrections, (std::vector<std::uint8_t>{0, 1, 0}));

  corrections.assign(3, 0xCC);
  EXPECT_NO_THROW(cudaq::qec::decoding::host::get_corrections(
      /*decoder_id=*/0, corrections.data(), corrections.size(),
      /*reset=*/false));
  EXPECT_EQ(corrections, (std::vector<std::uint8_t>{0, 0, 0}));

  config::finalize_decoders();
}

//==============================================================================
// Session-level hardware-pinning conflict tests (consensus gate in
// qec_realtime_session::initialize()).  Contract under test: decoders that
// disagree on an affinity knob must degrade LOUDLY (a CUDA_QEC_WARN naming the
// knob) but never break the session -- initialize() succeeds, every decoder
// still decodes correctly through the per-call guards, finalize() is clean.
//==============================================================================

namespace {

// H (3x2) shared by the conflict tests; syndrome {1,1,0} decodes to {1,0}
// (the same known-good pymatching mapping CheckRegularEdges relies on).
const std::vector<std::uint8_t> kConflictH = {1, 0, 1, 1, 0, 1};
constexpr std::size_t kConflictSyndromeSize = 3;
constexpr std::size_t kConflictBlockSize = 2;

// Two identical pymatching decoders with ids {0, 1}.  Affinity knobs are
// applied by each test via the public decoder::set_hardware_params, so the
// knobless (B4) path never touches the knob API at all.
DecoderVec make_session_pair() {
  auto a = make_pymatching_decoders(kConflictH, kConflictSyndromeSize,
                                    kConflictBlockSize);
  auto b = make_pymatching_decoders(kConflictH, kConflictSyndromeSize,
                                    kConflictBlockSize);
  b[0]->set_decoder_id(1);
  DecoderVec decoders;
  decoders.push_back(std::move(a[0]));
  decoders.push_back(std::move(b[0]));
  return decoders;
}

// HOST-mode sessions are process-global-exclusive (g_active_decoders), so the
// slot must be released on EVERY exit path, assertion failures included.
// finalize() is documented idempotent + destructor-safe, so an extra explicit
// finalize() in the happy path is harmless.
struct session_finalizer {
  cudaq::qec::realtime::qec_realtime_session &session;
  ~session_finalizer() { session.finalize(); }
};

// initialize() while capturing stderr.  Never lets an exception escape while
// the gtest capture is active (a dangling capture would poison later tests);
// the caller asserts on `threw` after the capture is closed.
std::string
initialize_capturing_stderr(cudaq::qec::realtime::qec_realtime_session &session,
                            bool &threw, std::string &what) {
  threw = false;
  what.clear();
  testing::internal::CaptureStderr();
  try {
    session.initialize();
  } catch (const std::exception &e) {
    threw = true;
    what = e.what();
  } catch (...) {
    threw = true;
    what = "non-std::exception thrown";
  }
  return testing::internal::GetCapturedStderr();
}

// Round-trip one known syndrome through `decoder_id` and require the exact
// pymatching correction for kConflictH.
// CUDA device count for the B2 gate.  This test target does not link cudart
// directly (a direct cudaGetDeviceCount reference fails to link with "DSO
// missing from command line") and CMake edits are out of scope, but
// libcudart.so is already in the process image as a dependency of
// libcudaq-qec -- so resolve the symbol dynamically for the probe only.
// Returns -1 when the symbol or the CUDA runtime is unavailable.
int probe_cuda_device_count() {
  void *sym = dlsym(RTLD_DEFAULT, "cudaGetDeviceCount");
  if (!sym)
    return -1;
  // ABI: cudaError_t is an int-sized enum; cudaSuccess == 0.
  auto get_count = reinterpret_cast<int (*)(int *)>(sym);
  int count = 0;
  if (get_count(&count) != 0)
    return -1;
  return count;
}

void expect_decoder_still_decodes(
    cudaq::qec::realtime::qec_realtime_session &session, std::size_t decoder_id,
    std::uint64_t tag) {
  const std::vector<std::uint8_t> syndrome{1, 1, 0};
  cudaq::qec::decoding::rpc_producer::enqueue_syndromes(
      session, decoder_id, syndrome.data(), syndrome.size(), tag);
  std::vector<std::uint8_t> corrections(kConflictBlockSize, 0xCC);
  cudaq::qec::decoding::rpc_producer::get_corrections(
      session, decoder_id, corrections.data(), corrections.size(),
      /*reset=*/1);
  EXPECT_EQ(corrections, (std::vector<std::uint8_t>{1, 0}))
      << "decoder " << decoder_id
      << " must still decode correctly after a session-level pinning conflict";
}

} // namespace

// Conflict gate: same NUMA node, different non-empty cpu_affinity lists.
TEST(PyMatchingRealtime, SessionCpuAffinityConflictWarnsAndStillDecodes) {
#if defined(__linux__)
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  ASSERT_EQ(sched_getaffinity(0, sizeof(allowed), &allowed), 0);
  if (!CPU_ISSET(0, &allowed) || !CPU_ISSET(1, &allowed))
    GTEST_SKIP() << "CPUs 0 and 1 are not both in the allowed cpuset";

  auto decoders = make_session_pair();
  cudaqx::heterogeneous_map knobs_a;
  knobs_a.insert("numa_node_id", 0); // node 0 always exists
  knobs_a.insert("cpu_affinity", std::vector<int>{0});
  decoders[0]->set_hardware_params(knobs_a);
  cudaqx::heterogeneous_map knobs_b;
  knobs_b.insert("numa_node_id", 0);
  knobs_b.insert("cpu_affinity", std::vector<int>{1});
  decoders[1]->set_hardware_params(knobs_b);

  cudaq::qec::realtime::qec_realtime_session session(decoders);
  session_finalizer fin{session};
  bool threw = false;
  std::string what;
  const std::string err = initialize_capturing_stderr(session, threw, what);
  ASSERT_FALSE(threw) << "conflicting cpu_affinity lists must degrade to "
                         "per-call guards, not throw: "
                      << what;
  EXPECT_NE(err.find("different cpu_affinity lists"), std::string::npos)
      << "conflict must be loud; captured stderr: [" << err << "]";

  expect_decoder_still_decodes(session, /*decoder_id=*/0, /*tag=*/1);
  expect_decoder_still_decodes(session, /*decoder_id=*/1, /*tag=*/2);
  EXPECT_NO_THROW(session.finalize());
#else
  GTEST_SKIP() << "Linux-only (sched_getaffinity probe)";
#endif
}

// Conflict gate: different cuda_device_ids (no numa knob -- exercises the
// dev-conflict leg; numa_node_id may be soft-derived from the device but on a
// conflict-free node set that leg stays quiet).
TEST(PyMatchingRealtime, SessionCudaDeviceConflictWarnsAndStillDecodes) {
  const int device_count = probe_cuda_device_count();
  if (device_count < 2)
    GTEST_SKIP() << "needs >= 2 CUDA devices, have " << device_count;

  auto decoders = make_session_pair();
  cudaqx::heterogeneous_map knobs_a;
  knobs_a.insert("cuda_device_id", 0);
  decoders[0]->set_hardware_params(knobs_a);
  cudaqx::heterogeneous_map knobs_b;
  knobs_b.insert("cuda_device_id", 1);
  decoders[1]->set_hardware_params(knobs_b);

  cudaq::qec::realtime::qec_realtime_session session(decoders);
  session_finalizer fin{session};
  bool threw = false;
  std::string what;
  const std::string err = initialize_capturing_stderr(session, threw, what);
  ASSERT_FALSE(threw) << "conflicting cuda_device_ids must degrade to "
                         "per-call guards, not throw: "
                      << what;
  EXPECT_NE(err.find("different cuda_device_ids"), std::string::npos)
      << "conflict must be loud; captured stderr: [" << err << "]";

  expect_decoder_still_decodes(session, /*decoder_id=*/0, /*tag=*/1);
  expect_decoder_still_decodes(session, /*decoder_id=*/1, /*tag=*/2);
  EXPECT_NO_THROW(session.finalize());
}

// Conflict gate: different numa_node_ids.  Requires a multi-node host (CI
// lane); must SKIP cleanly on single-node boxes.
TEST(PyMatchingRealtime, SessionNumaNodeConflictWarnsAndStillDecodes) {
  std::ifstream node1("/sys/devices/system/node/node1/cpulist");
  if (!node1.is_open())
    GTEST_SKIP() << "single NUMA node host; numa conflict not testable";

  auto decoders = make_session_pair();
  cudaqx::heterogeneous_map knobs_a;
  knobs_a.insert("numa_node_id", 0);
  decoders[0]->set_hardware_params(knobs_a);
  cudaqx::heterogeneous_map knobs_b;
  knobs_b.insert("numa_node_id", 1);
  decoders[1]->set_hardware_params(knobs_b);

  cudaq::qec::realtime::qec_realtime_session session(decoders);
  session_finalizer fin{session};
  bool threw = false;
  std::string what;
  const std::string err = initialize_capturing_stderr(session, threw, what);
  ASSERT_FALSE(threw) << "conflicting numa_node_ids must disable session NUMA "
                         "pinning, not throw: "
                      << what;
  EXPECT_NE(err.find("different numa_node_ids"), std::string::npos)
      << "conflict must be loud; captured stderr: [" << err << "]";

  expect_decoder_still_decodes(session, /*decoder_id=*/0, /*tag=*/1);
  expect_decoder_still_decodes(session, /*decoder_id=*/1, /*tag=*/2);
  EXPECT_NO_THROW(session.finalize());
}

// Silence contract: a session over decoders that never
// touched a pinning knob must emit ZERO affinity noise ("[cudaq-qec
// affinity]" is the fprintf marker of hardware_affinity.h's affinity_warn)
// across construct + initialize + finalize.
TEST(PyMatchingRealtime, SessionKnoblessDecodersEmitNoAffinityNoise) {
  auto decoders = make_session_pair(); // set_hardware_params never called
  bool threw = false;
  std::string what;
  testing::internal::CaptureStderr();
  {
    cudaq::qec::realtime::qec_realtime_session session(decoders);
    session_finalizer fin{session};
    try {
      session.initialize();
    } catch (const std::exception &e) {
      threw = true;
      what = e.what();
    }
  } // finalize() runs here, inside the capture
  const std::string err = testing::internal::GetCapturedStderr();
  ASSERT_FALSE(threw) << "knobless session must initialize cleanly: " << what;
  EXPECT_EQ(err.find("[cudaq-qec affinity]"), std::string::npos)
      << "knobless session must be affinity-silent, got: [" << err << "]";
}
