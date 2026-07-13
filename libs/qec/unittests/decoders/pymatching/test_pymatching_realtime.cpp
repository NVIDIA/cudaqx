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

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <span>
#include <variant>
#include <vector>

namespace {

using DecoderVec = std::vector<std::unique_ptr<cudaq::qec::decoder>>;

class RealtimeSessionDeviceRecordingDecoder final : public cudaq::qec::decoder {
public:
  std::atomic<int> last_decode_device{-2};

  RealtimeSessionDeviceRecordingDecoder(
      const cudaq::qec::sparse_binary_matrix &H,
      const cudaqx::heterogeneous_map &)
      : decoder(H) {}

  cudaq::qec::decoder_result
  decode(const std::vector<cudaq::qec::float_t> &syndrome) override {
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess)
      device = -1;
    last_decode_device.store(device, std::memory_order_release);
    cudaq::qec::decoder_result result;
    result.converged = true;
    result.result = {syndrome.at(0)};
    return result;
  }

  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      RealtimeSessionDeviceRecordingDecoder,
      static std::unique_ptr<cudaq::qec::decoder> create(
          const cudaq::qec::decoder_init &init,
          const cudaqx::heterogeneous_map &params) {
        return cudaq::qec::make_pcm_decoder<
            RealtimeSessionDeviceRecordingDecoder>(init, params);
      })
};
CUDAQ_EXT_PT_REGISTER_TYPE(RealtimeSessionDeviceRecordingDecoder)

class ScopedDeviceRestore {
public:
  ScopedDeviceRestore() {
    if (cudaGetDevice(&previous) != cudaSuccess)
      previous = -1;
  }
  ~ScopedDeviceRestore() {
    if (previous >= 0)
      (void)cudaSetDevice(previous);
  }

private:
  int previous = -1;
};

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

TEST(RealtimeSessionCudaDeviceId, HostDispatcherUsesConfiguredDevice) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count < 2)
    GTEST_SKIP() << "needs >= 2 CUDA devices";

  ScopedDeviceRestore restore;
  cudaqx::tensor<std::uint8_t> H({std::size_t{1}, std::size_t{1}});
  H.at({0, 0}) = 1;
  cudaqx::heterogeneous_map params;
  params.insert("cuda_device_id", 1);
  auto decoder = cudaq::qec::decoder::get(
      "RealtimeSessionDeviceRecordingDecoder", H, params);
  auto *recording =
      dynamic_cast<RealtimeSessionDeviceRecordingDecoder *>(decoder.get());
  ASSERT_NE(recording, nullptr);
  decoder->set_decoder_id(0);
  decoder->set_D_sparse(std::vector<std::vector<std::uint32_t>>{{0}});
  decoder->set_O_sparse(std::vector<std::vector<std::uint32_t>>{{0}});

  DecoderVec decoders;
  decoders.push_back(std::move(decoder));
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  cudaq::qec::realtime::qec_realtime_session session(decoders);
  session.initialize();

  expect_corrections(session, {1}, std::vector<std::uint8_t>{1},
                     /*counter=*/1);
  EXPECT_EQ(recording->last_decode_device.load(std::memory_order_acquire), 1);

  int current = -1;
  ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
  EXPECT_EQ(current, 0);
  session.finalize();
}
