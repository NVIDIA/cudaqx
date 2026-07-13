/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "DecodingServer.h"
#include "DecodingSession.h"
#include "RoundAccumulator.h"
#include "RpcDispatcher.h"
#include "RpcWireFormat.h"
#include "../lib/hardware_guards.h"

#include "cudaq/qec/decoder.h"
#include "cudaq/qec/sparse_binary_matrix.h"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace {

using namespace cudaq::qec::decoding_server;

class ControlledDecoder final : public cudaq::qec::decoder {
public:
  ControlledDecoder()
      : decoder(cudaq::qec::sparse_binary_matrix::from_csr(
            /*num_rows=*/1, /*num_cols=*/1, /*row_ptrs=*/{0, 1},
            /*col_indices=*/{0})) {
    set_O_sparse(std::vector<std::vector<uint32_t>>{{0}});
    // One detector is the parity of two incoming measurement bits, so a decode
    // completes only after two one-bit enqueue calls.
    set_D_sparse(std::vector<std::vector<uint32_t>>{{0, 1}});
  }

  cudaq::qec::decoder_result
  decode(const std::vector<cudaq::qec::float_t> &syndrome) override {
    if (throw_on_decode)
      throw std::runtime_error("controlled decoder failure");
    cudaq::qec::decoder_result result;
    result.converged = converged;
    result.result = {syndrome.at(0)};
    return result;
  }

  bool converged = false;
  bool throw_on_decode = false;
};

class ServerDeviceRecordingDecoder final : public cudaq::qec::decoder {
public:
  std::atomic<int> last_decode_device{-2};
  inline static std::atomic<int> last_destroy_device{-2};

  ServerDeviceRecordingDecoder(const cudaq::qec::sparse_binary_matrix &H,
                               const cudaqx::heterogeneous_map &)
      : decoder(H) {}

  ~ServerDeviceRecordingDecoder() override {
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess)
      device = -1;
    last_destroy_device.store(device, std::memory_order_release);
  }

  cudaq::qec::decoder_result
  decode(const std::vector<cudaq::qec::float_t> &) override {
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess)
      device = -1;
    last_decode_device.store(device, std::memory_order_release);
    cudaq::qec::decoder_result result;
    result.converged = true;
    result.result = std::vector<cudaq::qec::float_t>(block_size, 0.0);
    return result;
  }

  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      ServerDeviceRecordingDecoder,
      static std::unique_ptr<cudaq::qec::decoder> create(
          const cudaq::qec::decoder_init &init,
          const cudaqx::heterogeneous_map &params) {
        return cudaq::qec::make_pcm_decoder<ServerDeviceRecordingDecoder>(
            init, params);
      })
};
CUDAQ_EXT_PT_REGISTER_TYPE(ServerDeviceRecordingDecoder)

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

class CaptureTransceiver final : public ITransceiver {
public:
  RxFrame recv() override { return {}; }

  void send(const PeerId &, const uint8_t *data, std::size_t len) override {
    response.assign(data, data + len);
  }

  void shutdown() override {}

  std::vector<uint8_t> response;
};

std::pair<std::unique_ptr<DecodingSession>, ControlledDecoder *>
make_session() {
  auto decoder = std::make_unique<ControlledDecoder>();
  auto *raw_decoder = decoder.get();
  SyndromeMappingTable mappings{{0, {{}}}};
  return {DecodingSession::create(std::move(decoder), std::move(mappings)),
          raw_decoder};
}

WorkItem make_enqueue(CaptureTransceiver &transport, uint64_t counter,
                      const std::vector<uint8_t> &bits) {
  WorkItem item{};
  item.function_id = kEnqueueSyndromesFunctionId;
  item.request_id = static_cast<uint32_t>(counter + 1);
  item.response_transport = &transport;
  item.frame_buf.resize(sizeof(RPCHeader) + sizeof(EnqueuePayload) +
                        bit_packed_bytes(bits.size()));

  auto *request = reinterpret_cast<EnqueuePayload *>(item.frame_buf.data() +
                                                     sizeof(RPCHeader));
  request->decoder_id = 0;
  request->counter = static_cast<int64_t>(counter);
  request->syndrome_mapping_id = 0;
  request->num_syndromes = static_cast<int64_t>(bits.size());

  auto *packed =
      item.frame_buf.data() + sizeof(RPCHeader) + sizeof(EnqueuePayload);
  for (std::size_t i = 0; i < bits.size(); ++i)
    if (bits[i] & 1u)
      packed[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
  return item;
}

WorkItem make_get_corrections(CaptureTransceiver &transport, bool reset) {
  WorkItem item{};
  item.function_id = kGetCorrectionsFunctionId;
  item.request_id = 101;
  item.response_transport = &transport;
  item.frame_buf.resize(sizeof(RPCHeader) + sizeof(GetCorrectionsPayload));

  auto *request = reinterpret_cast<GetCorrectionsPayload *>(
      item.frame_buf.data() + sizeof(RPCHeader));
  request->decoder_id = 0;
  request->return_size = 1;
  request->reset = reset ? 1 : 0;
  return item;
}

WorkItem make_reset(CaptureTransceiver &transport) {
  WorkItem item{};
  item.function_id = kResetDecoderFunctionId;
  item.request_id = 202;
  item.response_transport = &transport;
  item.frame_buf.resize(sizeof(RPCHeader) + sizeof(ResetPayload));
  auto *request = reinterpret_cast<ResetPayload *>(item.frame_buf.data() +
                                                   sizeof(RPCHeader));
  request->decoder_id = 0;
  return item;
}

void expect_status(const CaptureTransceiver &transport, RpcStatus status) {
  ASSERT_GE(transport.response.size(), sizeof(RPCResponse));
  const auto *response =
      reinterpret_cast<const RPCResponse *>(transport.response.data());
  EXPECT_EQ(response->magic, kRPCResponseMagic);
  EXPECT_EQ(response->status, static_cast<int32_t>(status));
}

TEST(DecodingSessionStateTest, RequiresACompletedDecodeForEachResult) {
  auto [session, decoder] = make_session();
  CaptureTransceiver transport;

  session->on_get_corrections(make_get_corrections(transport, false));
  expect_status(transport, RpcStatus::NOT_READY);

  session->on_enqueue(make_enqueue(transport, 0, {1}));
  session->on_get_corrections(make_get_corrections(transport, false));
  expect_status(transport, RpcStatus::NOT_READY);

  // A completed decode is ready even when the algorithm reports that it did
  // not converge. Readiness and convergence are different contracts.
  ASSERT_FALSE(decoder->converged);
  session->on_enqueue(make_enqueue(transport, 1, {0}));
  session->on_get_corrections(make_get_corrections(transport, false));
  expect_status(transport, RpcStatus::OK);
  ASSERT_EQ(transport.response.size(), sizeof(RPCResponse) + 1);
  EXPECT_EQ(transport.response[sizeof(RPCResponse)] & 1u, 1u);

  // Accepting part of the next volume makes the previous result stale.
  session->on_enqueue(make_enqueue(transport, 2, {0}));
  session->on_get_corrections(make_get_corrections(transport, false));
  expect_status(transport, RpcStatus::NOT_READY);

  session->on_enqueue(make_enqueue(transport, 3, {0}));
  session->on_get_corrections(make_get_corrections(transport, true));
  expect_status(transport, RpcStatus::OK);
  session->on_get_corrections(make_get_corrections(transport, false));
  expect_status(transport, RpcStatus::NOT_READY);
}

TEST(DecodingSessionStateTest, KeepsFailuresStickyUntilReset) {
  auto [session, decoder] = make_session();
  CaptureTransceiver transport;

  decoder->throw_on_decode = true;
  session->on_enqueue(make_enqueue(transport, 0, {1}));
  session->on_enqueue(make_enqueue(transport, 1, {0}));
  session->on_get_corrections(make_get_corrections(transport, false));
  expect_status(transport, RpcStatus::INTERNAL_ERROR);
  session->on_get_corrections(make_get_corrections(transport, false));
  expect_status(transport, RpcStatus::INTERNAL_ERROR);

  decoder->throw_on_decode = false;
  session->on_reset(make_reset(transport));
  expect_status(transport, RpcStatus::OK);
  session->on_enqueue(make_enqueue(transport, 2, {1}));
  session->on_enqueue(make_enqueue(transport, 3, {0}));
  session->on_get_corrections(make_get_corrections(transport, false));
  expect_status(transport, RpcStatus::OK);

  session->latch_syndromes_dropped();
  session->on_get_corrections(make_get_corrections(transport, false));
  expect_status(transport, RpcStatus::SYNDROMES_DROPPED);
  session->on_get_corrections(make_get_corrections(transport, false));
  expect_status(transport, RpcStatus::SYNDROMES_DROPPED);

  session->on_reset(make_reset(transport));
  expect_status(transport, RpcStatus::OK);
  session->on_get_corrections(make_get_corrections(transport, false));
  expect_status(transport, RpcStatus::NOT_READY);
}

TEST(DecodingSessionStateTest, RejectsMeasurementVolumeOverflow) {
  auto [session, decoder] = make_session();
  CaptureTransceiver transport;
  (void)decoder;

  session->on_enqueue(make_enqueue(transport, 0, {1}));
  session->on_enqueue(make_enqueue(transport, 1, {0, 1}));
  session->on_get_corrections(make_get_corrections(transport, false));
  expect_status(transport, RpcStatus::INTERNAL_ERROR);
  session->on_get_corrections(make_get_corrections(transport, false));
  expect_status(transport, RpcStatus::INTERNAL_ERROR);

  session->on_reset(make_reset(transport));
  expect_status(transport, RpcStatus::OK);
  session->on_get_corrections(make_get_corrections(transport, false));
  expect_status(transport, RpcStatus::NOT_READY);
}

TEST(RoundAccumulatorTest, RejectsMultiVpPassThroughMappings) {
  const RoundKey key{.decoder_id = 0, .counter = 12, .syndrome_mapping_id = 0};
  const SyndromeMappingTable multi_vp{{0, {{}, {}}}};

  RoundAccumulator unequal_lengths;
  const uint8_t vp0[] = {1};
  EXPECT_THROW(unequal_lengths.ingest(key, 0, vp0, 1, multi_vp),
               std::invalid_argument);

  RoundAccumulator equal_lengths;
  const uint8_t vp0_equal[] = {1, 0};
  EXPECT_THROW(equal_lengths.ingest(key, 0, vp0_equal, 2, multi_vp),
               std::invalid_argument);

  RoundAccumulator single_vp;
  const SyndromeMappingTable supported{{0, {{}}}};
  auto completed = single_vp.ingest(key, 0, vp0_equal, 2, supported);
  ASSERT_TRUE(completed.has_value());
  EXPECT_EQ(completed->bits, (std::vector<uint8_t>{1, 0}));
}

TEST(RpcDispatcherTest, ConvertsHandlerExceptionsToErrorResponses) {
  constexpr uint32_t function_id = 0x12345678;
  RpcDispatcher dispatcher;
  dispatcher.register_handler(function_id,
                              [](RxFrame, ResponseWriter &) -> void {
                                throw std::runtime_error("handler failure");
                              });

  RxFrame frame;
  frame.buf.resize(sizeof(RPCHeader));
  auto *header = reinterpret_cast<RPCHeader *>(frame.buf.data());
  header->magic = kRPCRequestMagic;
  header->function_id = function_id;
  header->request_id = 55;

  CaptureTransceiver transport;
  EXPECT_NO_THROW(dispatcher.dispatch(std::move(frame), transport));
  expect_status(transport, RpcStatus::INTERNAL_ERROR);
}

TEST(DecodingSessionCudaDeviceId, WorkerAndTeardownUseConfiguredDevice) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count < 2)
    GTEST_SKIP() << "needs >= 2 CUDA devices";

  ScopedDeviceRestore restore;
  ServerDeviceRecordingDecoder::last_destroy_device.store(
      -2, std::memory_order_release);

  auto H = cudaq::qec::sparse_binary_matrix::from_csr(
      /*num_rows=*/1, /*num_cols=*/1, /*row_ptrs=*/{0, 1},
      /*col_indices=*/{0});
  cudaqx::heterogeneous_map params;
  params.insert("cuda_device_id", 1);
  auto decoder =
      cudaq::qec::decoder::get("ServerDeviceRecordingDecoder", H, params);
  auto *recording = dynamic_cast<ServerDeviceRecordingDecoder *>(decoder.get());
  ASSERT_NE(recording, nullptr);
  decoder->set_O_sparse(std::vector<std::vector<uint32_t>>{{0}});
  decoder->set_D_sparse(std::vector<std::vector<uint32_t>>{{0}});

  SyndromeMappingTable mappings{{0, {{}}}};
  auto session =
      DecodingSession::create(std::move(decoder), std::move(mappings));

  // Worker startup must not depend on the caller still being on device 1.
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  session->start_worker();
  CaptureTransceiver transport;
  ASSERT_TRUE(session->try_enqueue(make_enqueue(transport, 0, {1})));

  for (int attempt = 0; attempt < 200 && recording->last_decode_device.load(
                                             std::memory_order_acquire) < 0;
       ++attempt)
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  EXPECT_EQ(recording->last_decode_device.load(std::memory_order_acquire), 1);

  session->stop_worker();
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  session.reset();
  EXPECT_EQ(ServerDeviceRecordingDecoder::last_destroy_device.load(
                std::memory_order_acquire),
            1);

  int current = -1;
  ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
  EXPECT_EQ(current, 0);
}

TEST(GpuRoceDeviceReconcile, BothUnsetDefaultsToZero) {
  EXPECT_EQ(
      cudaq::qec::decoding_server::reconcile_gpu_roce_device(std::nullopt, -1),
      0);
}

TEST(GpuRoceDeviceReconcile, EnvOnlyWins) {
  EXPECT_EQ(cudaq::qec::decoding_server::reconcile_gpu_roce_device(2, -1), 2);
}

TEST(GpuRoceDeviceReconcile, PinOnlyWins) {
  EXPECT_EQ(
      cudaq::qec::decoding_server::reconcile_gpu_roce_device(std::nullopt, 3),
      3);
}

TEST(GpuRoceDeviceReconcile, AgreementPasses) {
  EXPECT_EQ(cudaq::qec::decoding_server::reconcile_gpu_roce_device(1, 1), 1);
}

TEST(GpuRoceDeviceReconcile, ConflictThrows) {
  EXPECT_THROW(cudaq::qec::decoding_server::reconcile_gpu_roce_device(0, 2),
               std::runtime_error);
}

TEST(SetCudaDeviceForDecode, UnpinnedIsNoOp) {
  // -1 = unpinned: must never touch the device or throw, even on a machine
  // with no CUDA devices at all.
  EXPECT_NO_THROW(cudaq::qec::detail_affinity::set_cuda_device_for_decode(-1));
}

TEST(SetCudaDeviceForDecode, ImpossibleDeviceThrows) {
  // The handshake's failure transport rides on this throw; an id beyond the
  // device count fails cudaSetDevice on any machine, including GPU-less CI.
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess)
    count = 0;
  EXPECT_THROW(
      cudaq::qec::detail_affinity::set_cuda_device_for_decode(count + 7),
      std::runtime_error);
}

TEST(DecodingSessionPinHandshake, PinnedWorkerStartsAndServes) {
  // start_worker() must resolve the pin handshake (throwing on failure per
  // its contract) and leave a live worker serving items.
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count < 1)
    GTEST_SKIP() << "needs >= 1 CUDA device";

  cudaqx::heterogeneous_map params;
  params.insert("cuda_device_id", 0);
  auto dec = cudaq::qec::decoder::get(
      "single_error_lut",
      cudaq::qec::sparse_binary_matrix::from_csr(1, 1, {0, 1}, {0}), params);
  dec->set_O_sparse(std::vector<std::vector<uint32_t>>{{0}});
  dec->set_D_sparse(std::vector<std::vector<uint32_t>>{{0, 1}});

  SyndromeMappingTable table;
  table[0] = {{}};
  auto session = DecodingSession::create(std::move(dec), std::move(table));
  ASSERT_NO_THROW(session->start_worker());

  CaptureTransceiver transport;
  ASSERT_TRUE(session->try_enqueue(make_reset(transport)));
  for (int i = 0; i < 200 && session->reset_count.load() == 0; ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  EXPECT_EQ(session->reset_count.load(), 1u)
      << "pinned worker did not serve the queued item";
}

} // namespace
