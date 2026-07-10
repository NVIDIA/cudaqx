/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "DecodingSession.h"
#include "RoundAccumulator.h"
#include "RpcDispatcher.h"
#include "RpcWireFormat.h"

#include "cudaq/qec/decoder.h"
#include "cudaq/qec/sparse_binary_matrix.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
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

} // namespace
