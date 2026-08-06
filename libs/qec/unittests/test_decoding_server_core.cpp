/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "CqrTransceiver.h"
#include "DecodingServer.h"
#include "DecodingSession.h"
#include "RoundAccumulator.h"
#include "RpcDispatcher.h"
#include "../lib/hardware_guards.h"

#include "cudaq/qec/decoder.h"
#include "cudaq/qec/realtime/decoder_rpc_wire_format.h"
#include "cudaq/qec/sparse_binary_matrix.h"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cuda_runtime_api.h>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace {

using namespace cudaq::qec::decoding_server;
using namespace cudaq::qec::decoding::rpc;
using cudaq::realtime::RPCHeader;
using cudaq::realtime::RPCResponse;

class ControlledDecoder final : public cudaq::qec::decoder {
public:
  ControlledDecoder()
      : decoder(
            cudaq::qec::decoder_inputs(
                /*H=*/cudaq::qec::sparse_binary_matrix::from_csr(1, 1, {0, 1},
                                                                 {0}),
                /*O=*/
                cudaq::qec::sparse_binary_matrix::from_csr(1, 1, {0, 1}, {0}),
                /*error_rates=*/{},
                // One detector is the parity of two incoming measurement
                // bits, so a decode completes only after two one-bit
                // enqueue calls.
                /*D=*/
                cudaq::qec::sparse_binary_matrix::from_csr(1, 2, {0, 2},
                                                           {0, 1})),
            cudaq::qec::decode_result_type::errors) {}

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
  item.frame_buf.resize(sizeof(RPCHeader) + sizeof(EnqueueRequestPayload) +
                        bit_packed_bytes(bits.size()));

  auto *request = reinterpret_cast<EnqueueRequestPayload *>(
      item.frame_buf.data() + sizeof(RPCHeader));
  request->decoder_id = 0;
  request->counter = static_cast<int64_t>(counter);
  request->syndrome_mapping_id = 0;
  request->num_syndromes = static_cast<int64_t>(bits.size());

  auto *packed =
      item.frame_buf.data() + sizeof(RPCHeader) + sizeof(EnqueueRequestPayload);
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
  item.frame_buf.resize(sizeof(RPCHeader) +
                        sizeof(GetCorrectionsRequestPayload));

  auto *request = reinterpret_cast<GetCorrectionsRequestPayload *>(
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
  item.frame_buf.resize(sizeof(RPCHeader) + sizeof(ResetRequestPayload));
  auto *request = reinterpret_cast<ResetRequestPayload *>(
      item.frame_buf.data() + sizeof(RPCHeader));
  request->decoder_id = 0;
  return item;
}

void expect_status(const CaptureTransceiver &transport, RpcStatus status) {
  ASSERT_GE(transport.response.size(), sizeof(RPCResponse));
  const auto *response =
      reinterpret_cast<const RPCResponse *>(transport.response.data());
  EXPECT_EQ(response->magic, cudaq::realtime::RPC_MAGIC_RESPONSE);
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
  header->magic = cudaq::realtime::RPC_MAGIC_REQUEST;
  header->function_id = function_id;
  header->request_id = 55;

  CaptureTransceiver transport;
  EXPECT_NO_THROW(dispatcher.dispatch(std::move(frame), transport));
  expect_status(transport, RpcStatus::INTERNAL_ERROR);
}

TEST(ResolveDecodeDevice, UnpinnedDefaultsToZero) {
  EXPECT_EQ(cudaq::qec::decoding_server::resolve_decode_device(-1), 0);
}

TEST(ResolveDecodeDevice, PinSelectsDevice) {
  EXPECT_EQ(cudaq::qec::decoding_server::resolve_decode_device(3), 3);
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

/// cuda_device_id_ is protected: setting an impossible id directly bypasses
/// decoder::get()'s construction-time range check, the only front door --
/// which is exactly what makes the handshake's failure path injectable here.
class MispinnedDecoder final : public cudaq::qec::decoder {
public:
  MispinnedDecoder()
      : decoder(
            cudaq::qec::decoder_inputs(
                /*H=*/cudaq::qec::sparse_binary_matrix::from_csr(1, 1, {0, 1},
                                                                 {0}),
                /*O=*/
                cudaq::qec::sparse_binary_matrix::from_csr(1, 1, {0, 1}, {0}),
                /*error_rates=*/{},
                // One detector is the parity of two incoming measurement
                // bits, so a decode completes only after two one-bit
                // enqueue calls.
                /*D=*/
                cudaq::qec::sparse_binary_matrix::from_csr(1, 2, {0, 2},
                                                           {0, 1})),
            cudaq::qec::decode_result_type::errors) {
    cuda_device_id_ = 1 << 20;
  }
  cudaq::qec::decoder_result
  decode(const std::vector<cudaq::qec::float_t> &) override {
    return {};
  }
};

TEST(DecodingSessionPinHandshake, UnhonorablePinFailsStartWorker) {
  // The contract under test: a worker that cannot pin must never serve, and
  // the failure must surface on the caller (server-startup) thread. This is
  // the test that fails if start_worker ever reverts to log-and-continue.
  SyndromeMappingTable table;
  table[0] = {{}};
  auto session = DecodingSession::create(std::make_unique<MispinnedDecoder>(),
                                         std::move(table));
  EXPECT_THROW(session->start_worker(), std::runtime_error);
  // The failed worker was joined inside start_worker; nothing is left to
  // serve and destruction must not hang.
  EXPECT_FALSE(session->worker.joinable());
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
      cudaq::qec::decoder_inputs(
          /*H=*/cudaq::qec::sparse_binary_matrix::from_csr(1, 1, {0, 1}, {0}),
          /*O=*/
          cudaq::qec::sparse_binary_matrix::from_csr(1, 1, {0, 1}, {0}),
          /*error_rates=*/{},
          // One detector is the parity of two incoming measurement
          // bits, so a decode completes only after two one-bit
          // enqueue calls.
          /*D=*/
          cudaq::qec::sparse_binary_matrix::from_csr(1, 2, {0, 2}, {0, 1})),
      params);

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

// ---------------------------------------------------------------------------
// CqrTransceiver direct dispatch (install_dispatch_sink): the handler runs
// inline on the inject() caller thread, with no recv-thread handoff.
// ---------------------------------------------------------------------------

// CQR request slot: RPCHeader (+ optional payload) as the CUDAQ dispatcher
// hands it to inject().
std::vector<uint8_t> make_cqr_slot(uint32_t function_id, uint32_t request_id,
                                   const std::vector<uint8_t> &payload = {}) {
  std::vector<uint8_t> slot(sizeof(RPCHeader) + payload.size());
  RPCHeader header{};
  header.magic = cudaq::realtime::RPC_MAGIC_REQUEST;
  header.function_id = function_id;
  header.arg_len = static_cast<uint32_t>(payload.size());
  header.request_id = request_id;
  std::memcpy(slot.data(), &header, sizeof(header));
  if (!payload.empty())
    std::memcpy(slot.data() + sizeof(header), payload.data(), payload.size());
  return slot;
}

TEST(CqrDirectDispatch, RunsBlockingHandlerInlineAndCompletesTxSlot) {
  CqrTransceiver transceiver;
  RpcDispatcher dispatcher;
  std::thread::id handler_tid{};
  dispatcher.register_handler(kGetCorrectionsFunctionId,
                              [&](RxFrame, ResponseWriter &writer) {
                                handler_tid = std::this_thread::get_id();
                                writer.write_error(RpcStatus::NOT_READY);
                              });
  ASSERT_TRUE(transceiver.install_dispatch_sink([&](RxFrame &&frame) {
    dispatcher.dispatch(std::move(frame), transceiver);
  }));

  const auto rx = make_cqr_slot(kGetCorrectionsFunctionId, /*request_id=*/42);
  std::vector<uint8_t> tx(sizeof(RPCResponse));
  // Completing this call single-threaded also locks in the invariant that
  // inject() does not hold its mutex across the sink: the handler's
  // write_error re-enters send(), which takes the same mutex -- a violation
  // deadlocks right here.
  transceiver.inject(rx.data(), tx.data(), rx.size(),
                     kGetCorrectionsFunctionId);

  EXPECT_EQ(handler_tid, std::this_thread::get_id())
      << "handler did not run inline on the inject() caller thread";
  const auto *response = reinterpret_cast<const RPCResponse *>(tx.data());
  EXPECT_EQ(response->magic, cudaq::realtime::RPC_MAGIC_RESPONSE);
  EXPECT_EQ(response->status, static_cast<int32_t>(RpcStatus::NOT_READY));
  EXPECT_EQ(response->request_id, 42u);
}

TEST(CqrDirectDispatch, EnqueueAcksImmediatelyAndDispatchesInline) {
  CqrTransceiver transceiver;
  RpcDispatcher dispatcher;
  std::thread::id handler_tid{};
  int32_t status_at_handler_time = -1;
  std::vector<uint8_t> handler_frame;
  std::vector<uint8_t> tx(sizeof(RPCResponse));
  dispatcher.register_handler(
      kEnqueueSyndromesFunctionId, [&](RxFrame frame, ResponseWriter &) {
        handler_tid = std::this_thread::get_id();
        // The fire-and-forget ACK must be complete BEFORE the handler runs:
        // an exception after a deferred ACK would stall the CUDAQ transport
        // on an unwritten tx slot.
        status_at_handler_time =
            reinterpret_cast<const RPCResponse *>(tx.data())->status;
        handler_frame = frame.buf;
      });
  ASSERT_TRUE(transceiver.install_dispatch_sink([&](RxFrame &&frame) {
    dispatcher.dispatch(std::move(frame), transceiver);
  }));

  // 4-arg CUDAQ enqueue wire format: [u64 decoder_id][u64 counter]
  // [u64 mapping_id][u64 num_syndromes][bit-packed bytes].
  std::vector<uint8_t> payload(4 * sizeof(uint64_t) + 1);
  const std::array<uint64_t, 4> fields = {/*decoder_id=*/3, /*counter=*/7,
                                          /*mapping_id=*/0,
                                          /*num_syndromes=*/1};
  std::memcpy(payload.data(), fields.data(), sizeof(fields));
  payload.back() = 1;
  const auto rx =
      make_cqr_slot(kEnqueueSyndromesFunctionId, /*request_id=*/23, payload);

  transceiver.inject(rx.data(), tx.data(), rx.size(),
                     kEnqueueSyndromesFunctionId);

  EXPECT_EQ(handler_tid, std::this_thread::get_id());
  EXPECT_EQ(status_at_handler_time, 0) << "enqueue ACK not written before "
                                          "the handler ran";
  ASSERT_EQ(handler_frame.size(),
            sizeof(RPCHeader) + sizeof(EnqueueRequestPayload) + 1);
  const auto *request = reinterpret_cast<const EnqueueRequestPayload *>(
      handler_frame.data() + sizeof(RPCHeader));
  EXPECT_EQ(request->decoder_id, 3);
  EXPECT_EQ(request->counter, 7);
  EXPECT_EQ(request->num_syndromes, 1);
  EXPECT_EQ(handler_frame.back(), 1);
}

TEST(CqrDirectDispatch, ShutdownReleasesBlockedInjectWaiter) {
  CqrTransceiver transceiver;
  RpcDispatcher dispatcher;
  // Handler intentionally produces no response: inject() parks on its
  // pending promise exactly as it does while a session worker decodes.
  dispatcher.register_handler(kGetCorrectionsFunctionId,
                              [](RxFrame, ResponseWriter &) {});
  ASSERT_TRUE(transceiver.install_dispatch_sink([&](RxFrame &&frame) {
    dispatcher.dispatch(std::move(frame), transceiver);
  }));

  const auto rx = make_cqr_slot(kGetCorrectionsFunctionId, /*request_id=*/7);
  std::vector<uint8_t> tx(sizeof(RPCResponse));
  std::thread waiter([&] {
    transceiver.inject(rx.data(), tx.data(), rx.size(),
                       kGetCorrectionsFunctionId);
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  transceiver.shutdown(); // drain must complete the slot and release waiter
  waiter.join();          // hangs here (test timeout) if the drain is broken

  const auto *response = reinterpret_cast<const RPCResponse *>(tx.data());
  EXPECT_EQ(response->magic, cudaq::realtime::RPC_MAGIC_RESPONSE);
  EXPECT_EQ(response->status, static_cast<int32_t>(RpcStatus::BAD_REQUEST));
  EXPECT_EQ(response->request_id, 7u);
}

TEST(CqrDirectDispatch, ConcurrentCollidingClientRidsCompleteTheirOwnSlots) {
  // Per-decoder rings mean one dispatcher thread per ring calls inject()
  // concurrently, and each ring's caller session numbers its requests
  // independently -- so distinct in-flight blocking calls legitimately carry
  // the SAME client request_id.  Every caller must still receive ITS OWN
  // response in ITS tx_slot (correlation is by the transceiver's unique
  // token), discriminated here by the echoed ptp_timestamp.  Under
  // client-rid keying this deadlocks or cross-completes slots.
  constexpr int kThreads = 4;
  CqrTransceiver transceiver;
  RpcDispatcher dispatcher;
  std::atomic<int> arrived{0};
  dispatcher.register_handler(
      kGetCorrectionsFunctionId, [&](RxFrame, ResponseWriter &writer) {
        // Hold every handler until all callers hold a live pending entry, so
        // the colliding-rid window genuinely overlaps.
        arrived.fetch_add(1, std::memory_order_acq_rel);
        while (arrived.load(std::memory_order_acquire) < kThreads)
          std::this_thread::yield();
        writer.write_error(RpcStatus::NOT_READY);
      });
  ASSERT_TRUE(transceiver.install_dispatch_sink([&](RxFrame &&frame) {
    dispatcher.dispatch(std::move(frame), transceiver);
  }));

  constexpr uint32_t kSharedClientRid = 42; // collides across all "rings"
  std::vector<std::vector<uint8_t>> rx(kThreads);
  std::vector<std::vector<uint8_t>> tx(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    rx[t] = make_cqr_slot(kGetCorrectionsFunctionId, kSharedClientRid);
    reinterpret_cast<RPCHeader *>(rx[t].data())->ptp_timestamp = 1000 + t;
    tx[t].assign(sizeof(RPCResponse), 0);
  }

  std::vector<std::thread> callers;
  for (int t = 0; t < kThreads; ++t)
    callers.emplace_back([&, t] {
      transceiver.inject(rx[t].data(), tx[t].data(), rx[t].size(),
                         kGetCorrectionsFunctionId);
    });
  for (auto &caller : callers)
    caller.join();

  for (int t = 0; t < kThreads; ++t) {
    const auto *response = reinterpret_cast<const RPCResponse *>(tx[t].data());
    EXPECT_EQ(response->magic, cudaq::realtime::RPC_MAGIC_RESPONSE) << t;
    EXPECT_EQ(response->status, static_cast<int32_t>(RpcStatus::NOT_READY))
        << t;
    EXPECT_EQ(response->request_id, kSharedClientRid)
        << "client rid not restored for caller " << t;
    EXPECT_EQ(response->ptp_timestamp, 1000u + t)
        << "response landed in the wrong caller's tx_slot";
  }
}

// ---------------------------------------------------------------------------
// Spin-then-block worker wait (SpinPolicy.h): both arrival modes must be
// served regardless of the configured budget (the env knob is parsed once
// per process, so this test is deliberately budget-agnostic).
// ---------------------------------------------------------------------------

TEST(DecodingSessionSpin, ProcessesBurstAndIdleArrivals) {
  auto [session, decoder] = make_session();
  (void)decoder;
  // ControlledDecoder is unpinned, so start_worker's CUDA pin is a no-op and
  // this test runs on driverless boxes too.
  ASSERT_NO_THROW(session->start_worker());

  CaptureTransceiver transport;
  // Burst: back-to-back arrivals land inside the worker's spin window when
  // spinning is enabled.
  constexpr uint64_t kBurst = 8;
  for (uint64_t i = 0; i < kBurst; ++i)
    ASSERT_TRUE(session->try_enqueue(make_reset(transport)));
  for (int i = 0; i < 400 && session->reset_count.load() < kBurst; ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  EXPECT_EQ(session->reset_count.load(), kBurst);

  // Idle arrival: the worker is far past any finite spin budget and parked
  // in its condvar wait; the push must still wake it (the blocking fallback
  // stays armed).
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  ASSERT_TRUE(session->try_enqueue(make_reset(transport)));
  for (int i = 0; i < 400 && session->reset_count.load() < kBurst + 1; ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  EXPECT_EQ(session->reset_count.load(), kBurst + 1);

  // stop_worker() must return promptly from an idle worker whether it is
  // mid-spin or parked (the spin predicate observes the shutdown flag).
  const auto t0 = std::chrono::steady_clock::now();
  session->stop_worker();
  const auto stop_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now() - t0)
                           .count();
  EXPECT_LT(stop_ms, 1000) << "stop_worker did not interrupt the wait";
}

} // namespace
