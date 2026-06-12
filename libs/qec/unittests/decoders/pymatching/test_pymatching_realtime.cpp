/*******************************************************************************
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "qec_realtime_session.h"
#include "rpc_producer.h"
#include "cudaq/qec/decoder.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <span>
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
      session, /*decoder_id=*/0, syndrome.data(), syndrome.size(), counter,
      /*syndrome_mapping_id=*/0);

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
                   oversized_syndrome.size(), /*counter=*/1,
                   /*syndrome_mapping_id=*/0),
               std::runtime_error);

  // A correctly-sized request to the same decoder is still accepted (confirms
  // the guard rejected only the oversized one and released the slot).
  std::vector<std::uint8_t> ok_syndrome(capacity, 0);
  EXPECT_NO_THROW(cudaq::qec::decoding::rpc_producer::enqueue_syndromes(
      session, /*decoder_id=*/0, ok_syndrome.data(), ok_syndrome.size(),
      /*counter=*/2, /*syndrome_mapping_id=*/0));

  session.finalize();
}
