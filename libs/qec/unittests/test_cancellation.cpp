/*******************************************************************************
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/cancellation.h"
#include "cudaq/qec/decoder.h"

#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <vector>

using cudaq::qec::cancellation_level;
using cudaq::qec::cancellation_source;
using cudaq::qec::cancellation_token;

TEST(CancellationTest, DefaultTokenIsNeverStopped) {
  cancellation_token tok;
  EXPECT_FALSE(tok.stop_possible());
  EXPECT_FALSE(tok.stop_requested());
}

TEST(CancellationTest, FreshSourceIsNotStopped) {
  cancellation_source src;
  auto tok = src.get_token();
  EXPECT_TRUE(tok.stop_possible());
  EXPECT_EQ(src.level(), cancellation_level::none);
  EXPECT_FALSE(tok.stop_requested());
}

TEST(CancellationTest, SoftStopIsAStopButNotAHardOne) {
  cancellation_source src;
  auto tok = src.get_token();
  EXPECT_TRUE(src.request_soft_stop());
  // stop_requested() is "a stop this token reports"; the level itself is only
  // visible on the source.
  EXPECT_TRUE(tok.stop_requested());
  EXPECT_EQ(src.level(), cancellation_level::soft);
  EXPECT_FALSE(tok.hard_only().stop_requested());
  // Repeating the same request reports no change.
  EXPECT_FALSE(src.request_soft_stop());
  EXPECT_EQ(src.level(), cancellation_level::soft);
}

TEST(CancellationTest, HardStopIsReportedByEveryView) {
  cancellation_source src;
  auto tok = src.get_token();
  EXPECT_TRUE(src.request_hard_stop());
  EXPECT_TRUE(tok.stop_requested());
  EXPECT_TRUE(tok.at_least(cancellation_level::soft).stop_requested());
  EXPECT_TRUE(tok.hard_only().stop_requested());
  EXPECT_FALSE(src.request_hard_stop());
}

TEST(CancellationTest, LevelsOnlyEscalate) {
  cancellation_source src;
  auto tok = src.get_token();
  EXPECT_TRUE(src.request_soft_stop());
  EXPECT_TRUE(src.request_hard_stop()); // soft -> hard is a change
  EXPECT_EQ(src.level(), cancellation_level::hard);
  EXPECT_FALSE(src.request_soft_stop()); // hard -> soft is refused
  EXPECT_FALSE(src.request(cancellation_level::none));
  EXPECT_EQ(src.level(), cancellation_level::hard);
}

TEST(CancellationTest, AtLeastViewHidesLevelsBelowItsThreshold) {
  cancellation_source src;
  auto tok = src.get_token();
  auto masked = tok.hard_only(); // == at_least(hard)
  EXPECT_EQ(tok.min_level(), cancellation_level::none);
  EXPECT_EQ(masked.min_level(), cancellation_level::hard);
  EXPECT_TRUE(masked.stop_possible()) << "a hard stop can still arrive";

  src.request_soft_stop();
  // A soft stop is invisible to the masked view but not to the unmasked one.
  EXPECT_TRUE(tok.stop_requested());
  EXPECT_FALSE(masked.stop_requested());

  src.request_hard_stop();
  EXPECT_TRUE(masked.stop_requested());
}

TEST(CancellationTest, AtLeastThresholdsComposeUpwards) {
  cancellation_source src;
  auto tok = src.get_token();
  // A threshold of `none` or `soft` is a no-op today: soft is the lowest
  // level a stop can have, so both views see exactly what the token sees.
  EXPECT_EQ(tok.at_least(cancellation_level::none).min_level(),
            cancellation_level::none);
  auto soft_view = tok.at_least(cancellation_level::soft);
  src.request_soft_stop();
  EXPECT_TRUE(soft_view.stop_requested());
  // Views only tighten: a stricter view of a view keeps the stricter
  // threshold, a milder one leaves it alone.
  auto hard_view = soft_view.at_least(cancellation_level::hard);
  EXPECT_EQ(hard_view.min_level(), cancellation_level::hard);
  EXPECT_EQ(hard_view.at_least(cancellation_level::soft).min_level(),
            cancellation_level::hard);
  EXPECT_FALSE(hard_view.stop_requested());
  src.request_hard_stop();
  EXPECT_TRUE(hard_view.stop_requested());
}

TEST(CancellationTest, AtLeastIsAViewNotACopyOfState) {
  cancellation_source src;
  auto masked = src.get_token().hard_only();
  // Masking is a property of the token value: the source, other tokens and
  // fresh tokens are unaffected, and copies of the view keep the threshold.
  src.request_soft_stop();
  EXPECT_EQ(src.level(), cancellation_level::soft);
  EXPECT_TRUE(src.get_token().stop_requested());
  cancellation_token copy = masked;
  EXPECT_EQ(copy.min_level(), cancellation_level::hard);
  EXPECT_FALSE(copy.stop_requested());
  // A default token masked is still a never-stopped token.
  EXPECT_FALSE(cancellation_token{}.hard_only().stop_possible());
}

TEST(CancellationTest, TokensShareStateWithSourceAndEachOther) {
  cancellation_source src;
  auto a = src.get_token();
  auto b = a;
  auto c = src.get_token();
  src.request_soft_stop();
  EXPECT_TRUE(a.stop_requested());
  EXPECT_TRUE(b.stop_requested());
  EXPECT_TRUE(c.stop_requested());
}

TEST(CancellationTest, CopiedSourcesShareState) {
  cancellation_source src;
  cancellation_source copy = src;
  auto tok = src.get_token();
  copy.request_hard_stop();
  EXPECT_EQ(src.level(), cancellation_level::hard);
  EXPECT_TRUE(tok.stop_requested());
}

TEST(CancellationTest, TokenOutlivesSource) {
  cancellation_token tok;
  {
    cancellation_source src;
    tok = src.get_token();
    src.request_hard_stop();
  }
  EXPECT_TRUE(tok.stop_possible());
  EXPECT_TRUE(tok.stop_requested());
}

TEST(CancellationTest, RequestFromAnotherThreadIsObserved) {
  cancellation_source src;
  auto tok = src.get_token();
  std::thread requester([src]() mutable { src.request_hard_stop(); });
  while (!tok.stop_requested())
    std::this_thread::yield();
  requester.join();
  EXPECT_TRUE(tok.hard_only().stop_requested());
}

namespace {
/// Minimal decoder that records which decode entry point was hit.
struct recording_decoder : public cudaq::qec::decoder {
  using decoder::decode;
  using decoder::decoder;
  std::atomic<int> plain_calls{0};

  cudaq::qec::decoder_result
  decode(const std::vector<cudaq::qec::float_t> &syndrome) override {
    ++plain_calls;
    cudaq::qec::decoder_result r;
    r.converged = true;
    r.result.assign(block_size, 0.0);
    return r;
  }
};

/// Decoder that overrides the cancellable overload and polls the token it was
/// handed; the caller picks the level it reacts to via the token's mask.
struct token_aware_decoder : public recording_decoder {
  using recording_decoder::decode;
  using recording_decoder::recording_decoder;
  bool saw_possible = false;

  std::optional<cudaq::qec::decoder_result>
  decode(const std::vector<cudaq::qec::float_t> &syndrome,
         cancellation_token tok) override {
    saw_possible = tok.stop_possible();
    if (tok.stop_requested())
      return std::nullopt;
    return recording_decoder::decode(syndrome);
  }
};

cudaq::qec::sparse_binary_matrix make_H() {
  cudaqx::tensor<uint8_t> H({2, 3});
  H.at({0, 0}) = 1;
  H.at({0, 1}) = 1;
  H.at({1, 1}) = 1;
  H.at({1, 2}) = 1;
  return cudaq::qec::sparse_binary_matrix(H);
}
} // namespace

TEST(CancellationTest, DecoderDefaultOverloadIgnoresToken) {
  recording_decoder d(make_H());
  cancellation_source src;
  src.request_hard_stop();
  std::vector<cudaq::qec::float_t> syn{1.0, 0.0};
  auto r = d.decode(syn, src.get_token());
  EXPECT_EQ(d.plain_calls, 1);
  ASSERT_TRUE(r.has_value());
  EXPECT_TRUE(r->converged);
  EXPECT_EQ(r->result.size(), 3u);
}

TEST(CancellationTest, DecoderTensorOverloadForwardsToken) {
  token_aware_decoder d(make_H());
  cancellation_source src;
  src.request_soft_stop();
  cudaqx::tensor<uint8_t> syn({2});
  syn.at({0}) = 1;
  // The mask survives the tensor -> vector conversion: a hard_only() view
  // ignores the soft stop, the unmasked token honors it.
  auto r = d.decode(syn, src.get_token().hard_only());
  EXPECT_TRUE(d.saw_possible);
  ASSERT_TRUE(r.has_value());
  EXPECT_TRUE(r->converged);
  EXPECT_EQ(d.plain_calls, 1);
  EXPECT_FALSE(d.decode(syn, src.get_token()).has_value());
  EXPECT_EQ(d.plain_calls, 1);
}

TEST(CancellationTest, DecoderTensorOverloadRejectsNonRank1) {
  token_aware_decoder d(make_H());
  cudaqx::tensor<uint8_t> syn({2, 1});
  EXPECT_THROW(d.decode(syn, cancellation_token{}), std::runtime_error);
}

TEST(CancellationTest, DecoderDefaultBatchOverloadIgnoresToken) {
  recording_decoder d(make_H());
  cancellation_source src;
  src.request_hard_stop();
  std::vector<std::vector<cudaq::qec::float_t>> syns{{1.0, 0.0}, {0.0, 1.0}};
  auto results = d.decode_batch(syns, src.get_token());
  ASSERT_TRUE(results.has_value());
  ASSERT_EQ(results->size(), 2u);
  EXPECT_EQ(d.plain_calls, 2);
  for (const auto &r : *results)
    EXPECT_TRUE(r.converged);
}

TEST(CancellationTest, TokenMaskPicksTheLevelADecoderReactsTo) {
  token_aware_decoder d(make_H());
  cancellation_source src;
  std::vector<cudaq::qec::float_t> syn{1.0, 0.0};
  src.request_soft_stop();
  EXPECT_FALSE(d.decode(syn, src.get_token()).has_value());
  EXPECT_TRUE(d.decode(syn, src.get_token().hard_only()).has_value());
  src.request_hard_stop();
  EXPECT_FALSE(d.decode(syn, src.get_token().hard_only()).has_value());
}
