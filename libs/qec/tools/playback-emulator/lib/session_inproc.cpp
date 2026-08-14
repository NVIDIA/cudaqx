/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// The in-process backend: one session per decoder,
/// dispatching straight to that decoder's own DecodingSession payload-level
/// cores, skipping shared-memory rings and CUDA graph dispatch entirely --
/// the emulator only needs the decoder's answer, not the GPU dispatch path.
/// A decoder's own decode work may run on its own thread internally (e.g.
/// dummy_sifl_decoder's background "decode" sleep); that is the decoder
/// implementation's business, not this class's -- an inproc_session is a
/// thin, synchronous dispatcher to one already-resolved DecodingSession,
/// mirroring how a udp_session is bound to exactly one remote decoder
/// (a session always corresponds to exactly one decoder).

#include "RpcSlot.h"
#include "SessionRegistry.h"
#include "cudaq/qec/playback/backends.h"

#include <cassert>

namespace cudaq::qec::playback {

namespace {

using cudaq::qec::decoding_server::DecodingSession;
using cudaq::qec::decoding_server::SessionRegistry;

/// Result of dispatching one raw request frame.
struct DispatchResult {
  /// False only for RPCs with no wire reply (enqueue); such calls are
  /// fire-and-forget, and `status`/`reply_len` are meaningless.
  bool has_reply = false;
  cudaq::qec::decoding::rpc::RpcStatus status =
      cudaq::qec::decoding::rpc::RpcStatus::BAD_REQUEST;
  std::size_t reply_len = 0;
};

/// Parses \p bytes as one of the three decoder RPCs and calls the matching
/// core on \p dec, copying (for get_corrections) the still-bit-packed
/// result into \p reply (capacity \p reply_capacity). The only place in the
/// playback emulator that knows which RPC a frame holds -- the UDP backend
/// is pure transport, and `session` stays opaque to operation semantics
/// (session.h) everywhere else.
DispatchResult dispatch_rpc(DecodingSession &dec, const std::uint8_t *bytes,
                            std::size_t size, std::uint8_t *reply,
                            std::size_t reply_capacity) {
  namespace slot = cudaq::qec::decoding_server::slot;
  using cudaq::qec::decoding::rpc::kEnqueueSyndromesFunctionId;
  using cudaq::qec::decoding::rpc::kGetCorrectionsFunctionId;
  using cudaq::qec::decoding::rpc::kResetDecoderFunctionId;
  using cudaq::qec::decoding::rpc::RpcStatus;
  using cudaq::realtime::RPCHeader;

  if (size < sizeof(RPCHeader))
    return {true, RpcStatus::BAD_REQUEST, 0};
  const auto *header = reinterpret_cast<const RPCHeader *>(bytes);

  if (header->function_id == kEnqueueSyndromesFunctionId) {
    slot::EnqueueView view;
    if (!slot::parse_enqueue(bytes, size, view))
      return {false, RpcStatus::BAD_REQUEST, 0};
    dec.enqueue_core(view);
    return {false, RpcStatus::OK, 0};
  }

  if (header->function_id == kGetCorrectionsFunctionId) {
    slot::GetCorrectionsView view;
    if (!slot::parse_get_corrections(bytes, size, view))
      return {true, RpcStatus::BAD_REQUEST, 0};
    std::size_t reply_len = 0;
    auto status = dec.get_corrections_core(view.return_size, view.reset,
                                           reply, reply_capacity, reply_len);
    return {true, status, reply_len};
  }

  if (header->function_id == kResetDecoderFunctionId) {
    slot::ResetView view;
    if (!slot::parse_reset(bytes, size, view))
      return {true, RpcStatus::BAD_REQUEST, 0};
    return {true, dec.reset_core(), 0};
  }

  return {true, RpcStatus::BAD_REQUEST, 0};
}

class inproc_session : public session {
public:
  inproc_session(std::shared_ptr<SessionRegistry> registry,
                 std::uint64_t decoder_id, std::uint32_t observables)
      : registry_(std::move(registry)), observables_(observables) {
    dec_ = registry_->find(decoder_id);
    assert(dec_ && "make_inproc_sessions() must only construct a session "
                  "for a decoder_id actually present in the config");
  }

  void send_async(const frame &f) override {
    dispatch_rpc(*dec_, f.bytes, f.size, nullptr, 0);
  }

  RpcStatus send_sync(const frame &f, std::span<std::uint8_t> reply,
                      std::size_t &reply_len) override {
    auto result = dispatch_rpc(*dec_, f.bytes, f.size, reply.data(), reply.size());
    reply_len = result.reply_len;
    return result.status;
  }

  capabilities caps() const override {
    capabilities c;
    c.reports_not_ready = true; // get_corrections_core answers NOT_READY
                                 // while a decode is still in flight.
    c.max_frame_bytes = 0; // unbounded; no ring slot constrains this path.
    c.observables = observables_;
    return c;
  }

  // dec_ is already resolved at construction; nothing left to fault in.
  void warm_up() override {}

private:
  std::shared_ptr<SessionRegistry> registry_; // keeps every decoder alive
  DecodingSession *dec_ = nullptr;
  std::uint32_t observables_ = 0;
};

} // namespace

std::vector<std::pair<std::uint64_t, std::unique_ptr<session>>>
make_inproc_sessions(
    const cudaq::qec::decoding::config::multi_decoder_config &config) {
  auto registry = std::make_shared<SessionRegistry>();
  registry->load_from_config(config, "playback-emulator");

  std::vector<std::pair<std::uint64_t, std::unique_ptr<session>>> out;
  out.reserve(config.decoders.size());
  for (const auto &d : config.decoders) {
    const auto id = static_cast<std::uint64_t>(d.id);
    out.emplace_back(
        id, std::make_unique<inproc_session>(
                registry, id, static_cast<std::uint32_t>(d.block_size)));
  }
  return out;
}

} // namespace cudaq::qec::playback
