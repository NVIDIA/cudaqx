/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 ******************************************************************************/

/// Tests the session backends: session.h's ordering contract (submission
/// order is delivery order; a waiting caller holds back nobody), the `null`
/// jitter floor, and the UDP backend (timeouts, max_frame_bytes
/// trustworthiness, plan()'s oversized-enqueue rejection).

#include "session.h"
#include "emulator.h"
#include "syndrome_source.h"

#include <algorithm>
#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <cstring>
#include <gtest/gtest.h>
#include <mutex>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace cudaq::qec::playback;

// ─── SessionOrdering ────────────────────────────────────────────────────────
//
// Without both halves of the ordering contract a schedule can't place a read
// relative to its syndromes without stalling for a whole decode, and a late
// read is worse than useless (DecodingSession drops an unread result once
// the next round lands). The recording server here only tests the client's
// transport and dispatch order, not decoder semantics.

namespace {

using cudaq::qec::decoding::rpc::EnqueueRequestPayload;
using cudaq::qec::decoding::rpc::GetCorrectionsRequestPayload;
using cudaq::qec::decoding::rpc::kEnqueueSyndromesFunctionId;
using cudaq::qec::decoding::rpc::kGetCorrectionsFunctionId;
using cudaq::realtime::RPCHeader;
using cudaq::realtime::RPCResponse;

constexpr std::chrono::milliseconds kDecodeTime{120};

/// A loopback UDP socket that logs the function_id of every frame it
/// receives, in arrival order, and answers reads after `decode_time` --
/// long enough that a client which waits for one instead of moving on shows
/// up as a reordering, not as a scheduling wobble.
class recording_server {
public:
  explicit recording_server(std::chrono::milliseconds decode_time = kDecodeTime)
      : decode_time_(decode_time) {
    fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    ::bind(fd_, reinterpret_cast<sockaddr *>(&addr), sizeof(addr));
    socklen_t len = sizeof(addr);
    ::getsockname(fd_, reinterpret_cast<sockaddr *>(&addr), &len);
    endpoint_ = "127.0.0.1:" + std::to_string(ntohs(addr.sin_port));

    timeval tv{};
    tv.tv_usec = 50'000;
    ::setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    thread_ = std::thread([this] { serve(); });
  }

  ~recording_server() {
    stop_.store(true);
    if (thread_.joinable())
      thread_.join();
    ::close(fd_);
  }

  const std::string &endpoint() const { return endpoint_; }

  std::vector<std::uint32_t> arrivals() const {
    std::lock_guard<std::mutex> lock(mu_);
    return arrivals_;
  }

private:
  void serve() {
    std::vector<std::uint8_t> buf(65536);
    while (!stop_.load()) {
      sockaddr_in from{};
      socklen_t from_len = sizeof(from);
      const ssize_t n = ::recvfrom(fd_, buf.data(), buf.size(), 0,
                                   reinterpret_cast<sockaddr *>(&from), &from_len);
      if (n < static_cast<ssize_t>(sizeof(RPCHeader)))
        continue;
      RPCHeader hdr{};
      std::memcpy(&hdr, buf.data(), sizeof(hdr));
      {
        std::lock_guard<std::mutex> lock(mu_);
        arrivals_.push_back(hdr.function_id);
      }
      if (hdr.function_id != kGetCorrectionsFunctionId)
        continue; // enqueue has no reply on the wire

      // Answering inline is what makes this a faithful stand-in: the real
      // server runs one dispatcher thread per ring, so a slow decode delays
      // everything behind it on that ring and nothing else.
      std::this_thread::sleep_for(decode_time_);
      std::vector<std::uint8_t> out(sizeof(RPCResponse) + 1, 0);
      RPCResponse resp{};
      resp.magic = cudaq::realtime::RPC_MAGIC_RESPONSE;
      resp.status = 0;
      resp.result_len = 1;
      resp.request_id = hdr.request_id;
      std::memcpy(out.data(), &resp, sizeof(resp));
      ::sendto(fd_, out.data(), out.size(), 0,
               reinterpret_cast<sockaddr *>(&from), from_len);
    }
  }

  int fd_ = -1;
  std::string endpoint_;
  std::chrono::milliseconds decode_time_;
  mutable std::mutex mu_;
  std::vector<std::uint32_t> arrivals_;
  std::atomic<bool> stop_{false};
  std::thread thread_;
};

std::vector<std::uint8_t> make_read_frame(std::uint32_t request_id) {
  std::vector<std::uint8_t> buf(sizeof(RPCHeader) +
                                sizeof(GetCorrectionsRequestPayload));
  RPCHeader hdr{};
  hdr.magic = cudaq::realtime::RPC_MAGIC_REQUEST;
  hdr.function_id = kGetCorrectionsFunctionId;
  hdr.arg_len = sizeof(GetCorrectionsRequestPayload);
  hdr.request_id = request_id;
  std::memcpy(buf.data(), &hdr, sizeof(hdr));
  GetCorrectionsRequestPayload payload{0, 1, 1};
  std::memcpy(buf.data() + sizeof(hdr), &payload, sizeof(payload));
  return buf;
}

std::vector<std::uint8_t> make_enqueue_frame(std::uint32_t request_id) {
  std::vector<std::uint8_t> buf(sizeof(RPCHeader) +
                                sizeof(EnqueueRequestPayload) + 1);
  RPCHeader hdr{};
  hdr.magic = cudaq::realtime::RPC_MAGIC_REQUEST;
  hdr.function_id = kEnqueueSyndromesFunctionId;
  hdr.arg_len = sizeof(EnqueueRequestPayload) + 1;
  hdr.request_id = request_id;
  std::memcpy(buf.data(), &hdr, sizeof(hdr));
  EnqueueRequestPayload payload{0, 0, 0, 8};
  std::memcpy(buf.data() + sizeof(hdr), &payload, sizeof(payload));
  return buf;
}

std::unique_ptr<session> connect(const recording_server &server) {
  auto sessions = make_udp_sessions(
      std::unordered_map<std::uint64_t, std::string>{{0, server.endpoint()}},
      /*timeout_ms=*/5000);
  return std::move(sessions[0].second);
}

long long ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now() - t0)
      .count();
}

} // namespace

TEST(SessionOrdering, AWaitingCallerDoesNotHoldBackTheNextSend) {
  // The regression: with the reply collected inside the send, syndromes
  // meant to follow a read went out while it was still in flight, so the
  // decoder saw them the wrong way round.
  recording_server server;
  auto s = connect(server);

  const auto read = make_read_frame(1);
  const auto enqueue = make_enqueue_frame(2);

  const auto t0 = std::chrono::steady_clock::now();
  const std::uint32_t t = s->submit({read.data(), read.size()});
  const auto after_submit = ms_since(t0);
  s->send_async({enqueue.data(), enqueue.size()});
  const auto after_enqueue = ms_since(t0);

  EXPECT_LT(after_submit, kDecodeTime.count() / 2)
      << "submit() waited for the reply instead of just publishing the frame";
  EXPECT_LT(after_enqueue, kDecodeTime.count() / 2)
      << "the next frame was held back by a read that had not answered yet";

  std::vector<std::uint8_t> reply(1);
  std::size_t reply_len = 0;
  EXPECT_EQ(s->await(t, reply, reply_len), RpcStatus::OK);
  EXPECT_GE(ms_since(t0), kDecodeTime.count()); // it really did have to wait

  EXPECT_EQ(server.arrivals(),
            (std::vector<std::uint32_t>{kGetCorrectionsFunctionId,
                                        kEnqueueSyndromesFunctionId}));
}

TEST(SessionOrdering, ConcurrentSubmissionsEachGetTheirOwnReply) {
  // Two outstanding reads at once must both work: matching replies by
  // request_id, not by arrival order, is what makes that safe.
  recording_server server;
  auto s = connect(server);

  const auto first = make_read_frame(11);
  const auto second = make_read_frame(22);
  const std::uint32_t a = s->submit({first.data(), first.size()});
  const std::uint32_t b = s->submit({second.data(), second.size()});
  EXPECT_EQ(a, 11u);
  EXPECT_EQ(b, 22u);

  // Collected in the opposite order to make the point that a request id, not
  // arrival order, is what says which reply belongs to whom.
  std::vector<std::uint8_t> reply(1);
  std::size_t reply_len = 0;
  EXPECT_EQ(s->await(b, reply, reply_len), RpcStatus::OK);
  EXPECT_EQ(s->await(a, reply, reply_len), RpcStatus::OK);
  EXPECT_EQ(server.arrivals().size(), 2u);
}

TEST(SessionOrdering, AnUnblockingReadKeepsItsPlaceInTheSyndromeStream) {
  // Schedule level: a `signal=` read sits on the timeline that owns the
  // syndromes, between the rounds it must separate, and hands its answer
  // back through a signal. What the decoder sees must be enqueue, read,
  // enqueue -- and the timeline must not have stopped for the decode.
  recording_server server;
  auto sessions = make_udp_sessions(
      std::unordered_map<std::uint64_t, std::string>{{0, server.endpoint()}},
      /*timeout_ms=*/5000);
  std::unordered_map<std::uint64_t, session *> router{{0, sessions[0].second.get()}};
  static_source src(std::vector<std::vector<std::uint8_t>>(8, {1}));

  auto sched = parse("0 stream source=0 rounds=1\n"
                     "+0 get_corrections return_size=1 signal=shot\n"
                     "+0 stream source=0 rounds=1\n",
                     {0}, 1000);
  auto result = run(plan(sched, router, {{0, &src}}, {}));

  EXPECT_EQ(server.arrivals(),
            (std::vector<std::uint32_t>{kEnqueueSyndromesFunctionId,
                                        kGetCorrectionsFunctionId,
                                        kEnqueueSyndromesFunctionId}));

  const auto &read = result.records[1];
  const auto &next_round = result.records[2];
  EXPECT_LT(next_round.call_ns - read.call_ns,
            static_cast<std::uint64_t>(kDecodeTime.count()) * 1'000'000 / 2)
      << "the timeline stopped to wait for the answer instead of moving on";
  EXPECT_TRUE(read.read_completed); // filled in by the reader thread
  EXPECT_EQ(read.correction_count, 1u);
  EXPECT_GE(read.return_ns - read.call_ns,
            static_cast<std::uint64_t>(kDecodeTime.count()) * 1'000'000)
      << "the record should still span the whole read";
}

TEST(SessionOrdering, ParserAndPlanRejectAnUncollectableRead) {
  EXPECT_THROW(parse("0 get_corrections return_size=1 signal=\n", {0}, 1000),
               std::invalid_argument); // `signal=` with nothing to raise
  EXPECT_THROW(parse("0 get_corrections return_size=1 signal=a signal=b\n",
                     {0}, 1000),
               std::invalid_argument); // one read, one answer, one signal
  EXPECT_NO_THROW(
      parse("0 get_corrections return_size=1 signal=done\n", {0}, 1000));

  // A signal an unblocking read raises counts as raised for the schedule's
  // wait graph: a later `until=` has something that does eventually bring
  // it up.
  auto null = make_null_session();
  std::unordered_map<std::uint64_t, session *> router{{0, null.get()}};
  static_source src(std::vector<std::vector<std::uint8_t>>(4, {1}));
  EXPECT_NO_THROW(plan(parse("0 get_corrections return_size=1 signal=done\n"
                             "+0 stream source=0 until=done\n",
                             {0}, 1000),
                       router, {{0, &src}}, {}));
}

// ─── NullBackendAdvanced ────────────────────────────────────────────────────
//
// Adversarial tests for the `null` backend (jitter floor): max_frame_bytes=0
// (unbounded), discards every frame via an atomic checksum. Probes the
// edges -- empty/huge frames, reply-buffer bounds, repeated calls -- since a
// backend whose job is to cost nothing is the likeliest to skip a bounds check.

TEST(NullBackendAdvanced, FramesOfEverySizeAreDiscardedWithoutCrashingOrReplying) {
  // 0 is the interesting end (a null pointer with no bytes to checksum) and
  // the odd sizes either side of a power of two are the other: the checksum
  // loop is per-byte, so a partial trailing word is where it would run off.
  auto s = make_null_session();
  for (std::size_t size : {0u, 1u, 3u, 7u, 63u, 65u, 4095u, 4097u, 8192u}) {
    SCOPED_TRACE(size);
    std::vector<std::uint8_t> buf(size);
    for (std::size_t i = 0; i < size; ++i)
      buf[i] = static_cast<std::uint8_t>(i * 37 + 1);
    const frame f{size ? buf.data() : nullptr, size};

    EXPECT_NO_THROW(s->send_async(f));

    // Sentinel-filled, so a reply that is never written cannot be mistaken
    // for one that was written with zeros.
    std::vector<std::uint8_t> reply(16, 0xAB);
    std::size_t reply_len = 999;
    RpcStatus status = RpcStatus::BUSY;
    EXPECT_NO_THROW(status = s->await(s->submit(f), reply, reply_len));
    EXPECT_EQ(status, RpcStatus::OK);
    EXPECT_EQ(reply_len, reply.size());
    for (auto b : reply)
      EXPECT_EQ(b, 0u) << "null zero-fills the reply it discards";
  }
}

TEST(NullBackendAdvanced, AwaitNeverWritesPastTheSuppliedReplySpan) {
  // Canary bytes surround a small reply buffer inside one contiguous
  // allocation; await must only ever touch the span it was given.
  struct Canaried {
    std::uint8_t before[32];
    std::uint8_t reply[8];
    std::uint8_t after[32];
  } buf;
  std::memset(buf.before, 0xCC, sizeof(buf.before));
  std::memset(buf.reply, 0xAB, sizeof(buf.reply));
  std::memset(buf.after, 0xCC, sizeof(buf.after));

  auto s = make_null_session();
  std::vector<std::uint8_t> req(16, 0x11);
  frame f{req.data(), req.size()};
  std::span<std::uint8_t> reply_span(buf.reply, sizeof(buf.reply));
  std::size_t reply_len = 999;
  RpcStatus status = s->await(s->submit(f), reply_span, reply_len);

  EXPECT_EQ(status, RpcStatus::OK);
  EXPECT_EQ(reply_len, sizeof(buf.reply)); // the whole zero-filled span.
  for (auto b : buf.reply)
    EXPECT_EQ(b, 0u); // reply itself is zero-filled by null_session
  for (auto b : buf.before)
    EXPECT_EQ(b, 0xCC) << "await wrote before the reply span";
  for (auto b : buf.after)
    EXPECT_EQ(b, 0xCC) << "await wrote past the reply span";
}

TEST(NullBackendAdvanced, MaxFrameBytesIsUnboundedAndStaysThatWayUnderLoad) {
  // 0 = unbounded, per session.h's comment. Interleaving hundreds of
  // send_async and submit/await calls with varying frame contents must not
  // disturb it: nothing mutable is shared between calls except the internal
  // checksum accumulator.
  auto s = make_null_session();
  EXPECT_EQ(s->max_frame_bytes, 0u);
  for (int i = 0; i < 500; ++i) {
    std::vector<std::uint8_t> buf(static_cast<std::size_t>(1 + (i % 37)),
                                  static_cast<std::uint8_t>(i));
    frame f{buf.data(), buf.size()};
    if (i % 2 == 0) {
      s->send_async(f);
    } else {
      std::vector<std::uint8_t> reply(4);
      std::size_t reply_len = 0;
      EXPECT_EQ(s->await(s->submit(f), reply, reply_len), RpcStatus::OK);
    }
  }
  EXPECT_EQ(s->max_frame_bytes, 0u);
}

// ─── UdpBackendAdvanced ─────────────────────────────────────────────────────
//
// Adversarial tests for the UDP backend ("Replies are matched by
// request_id, never by arrival order... This is not optional"): a
// genuinely-closed-port timeout, max_frame_bytes trustworthiness, and
// plan()'s oversized-enqueue rejection trusting that number.

namespace {

/// Hand-builds a raw [RPCHeader][ResetRequestPayload] frame -- RpcSlot.h's
/// parse_reset layout -- so tests can drive udp_session::submit()/await()
/// directly, without plan()/run(), for precise control over call count and
/// wall-clock timing.
std::vector<std::uint8_t> make_reset_frame(std::uint64_t decoder_id,
                                           std::uint32_t request_id) {
  using cudaq::qec::decoding::rpc::kResetDecoderFunctionId;
  using cudaq::qec::decoding::rpc::ResetRequestPayload;
  using cudaq::realtime::RPCHeader;

  std::vector<std::uint8_t> buf(sizeof(RPCHeader) + sizeof(ResetRequestPayload));
  RPCHeader hdr{};
  hdr.magic = cudaq::realtime::RPC_MAGIC_REQUEST;
  hdr.function_id = kResetDecoderFunctionId;
  hdr.arg_len = sizeof(ResetRequestPayload);
  hdr.request_id = request_id;
  hdr.ptp_timestamp = 0;
  std::memcpy(buf.data(), &hdr, sizeof(hdr));

  ResetRequestPayload payload{};
  payload.decoder_id = static_cast<std::int64_t>(decoder_id);
  std::memcpy(buf.data() + sizeof(hdr), &payload, sizeof(payload));
  return buf;
}

/// Binds and immediately closes a loopback UDP socket to obtain a port
/// number that is (at the moment of the check) genuinely unbound -- more
/// reliable than guessing an arbitrary high port.
std::string closed_loopback_endpoint() {
  int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = 0;
  ::bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr));
  socklen_t len = sizeof(addr);
  ::getsockname(fd, reinterpret_cast<sockaddr *>(&addr), &len);
  auto port = ntohs(addr.sin_port);
  ::close(fd);
  return "127.0.0.1:" + std::to_string(port);
}

} // namespace

TEST(UdpBackendAdvanced, AwaitOnAGenuinelyClosedPortTimesOutBoundedAndReturnsAnError) {
  auto endpoint = closed_loopback_endpoint();
  constexpr std::uint32_t kTimeoutMs = 80;
  auto sessions = make_udp_sessions(
      std::unordered_map<std::uint64_t, std::string>{{0, endpoint}},
      kTimeoutMs);
  ASSERT_EQ(sessions.size(), 1u);
  auto &sess = sessions[0].second;

  auto req = make_reset_frame(0, /*request_id=*/1);
  frame f{req.data(), req.size()};
  std::vector<std::uint8_t> reply(64);
  std::size_t reply_len = 0;

  auto t0 = std::chrono::steady_clock::now();
  RpcStatus status = sess->await(sess->submit(f), reply, reply_len);
  auto elapsed = std::chrono::steady_clock::now() - t0;
  auto elapsed_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

  // session_udp.cpp: a socket that has stayed silent for longer than the
  // timeout is synthesized as RpcStatus::INTERNAL_ERROR -- there is no
  // dedicated "local timeout" status. Must NOT be OK, and must return in
  // bounded time, not hang.
  EXPECT_NE(status, RpcStatus::OK);
  EXPECT_EQ(status, RpcStatus::INTERNAL_ERROR);
  EXPECT_LT(elapsed_ms, 2 * static_cast<long long>(kTimeoutMs));
}

TEST(UdpBackendAdvanced, TheReceiverThreadSurvivesATransientErrorAndKeepsListening) {
  auto endpoint = closed_loopback_endpoint();
  constexpr std::uint32_t kTimeoutMs = 150;
  auto sessions = make_udp_sessions(
      std::unordered_map<std::uint64_t, std::string>{{0, endpoint}},
      kTimeoutMs);
  ASSERT_EQ(sessions.size(), 1u);
  auto &sess = sessions[0].second;

  auto first = make_reset_frame(0, 1);
  frame f1{first.data(), first.size()};
  std::vector<std::uint8_t> reply(64);
  std::size_t reply_len = 0;
  ASSERT_EQ(sess->await(sess->submit(f1), reply, reply_len),
            RpcStatus::INTERNAL_ERROR);

  const auto port = endpoint.substr(endpoint.rfind(':') + 1);
  int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = htons(static_cast<std::uint16_t>(std::stoi(port)));
  ASSERT_EQ(::bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)), 0);

  std::thread server([fd] {
    std::vector<std::uint8_t> buf(256);
    sockaddr_in from{};
    socklen_t from_len = sizeof(from);
    const ssize_t n = ::recvfrom(fd, buf.data(), buf.size(), 0,
                                 reinterpret_cast<sockaddr *>(&from), &from_len);
    if (n < static_cast<ssize_t>(sizeof(RPCHeader)))
      return;
    RPCHeader hdr{};
    std::memcpy(&hdr, buf.data(), sizeof(hdr));
    RPCResponse resp{};
    resp.magic = cudaq::realtime::RPC_MAGIC_RESPONSE;
    resp.status = 0;
    resp.result_len = 0;
    resp.request_id = hdr.request_id;
    ::sendto(fd, &resp, sizeof(resp), 0,
            reinterpret_cast<sockaddr *>(&from), from_len);
  });

  auto second = make_reset_frame(0, 2);
  frame f2{second.data(), second.size()};
  const auto request_id = sess->submit(f2);
  server.join();
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  ::close(fd);
  const auto status = sess->await(request_id, reply, reply_len);
  EXPECT_EQ(status, RpcStatus::OK);
}

TEST(UdpBackendAdvanced, MaxFrameBytesIsTheMaxUdpDatagramPayloadNotZeroOrUnbounded) {
  // No listener needed: connect() on a UDP socket just records the default
  // destination, it doesn't require a live peer.
  auto endpoint = closed_loopback_endpoint();
  auto sessions = make_udp_sessions(
      std::unordered_map<std::uint64_t, std::string>{{0, endpoint}}, 50);
  ASSERT_EQ(sessions.size(), 1u);

  // session_udp.cpp: kMaxDatagram = 65535 - 8 (UDP header) - 20 (IPv4
  // header) = 65507.
  EXPECT_EQ(sessions[0].second->max_frame_bytes, 65507u);
  EXPECT_NE(sessions[0].second->max_frame_bytes, 0u) << "0 would mean unbounded, which UDP "
                                        "is not";
}

TEST(UdpBackendAdvanced,
     PlanTimeCheckTrustsMaxFrameBytesAndRejectsAnOversizedEnqueue) {
  // Really about proving max_frame_bytes is trustworthy, since plan() trusts
  // it wholesale to keep oversized frames off the wire. Build an enqueue
  // frame (24B header + 32B payload + packed bits) exceeding kMaxDatagram
  // (65507) by construction: n > 523608 bits.
  auto endpoint = closed_loopback_endpoint();
  auto sessions = make_udp_sessions(
      std::unordered_map<std::uint64_t, std::string>{{0, endpoint}}, 50);
  ASSERT_EQ(sessions.size(), 1u);
  ASSERT_EQ(sessions[0].second->max_frame_bytes, 65507u);

  constexpr std::size_t kBits = 523'616; // 56 + 523616/8 = 65508 > 65507
  std::string sched_text = "0 enqueue source=0b" + std::string(kBits, '1') + "\n";
  auto sched = parse(sched_text, {0}, 1000);
  std::unordered_map<std::uint64_t, session *> router;
  router[0] = sessions[0].second.get();
  EXPECT_THROW(plan(sched, router, {}), std::invalid_argument);
}

TEST(UdpBackendAdvanced, SubmitRejectsAFrameShorterThanAnRPCHeader) {
  auto endpoint = closed_loopback_endpoint();
  auto sessions = make_udp_sessions(
      std::unordered_map<std::uint64_t, std::string>{{0, endpoint}}, 50);
  ASSERT_EQ(sessions.size(), 1u);
  auto &sess = sessions[0].second;

  std::vector<std::uint8_t> short_frame(sizeof(RPCHeader) - 1, 0);
  frame f{short_frame.data(), short_frame.size()};
  EXPECT_THROW(sess->submit(f), std::invalid_argument);
}
