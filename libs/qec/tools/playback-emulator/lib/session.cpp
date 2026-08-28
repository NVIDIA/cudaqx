/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file session.cpp
/// @brief All three session backends: `null` (discard every frame), inproc
/// (dispatch to a decoder's own DecodingSession), and UDP (a decoding
/// server). Concrete classes are anonymous-namespace private; the factories
/// in session.h are the only public surface.

#include "RpcSlot.h"
#include "SessionRegistry.h"
#include "session.h"

#include <algorithm>
#include <arpa/inet.h>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <netdb.h>
#include <stdexcept>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>

namespace cudaq::qec::playback {

// ─── null_session ──────────────────────────────────────────────────────────
//
// The `null` backend. It discards every frame, but touches every byte of it
// through an atomic checksum so the optimizer cannot prove the frame is dead
// and elide the serialization work the real backends would also have to pay
// for.

namespace {

class null_session : public session {
public:
  // Echoes the frame's own request_id rather than inventing one, so a
  // reader can key completions by id the same way every other backend does.
  std::uint32_t submit(const frame &f) override {
    checksum(f);
    const std::uint32_t rid =
        f.size >= sizeof(cudaq::realtime::RPCHeader)
            ? reinterpret_cast<const cudaq::realtime::RPCHeader *>(f.bytes)
                  ->request_id
            : 0;
    {
      std::lock_guard<std::mutex> lock(mu_);
      completed_.push_back(rid);
    }
    completed_cv_.notify_one();
    return rid;
  }

  bool wait_next_completion(std::uint32_t &request_id,
                            std::chrono::milliseconds timeout) override {
    std::unique_lock<std::mutex> lock(mu_);
    if (!completed_cv_.wait_for(lock, timeout,
                               [&] { return !completed_.empty(); }))
      return false;
    request_id = completed_.front();
    completed_.pop_front();
    return true;
  }

  RpcStatus await(std::uint32_t, std::span<std::uint8_t> reply,
                  std::size_t &reply_len) override {
    std::fill(reply.begin(), reply.end(), std::uint8_t{0});
    reply_len = reply.size();
    return RpcStatus::OK;
  }

private:
  // Folds every byte of the frame into an atomic accumulator so the compiler
  // cannot elide the build/serialize step even though the result is unused.
  void checksum(const frame &f) {
    std::uint64_t acc = 0;
    for (std::size_t i = 0; i < f.size; ++i)
      acc ^= (std::uint64_t(f.bytes[i]) << (8 * (i & 7)));
    checksum_.fetch_xor(acc, std::memory_order_relaxed);
  }

  std::atomic<std::uint64_t> checksum_{0};
  std::mutex mu_;
  std::condition_variable completed_cv_;
  std::deque<std::uint32_t> completed_;
};

} // namespace

std::unique_ptr<session> make_null_session() {
  return std::make_unique<null_session>();
}

std::vector<std::pair<std::uint64_t, std::unique_ptr<session>>>
make_null_sessions(const std::vector<std::uint64_t> &decoder_ids) {
  std::vector<std::pair<std::uint64_t, std::unique_ptr<session>>> out;
  out.reserve(decoder_ids.size());
  for (auto id : decoder_ids)
    out.emplace_back(id, make_null_session());
  return out;
}

void route_sessions(
    const std::vector<std::pair<std::uint64_t, std::unique_ptr<session>>> &sessions,
    std::unordered_map<std::uint64_t, session *> &router) {
  for (const auto &[id, s] : sessions)
    router[id] = s.get();
}

// ─── inproc_session ────────────────────────────────────────────────────────
//
// One session per decoder, dispatching straight to that decoder's own
// DecodingSession payload-level cores (skipping shared-memory rings and CUDA
// graph dispatch). A thin, synchronous dispatcher to one already-resolved
// DecodingSession -- any internal decode threading is that class's business.

namespace {

using cudaq::qec::decoding_server::DecodingSession;
using cudaq::qec::decoding_server::SessionRegistry;

/// Result of dispatching one raw request frame -- every RPC gets a reply.
struct DispatchResult {
  cudaq::qec::decoding::rpc::RpcStatus status =
      cudaq::qec::decoding::rpc::RpcStatus::BAD_REQUEST;
  std::size_t reply_len = 0;
};

/// Parses \p bytes as one of the three decoder RPCs and calls the matching
/// core on \p dec, copying (for get_corrections) the still-bit-packed
/// result into \p reply (capacity \p reply_capacity). 
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
    return {RpcStatus::BAD_REQUEST, 0};
  const auto *header = reinterpret_cast<const RPCHeader *>(bytes);

  if (header->function_id == kEnqueueSyndromesFunctionId) {
    // Mirrors DecodingSession::handle_enqueue: reject anything but the
    // identity syndrome mapping, and ack OK even on an internal decode
    // failure -- that surfaces at the next get_corrections instead.
    slot::EnqueueView view;
    if (!slot::parse_enqueue(bytes, size, view))
      return {RpcStatus::BAD_REQUEST, 0};
    if (view.syndrome_mapping_id != 0)
      return {RpcStatus::BAD_REQUEST, 0};
    dec.enqueue_core(view);
    return {RpcStatus::OK, 0};
  }

  if (header->function_id == kGetCorrectionsFunctionId) {
    slot::GetCorrectionsView view;
    if (!slot::parse_get_corrections(bytes, size, view))
      return {RpcStatus::BAD_REQUEST, 0};
    std::size_t reply_len = 0;
    auto status = dec.get_corrections_core(view.return_size, view.reset,
                                           reply, reply_capacity, reply_len);
    return {status, reply_len};
  }

  if (header->function_id == kResetDecoderFunctionId) {
    slot::ResetView view;
    if (!slot::parse_reset(bytes, size, view))
      return {RpcStatus::BAD_REQUEST, 0};
    return {dec.reset_core(), 0};
  }

  return {RpcStatus::BAD_REQUEST, 0};
}

/// How big a reply this frame can produce, so a submitted request can carry
/// its own result buffer.
std::size_t reply_capacity_for(const std::uint8_t *bytes, std::size_t size) {
  namespace slot = cudaq::qec::decoding_server::slot;
  using cudaq::realtime::RPCHeader;
  if (size < sizeof(RPCHeader) ||
      reinterpret_cast<const RPCHeader *>(bytes)->function_id !=
          cudaq::qec::decoding::rpc::kGetCorrectionsFunctionId)
    return 0;
  slot::GetCorrectionsView view;
  if (!slot::parse_get_corrections(bytes, size, view))
    return 0;
  return cudaq::qec::decoding::rpc::bit_packed_bytes(
      static_cast<std::size_t>(std::max<std::int64_t>(view.return_size, 0)));
}

/// One decoder's input queue and the single thread that drains it, in
/// submission order. 
class inproc_session : public session {
public:
  inproc_session(std::shared_ptr<SessionRegistry> registry,
                 std::uint64_t decoder_id)
      : registry_(std::move(registry)) {
    dec_ = registry_->find(decoder_id);
    assert(dec_ && "make_inproc_sessions() must only construct a session "
                  "for a decoder_id actually present in the config");
    dispatcher_ = std::thread([this] { drain(); });
  }

  ~inproc_session() override {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_ = true;
    }
    work_.notify_all();
    if (dispatcher_.joinable())
      dispatcher_.join();
  }

  std::uint32_t submit(const frame &f) override { return push(f); }

  bool wait_next_completion(std::uint32_t &request_id,
                            std::chrono::milliseconds timeout) override {
    std::unique_lock<std::mutex> lock(mu_);
    if (!completed_cv_.wait_for(lock, timeout,
                               [&] { return !completed_.empty(); }))
      return false;
    request_id = completed_.front();
    completed_.pop_front();
    return true;
  }

  RpcStatus await(std::uint32_t request_id, std::span<std::uint8_t> reply,
                  std::size_t &reply_len) override {
    reply_len = 0;
    std::shared_ptr<job> j;
    {
      std::lock_guard<std::mutex> lock(mu_);
      auto it = pending_.find(request_id);
      if (it == pending_.end())
        return RpcStatus::INTERNAL_ERROR; // no such submission
      j = it->second;
      pending_.erase(it);
    }
    std::unique_lock<std::mutex> lock(j->mu);
    j->done_cv.wait(lock, [&] { return j->done; });
    reply_len = std::min(j->reply_len, reply.size());
    std::copy_n(j->reply.begin(), reply_len, reply.begin());
    return j->status;
  }

private:
  /// A submitted request, from the caller's thread to the dispatcher's and
  /// (for a reply) back. The frame is copied because the caller's buffer may
  /// well be a local that dies the moment submit() returns.
  struct job {
    std::uint32_t rid = 0;
    std::vector<std::uint8_t> bytes;
    std::vector<std::uint8_t> reply;
    std::size_t reply_len = 0;
    RpcStatus status = RpcStatus::OK;
    bool done = false;
    std::mutex mu;
    std::condition_variable done_cv;
  };

  /// Publish one frame.
  std::uint32_t push(const frame &f) {
    // Too short to carry a request_id -- see udp_session::submit for why
    // this is rejected outright rather than keyed under a placeholder id.
    if (f.size < sizeof(cudaq::realtime::RPCHeader))
      throw std::invalid_argument(
          "inproc_session: frame is smaller than RPCHeader");
    auto j = std::make_shared<job>();
    j->rid = reinterpret_cast<const cudaq::realtime::RPCHeader *>(f.bytes)
                 ->request_id;
    j->bytes.assign(f.bytes, f.bytes + f.size);
    j->reply.resize(reply_capacity_for(f.bytes, f.size));
    {
      std::lock_guard<std::mutex> lock(mu_);
      queue_.push_back(j);
      pending_.emplace(j->rid, j);
    }
    work_.notify_one();
    return j->rid;
  }

  void drain() {
    for (;;) {
      std::shared_ptr<job> j;
      {
        std::unique_lock<std::mutex> lock(mu_);
        work_.wait(lock, [&] { return stop_ || !queue_.empty(); });
        if (queue_.empty())
          return; 
        j = std::move(queue_.front());
        queue_.pop_front();
      }
      const auto result = dispatch_rpc(*dec_, j->bytes.data(), j->bytes.size(),
                                       j->reply.data(), j->reply.size());
      {
        std::lock_guard<std::mutex> lock(j->mu);
        j->status = result.status;
        j->reply_len = result.reply_len;
        j->done = true;
      }
      j->done_cv.notify_all();
      {
        std::lock_guard<std::mutex> lock(mu_);
        completed_.push_back(j->rid);
      }
      completed_cv_.notify_one();
    }
  }

  std::shared_ptr<SessionRegistry> registry_; // keeps every decoder alive
  DecodingSession *dec_ = nullptr;

  std::mutex mu_;
  std::condition_variable work_;
  std::deque<std::shared_ptr<job>> queue_;
  std::unordered_map<std::uint32_t, std::shared_ptr<job>> pending_;
  std::condition_variable completed_cv_;
  std::deque<std::uint32_t> completed_;
  bool stop_ = false;
  std::thread dispatcher_; // last, so drain() only sees initialized members
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
    out.emplace_back(id, std::make_unique<inproc_session>(registry, id));
  }
  return out;
}

// ─── udp_session ───────────────────────────────────────────────────────────
//
// Connected UDP socket(s) to a decoding server speaking decoder_rpc_wire_
// format.h; replies are matched by request_id. Pure transport: this file
// looks no further than the generic RPCHeader/RPCResponse framing and has
// no notion of which RPC a frame holds.

namespace {

using cudaq::realtime::RPC_MAGIC_RESPONSE;
using cudaq::realtime::RPCHeader;
using cudaq::realtime::RPCResponse;

// A UDP datagram is at most 65507 bytes of payload (65535 - 8-byte UDP
// header - 20-byte IPv4 header); size scratch buffers to that.
constexpr std::size_t kMaxDatagram = 65507;

/// Splits "host:port" on the LAST ':' 
void split_endpoint(const std::string &endpoint, std::string &host,
                    std::string &port) {
  auto pos = endpoint.rfind(':');
  if (pos == std::string::npos)
    throw std::runtime_error("playback UDP endpoint missing ':port': " +
                             endpoint);
  host = endpoint.substr(0, pos);
  port = endpoint.substr(pos + 1);
}

int make_connected_udp_socket(const std::string &endpoint,
                              std::uint32_t timeout_ms) {
  std::string host, port;
  split_endpoint(endpoint, host, port);

  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_DGRAM;
  addrinfo *res = nullptr;
  int rc = getaddrinfo(host.c_str(), port.c_str(), &hints, &res);
  if (rc != 0 || !res)
    throw std::runtime_error("playback UDP: failed to resolve endpoint '" +
                             endpoint + "': " + gai_strerror(rc));

  int fd = -1;
  for (addrinfo *p = res; p; p = p->ai_next) {
    fd = ::socket(p->ai_family, p->ai_socktype, p->ai_protocol);
    if (fd < 0)
      continue;
    if (::connect(fd, p->ai_addr, p->ai_addrlen) == 0)
      break;
    ::close(fd);
    fd = -1;
  }
  freeaddrinfo(res);
  if (fd < 0)
    throw std::runtime_error("playback UDP: failed to connect to '" +
                             endpoint + "'");

  int rcvbuf = 1 << 20; // generous SO_RCVBUF
  ::setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));

  timeval tv{};
  tv.tv_sec = timeout_ms / 1000;
  tv.tv_usec = (timeout_ms % 1000) * 1000;
  ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

  return fd;
}

/// One connected UDP socket talks to one decoder on the decoding server.
///
/// Exactly one thread publishes on a session (see session.h).
class udp_session : public session {
public:
  udp_session(int fd, std::uint32_t timeout_ms)
      : fd_(fd), timeout_(std::chrono::milliseconds(timeout_ms)) {
    max_frame_bytes = static_cast<std::uint32_t>(kMaxDatagram);
    receiver_ = std::thread([this] { receive_loop(); });
  }

  ~udp_session() override {
    stop_.store(true, std::memory_order_release);
    if (fd_ >= 0)
      ::shutdown(fd_, SHUT_RDWR); // unblock a receiver parked in recv()
    if (receiver_.joinable())
      receiver_.join();
    if (fd_ >= 0)
      ::close(fd_);
  }

  std::uint32_t submit(const frame &f) override {
    if (f.size < sizeof(RPCHeader))
      throw std::invalid_argument(
          "udp_session::submit: frame is smaller than RPCHeader");
    const std::uint32_t request_id =
        reinterpret_cast<const RPCHeader *>(f.bytes)->request_id;
    auto w = std::make_shared<waiter>();
    w->deadline = std::chrono::steady_clock::now() + timeout_;
    {
      std::lock_guard<std::mutex> lock(mu_);
      pending_[request_id] = w;
    }
    if (::send(fd_, f.bytes, f.size, 0) < 0) {
      {
        std::lock_guard<std::mutex> lock(mu_);
        pending_.erase(request_id);
      }
      finish(request_id, *w, RpcStatus::INTERNAL_ERROR, nullptr, 0);
    }
    return request_id;
  }

  bool wait_next_completion(std::uint32_t &request_id,
                            std::chrono::milliseconds timeout) override {
    std::unique_lock<std::mutex> lock(mu_);
    if (!completed_cv_.wait_for(lock, timeout,
                               [&] { return !completed_.empty(); }))
      return false;
    request_id = completed_.front();
    completed_.pop_front();
    return true;
  }

  RpcStatus await(std::uint32_t request_id, std::span<std::uint8_t> reply,
                  std::size_t &reply_len) override {
    reply_len = 0;
    std::shared_ptr<waiter> w;
    {
      std::lock_guard<std::mutex> lock(mu_);
      auto it = pending_.find(request_id);
      if (it == pending_.end())
        return RpcStatus::INTERNAL_ERROR; // never submitted, or already taken
      w = it->second;
    }
    // No self-timeout here: receive_loop's own periodic wake (bounded by
    // SO_RCVTIMEO) sweeps a silent socket's stale requests on its own, so
    // this always completes even if nobody else is watching this id.
    std::unique_lock<std::mutex> lock(w->mu);
    w->done_cv.wait(lock, [&] { return w->done; });
    reply_len = std::min(w->reply.size(), reply.size());
    std::copy_n(w->reply.begin(), reply_len, reply.begin());
    const auto status = w->status;
    lock.unlock();
    // Only await() removes a completed entry -- a reply/sweep marks it done
    // in place, so wait_next_completion() can report it and this call can
    // still find and read it afterward.
    std::lock_guard<std::mutex> lock2(mu_);
    pending_.erase(request_id);
    return status;
  }

private:
  /// One outstanding request: filled in by the receiver, collected by await.
  /// `deadline` is fixed at submission and never written again, so sweep_stale
  /// can read it without locking.
  struct waiter {
    std::chrono::steady_clock::time_point deadline;
    std::mutex mu;
    std::condition_variable done_cv;
    std::vector<std::uint8_t> reply;
    RpcStatus status = RpcStatus::OK;
    bool done = false;
  };

  /// Finishes a request and makes it visible to wait_next_completion. A
  /// genuine reply leaves `request_id` in `pending_` for await() to erase
  /// once it has consumed the reply; a send failure or a sweep -- which
  /// give up on a request rather than reporting a real reply -- erase it
  /// first instead, since nothing else will.
  void finish(std::uint32_t request_id, waiter &w, RpcStatus status,
             const std::uint8_t *body, std::size_t len) {
    {
      std::lock_guard<std::mutex> lock(w.mu);
      w.status = status;
      w.reply.assign(body, body + len);
      w.done = true;
    }
    w.done_cv.notify_all();
    {
      std::lock_guard<std::mutex> lock(mu_);
      completed_.push_back(request_id);
    }
    completed_cv_.notify_one();
  }

  /// No dedicated "local timeout" status exists in RpcStatus; a client-side
  /// synthesized timeout/hard-error is reported as INTERNAL_ERROR. Each
  /// request's own deadline (submission + timeout_) governs it, not overall
  /// socket silence -- unrelated traffic (another request's reply, or ICMP
  /// errors for a dead peer) must never mask one dead request forever.
  void sweep_stale() {
    std::vector<std::pair<std::uint32_t, std::shared_ptr<waiter>>> snapshot;
    {
      std::lock_guard<std::mutex> lock(mu_);
      snapshot.assign(pending_.begin(), pending_.end());
    }
    const auto now = std::chrono::steady_clock::now();
    for (auto &[id, w] : snapshot) {
      if (now < w->deadline)
        continue; // still within its own grace period
      bool already_done = false;
      {
        std::lock_guard<std::mutex> lock(w->mu);
        already_done = w->done;
      }
      // A reply that landed but is not yet await()'d stays in `pending_` for
      // that eventual await() to find; sweeping it here would clobber its
      // real status with INTERNAL_ERROR.
      if (already_done)
        continue;
      {
        std::lock_guard<std::mutex> lock(mu_);
        pending_.erase(id);
      }
      finish(id, *w, RpcStatus::INTERNAL_ERROR, nullptr, 0);
    }
  }

  void receive_loop() {
    std::vector<std::uint8_t> scratch(kMaxDatagram);
    while (!stop_.load(std::memory_order_acquire)) {
      // Runs every iteration -- including a tight run of async ICMP errors
      // for a dead peer (e.g. one per rejected send), not just a genuine
      // SO_RCVTIMEO silence -- since each request's own deadline is what
      // actually gates whether sweep_stale() does anything.
      sweep_stale();
      const ssize_t n = ::recv(fd_, scratch.data(), scratch.size(), 0);
      if (n < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)
          continue; // SO_RCVTIMEO tick: just re-check stop_
        if (stop_.load(std::memory_order_acquire))
          return; // intentional shutdown
        // A connected UDP socket also surfaces async ICMP errors here (e.g.
        // ECONNREFUSED while the server is briefly down); that is not a
        // reason to stop listening.
        continue;
      }
      if (static_cast<std::size_t>(n) < sizeof(RPCResponse))
        continue; // short/garbage datagram
      RPCResponse resp;
      std::memcpy(&resp, scratch.data(), sizeof(RPCResponse));
      if (resp.magic != RPC_MAGIC_RESPONSE)
        continue; // garbage/truncated datagram

      std::shared_ptr<waiter> w;
      {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = pending_.find(resp.request_id);
        if (it == pending_.end())
          continue; // stale reply: nobody is waiting on this id any more
        w = it->second;
      }
      const std::size_t avail = static_cast<std::size_t>(n) - sizeof(RPCResponse);
      finish(resp.request_id, *w, static_cast<RpcStatus>(resp.status),
             scratch.data() + sizeof(RPCResponse),
             std::min<std::size_t>(resp.result_len, avail));
    }
  }

  int fd_ = -1;
  std::chrono::milliseconds timeout_;
  std::condition_variable completed_cv_;
  std::deque<std::uint32_t> completed_;
  // Guards `pending_` and `completed_`
  std::mutex mu_;
  std::unordered_map<std::uint32_t, std::shared_ptr<waiter>> pending_;
  std::atomic<bool> stop_{false};
  std::thread receiver_; // last, so receive_loop() sees initialized members
};

} // namespace

std::vector<std::pair<std::uint64_t, std::unique_ptr<session>>>
make_udp_sessions(const std::unordered_map<std::uint64_t, std::string> &endpoints,
                  std::uint32_t timeout_ms) {
  std::vector<std::pair<std::uint64_t, std::unique_ptr<session>>> out;
  out.reserve(endpoints.size());
  for (const auto &[id, endpoint] : endpoints) {
    int fd = make_connected_udp_socket(endpoint, timeout_ms);
    out.emplace_back(id, std::make_unique<udp_session>(fd, timeout_ms));
  }
  return out;
}

} // namespace cudaq::qec::playback
