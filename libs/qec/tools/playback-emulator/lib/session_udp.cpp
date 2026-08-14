/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// The UDP backend: connected datagram socket(s) to a
/// decoding server speaking the wire format in decoder_rpc_wire_format.h.
/// Replies are matched by request_id
///
/// Pure transport: this file never looks at a frame's contents beyond the
/// generic RPCHeader/RPCResponse framing needed to match a reply to its
/// request. It has no notion of which RPC a frame holds.

#include "cudaq/qec/playback/backends.h"

#include <algorithm>
#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <mutex>
#include <netdb.h>
#include <stdexcept>
#include <sys/socket.h>
#include <unistd.h>

namespace cudaq::qec::playback {

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

/// One connected UDP socket talks to one decoder on the decoding server
class udp_session : public session {
public:
  explicit udp_session(int fd) : fd_(fd) {}

  ~udp_session() override {
    if (fd_ >= 0)
      ::close(fd_);
  }

  void send_async(const frame &f) override {
    // Fire-and-forget: enqueue has no reply on the wire (decoder_server_
    // runtime.md), so nothing is read back here.
    ::send(fd_, f.bytes, f.size, 0);
  }

  RpcStatus send_sync(const frame &f, std::span<std::uint8_t> reply,
                      std::size_t &reply_len) override {
    reply_len = 0;
    if (f.size < sizeof(RPCHeader))
      return RpcStatus::BAD_REQUEST;
    const auto request_id =
        reinterpret_cast<const RPCHeader *>(f.bytes)->request_id;

    std::lock_guard<std::mutex> lock(mu_); // serialize this socket's use
    if (::send(fd_, f.bytes, f.size, 0) < 0)
      return RpcStatus::INTERNAL_ERROR;

    if (recv_scratch_.empty())
      recv_scratch_.resize(kMaxDatagram);
    for (;;) {
      ssize_t n = ::recv(fd_, recv_scratch_.data(), recv_scratch_.size(), 0);
      if (n < 0) {
        // No dedicated "local timeout" status exists in RpcStatus; a
        // client-side synthesized timeout/hard-error is reported as
        // INTERNAL_ERROR rather than hanging 
        return RpcStatus::INTERNAL_ERROR;
      }
      if (static_cast<std::size_t>(n) < sizeof(RPCResponse))
        continue; // short/garbage datagram; keep waiting for the real reply
      RPCResponse resp;
      std::memcpy(&resp, recv_scratch_.data(), sizeof(RPCResponse));
      if (resp.magic != RPC_MAGIC_RESPONSE)
        continue; // garbage/truncated datagram; keep waiting for the real reply
      if (resp.request_id != request_id)
        continue; // stale or reordered reply to a different request; discard
      const std::size_t avail =
          static_cast<std::size_t>(n) - sizeof(RPCResponse);
      const std::size_t copy_len = std::min<std::size_t>(
          {resp.result_len, avail, reply.size()});
      if (copy_len > 0)
        std::memcpy(reply.data(), recv_scratch_.data() + sizeof(RPCResponse),
                    copy_len);
      reply_len = copy_len;
      return static_cast<RpcStatus>(resp.status);
    }
  }

  capabilities caps() const override {
    capabilities c;
    c.reports_not_ready = true; // the remote server's get_corrections_core
                                // returns NOT_READY the same as inproc.
    c.max_frame_bytes = static_cast<std::uint32_t>(kMaxDatagram);
    return c;
  }

private:
  int fd_ = -1;
  std::mutex mu_;
  std::vector<std::uint8_t> recv_scratch_;
};

} // namespace

std::vector<std::pair<std::uint64_t, std::unique_ptr<session>>>
make_udp_sessions(const std::unordered_map<std::uint64_t, std::string> &endpoints,
                  std::uint32_t timeout_ms) {
  std::vector<std::pair<std::uint64_t, std::unique_ptr<session>>> out;
  out.reserve(endpoints.size());
  for (const auto &[id, endpoint] : endpoints) {
    int fd = make_connected_udp_socket(endpoint, timeout_ms);
    out.emplace_back(id, std::make_unique<udp_session>(fd));
  }
  return out;
}

} // namespace cudaq::qec::playback
