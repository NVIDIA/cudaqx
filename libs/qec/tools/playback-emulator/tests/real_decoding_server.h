/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// Test-only infrastructure: spawns the REAL `decoding_server` binary
/// (libs/qec/tools/decoding-server/) as a subprocess over the `udp`
/// transport, so the UDP backend's client code (session_udp.cpp's
/// udp_session) is tested in lockstep against the actual production server
/// rather than a hand-rolled stand-in. No frame parsing or RPC-routing logic
/// lives here at all -- this is a process launcher and a `QEC_DECODING_
/// SERVER_READY` line parser, nothing else.
///
/// These tests do not run automatically: set QEC_PLAYBACK_ENABLE_REAL_
/// SERVER_TESTS=1 to opt in (see real_server_tests_enabled() below). They
/// spawn a genuine subprocess per test, which is much slower than the
/// in-process backends and requires the decoding_server binary to have been
/// built.

#include <cstdint>
#include <string>
#include <unordered_map>

namespace cudaq::qec::playback::testing {

/// True iff QEC_PLAYBACK_ENABLE_REAL_SERVER_TESTS is set to a non-empty,
/// non-"0" value. Tests that spawn a real_decoding_server should
/// GTEST_SKIP() when this is false (see QEC_SKIP_UNLESS_REAL_SERVER below).
bool real_server_tests_enabled();

/// Owns the decoding_server subprocess. Move-only; the destructor sends
/// SIGTERM (then SIGKILL if it doesn't exit promptly) and reaps the child.
struct real_decoding_server {
  real_decoding_server() = default;
  real_decoding_server(const real_decoding_server &) = delete;
  real_decoding_server &operator=(const real_decoding_server &) = delete;
  real_decoding_server(real_decoding_server &&other) noexcept;
  real_decoding_server &operator=(real_decoding_server &&other) noexcept;
  ~real_decoding_server();

  /// "127.0.0.1:<port>" of decoder 0's ring (the first "port=" the server
  /// prints). For a multi-decoder config, use endpoint_for(decoder_id)
  /// instead -- decoding_server opens one socket per decoder, never a
  /// shared one.
  std::string endpoint;

  /// "127.0.0.1:<port>" of `decoder_id`'s own ring. Throws std::out_of_range
  /// if `decoder_id` wasn't in the config this server was started with.
  std::string endpoint_for(std::uint64_t decoder_id) const;

  struct impl;
  impl *impl_ = nullptr;
};

/// Writes `yaml_config` to a temp file and spawns decoding_server
/// (--config=<temp file> --transport=udp --port=0) with it, waiting up to
/// `ready_timeout_ms` for the QEC_DECODING_SERVER_READY line. Throws
/// std::runtime_error (naming the QEC_DECODING_SERVER_PATH binary and any
/// captured stderr) if the binary is missing or never becomes ready.
real_decoding_server
start_real_decoding_server(const std::string &yaml_config,
                           std::uint32_t ready_timeout_ms = 15000);

} // namespace cudaq::qec::playback::testing

/// Place at the top of a TEST()/TEST_F() body that spawns a
/// real_decoding_server: skips (rather than running, and rather than
/// failing on a missing binary) unless the test has been explicitly opted
/// in via QEC_PLAYBACK_ENABLE_REAL_SERVER_TESTS=1.
#define QEC_SKIP_UNLESS_REAL_SERVER()                                        \
  do {                                                                       \
    if (!cudaq::qec::playback::testing::real_server_tests_enabled())         \
      GTEST_SKIP() << "set QEC_PLAYBACK_ENABLE_REAL_SERVER_TESTS=1 to run "  \
                      "this test against the real decoding_server binary";   \
  } while (0)
