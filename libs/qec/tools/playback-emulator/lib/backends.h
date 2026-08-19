/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file backends.h
/// @brief Factories for the three session backends 
/// `null` (testing, discard every frame), in-process (shared-memory-ring-free: dispatches
/// straight to DecodingSession cores), and UDP (connected datagram sockets to
/// a decoding server). Only factory functions are public

#include "session.h"
#include "cudaq/qec/realtime/decoding_config.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace cudaq::qec::playback {

/// Discards everything, but still builds and serializes a full frame per
/// event and checksums it so the compiler cannot elide the work. 
std::unique_ptr<session> make_null_session();

/// Realizes the decoders named in `config` in this process, one session per
/// decoder, each dispatching directly to that decoder's own DecodingSession
std::vector<std::pair<std::uint64_t, std::unique_ptr<session>>>
make_inproc_sessions(
    const cudaq::qec::decoding::config::multi_decoder_config &config);

/// Connected UDP client session(s) to a decoding server speaking the wire
/// format in decoder_rpc_wire_format.h. `endpoints` maps decoder_id ->
/// "host:port"; one session per decoder_id
/// `timeout_ms` bounds one send_sync() round trip.
std::vector<std::pair<std::uint64_t, std::unique_ptr<session>>>
make_udp_sessions(const std::unordered_map<std::uint64_t, std::string> &endpoints,
                   std::uint32_t timeout_ms = 200);

} // namespace cudaq::qec::playback
