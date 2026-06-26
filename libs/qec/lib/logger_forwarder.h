/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq/qec/logger.h"
#include <array>
#include <cstdint>
#include <thread>

/// @file
/// @brief Internal asynchronous forwarder API for the QEC logger.
/// @details
/// This header is intentionally private to the logger implementation. It
/// exposes the lock-free queue payload type and a minimal control surface used
/// by `logger.cpp` to enqueue preformatted records and query forwarding stats.

namespace cudaq::qec::detail::forwarder_internal {

/// @brief Fixed-size queue record used by the forwarder ring buffer.
/// @details Uses bounded arrays so producer-side enqueue can avoid heap
/// allocation when forwarding is enabled.
struct QueuedLogRecord {
  LogLevel level = LogLevel::info;
  std::uint64_t timestampNs = 0;
  int lineNo = 0;
  std::thread::id threadId;
  std::array<char, 128> fileName{};
  std::size_t fileNameLen = 0;
  std::array<char, kRealtimeForwarderMaxMessageCapacity> message{};
  std::size_t messageLen = 0;
};

void set(ForwarderConfig config);
void clear();
bool isEnabled();
void resetStats();
ForwarderStats stats();
void flush();
void enqueue(QueuedLogRecord record);
std::size_t messageCapacity();
void noteTruncation();

} // namespace cudaq::qec::detail::forwarder_internal
