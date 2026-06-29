/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/logger.h"
#include "logger_forwarder.h"
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <mutex>
#include <optional>
#include <sstream>

/// @file
/// @brief Core QEC logger front-end.
/// @details
/// Owns level parsing, timestamp/source formatting, and synchronous sink
/// output. When forwarding is enabled, this file packs records into bounded
/// buffers and delegates asynchronous delivery to `logger_forwarder.cpp`.

namespace cudaq::qec::detail {
namespace {

using Clock = std::chrono::system_clock;

std::atomic<LogLevel> gLogLevel{LogLevel::warn};
std::once_flag gLogLevelInitFlag;

// Convert a log level token to lowercase for robust parsing.
std::string toLower(std::string value) {
  for (auto &ch : value)
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  return value;
}

// Parse CUDAQ_LOG_LEVEL text into internal enum.
std::optional<LogLevel> parseLogLevel(const std::string_view level) {
  const std::string lower = toLower(std::string(level));
  if (lower == "trace")
    return LogLevel::trace;
  if (lower == "debug")
    return LogLevel::debug;
  if (lower == "info")
    return LogLevel::info;
  if (lower == "warn" || lower == "warning")
    return LogLevel::warn;
  if (lower == "error")
    return LogLevel::error;
  return std::nullopt;
}

// Convert internal level enum to stable output label.
const char *logLevelName(const LogLevel level) {
  switch (level) {
  case LogLevel::trace:
    return "trace";
  case LogLevel::debug:
    return "debug";
  case LogLevel::info:
    return "info";
  case LogLevel::warn:
    return "warn";
  case LogLevel::error:
    return "error";
  }
  return "info";
}

// Return current wall-clock timestamp in nanoseconds.
std::uint64_t nowNs() {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          Clock::now().time_since_epoch())
          .count());
}

// Render local-time timestamp with microsecond precision.
std::string formatTimestamp(const Clock::time_point tp) {
  const auto seconds = std::chrono::time_point_cast<std::chrono::seconds>(tp);
  const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(
                          tp.time_since_epoch() - seconds.time_since_epoch())
                          .count();

  const std::time_t t = Clock::to_time_t(tp);
  std::tm tm{};
#if defined(_WIN32)
  localtime_s(&tm, &t);
#else
  localtime_r(&t, &tm);
#endif
  std::ostringstream out;
  out << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << '.' << std::setw(6)
      << std::setfill('0') << micros;
  return out.str();
}

// Build one final text log line with timestamp/source metadata. The
// `[file:line]` segment is omitted when no filename is supplied (empty or
// null), allowing source-free log lines via the `log()` helper.
std::string composeLine(const LogLevel level, const std::string_view message,
                        const char *fileName, const int lineNo,
                        const Clock::time_point ts) {
  std::ostringstream out;
  out << '[' << formatTimestamp(ts) << "] [" << logLevelName(level) << "] ";
  if (fileName != nullptr && fileName[0] != '\0')
    out << '[' << pathToFileName(fileName) << ':' << lineNo << "] ";
  out << message;
  return out.str();
}

// Lightweight filename extraction view to avoid allocations.
std::string_view pathToFileNameView(const std::string_view fullFilePath) {
  const auto pos = fullFilePath.find_last_of("/\\");
  if (pos == std::string_view::npos)
    return fullFilePath;
  return fullFilePath.substr(pos + 1);
}

template <std::size_t N>
std::size_t copyToFixed(std::array<char, N> &dest, const std::string_view src,
                        std::size_t cap, bool *truncated = nullptr) {
  static_assert(N > 0);
  cap = std::min(cap, N - 1);
  const std::size_t copied = std::min(src.size(), cap);
  if (copied > 0)
    std::memcpy(dest.data(), src.data(), copied);
  dest[copied] = '\0';
  if (truncated)
    *truncated = src.size() > copied;
  return copied;
}

// Lazily initialize runtime log level from CUDAQ_LOG_LEVEL once.
void initializeLogLevelFromEnv() {
  std::call_once(gLogLevelInitFlag, [] {
    if (const char *env = std::getenv("CUDAQ_LOG_LEVEL")) {
      if (const auto parsed = parseLogLevel(env))
        gLogLevel.store(*parsed, std::memory_order_relaxed);
    }
  });
}

void emit(const LogLevel level, const std::string_view rawMessage,
          const char *fileName, const int lineNo) {
  // Forwarder mode uses fixed-size packing to keep producer-side allocation
  // predictable and bounded.
  if (forwarder_internal::isEnabled()) {
    forwarder_internal::QueuedLogRecord record;
    const std::size_t messageCap = forwarder_internal::messageCapacity();
    record.level = level;
    record.timestampNs = nowNs();
    record.fileNameLen =
        copyToFixed(record.fileName, pathToFileNameView(fileName),
                    record.fileName.size() - 1);
    record.lineNo = lineNo;
    bool truncated = false;
    record.messageLen =
        copyToFixed(record.message, rawMessage, messageCap, &truncated);
    if (truncated) {
      forwarder_internal::noteTruncation();
      if (record.messageLen >= kRealtimeTruncationSuffix.size()) {
        const std::size_t start =
            record.messageLen - kRealtimeTruncationSuffix.size();
        std::copy(kRealtimeTruncationSuffix.begin(),
                  kRealtimeTruncationSuffix.end(),
                  record.message.begin() + start);
      }
    }
    record.threadId = std::this_thread::get_id();
    forwarder_internal::enqueue(std::move(record));
    return;
  }

  const auto ts = Clock::now();
  const std::string fullLine =
      composeLine(level, rawMessage, fileName, lineNo, ts);
  FILE *stream =
      (level == LogLevel::warn || level == LogLevel::error) ? stderr : stdout;
  std::fputs(fullLine.c_str(), stream);
  std::fputc('\n', stream);
}

} // namespace

// Compare candidate level against current runtime threshold.
bool should_log(const LogLevel logLevel) {
  initializeLogLevelFromEnv();
  return static_cast<int>(logLevel) >=
         static_cast<int>(gLogLevel.load(std::memory_order_relaxed));
}

// Public API: install/replace asynchronous forwarding callback.
void setForwarder(ForwarderConfig config) {
  forwarder_internal::set(std::move(config));
  forwarder_internal::resetStats();
}

// Public API: enable forwarding with default stdout/stderr callback.
void setForwarder() {
  setForwarder(ForwarderConfig{
      .callback =
          [](const ForwardedLogRecord &record) {
            const auto tp = Clock::time_point(std::chrono::nanoseconds(
                static_cast<std::int64_t>(record.timestampNs)));
            const std::string fullLine =
                composeLine(record.level, record.message,
                            record.fileName.c_str(), record.lineNo, tp);
            FILE *stream = (record.level == LogLevel::warn ||
                            record.level == LogLevel::error)
                               ? stderr
                               : stdout;
            std::fputs(fullLine.c_str(), stream);
            std::fputc('\n', stream);
          },
      .queueCapacity = ForwarderConfig{}.queueCapacity,
      .dropPolicy = ForwarderConfig{}.dropPolicy});
}

// Public API: disable asynchronous forwarding.
void clearForwarder() { forwarder_internal::clear(); }
// Public API: report whether forwarding is active.
bool isForwarderEnabled() { return forwarder_internal::isEnabled(); }
std::size_t getForwarderMessageCapacity() {
  return forwarder_internal::messageCapacity();
}
void recordForwarderMessageTruncation() {
  forwarder_internal::noteTruncation();
}
// Public API: return forwarding counters snapshot.
ForwarderStats getForwarderStats() { return forwarder_internal::stats(); }

// Public API: direct sinks for already-formatted messages.
void trace(const std::string_view msg) {
  emit(LogLevel::trace, msg, "<unknown>", 0);
}
void info(const std::string_view msg) {
  emit(LogLevel::info, msg, "<unknown>", 0);
}
void debug(const std::string_view msg) {
  emit(LogLevel::debug, msg, "<unknown>", 0);
}
void warn(const std::string_view msg) {
  emit(LogLevel::warn, msg, "<unknown>", 0);
}
void error(const std::string_view msg) {
  emit(LogLevel::error, msg, "<unknown>", 0);
}

// Strip directory prefix to keep compact source metadata output.
std::string pathToFileName(const std::string_view fullFilePath) {
  const auto pos = fullFilePath.find_last_of("/\\");
  if (pos == std::string_view::npos)
    return std::string(fullFilePath);
  return std::string(fullFilePath.substr(pos + 1));
}

// Override runtime logging threshold (primarily used in tests).
void setLogLevel(const LogLevel level) {
  gLogLevel.store(level, std::memory_order_relaxed);
}

// Return current runtime logging threshold.
LogLevel getLogLevel() { return gLogLevel.load(std::memory_order_relaxed); }

// Flush primary sinks and wait for forwarded queue drain.
void flushLogs() {
  std::fflush(stdout);
  std::fflush(stderr);
  forwarder_internal::flush();
}

// Entry points used by templated header helpers after formatting.
void logMessageFormatted(LogLevel logLevel, std::string formattedMessage,
                         const char *fileName, int lineNo) {
  emit(logLevel, formattedMessage, fileName, lineNo);
}

void logMessageBuffer(LogLevel logLevel, const char *formattedMessage,
                      std::size_t messageLen, const char *fileName,
                      int lineNo) {
  emit(logLevel, std::string_view(formattedMessage, messageLen), fileName,
       lineNo);
}

void logWithTimestampFormatted(std::string formattedMessage,
                               const char *fileName, int lineNo) {
  emit(LogLevel::info, formattedMessage, fileName, lineNo);
}

void logWithTimestampBuffer(const char *formattedMessage,
                            std::size_t messageLen, const char *fileName,
                            int lineNo) {
  emit(LogLevel::info, std::string_view(formattedMessage, messageLen), fileName,
       lineNo);
}

} // namespace cudaq::qec::detail
