/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

/// @file
/// @brief QEC logging interface used by runtime and plugin code.
/// @details
/// Declares log levels, forwarding configuration, and lightweight templated
/// formatting helpers used by logging macros (`CUDA_QEC_INFO`, `CUDA_QEC_WARN`,
/// `CUDA_QEC_ERROR`, `CUDA_QEC_DBG`) and direct
/// `cudaq::qec::info/warn/error/debug` calls.

namespace cudaq::qec {

// Keep all spdlog headers hidden in the implementation file.
namespace detail {
// This enum must match spdlog::level enums. This is checked via static_assert
// in logger.cpp.
/// @brief Severity levels supported by the QEC logger.
enum class LogLevel { trace, debug, info, warn, error };
inline constexpr std::size_t kRealtimeForwarderDefaultMessageCapacity = 512;
inline constexpr std::size_t kRealtimeForwarderMaxMessageCapacity = 4096;
inline constexpr std::string_view kRealtimeTruncationSuffix = " ...[truncated]";

/// @brief Log payload forwarded to an optional background callback.
struct ForwardedLogRecord {
  LogLevel level;
  std::uint64_t timestampNs;
  std::string fileName;
  int lineNo;
  std::string message;
  std::thread::id threadId;
};

/// @brief Queue backpressure policy for forwarded records.
enum class ForwardDropPolicy { dropNewest, dropOldest };

/// @brief Runtime configuration for asynchronous forwarding.
struct ForwarderConfig {
  std::function<void(const ForwardedLogRecord &)> callback;
  std::size_t queueCapacity = 1024;
  std::size_t messageCapacity = kRealtimeForwarderDefaultMessageCapacity;
  ForwardDropPolicy dropPolicy = ForwardDropPolicy::dropNewest;
};

/// @brief Runtime counters for asynchronous forwarding behavior.
struct ForwarderStats {
  std::uint64_t enqueuedRecords = 0;
  std::uint64_t droppedRecords = 0;
  std::uint64_t truncatedRecords = 0;
  std::uint64_t forwardFailures = 0;
};

/// @brief Return true if the given level is currently enabled.
bool should_log(const LogLevel logLevel);
/// @brief Install or replace the optional async forwarding callback.
void setForwarder(ForwarderConfig config);
/// @brief Enable forwarding with a default worker callback to stdout/stderr.
void setForwarder();
/// @brief Disable forwarding and tear down the forwarding worker.
void clearForwarder();
/// @brief Return true when forwarding is enabled.
bool isForwarderEnabled();
/// @brief Return active forwarded-message capacity in bytes.
std::size_t getForwarderMessageCapacity();
/// @brief Return a snapshot of forwarding counters.
ForwarderStats getForwarderStats();
/// @brief Record that a forwarded message was truncated at format time.
void recordForwarderMessageTruncation();
/// @brief Emit a preformatted trace message.
void trace(const std::string_view msg);
/// @brief Emit a preformatted info message.
void info(const std::string_view msg);
/// @brief Emit a preformatted debug message.
void debug(const std::string_view msg);
/// @brief Emit a preformatted warning message.
void warn(const std::string_view msg);
/// @brief Emit a preformatted error message.
void error(const std::string_view msg);
/// @brief Extract filename from a path-like string.
std::string pathToFileName(const std::string_view fullFilePath);

// Test/debug helpers. Production callers configure the level via the
// CUDAQ_LOG_LEVEL environment variable.
/// @brief Override log level at runtime.
void setLogLevel(LogLevel level);
/// @brief Return the current runtime log level.
LogLevel getLogLevel();

// Flushes any buffered log output. Useful in tests that need to inspect
// captured stdout immediately after emitting an info/debug message
// (initializeLogger only enables flush_on(warn)).
/// @brief Flush logger sinks and pending forwarded records.
void flushLogs();

/// @brief Emit a formatted message with file and line metadata.
void logMessageFormatted(LogLevel logLevel, std::string formattedMessage,
                         const char *fileName, int lineNo);
/// @brief Emit a preformatted view with file and line metadata.
void logMessageView(LogLevel logLevel, std::string_view formattedMessage,
                    const char *fileName, int lineNo);
/// @brief Emit a formatted timestamped message with file and line metadata.
void logWithTimestampFormatted(std::string formattedMessage,
                               const char *fileName, int lineNo);
/// @brief Emit a preformatted timestamped message view.
void logWithTimestampView(std::string_view formattedMessage,
                          const char *fileName, int lineNo);

/// @brief Format a message and arguments using `fmt`.
template <typename... Args>
std::string formatMessage(const std::string_view message, Args &&...args) {
  return fmt::vformat(message, fmt::make_format_args(args...));
}

/// @brief Format and emit a message for the provided log level.
template <typename... Args>
void logMessage(LogLevel logLevel, const std::string_view message,
                const char *fileName, int lineNo, Args &&...args) {
  if (isForwarderEnabled()) {
    std::array<char, kRealtimeForwarderMaxMessageCapacity> buffer{};
    const std::size_t cap =
        std::min(getForwarderMessageCapacity(), buffer.size() - 1);
    auto result = fmt::vformat_to_n(buffer.data(), cap, message,
                                    fmt::make_format_args(args...));
    auto used = std::min<std::size_t>(result.size, cap);
    if (result.size > cap) {
      recordForwarderMessageTruncation();
      if (used >= kRealtimeTruncationSuffix.size()) {
        const std::size_t start = used - kRealtimeTruncationSuffix.size();
        std::copy(kRealtimeTruncationSuffix.begin(),
                  kRealtimeTruncationSuffix.end(), buffer.begin() + start);
      }
    }
    buffer[used] = '\0';
    logMessageView(logLevel, std::string_view(buffer.data(), used), fileName,
                   lineNo);
    return;
  }
  logMessageFormatted(logLevel,
                      formatMessage(message, std::forward<Args>(args)...),
                      fileName, lineNo);
}
} // namespace detail

/// These types seek to enable automated injection of the source location of the
/// `cudaq::qec::info()` or `debug()` call. The actual formatting is out-of-line
/// in logger.cpp so callers do not need to parse `fmt` or `chrono` headers.
#define CUDAQ_LOGGER_DEDUCTION_STRUCT(NAME)                                    \
  template <typename... Args>                                                  \
  struct NAME {                                                                \
    NAME(const std::string_view message, Args &&...args,                       \
         const char *fileName = __builtin_FILE(),                              \
         int lineNo = __builtin_LINE()) {                                      \
      if (detail::should_log(detail::LogLevel::NAME))                          \
        detail::logMessage(detail::LogLevel::NAME, message, fileName, lineNo,  \
                           std::forward<Args>(args)...);                       \
    }                                                                          \
  };                                                                           \
  template <typename... Args>                                                  \
  NAME(const std::string_view, Args &&...) -> NAME<Args...>;

CUDAQ_LOGGER_DEDUCTION_STRUCT(info);
CUDAQ_LOGGER_DEDUCTION_STRUCT(warn);
CUDAQ_LOGGER_DEDUCTION_STRUCT(error);

#ifdef CUDAQ_DEBUG
CUDAQ_LOGGER_DEDUCTION_STRUCT(debug);
#else
// Remove cudaq::debug log messages from Release binaries.
template <typename... Args>
void debug(const std::string_view, Args &&...) {}
#endif

/// @brief Log a message with timestamp.
// Note 1: This will always log the message regardless of the logging level.
// Note 2: File and line info is not included in the log line.
template <typename... Args>
void log(const std::string_view message, Args &&...args) {
  if (detail::isForwarderEnabled()) {
    std::array<char, detail::kRealtimeForwarderMaxMessageCapacity> buffer{};
    const std::size_t cap =
        std::min(detail::getForwarderMessageCapacity(), buffer.size() - 1);
    auto result = fmt::vformat_to_n(buffer.data(), cap, message,
                                    fmt::make_format_args(args...));
    auto used = std::min<std::size_t>(result.size, cap);
    if (result.size > cap) {
      detail::recordForwarderMessageTruncation();
      if (used >= detail::kRealtimeTruncationSuffix.size()) {
        const std::size_t start =
            used - detail::kRealtimeTruncationSuffix.size();
        std::copy(detail::kRealtimeTruncationSuffix.begin(),
                  detail::kRealtimeTruncationSuffix.end(),
                  buffer.begin() + start);
      }
    }
    buffer[used] = '\0';
    detail::logWithTimestampView(std::string_view(buffer.data(), used),
                                 __builtin_FILE(), __builtin_LINE());
    return;
  }
  detail::logWithTimestampFormatted(
      detail::formatMessage(message, std::forward<Args>(args)...),
      __builtin_FILE(), __builtin_LINE());
}

} // namespace cudaq::qec

// The following macros avoid the unnecessary processing cost of argument
// evaluation and string formation until after the log level check is done.
#define CUDA_QEC_LOG_IMPL(LEVEL, msg, ...)                                     \
  do {                                                                         \
    if (::cudaq::qec::detail::should_log(                                      \
            ::cudaq::qec::detail::LogLevel::LEVEL)) {                          \
      ::cudaq::qec::detail::logMessage(::cudaq::qec::detail::LogLevel::LEVEL,  \
                                       msg, __FILE__,                          \
                                       __LINE__ __VA_OPT__(, ) __VA_ARGS__);   \
    }                                                                          \
  } while (false)

#define CUDA_QEC_ERROR_IMPL(msg, ...)                                          \
  do {                                                                         \
    ::cudaq::qec::detail::logMessage(::cudaq::qec::detail::LogLevel::error,    \
                                     msg, __FILE__,                            \
                                     __LINE__ __VA_OPT__(, ) __VA_ARGS__);     \
    throw std::runtime_error(                                                  \
        ::cudaq::qec::detail::formatMessage(msg __VA_OPT__(, ) __VA_ARGS__));  \
  } while (false)

#define CUDA_QEC_ERROR(...) CUDA_QEC_ERROR_IMPL(__VA_ARGS__)
#define CUDA_QEC_WARN(...) CUDA_QEC_LOG_IMPL(warn, __VA_ARGS__)
#define CUDA_QEC_INFO(...) CUDA_QEC_LOG_IMPL(info, __VA_ARGS__)

#ifdef CUDAQ_DEBUG
#define CUDA_QEC_DBG(...) CUDA_QEC_LOG_IMPL(debug, __VA_ARGS__)
#else
#define CUDA_QEC_DBG(...)
#endif

// Backward-compatible aliases for existing QEC callsites. If runtime logger
// macros are already present in this translation unit, do not redefine them.
#ifndef CUDAQ_LOG_IMPL
#define CUDAQ_LOG_IMPL(LEVEL, msg, ...)                                        \
  CUDA_QEC_LOG_IMPL(LEVEL, msg __VA_OPT__(, ) __VA_ARGS__)
#endif
#ifndef CUDAQ_ERROR_IMPL
#define CUDAQ_ERROR_IMPL(msg, ...)                                             \
  CUDA_QEC_ERROR_IMPL(msg __VA_OPT__(, ) __VA_ARGS__)
#endif
#ifndef CUDAQ_ERROR
#define CUDAQ_ERROR(...) CUDA_QEC_ERROR(__VA_ARGS__)
#endif
#ifndef CUDAQ_WARN
#define CUDAQ_WARN(...) CUDA_QEC_WARN(__VA_ARGS__)
#endif
#ifndef CUDAQ_INFO
#define CUDAQ_INFO(...) CUDA_QEC_INFO(__VA_ARGS__)
#endif
#ifndef CUDAQ_DBG
#define CUDAQ_DBG(...) CUDA_QEC_DBG(__VA_ARGS__)
#endif
