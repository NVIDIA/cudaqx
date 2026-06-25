/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/logger.h"
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <mutex>
#include <optional>
#include <sstream>
#include <vector>

/// @file
/// @brief QEC logger implementation.
/// @details
/// Implements runtime level filtering, synchronous stdout/stderr emission,
/// optional asynchronous forwarding to a worker thread, and helper utilities
/// for formatting timestamped log lines with source metadata.

namespace cudaq::qec::detail {
namespace {

using Clock = std::chrono::system_clock;

std::atomic<LogLevel> gLogLevel{LogLevel::warn};
std::once_flag gLogLevelInitFlag;

// Convert a level token to lowercase before parsing.
std::string toLower(std::string value) {
  for (auto &ch : value)
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  return value;
}

// Parse text level from env/config to an internal enum.
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
} // end - parseLogLevel()

// Convert internal log level to stable label in log lines.
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
} // end - logLevelName()

// Return current wall-clock timestamp in nanoseconds since epoch.
std::uint64_t nowNs() {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          Clock::now().time_since_epoch())
          .count());
}

// Render local-time timestamp with microsecond precision for log prefixes.
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
} // end - formatTimestamp()

// Build one final log line with timestamp, level, source, and payload.
std::string composeLine(const LogLevel level, const std::string_view message,
                        const char *fileName, const int lineNo,
                        const Clock::time_point ts) {
  std::ostringstream out;
  out << '[' << formatTimestamp(ts) << "] [" << logLevelName(level) << "] ["
      << pathToFileName(fileName) << ':' << lineNo << "] " << message;
  return out.str();
}

class AsyncForwarder {
public:
  // Default-construct the forwarder with forwarding disabled.
  AsyncForwarder() = default;
  // Ensure worker thread is stopped before object destruction.
  ~AsyncForwarder() { clear(); }

  // Install callback + queue configuration and start worker thread.
  void set(ForwarderConfig config) {
    clear();
    {
      std::lock_guard<std::mutex> lock(mMutex);
      mConfig = std::move(config);
      if (!mConfig.callback)
        return;
      if (mConfig.queueCapacity == 0)
        mConfig.queueCapacity = 1;
      mQueue.assign(mConfig.queueCapacity, ForwardedLogRecord{});
      mHead = 0;
      mSize = 0;
      mStop = false;
      mEnabled.store(true, std::memory_order_release);
      mWorker = std::thread([this] { runWorker(); });
    }
  } // end - set()

  // Stop worker thread and reset forwarding state.
  void clear() {
    mEnabled.store(false, std::memory_order_release);
    {
      std::lock_guard<std::mutex> lock(mMutex);
      mStop = true;
    }
    mCv.notify_all();
    if (mWorker.joinable())
      mWorker.join();
    std::lock_guard<std::mutex> lock(mMutex);
    mQueue.clear();
    mHead = 0;
    mSize = 0;
    mStop = false;
    mConfig = {};
  } // end - clear()

  // Fast path atomic check used by producers.
  bool isEnabled() const { return mEnabled.load(std::memory_order_relaxed); }

  // Return lock-free counters snapshot for diagnostics/tests.
  ForwarderStats stats() const {
    ForwarderStats stats;
    stats.enqueuedRecords = mEnqueuedRecords.load(std::memory_order_relaxed);
    stats.droppedRecords = mDroppedRecords.load(std::memory_order_relaxed);
    stats.forwardFailures = mForwardFailures.load(std::memory_order_relaxed);
    return stats;
  }

  // Reset counters when installing a new forwarder config.
  void resetStats() {
    mEnqueuedRecords.store(0, std::memory_order_relaxed);
    mDroppedRecords.store(0, std::memory_order_relaxed);
    mForwardFailures.store(0, std::memory_order_relaxed);
  }

  // Wait until queue drains, used in tests and explicit flush.
  void flush() {
    std::unique_lock<std::mutex> lock(mMutex);
    mCv.wait(lock, [&] { return mSize == 0 || !mEnabled.load(); });
  }

  // Best-effort non-blocking enqueue; drops when lock/queue is unavailable.
  void tryEnqueue(ForwardedLogRecord record) {
    if (!isEnabled())
      return;
    if (!mMutex.try_lock()) {
      mDroppedRecords.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    std::unique_lock<std::mutex> lock(mMutex, std::adopt_lock);
    if (!mEnabled.load(std::memory_order_relaxed) || !mConfig.callback)
      return;

    if (mQueue.empty()) {
      mDroppedRecords.fetch_add(1, std::memory_order_relaxed);
      return;
    }

    if (mSize == mQueue.size()) {
      if (mConfig.dropPolicy == ForwardDropPolicy::dropNewest) {
        mDroppedRecords.fetch_add(1, std::memory_order_relaxed);
        return;
      }
      // Drop oldest.
      mQueue[mHead] = std::move(record);
      mHead = (mHead + 1) % mQueue.size();
      mDroppedRecords.fetch_add(1, std::memory_order_relaxed);
      mEnqueuedRecords.fetch_add(1, std::memory_order_relaxed);
      lock.unlock();
      mCv.notify_one();
      return;
    } // end - if(mSize == mQueue.size())

    const std::size_t tail = (mHead + mSize) % mQueue.size();
    mQueue[tail] = std::move(record);
    ++mSize;
    mEnqueuedRecords.fetch_add(1, std::memory_order_relaxed);
    lock.unlock();
    mCv.notify_one();
  } // end - tryEnqueue()

private:
  // Consume queued records and invoke callback on background thread.
  void runWorker() {
    while (true) {
      ForwardedLogRecord record;
      {
        std::unique_lock<std::mutex> lock(mMutex);
        mCv.wait(lock, [&] { return mStop || mSize > 0; });
        if (mStop && mSize == 0)
          return;
        record = std::move(mQueue[mHead]);
        mHead = (mHead + 1) % mQueue.size();
        --mSize;
      }

      try {
        if (mConfig.callback)
          mConfig.callback(record);
      } catch (...) {
        mForwardFailures.fetch_add(1, std::memory_order_relaxed);
      }

      mCv.notify_all();
    } // end - while(true)
  } // end - runWorker()

  mutable std::mutex mMutex;
  std::condition_variable mCv;
  ForwarderConfig mConfig;
  std::vector<ForwardedLogRecord> mQueue;
  std::size_t mHead = 0;
  std::size_t mSize = 0;
  bool mStop = false;
  std::thread mWorker;
  std::atomic<bool> mEnabled{false};
  std::atomic<std::uint64_t> mEnqueuedRecords{0};
  std::atomic<std::uint64_t> mDroppedRecords{0};
  std::atomic<std::uint64_t> mForwardFailures{0};
};

// Return singleton forwarder shared by all logger call sites.
AsyncForwarder &forwarder() {
  static AsyncForwarder instance;
  return instance;
}

// Lazily initialize default log level from CUDAQ_LOG_LEVEL env var.
void initializeLogLevelFromEnv() {
  std::call_once(gLogLevelInitFlag, [] {
    if (const char *env = std::getenv("CUDAQ_LOG_LEVEL")) {
      if (const auto parsed = parseLogLevel(env))
        gLogLevel.store(*parsed, std::memory_order_relaxed);
    }
  });
}

// Emit to forwarding sink when enabled; otherwise use stdout/stderr.
void emit(const LogLevel level, const std::string_view rawMessage,
          const char *fileName, const int lineNo) {
  if (forwarder().isEnabled()) {
    ForwardedLogRecord record;
    record.level = level;
    record.timestampNs = nowNs();
    record.fileName = pathToFileName(fileName);
    record.lineNo = lineNo;
    record.message = std::string(rawMessage);
    record.threadId = std::this_thread::get_id();
    forwarder().tryEnqueue(std::move(record));
    return;
  } // end - if(forwarder().isEnabled())

  const auto ts = Clock::now();
  const std::string fullLine =
      composeLine(level, rawMessage, fileName, lineNo, ts);
  FILE *stream =
      (level == LogLevel::warn || level == LogLevel::error) ? stderr : stdout;
  std::fputs(fullLine.c_str(), stream);
  std::fputc('\n', stream);
} // end - emit()

} // namespace

// Compare candidate level against current runtime threshold.
bool should_log(const LogLevel logLevel) {
  initializeLogLevelFromEnv();
  return static_cast<int>(logLevel) >=
         static_cast<int>(gLogLevel.load(std::memory_order_relaxed));
}

// Public API: install/replace forwarding callback.
void setForwarder(ForwarderConfig config) {
  forwarder().set(std::move(config));
  forwarder().resetStats();
}

// Public API: enable forwarding with a default stdout/stderr callback.
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
void clearForwarder() { forwarder().clear(); }

// Public API: report whether forwarding callback is active.
bool isForwarderEnabled() { return forwarder().isEnabled(); }

// Public API: return forwarding counters snapshot.
ForwarderStats getForwarderStats() { return forwarder().stats(); }

// Public API: direct trace sink for already-formatted messages.
void trace(const std::string_view msg) {
  emit(LogLevel::trace, msg, "<unknown>", 0);
}
// Public API: direct info sink for already-formatted messages.
void info(const std::string_view msg) {
  emit(LogLevel::info, msg, "<unknown>", 0);
}
// Public API: direct debug sink for already-formatted messages.
void debug(const std::string_view msg) {
  emit(LogLevel::debug, msg, "<unknown>", 0);
}
// Public API: direct warn sink for already-formatted messages.
void warn(const std::string_view msg) {
  emit(LogLevel::warn, msg, "<unknown>", 0);
}
// Public API: direct error sink for already-formatted messages.
void error(const std::string_view msg) {
  emit(LogLevel::error, msg, "<unknown>", 0);
}

// Strip directory prefix to keep compact source location output.
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

// Flush primary sink and wait for forwarding queue to drain.
void flushLogs() {
  std::fflush(stdout);
  std::fflush(stderr);
  forwarder().flush();
}

// Entry point used by templated call sites after message formatting.
void logMessageFormatted(LogLevel logLevel, std::string formattedMessage,
                         const char *fileName, int lineNo) {
  emit(logLevel, formattedMessage, fileName, lineNo);
}

// Entry point for timestamped log helper after message formatting.
void logWithTimestampFormatted(std::string formattedMessage,
                               const char *fileName, int lineNo) {
  emit(LogLevel::info, formattedMessage, fileName, lineNo);
}

} // namespace cudaq::qec::detail
