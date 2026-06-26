/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/logger.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>

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

struct QueuedLogRecord {
  LogLevel level = LogLevel::info;
  std::uint64_t timestampNs = 0;
  int lineNo = 0;
  std::thread::id threadId;
  std::array<char, 128> fileName{};
  std::size_t fileNameLen = 0;
  std::array<char, detail::kRealtimeForwarderMaxMessageCapacity> message{};
  std::size_t messageLen = 0;
};

class AsyncForwarder {
public:
  // Default-construct the forwarder with forwarding disabled.
  AsyncForwarder() = default;
  // Ensure worker thread is stopped before object destruction.
  ~AsyncForwarder() { clear(); }

  // Install callback + queue configuration and start worker thread.
  void set(ForwarderConfig config) {
    clear();
    mConfig = std::move(config);
    if (!mConfig.callback)
      return;
    if (mConfig.queueCapacity == 0)
      mConfig.queueCapacity = 1;
    if (mConfig.messageCapacity == 0)
      mConfig.messageCapacity = 1;
    mConfig.messageCapacity =
        std::min(mConfig.messageCapacity, kRealtimeForwarderMaxMessageCapacity);
    mMessageCapacity.store(mConfig.messageCapacity, std::memory_order_relaxed);

    mCapacity = roundUpToPow2(mConfig.queueCapacity);
    mRing = std::make_unique<RingSlot[]>(mCapacity);
    for (std::size_t i = 0; i < mCapacity; ++i)
      mRing[i].sequence.store(i, std::memory_order_relaxed);

    mMask = mCapacity - 1;
    mEnqueuePos.store(0, std::memory_order_relaxed);
    mDequeuePos.store(0, std::memory_order_relaxed);
    mStop.store(false, std::memory_order_relaxed);
    mEnabled.store(true, std::memory_order_release);
    mWorker = std::thread([this] { runWorker(); });
  } // end - set()

  // Stop worker thread and reset forwarding state.
  void clear() {
    mEnabled.store(false, std::memory_order_release);
    while (mActiveProducers.load(std::memory_order_acquire) != 0)
      std::this_thread::yield();
    mStop.store(true, std::memory_order_release);
    mCv.notify_all();
    if (mWorker.joinable())
      mWorker.join();
    mRing.reset();
    mCapacity = 0;
    mMask = 0;
    mEnqueuePos.store(0, std::memory_order_relaxed);
    mDequeuePos.store(0, std::memory_order_relaxed);
    mStop.store(false, std::memory_order_relaxed);
    mMessageCapacity.store(kRealtimeForwarderDefaultMessageCapacity,
                           std::memory_order_relaxed);
    mConfig = {};
  } // end - clear()

  // Fast path atomic check used by producers.
  bool isEnabled() const { return mEnabled.load(std::memory_order_relaxed); }

  // Return lock-free counters snapshot for diagnostics/tests.
  ForwarderStats stats() const {
    ForwarderStats stats;
    stats.enqueuedRecords = mEnqueuedRecords.load(std::memory_order_relaxed);
    stats.droppedRecords = mDroppedRecords.load(std::memory_order_relaxed);
    stats.truncatedRecords = mTruncatedRecords.load(std::memory_order_relaxed);
    stats.forwardFailures = mForwardFailures.load(std::memory_order_relaxed);
    return stats;
  }

  // Reset counters when installing a new forwarder config.
  void resetStats() {
    mEnqueuedRecords.store(0, std::memory_order_relaxed);
    mDroppedRecords.store(0, std::memory_order_relaxed);
    mTruncatedRecords.store(0, std::memory_order_relaxed);
    mForwardFailures.store(0, std::memory_order_relaxed);
    mDropWarningPending.store(false, std::memory_order_relaxed);
    mDropWarningEmitted.store(false, std::memory_order_relaxed);
  }

  std::size_t messageCapacity() const {
    return mMessageCapacity.load(std::memory_order_relaxed);
  }

  void noteTruncation() {
    mTruncatedRecords.fetch_add(1, std::memory_order_relaxed);
  }

  // Wait until queue drains, used in tests and explicit flush.
  void flush() {
    std::unique_lock<std::mutex> lock(mWaitMutex);
    mCv.wait(lock, [&] { return isQueueEmpty() || !mEnabled.load(); });
  }

  // Producer path: enqueue into ring buffer, drop only when saturated.
  void tryEnqueue(QueuedLogRecord record) {
    if (!isEnabled())
      return;
    ProducerGuard producerGuard(mActiveProducers);
    if (!isEnabled() || !mConfig.callback || !mRing)
      return;

    while (true) {
      if (tryEnqueueOne(record)) {
        mEnqueuedRecords.fetch_add(1, std::memory_order_relaxed);
        mCv.notify_one();
        return;
      }

      if (mConfig.dropPolicy == ForwardDropPolicy::dropOldest) {
        QueuedLogRecord ignored;
        if (tryDequeueOne(ignored)) {
          notifyDrop();
          continue;
        }
      }

      notifyDrop();
      return;
    }
  } // end - tryEnqueue()

private:
  struct RingSlot {
    std::atomic<std::size_t> sequence{0};
    QueuedLogRecord record;
  };

  struct ProducerGuard {
    explicit ProducerGuard(std::atomic<std::uint64_t> &counterRef)
        : counter(counterRef) {
      counter.fetch_add(1, std::memory_order_acq_rel);
    }
    ~ProducerGuard() { counter.fetch_sub(1, std::memory_order_acq_rel); }
    std::atomic<std::uint64_t> &counter;
  };

  static std::size_t roundUpToPow2(std::size_t value) {
    // This queue uses `index = position & (capacity - 1)` and per-slot sequence
    // arithmetic (Vyukov-style ring). That mapping is equivalent to modulo only
    // for power-of-two capacities; non-power-of-two sizes break slot mapping
    // and can corrupt full/empty detection.
    std::size_t rounded = 2;
    while (rounded < value)
      rounded <<= 1;
    return rounded;
  }

  bool tryEnqueueOne(QueuedLogRecord &record) {
    std::size_t pos = mEnqueuePos.load(std::memory_order_relaxed);
    while (true) {
      RingSlot &slot = mRing[pos & mMask];
      const std::size_t sequence =
          slot.sequence.load(std::memory_order_acquire);
      const std::intptr_t dif = static_cast<std::intptr_t>(sequence) -
                                static_cast<std::intptr_t>(pos);

      if (dif == 0) {
        if (mEnqueuePos.compare_exchange_weak(pos, pos + 1,
                                              std::memory_order_relaxed))
          break;
        continue;
      }
      if (dif < 0)
        return false;
      pos = mEnqueuePos.load(std::memory_order_relaxed);
    }

    RingSlot &slot = mRing[pos & mMask];
    slot.record = std::move(record);
    slot.sequence.store(pos + 1, std::memory_order_release);
    return true;
  }

  bool tryDequeueOne(QueuedLogRecord &record) {
    std::size_t pos = mDequeuePos.load(std::memory_order_relaxed);
    while (true) {
      RingSlot &slot = mRing[pos & mMask];
      const std::size_t sequence =
          slot.sequence.load(std::memory_order_acquire);
      const std::intptr_t dif = static_cast<std::intptr_t>(sequence) -
                                static_cast<std::intptr_t>(pos + 1);

      if (dif == 0) {
        if (mDequeuePos.compare_exchange_weak(pos, pos + 1,
                                              std::memory_order_relaxed))
          break;
        continue;
      }
      if (dif < 0)
        return false;
      pos = mDequeuePos.load(std::memory_order_relaxed);
    }

    RingSlot &slot = mRing[pos & mMask];
    record = std::move(slot.record);
    slot.sequence.store(pos + mCapacity, std::memory_order_release);
    return true;
  }

  bool isQueueEmpty() const {
    return mEnqueuePos.load(std::memory_order_acquire) ==
           mDequeuePos.load(std::memory_order_acquire);
  }

  // Emit a one-time diagnostic when forwarding drops are first observed.
  void notifyDrop() {
    mDroppedRecords.fetch_add(1, std::memory_order_relaxed);
    mDropWarningPending.store(true, std::memory_order_relaxed);
    mCv.notify_one();
  }

  void emitDropWarningIfPending() {
    if (!mDropWarningPending.exchange(false, std::memory_order_relaxed))
      return;
    bool expected = false;
    if (!mDropWarningEmitted.compare_exchange_strong(expected, true,
                                                     std::memory_order_relaxed))
      return;
    std::fputs("[cudaq::qec::logger] forwarder dropped log records "
               "(queue full); increase ForwarderConfig::queueCapacity or "
               "reduce callback latency.\n",
               stderr);
  }

  // Consume queued records and invoke callback on background thread.
  void runWorker() {
    while (true) {
      emitDropWarningIfPending();
      QueuedLogRecord queuedRecord;
      if (!tryDequeueOne(queuedRecord)) {
        std::unique_lock<std::mutex> lock(mWaitMutex);
        mCv.wait(lock, [&] {
          return mStop.load(std::memory_order_acquire) || !isQueueEmpty();
        });
        if (mStop.load(std::memory_order_acquire) && isQueueEmpty())
          return;
        continue;
      }

      ForwardedLogRecord record;
      record.level = queuedRecord.level;
      record.timestampNs = queuedRecord.timestampNs;
      record.fileName.assign(queuedRecord.fileName.data(),
                             queuedRecord.fileNameLen);
      record.lineNo = queuedRecord.lineNo;
      record.message.assign(queuedRecord.message.data(),
                            queuedRecord.messageLen);
      record.threadId = queuedRecord.threadId;

      try {
        if (mConfig.callback)
          mConfig.callback(record);
      } catch (...) {
        mForwardFailures.fetch_add(1, std::memory_order_relaxed);
      }

      emitDropWarningIfPending();
      mCv.notify_all();
    } // end - while(true)
  } // end - runWorker()

  std::condition_variable_any mCv;
  mutable std::mutex mWaitMutex;
  ForwarderConfig mConfig;
  std::unique_ptr<RingSlot[]> mRing;
  std::size_t mCapacity = 0;
  std::size_t mMask = 0;
  std::atomic<std::size_t> mEnqueuePos{0};
  std::atomic<std::size_t> mDequeuePos{0};
  std::atomic<bool> mStop{false};
  std::thread mWorker;
  std::atomic<bool> mEnabled{false};
  std::atomic<std::uint64_t> mEnqueuedRecords{0};
  std::atomic<std::uint64_t> mDroppedRecords{0};
  std::atomic<std::uint64_t> mTruncatedRecords{0};
  std::atomic<std::uint64_t> mForwardFailures{0};
  std::atomic<std::uint64_t> mActiveProducers{0};
  std::atomic<bool> mDropWarningPending{false};
  std::atomic<bool> mDropWarningEmitted{false};
  std::atomic<std::size_t> mMessageCapacity{
      kRealtimeForwarderDefaultMessageCapacity};
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
    QueuedLogRecord record;
    const std::size_t messageCap = forwarder().messageCapacity();
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
      forwarder().noteTruncation();
      if (record.messageLen >= kRealtimeTruncationSuffix.size()) {
        const std::size_t start =
            record.messageLen - kRealtimeTruncationSuffix.size();
        std::copy(kRealtimeTruncationSuffix.begin(),
                  kRealtimeTruncationSuffix.end(),
                  record.message.begin() + start);
      }
    }
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

std::size_t getForwarderMessageCapacity() {
  return forwarder().messageCapacity();
}

void recordForwarderMessageTruncation() { forwarder().noteTruncation(); }

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

void logMessageView(LogLevel logLevel, std::string_view formattedMessage,
                    const char *fileName, int lineNo) {
  emit(logLevel, formattedMessage, fileName, lineNo);
}

// Entry point for timestamped log helper after message formatting.
void logWithTimestampFormatted(std::string formattedMessage,
                               const char *fileName, int lineNo) {
  emit(LogLevel::info, formattedMessage, fileName, lineNo);
}

void logWithTimestampView(std::string_view formattedMessage,
                          const char *fileName, int lineNo) {
  emit(LogLevel::info, formattedMessage, fileName, lineNo);
}

} // namespace cudaq::qec::detail
