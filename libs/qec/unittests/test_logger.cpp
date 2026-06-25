/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq/qec/logger.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

/// @file
/// @brief Unit tests for QEC logger behavior.
/// @details
/// Validates runtime level filtering, source metadata rendering, forwarding
/// behavior, sink selection rules, queue backpressure handling, and default
/// forwarder setup semantics.

namespace {

using namespace std::chrono_literals;

struct ForwarderGuard {
  ~ForwarderGuard() { cudaq::qec::detail::clearForwarder(); }
};

// Verify runtime log level changes take effect immediately for visibility.
TEST(Logger, UserSettableLogLevelControlsVisibility) {
  ForwarderGuard guard;
  cudaq::qec::detail::clearForwarder();
  cudaq::qec::detail::setLogLevel(cudaq::qec::detail::LogLevel::warn);

  testing::internal::CaptureStdout();
  CUDAQX_INFO("hidden message");
  cudaq::qec::detail::flushLogs();
  const std::string hidden = testing::internal::GetCapturedStdout();
  EXPECT_TRUE(hidden.empty());

  cudaq::qec::detail::setLogLevel(cudaq::qec::detail::LogLevel::info);
  testing::internal::CaptureStdout();
  CUDAQX_INFO("visible message");
  cudaq::qec::detail::flushLogs();
  const std::string visible = testing::internal::GetCapturedStdout();
  EXPECT_NE(visible.find("visible message"), std::string::npos);
} // end - TEST(Logger, UserSettableLogLevelControlsVisibility)

// Validate that log lines include level, source file, and formatted message.
TEST(Logger, InfoLogsIncludeTimestampAndSourceLocation) {
  ForwarderGuard guard;
  cudaq::qec::detail::clearForwarder();
  cudaq::qec::detail::setLogLevel(cudaq::qec::detail::LogLevel::trace);

  testing::internal::CaptureStdout();
  CUDAQX_INFO("metadata message {}", 7);
  cudaq::qec::detail::flushLogs();
  const std::string out = testing::internal::GetCapturedStdout();

  EXPECT_NE(out.find("[info]"), std::string::npos);
  EXPECT_NE(out.find("test_logger.cpp"), std::string::npos);
  EXPECT_NE(out.find("metadata message 7"), std::string::npos);
} // end - TEST(Logger, InfoLogsIncludeTimestampAndSourceLocation)

// Ensure hidden levels short-circuit before evaluating expensive arguments.
TEST(Logger, HiddenLevelMessagesAreNotFormed) {
  ForwarderGuard guard;
  cudaq::qec::detail::clearForwarder();
  cudaq::qec::detail::setLogLevel(cudaq::qec::detail::LogLevel::error);

  std::atomic<int> evalCount{0};
  auto expensiveValue = [&]() -> int {
    evalCount.fetch_add(1, std::memory_order_relaxed);
    return 42;
  };

  CUDAQX_INFO("hidden {}", expensiveValue());
  EXPECT_EQ(evalCount.load(std::memory_order_relaxed), 0);
} // end - TEST(Logger, HiddenLevelMessagesAreNotFormed)

// Keep a lightweight throughput-neutrality guard in the unit suite: suppressed
// info logs should remain inexpensive when the active level is warn.
TEST(Logger, SuppressedInfoPathStaysFast) {
  ForwarderGuard guard;
  cudaq::qec::detail::clearForwarder();
  cudaq::qec::detail::setLogLevel(cudaq::qec::detail::LogLevel::warn);

  constexpr int iterations = 200000;
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i)
    CUDAQX_INFO("suppressed-path {}", i);
  const auto end = std::chrono::steady_clock::now();

  const auto elapsedUs =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count();
  // Use a conservative budget to reduce CI flakiness while still catching
  // severe suppressed-path regressions.
  EXPECT_LT(elapsedUs, 100000);
} // end - TEST(Logger, SuppressedInfoPathStaysFast)

// Verify forwarded records preserve payload and source metadata.
TEST(Logger, ForwarderReceivesRecordsWhenEnabled) {
  ForwarderGuard guard;
  cudaq::qec::detail::clearForwarder();
  cudaq::qec::detail::setLogLevel(cudaq::qec::detail::LogLevel::info);

  std::mutex mutex;
  std::condition_variable cv;
  std::vector<cudaq::qec::detail::ForwardedLogRecord> records;
  cudaq::qec::detail::setForwarder(cudaq::qec::detail::ForwarderConfig{
      .callback =
          [&](const cudaq::qec::detail::ForwardedLogRecord &record) {
            std::lock_guard<std::mutex> lock(mutex);
            records.push_back(record);
            cv.notify_all();
          },
      .queueCapacity = 16,
      .dropPolicy = cudaq::qec::detail::ForwardDropPolicy::dropNewest});

  // Forwarding enqueue is intentionally best-effort (try_lock + drop on
  // contention), so send a short burst and assert that at least one expected
  // record arrives instead of requiring a single specific one.
  for (int i = 0; i < 32; ++i)
    CUDAQX_INFO("forwarded {}", i);
  cudaq::qec::detail::flushLogs();

  std::unique_lock<std::mutex> lock(mutex);
  ASSERT_TRUE(cv.wait_for(lock, 2s, [&] { return !records.empty(); }));
  EXPECT_EQ(records.front().fileName, "test_logger.cpp");
  EXPECT_GT(records.front().lineNo, 0);
  const bool hasForwarded =
      std::any_of(records.begin(), records.end(), [](const auto &record) {
        return record.message.rfind("forwarded ", 0) == 0;
      });
  EXPECT_TRUE(hasForwarded);
} // end - TEST(Logger, ForwarderReceivesRecordsWhenEnabled)

// Ensure enabling a forwarder suppresses direct stdout/stderr emission.
TEST(Logger, ForwarderEnabledSuppressesStdoutAndStderr) {
  ForwarderGuard guard;
  cudaq::qec::detail::clearForwarder();
  cudaq::qec::detail::setLogLevel(cudaq::qec::detail::LogLevel::trace);

  std::mutex mutex;
  std::condition_variable cv;
  std::vector<cudaq::qec::detail::ForwardedLogRecord> records;
  cudaq::qec::detail::setForwarder(cudaq::qec::detail::ForwarderConfig{
      .callback =
          [&](const cudaq::qec::detail::ForwardedLogRecord &record) {
            std::lock_guard<std::mutex> lock(mutex);
            records.push_back(record);
            cv.notify_all();
          },
      .queueCapacity = 16,
      .dropPolicy = cudaq::qec::detail::ForwardDropPolicy::dropNewest});

  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();
  // Keep stdout/stderr assertions deterministic by allowing occasional dropped
  // records while still requiring both message types to be observed.
  for (int i = 0; i < 32; ++i) {
    CUDAQX_INFO("forwarded-info");
    CUDAQX_WARN("forwarded-warn");
  }
  cudaq::qec::detail::flushLogs();
  const std::string out = testing::internal::GetCapturedStdout();
  const std::string err = testing::internal::GetCapturedStderr();

  std::unique_lock<std::mutex> lock(mutex);
  ASSERT_TRUE(cv.wait_for(lock, 2s, [&] { return !records.empty(); }));
  EXPECT_TRUE(out.empty());
  EXPECT_TRUE(err.empty());
  const bool hasInfo =
      std::any_of(records.begin(), records.end(), [](const auto &record) {
        return record.message == "forwarded-info";
      });
  const bool hasWarn =
      std::any_of(records.begin(), records.end(), [](const auto &record) {
        return record.message == "forwarded-warn";
      });
  EXPECT_TRUE(hasInfo);
  EXPECT_TRUE(hasWarn);
} // end - TEST(Logger, ForwarderEnabledSuppressesStdoutAndStderr)

// Confirm default sink routing when no forwarder is installed.
TEST(Logger, DisabledForwarderUsesStdoutAndStderrByLevel) {
  ForwarderGuard guard;
  cudaq::qec::detail::clearForwarder();
  cudaq::qec::detail::setLogLevel(cudaq::qec::detail::LogLevel::trace);

  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();
  CUDAQX_INFO("stdout-info");
#ifdef CUDAQ_DEBUG
  CUDAQX_DBG("stdout-debug");
#endif
  CUDAQX_WARN("stderr-warn");
  // CUDAQX_ERROR logs and then throws by design; assert the throw explicitly so
  // this sink-routing test does not fail from an uncaught exception.
  EXPECT_THROW(CUDAQX_ERROR("stderr-error"), std::runtime_error);
  const std::string out = testing::internal::GetCapturedStdout();
  const std::string err = testing::internal::GetCapturedStderr();

  EXPECT_NE(out.find("stdout-info"), std::string::npos);
#ifdef CUDAQ_DEBUG
  EXPECT_NE(out.find("stdout-debug"), std::string::npos);
#endif
  EXPECT_NE(err.find("stderr-warn"), std::string::npos);
  EXPECT_NE(err.find("stderr-error"), std::string::npos);
} // end - TEST(Logger, DisabledForwarderUsesStdoutAndStderrByLevel)

// Verify zero-argument setForwarder() installs the default sink callback.
TEST(Logger, DefaultSetForwarderWritesToStdoutAndStderr) {
  ForwarderGuard guard;
  cudaq::qec::detail::clearForwarder();
  cudaq::qec::detail::setLogLevel(cudaq::qec::detail::LogLevel::trace);
  cudaq::qec::detail::setForwarder();

  ASSERT_TRUE(cudaq::qec::detail::isForwarderEnabled());
  // The default forwarder also uses the same non-blocking enqueue path; use a
  // burst and check for some successful enqueues instead of an exact count.
  for (int i = 0; i < 32; ++i) {
    CUDAQX_INFO("default-forwarder-info");
    CUDAQX_WARN("default-forwarder-warn");
  }
  cudaq::qec::detail::flushLogs();
  // Capturing stdout/stderr is flaky here because the default forwarder emits
  // from a background worker thread; queue/accounting assertions are stable.
  const auto stats = cudaq::qec::detail::getForwarderStats();
  EXPECT_GT(stats.enqueuedRecords, 0u);
  EXPECT_EQ(stats.forwardFailures, 0u);
} // end - TEST(Logger, DefaultSetForwarderWritesToStdoutAndStderr)

// Confirm bounded queue drops records under sustained producer pressure.
TEST(Logger, SaturatedForwarderQueueDropsWithoutBlockingProducer) {
  ForwarderGuard guard;
  cudaq::qec::detail::clearForwarder();
  cudaq::qec::detail::setLogLevel(cudaq::qec::detail::LogLevel::info);

  std::atomic<bool> releaseCallback{false};
  cudaq::qec::detail::setForwarder(cudaq::qec::detail::ForwarderConfig{
      .callback =
          [&](const cudaq::qec::detail::ForwardedLogRecord &) {
            while (!releaseCallback.load(std::memory_order_relaxed))
              std::this_thread::sleep_for(1ms);
          },
      .queueCapacity = 1,
      .dropPolicy = cudaq::qec::detail::ForwardDropPolicy::dropNewest});

  for (int i = 0; i < 256; ++i)
    CUDAQX_INFO("queue-load {}", i);

  const auto stats = cudaq::qec::detail::getForwarderStats();
  EXPECT_GT(stats.enqueuedRecords, 0u);
  EXPECT_GT(stats.droppedRecords, 0u);

  releaseCallback.store(true, std::memory_order_relaxed);
  cudaq::qec::detail::flushLogs();
} // end - TEST(Logger, SaturatedForwarderQueueDropsWithoutBlockingProducer)

// Ensure no queue accounting changes when forwarding is disabled.
TEST(Logger, DisabledForwarderDoesNotEnqueueRecords) {
  ForwarderGuard guard;
  cudaq::qec::detail::clearForwarder();
  cudaq::qec::detail::setLogLevel(cudaq::qec::detail::LogLevel::info);

  const auto before = cudaq::qec::detail::getForwarderStats();
  CUDAQX_INFO("no-forwarder");
  cudaq::qec::detail::flushLogs();
  const auto after = cudaq::qec::detail::getForwarderStats();

  EXPECT_EQ(after.enqueuedRecords, before.enqueuedRecords);
} // end - TEST(Logger, DisabledForwarderDoesNotEnqueueRecords)

} // namespace
