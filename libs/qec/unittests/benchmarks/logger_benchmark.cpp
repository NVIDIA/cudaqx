#include "cudaq/qec/logger.h"

#include <chrono>
#include <iostream>

/// @file
/// @brief Lightweight microbenchmark for suppressed logger hot path.
/// @details
/// Measures the cost of invoking info-level log callsites when the runtime log
/// level hides those messages (no formatting/sink emission expected).

// Run a simple suppressed-path throughput measurement for logger calls.
int main() {
  // 200k iterations provide a stable microsecond-scale measurement while
  // keeping this benchmark fast enough for local CI/debug runs.
  constexpr int iterations = 200000;
  cudaq::qec::detail::clearForwarder();
  cudaq::qec::detail::setLogLevel(cudaq::qec::detail::LogLevel::warn);

  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i)
    CUDAQX_INFO("suppressed-path {}", i);
  const auto end = std::chrono::steady_clock::now();

  const auto elapsedUs =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count();
  std::cout << "suppressed_path_iterations=" << iterations
            << " elapsed_us=" << elapsedUs << '\n';
  return 0;
} // end - main()
