/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// Two-process realtime decoding-server test: this process runs only the
/// caller (simulated QPU) side; the decoder lives in a separate
/// qec_decoding_daemon process reached through a CUDA-Q device_call
/// channel.  Both the decoder and the transport are configuration:
///   - the decoder comes from the YAML config file handed to the daemon
///     (swapping decoders is a config-file change, not a code change);
///   - the transport defaults to `udp` (loopback; runs anywhere) and can be
///     switched to the CPU RoCE RDMA wire with
///       QEC_DECODING_SERVER_TRANSPORT=cpu_roce
///     plus the RDMA topology env vars shared with CUDA-Q's
///     CpuRoceChannelTester:
///       CUDAQ_CPU_ROCE_TEST_CHANNEL_DEVICE / CUDAQ_CPU_ROCE_TEST_CHANNEL_IP
///       CUDAQ_CPU_ROCE_TEST_DAEMON_DEVICE  / CUDAQ_CPU_ROCE_TEST_DAEMON_IP
///
/// The daemon is spawned as a subprocess, its ephemeral port read from the
/// QEC_DECODING_DAEMON_READY stdout line, and its dispatch count (printed at
/// shutdown) is the proof the device_calls crossed the process boundary --
/// there is no decoder configured in this process at all.
///
/// The kernel's block/syndrome size and expected correction must stay
/// consistent with the H/O/D matrices in the config file (3-bit identity:
/// syndrome bit 1 set -> correction bit 1 set for any sane decoder).

#include "cudaq.h"
#include "cudaq/qec/realtime/decoding.h"
#include "cudaq/realtime.h"
#include <fstream>
#include <gtest/gtest.h>

#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr std::uint64_t kRunShots = 1;
constexpr std::int64_t kExpectedCorrection = 1;

__qpu__ std::int64_t decoding_server_udp_kernel() {
  constexpr std::uint64_t kKernelDecoderId = 0;
  constexpr std::uint64_t kKernelBlockSize = 3;
  constexpr std::uint64_t kKernelSyndromeSize = 3;
  constexpr std::uint64_t kKernelSyndromeTag = 1;
  constexpr std::size_t kKernelActiveSyndromeIndex = 1;

  cudaq::qec::decoding::reset_decoder(/*decoder_id=*/kKernelDecoderId);

  std::vector<bool> syndrome(kKernelSyndromeSize);
  for (std::size_t i = 0; i < kKernelSyndromeSize; ++i)
    syndrome[i] = false;
  syndrome[kKernelActiveSyndromeIndex] = true;
  cudaq::qec::decoding::enqueue_syndromes_test(
      /*decoder_id=*/kKernelDecoderId, syndrome, /*tag=*/kKernelSyndromeTag);

  auto corrections = cudaq::qec::decoding::get_corrections(
      /*decoder_id=*/kKernelDecoderId, /*return_size=*/kKernelBlockSize,
      /*reset=*/true);
  return corrections[kKernelActiveSyndromeIndex] ? std::int64_t{1}
                                                 : std::int64_t{0};
}

std::string env_or(const char *name, const std::string &fallback) {
  const char *value = std::getenv(name);
  return value && *value ? std::string(value) : fallback;
}

// The daemon binary path is baked in at configure time (the daemon is built
// from libs/qec/tools/decoding-server); QEC_DECODING_DAEMON overrides it. The
// example decoder configs are placed in the same directory as the daemon.
std::string daemon_path() {
  return env_or("QEC_DECODING_DAEMON", QEC_DECODING_DAEMON_PATH);
}

std::string daemon_dir() {
  const std::string path = daemon_path();
  const auto slash = path.find_last_of('/');
  return slash == std::string::npos ? "." : path.substr(0, slash);
}

// Spawns qec_decoding_daemon with the given decoder config file, hands
// back its UDP port, and collects its stdout (for the shutdown dispatch-count
// line).
class DaemonProcess {
public:
  bool start(const std::string &config_file, std::string &error,
             int ready_timeout_ms = 15000) {
    int out_pipe[2] = {-1, -1};
    if (::pipe(out_pipe) != 0) {
      error = "pipe() failed";
      return false;
    }
    pid = ::fork();
    if (pid < 0) {
      error = "fork() failed";
      return false;
    }
    if (pid == 0) {
      ::dup2(out_pipe[1], STDOUT_FILENO);
      ::close(out_pipe[0]);
      ::close(out_pipe[1]);
      const std::string daemon = daemon_path();
      const std::string config_arg =
          "--config=" + (!config_file.empty() && config_file[0] == '/'
                             ? config_file
                             : daemon_dir() + "/" + config_file);
      const std::string transport_arg =
          "--transport=" + env_or("QEC_DECODING_SERVER_TRANSPORT", "udp");
      const std::string device_arg =
          "--device=" + env_or("CUDAQ_CPU_ROCE_TEST_DAEMON_DEVICE", "mlx5_0");
      const std::string local_ip_arg =
          "--local-ip=" + env_or("CUDAQ_CPU_ROCE_TEST_DAEMON_IP", "10.0.0.2");
      ::execl(daemon.c_str(), daemon.c_str(), config_arg.c_str(),
              transport_arg.c_str(), device_arg.c_str(), local_ip_arg.c_str(),
              "--port=0", "--timeout=60", static_cast<char *>(nullptr));
      std::perror("execl qec_decoding_daemon");
      _exit(127);
    }
    ::close(out_pipe[1]);
    outFd = out_pipe[0];

    // Read stdout until the READY line (or the deadline; decoders that
    // build TensorRT engines at startup need more than the default 15 s).
    std::string ready_line;
    if (!readLineWithPrefix("QEC_DECODING_DAEMON_READY", ready_timeout_ms,
                            ready_line)) {
      error = "daemon did not print QEC_DECODING_DAEMON_READY; output so "
              "far: " +
              captured;
      return false;
    }
    if (std::sscanf(ready_line.c_str(), "QEC_DECODING_DAEMON_READY port=%hu",
                    &port) != 1) {
      error = "could not parse daemon port from: " + ready_line;
      return false;
    }
    return true;
  }

  // Terminate the daemon and return its dispatched-request count (-1 if the
  // shutdown line never appeared). Also captures the per-decoder-worker
  // concurrency high-water mark into max_concurrent_decoders.
  std::int64_t stopAndGetDispatchCount() {
    if (pid <= 0)
      return -1;
    ::kill(pid, SIGTERM);
    std::string line;
    std::int64_t count = -1;
    if (readLineWithPrefix("QEC_DECODING_DAEMON_DISPATCHED", 10000, line)) {
      long long parsed = -1;
      if (std::sscanf(line.c_str(),
                      "QEC_DECODING_DAEMON_DISPATCHED count=%lld",
                      &parsed) == 1)
        count = parsed;
    }
    if (readLineWithPrefix("QEC_DECODING_DAEMON_MAX_CONCURRENT_DECODERS",
                           5000, line)) {
      long long parsed = -1;
      if (std::sscanf(
              line.c_str(),
              "QEC_DECODING_DAEMON_MAX_CONCURRENT_DECODERS count=%lld",
              &parsed) == 1)
        max_concurrent_decoders = parsed;
    }
    int status = 0;
    ::waitpid(pid, &status, 0);
    pid = -1;
    if (outFd >= 0) {
      ::close(outFd);
      outFd = -1;
    }
    return count;
  }

  ~DaemonProcess() {
    if (pid > 0) {
      ::kill(pid, SIGKILL);
      int status = 0;
      ::waitpid(pid, &status, 0);
    }
    if (outFd >= 0)
      ::close(outFd);
  }

  std::uint16_t port = 0;
  std::string captured;
  std::int64_t max_concurrent_decoders = -1;

private:
  bool readLineWithPrefix(const char *prefix, int timeout_ms,
                          std::string &line_out) {
    std::string line;
    const auto deadline_ms = timeout_ms;
    int waited_ms = 0;
    while (waited_ms < deadline_ms) {
      pollfd pfd{outFd, POLLIN, 0};
      const int ready = ::poll(&pfd, 1, 100);
      waited_ms += 100;
      if (ready <= 0)
        continue;
      char c = 0;
      while (::read(outFd, &c, 1) == 1) {
        captured.push_back(c);
        if (c == '\n') {
          if (line.rfind(prefix, 0) == 0) {
            line_out = line;
            return true;
          }
          line.clear();
        } else {
          line.push_back(c);
        }
        pollfd inner{outFd, POLLIN, 0};
        if (::poll(&inner, 1, 0) <= 0)
          break;
      }
    }
    return false;
  }

  pid_t pid = -1;
  int outFd = -1;
};

struct RealtimeGuard {
  bool armed = false;
  ~RealtimeGuard() {
    if (armed)
      cudaq::realtime::finalize();
  }
};

} // namespace

// Caller-side device_call channel arguments for the selected transport.
std::vector<std::string> channel_arguments(std::uint16_t daemon_port) {
  const std::string transport = env_or("QEC_DECODING_SERVER_TRANSPORT", "udp");
  if (transport == "cpu_roce") {
    // The daemon's READY port is its TCP rendezvous port; the RDMA topology
    // comes from the same env vars CUDA-Q's CpuRoceChannelTester uses.
    return {
        "--cudaq-device-call=cpu_roce",
        "ib-device=" + env_or("CUDAQ_CPU_ROCE_TEST_CHANNEL_DEVICE", "mlx5_0"),
        "local-ip=" + env_or("CUDAQ_CPU_ROCE_TEST_CHANNEL_IP", "10.0.0.1"),
        "rendezvous-host=" +
            env_or("CUDAQ_CPU_ROCE_TEST_DAEMON_IP", "10.0.0.2"),
        "rendezvous-port=" + std::to_string(daemon_port)};
  }
  return {"--cudaq-device-call=udp", "udp-host=127.0.0.1",
          "udp-port=" + std::to_string(daemon_port)};
}

// Runs the full two-process round-trip against a daemon configured with
// `config_file`. Each invocation spawns a fresh daemon on an ephemeral port.
void run_two_process_decode_test(const std::string &config_file) {
  DaemonProcess daemon;
  std::string error;
  ASSERT_TRUE(daemon.start(config_file, error)) << error;

  std::vector<std::string> args = {"test_decoding_server"};
  for (auto &arg : channel_arguments(daemon.port))
    args.push_back(std::move(arg));
  std::vector<char *> argv;
  for (auto &arg : args)
    argv.push_back(arg.data());
  argv.push_back(nullptr);
  int argc = static_cast<int>(args.size());
  cudaq::realtime::initialize(argc, argv.data());
  RealtimeGuard realtime_guard{true};

  const auto results = cudaq::run(kRunShots, decoding_server_udp_kernel);
  ASSERT_EQ(results.size(), kRunShots);
  EXPECT_EQ(results[0], kExpectedCorrection);

  // Two-process self-verification: the decode can only have happened in the
  // daemon (this process configured no decoders), and the daemon's dispatch
  // counter proves the device_calls crossed the UDP transport. Three calls:
  // reset_decoder, enqueue_syndromes, get_corrections.
  const std::int64_t dispatched = daemon.stopAndGetDispatchCount();
  EXPECT_GE(dispatched, 3) << "daemon output:\n" << daemon.captured;
}

TEST(DecodingServerUdp, TwoProcessHostDispatch) {
  run_two_process_decode_test("decoding_server_config.yaml");
}

TEST(DecodingServerUdp, TwoProcessHostDispatchMultiErrorLut) {
  run_two_process_decode_test("decoding_server_config_multi_error_lut.yaml");
}

// ---------------------------------------------------------------------------
// Two decoders (two logical qubits) in ONE daemon, driven from ONE __qpu__
// kernel: qubit A uses decoder 0 and qubit B uses decoder 1, with different
// active syndrome bits. On the server each decoder executes on its own
// per-decoder worker thread; the daemon's shutdown
// QEC_DECODING_DAEMON_MAX_CONCURRENT_DECODERS line reports the busy
// high-water mark of those workers.
// ---------------------------------------------------------------------------

__qpu__ std::int64_t dual_decoding_server_udp_kernel() {
  constexpr std::uint64_t kDecoderA = 0;
  constexpr std::uint64_t kDecoderB = 1;
  constexpr std::uint64_t kBlockSize = 3;
  constexpr std::uint64_t kSyndromeSize = 3;
  constexpr std::size_t kActiveA = 1;
  constexpr std::size_t kActiveB = 2;

  cudaq::qec::decoding::reset_decoder(kDecoderA);
  cudaq::qec::decoding::reset_decoder(kDecoderB);

  std::vector<bool> syndrome_a(kSyndromeSize);
  std::vector<bool> syndrome_b(kSyndromeSize);
  for (std::size_t i = 0; i < kSyndromeSize; ++i) {
    syndrome_a[i] = false;
    syndrome_b[i] = false;
  }
  syndrome_a[kActiveA] = true;
  syndrome_b[kActiveB] = true;
  cudaq::qec::decoding::enqueue_syndromes_test(kDecoderA, syndrome_a,
                                               /*tag=*/1);
  cudaq::qec::decoding::enqueue_syndromes_test(kDecoderB, syndrome_b,
                                               /*tag=*/1);

  auto corr_a = cudaq::qec::decoding::get_corrections(kDecoderA, kBlockSize,
                                                      /*reset=*/true);
  auto corr_b = cudaq::qec::decoding::get_corrections(kDecoderB, kBlockSize,
                                                      /*reset=*/true);
  std::int64_t out = 0;
  if (corr_a[kActiveA])
    out = out + 1; // bit 0: decoder 0 corrected its active bit
  if (corr_b[kActiveB])
    out = out + 2; // bit 1: decoder 1 corrected its active bit
  if (corr_a[kActiveB])
    out = out + 4; // bit 2 set = cross-talk (decoder 0 saw B's syndrome)
  if (corr_b[kActiveA])
    out = out + 8; // bit 3 set = cross-talk (decoder 1 saw A's syndrome)
  return out;
}

TEST(DecodingServerUdp, TwoProcessHostDispatchDualDecoders) {
  // Two identical 3-bit-identity pymatching decoders (ids 0 and 1) in one
  // daemon -- one per logical qubit.
  const std::string config_path =
      ::testing::TempDir() + "/decoding_server_dual_config.yaml";
  {
    std::ofstream config_file(config_path);
    config_file << "decoders:\n";
    for (int id = 0; id < 2; ++id) {
      config_file << "  - id: " << id << "\n"
                  << "    type: pymatching\n"
                  << "    block_size: 3\n"
                  << "    syndrome_size: 3\n"
                  << "    H_sparse: [0, -1, 1, -1, 2, -1]\n"
                  << "    O_sparse: [0, -1, 1, -1, 2, -1]\n"
                  << "    D_sparse: [0, -1, 1, -1, 2, -1]\n"
                  << "    decoder_custom_args:\n"
                  << "      merge_strategy: smallest_weight\n"
                  << "      error_rate_vec: [0.1, 0.1, 0.1]\n";
    }
  }

  DaemonProcess daemon;
  std::string error;
  ASSERT_TRUE(daemon.start(config_path, error)) << error;

  std::vector<std::string> args = {"test_decoding_server"};
  for (auto &arg : channel_arguments(daemon.port))
    args.push_back(std::move(arg));
  std::vector<char *> argv;
  for (auto &arg : args)
    argv.push_back(arg.data());
  argv.push_back(nullptr);
  int argc = static_cast<int>(args.size());
  cudaq::realtime::initialize(argc, argv.data());
  RealtimeGuard realtime_guard{true};

  const auto results = cudaq::run(kRunShots, dual_decoding_server_udp_kernel);
  ASSERT_EQ(results.size(), kRunShots);
  EXPECT_EQ(results[0], 3);

  // Six calls crossed the wire (reset/enqueue/get per decoder), and both
  // decoders' execution workers ran (high-water mark >= 1; == 2 when the
  // decodes genuinely overlapped, which tiny identity decodes need not).
  const std::int64_t dispatched = daemon.stopAndGetDispatchCount();
  EXPECT_GE(dispatched, 6) << "daemon output:\n" << daemon.captured;
  EXPECT_GE(daemon.max_concurrent_decoders, 1)
      << "daemon output:\n" << daemon.captured;
}
