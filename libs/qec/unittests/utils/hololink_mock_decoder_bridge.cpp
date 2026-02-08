/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file hololink_mock_decoder_bridge.cpp
/// @brief Bridge tool connecting Hololink GPU-RoCE Transceiver to the cudaq
///        realtime mock decoder via the dispatch kernel.
///
/// Architecture:
///   FPGA --(RDMA WRITE)--> [Hololink RX Kernel] --> rx_ring_flag/rx_ring_buf
///   [cudaq Dispatch Kernel]: polls rx_ring_flag, calls mock_decode_rpc,
///                            writes response to tx_ring_buf, signals tx_ring_flag
///   [Hololink TX Kernel]: polls tx_ring_flag, sends corrections via RDMA SEND
///
/// All three kernels run concurrently on the GPU. Hololink manages the RX and
/// TX kernels internally via blocking_monitor(). The dispatch kernel is launched
/// separately via the cudaq host API.
///
/// Usage (FPGA mode):
///   ./hololink_mock_decoder_bridge \
///       --device=rocep1s0f0 \
///       --peer-ip=10.0.0.2 \
///       --remote-qp=2 \
///       --gpu=0 \
///       --config=path/to/config_multi_err_lut.yml \
///       --syndromes=path/to/syndromes_multi_err_lut.txt \
///       --timeout=60

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>

// cuda-quantum host API
#include "cudaq/nvqlink/daemon/dispatcher/cudaq_realtime.h"

// cuda-quantum RPC types/hash helper
#include "cudaq/nvqlink/daemon/dispatcher/dispatch_kernel_launch.h"

// cudaqx mock decoder (host-callable wrappers)
#include "cudaq/qec/realtime/mock_decode_handler.cuh"

// Shared setup helpers (config parsing, syndrome loading, GPU context, launch wrapper)
#include "cudaq/qec/realtime/mock_decode_setup.h"

// Hololink wrapper (C interface to avoid fmt conflicts)
#include "hololink_wrapper.h"

//==============================================================================
// CUDA Error Checking
//==============================================================================

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "    \
                << cudaGetErrorString(err) << std::endl;                       \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

//==============================================================================
// Global Signal Handler
//==============================================================================

static std::atomic<bool> g_shutdown{false};

static void signal_handler(int) { g_shutdown = true; }

//==============================================================================
// Command-Line Arguments
//==============================================================================

struct BridgeArgs {
  std::string device = "rocep1s0f0"; // IB device name
  std::string peer_ip = "10.0.0.2";  // FPGA/emulator IP
  uint32_t remote_qp = 0x2;          // Remote QP (FPGA default: 2)
  int gpu_id = 0;                     // GPU device ID
  std::string config_path;            // Path to config YAML
  std::string syndrome_path;          // Path to syndromes text file
  int timeout_sec = 60;              // Runtime timeout in seconds
  size_t page_size = 256;            // Ring buffer slot size (bytes)
  unsigned num_pages = 64;           // Number of ring buffer slots
  bool exchange_qp = false;          // Use QP exchange protocol (emulator mode)
  int exchange_port = 12345;         // TCP port for QP exchange
};

static BridgeArgs parse_args(int argc, char *argv[]) {
  BridgeArgs args;
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg.find("--device=") == 0)
      args.device = arg.substr(9);
    else if (arg.find("--peer-ip=") == 0)
      args.peer_ip = arg.substr(10);
    else if (arg.find("--remote-qp=") == 0)
      args.remote_qp = std::stoul(arg.substr(12), nullptr, 0);
    else if (arg.find("--gpu=") == 0)
      args.gpu_id = std::stoi(arg.substr(6));
    else if (arg.find("--config=") == 0)
      args.config_path = arg.substr(9);
    else if (arg.find("--syndromes=") == 0)
      args.syndrome_path = arg.substr(12);
    else if (arg.find("--timeout=") == 0)
      args.timeout_sec = std::stoi(arg.substr(10));
    else if (arg.find("--page-size=") == 0)
      args.page_size = std::stoull(arg.substr(12));
    else if (arg.find("--num-pages=") == 0)
      args.num_pages = std::stoul(arg.substr(12));
    else if (arg == "--exchange-qp")
      args.exchange_qp = true;
    else if (arg.find("--exchange-port=") == 0)
      args.exchange_port = std::stoi(arg.substr(16));
    else if (arg == "--help" || arg == "-h") {
      std::cout
          << "Usage: " << argv[0] << " [options]\n"
          << "\n"
          << "Bridge connecting Hololink GPU-RoCE Transceiver to cudaq mock "
             "decoder.\n"
          << "\n"
          << "Options:\n"
          << "  --device=NAME         IB device (default: rocep1s0f0)\n"
          << "  --peer-ip=ADDR        FPGA/emulator IP (default: 10.0.0.2)\n"
          << "  --remote-qp=N         Remote QP number (default: 0x2 for "
             "FPGA)\n"
          << "  --gpu=N               GPU device ID (default: 0)\n"
          << "  --config=PATH         Path to config YAML (required)\n"
          << "  --syndromes=PATH      Path to syndromes text file (required)\n"
          << "  --timeout=N           Timeout in seconds (default: 60)\n"
          << "  --page-size=N         Ring buffer slot size (default: 256)\n"
          << "  --num-pages=N         Number of ring buffer slots (default: "
             "64)\n"
          << "  --exchange-qp         Enable QP exchange protocol (emulator "
             "mode)\n"
          << "  --exchange-port=N     TCP port for QP exchange (default: "
             "12345)\n";
      exit(0);
    }
  }
  return args;
}

//==============================================================================
// MAIN
//==============================================================================

int main(int argc, char *argv[]) {
  signal(SIGINT, signal_handler);
  signal(SIGTERM, signal_handler);

  try {
    auto args = parse_args(argc, argv);

    std::cout << "=== Hololink Mock Decoder Bridge ===" << std::endl;
    std::cout << "Device: " << args.device << std::endl;
    std::cout << "Peer IP: " << args.peer_ip << std::endl;
    std::cout << "Remote QP: 0x" << std::hex << args.remote_qp << std::dec
              << std::endl;
    std::cout << "GPU: " << args.gpu_id << std::endl;

    // Validate required arguments
    if (args.config_path.empty()) {
      std::cerr << "ERROR: --config=PATH is required" << std::endl;
      return 1;
    }
    if (args.syndrome_path.empty()) {
      std::cerr << "ERROR: --syndromes=PATH is required" << std::endl;
      return 1;
    }

    //==========================================================================
    // [1/5] Initialize CUDA and load config
    //==========================================================================
    std::cout << "\n[1/5] Initializing CUDA and loading config..." << std::endl;

    CUDA_CHECK(cudaSetDevice(args.gpu_id));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, args.gpu_id));
    std::cout << "  GPU: " << prop.name << std::endl;

    // Load config file
    std::ifstream config_file(args.config_path);
    if (!config_file.good()) {
      std::cerr << "ERROR: Cannot open config file: " << args.config_path
                << std::endl;
      return 1;
    }
    std::string config_content((std::istreambuf_iterator<char>(config_file)),
                               std::istreambuf_iterator<char>());

    std::size_t syndrome_size =
        cudaq::qec::realtime::parse_scalar(config_content, "syndrome_size");
    std::size_t block_size =
        cudaq::qec::realtime::parse_scalar(config_content, "block_size");
    if (syndrome_size == 0 || block_size == 0) {
      std::cerr << "ERROR: Invalid config (syndrome_size=" << syndrome_size
                << ", block_size=" << block_size << ")" << std::endl;
      return 1;
    }
    std::cout << "  syndrome_size: " << syndrome_size << std::endl;
    std::cout << "  block_size: " << block_size << std::endl;

    // Load syndromes for mock decoder lookup table
    auto syndromes =
        cudaq::qec::realtime::load_syndromes(args.syndrome_path, syndrome_size);
    if (syndromes.empty()) {
      std::cerr << "ERROR: No syndrome data loaded from: "
                << args.syndrome_path << std::endl;
      return 1;
    }
    std::cout << "  Loaded " << syndromes.size() << " syndrome entries"
              << std::endl;

    //==========================================================================
    // [2/5] Set up mock decoder context on GPU
    //==========================================================================
    std::cout << "\n[2/5] Setting up mock decoder context..." << std::endl;

    cudaq::qec::realtime::MockDecoderGpuResources gpu_resources;
    cudaError_t ctx_err =
        cudaq::qec::realtime::setup_mock_decoder_from_syndromes(
            syndromes, syndrome_size, gpu_resources);
    if (ctx_err != cudaSuccess) {
      std::cerr << "ERROR: Failed to set up mock decoder on GPU: "
                << cudaGetErrorString(ctx_err) << std::endl;
      return 1;
    }
    std::cout << "  Mock decoder context initialized" << std::endl;

    //==========================================================================
    // [3/5] Create Hololink transceiver (RX + TX kernels)
    //==========================================================================
    std::cout << "\n[3/5] Creating Hololink transceiver..." << std::endl;

    // Frame size = enough to hold RPCHeader + syndrome data
    std::size_t frame_size =
        sizeof(cudaq::nvqlink::RPCHeader) +
        std::max(syndrome_size, static_cast<std::size_t>(256));

    // Ensure page_size is at least as large as frame_size
    if (args.page_size < frame_size) {
      std::cout << "  Adjusting page_size from " << args.page_size << " to "
                << frame_size << " to fit frame" << std::endl;
      args.page_size = frame_size;
    }

    std::cout << "  Frame size: " << frame_size << " bytes" << std::endl;
    std::cout << "  Page size: " << args.page_size << " bytes" << std::endl;
    std::cout << "  Num pages: " << args.num_pages << std::endl;

    // Create transceiver in deferred connection mode ("0.0.0.0") so we can
    // specify the correct remote QP number.  The default hardcodes QP=0x2
    // (physical FPGA), but in emulator mode the QP is dynamic.
    hololink_transceiver_t transceiver = hololink_create_transceiver(
        args.device.c_str(), 1, // ib_port
        frame_size, args.page_size, args.num_pages,
        "0.0.0.0",  // deferred connection
        0, // forward = false
        1, // rx_only = true
        1  // tx_only = true
    );

    if (!transceiver) {
      std::cerr << "ERROR: Failed to create Hololink transceiver" << std::endl;
      return 1;
    }

    if (!hololink_start(transceiver)) {
      std::cerr << "ERROR: Failed to start Hololink transceiver" << std::endl;
      hololink_destroy_transceiver(transceiver);
      return 1;
    }

    // Now connect the QP to the actual remote peer
    {
      uint8_t remote_gid[16] = {};
      // Build IPv4-mapped GID: ::ffff:a.b.c.d
      remote_gid[10] = 0xff;
      remote_gid[11] = 0xff;
      inet_pton(AF_INET, args.peer_ip.c_str(), &remote_gid[12]);

      std::cout << "  Connecting QP to remote QP 0x" << std::hex
                << args.remote_qp << std::dec
                << " at " << args.peer_ip << "..." << std::endl;

      if (!hololink_reconnect_qp(transceiver, remote_gid, args.remote_qp)) {
        std::cerr << "ERROR: Failed to connect QP to remote peer" << std::endl;
        hololink_destroy_transceiver(transceiver);
        return 1;
      }
      std::cout << "  QP connected to remote peer" << std::endl;
    }

    // Get QP info (for display and potential future use by FPGA stimulus tool)
    uint32_t our_qp = hololink_get_qp_number(transceiver);
    uint32_t our_rkey = hololink_get_rkey(transceiver);
    uint64_t our_buffer = hololink_get_buffer_addr(transceiver);

    std::cout << "  Hololink QP Number: 0x" << std::hex << our_qp << std::dec
              << std::endl;
    std::cout << "  Hololink RKey: " << our_rkey << std::endl;
    std::cout << "  Hololink Buffer Addr: 0x" << std::hex << our_buffer
              << std::dec << std::endl;

    // Get ring buffer pointers from Hololink
    uint8_t *rx_ring_data = reinterpret_cast<uint8_t *>(
        hololink_get_rx_ring_data_addr(transceiver));
    uint64_t *rx_ring_flag = hololink_get_rx_ring_flag_addr(transceiver);
    uint8_t *tx_ring_data = reinterpret_cast<uint8_t *>(
        hololink_get_tx_ring_data_addr(transceiver));
    uint64_t *tx_ring_flag = hololink_get_tx_ring_flag_addr(transceiver);

    std::cout << "  RX ring data: " << (void *)rx_ring_data << std::endl;
    std::cout << "  RX ring flag: " << (void *)rx_ring_flag << std::endl;
    std::cout << "  TX ring data: " << (void *)tx_ring_data << std::endl;
    std::cout << "  TX ring flag: " << (void *)tx_ring_flag << std::endl;

    if (!rx_ring_data || !rx_ring_flag || !tx_ring_data || !tx_ring_flag) {
      std::cerr << "ERROR: Failed to get ring buffer pointers" << std::endl;
      hololink_destroy_transceiver(transceiver);
      return 1;
    }

    //==========================================================================
    // Force eager CUDA module loading via occupancy queries.
    // Without this, lazy module loading can deadlock when persistent kernels
    // (dispatch, rx_only, tx_only) are running and another module tries to load.
    //==========================================================================
    std::cout << "\n  Forcing CUDA module loading (occupancy queries)..."
              << std::endl;
    {
      // Dispatch kernel module
      int dispatch_blocks = 0;
      cudaError_t occ_err = cudaq_dispatch_kernel_query_occupancy(
          &dispatch_blocks, 1);
      if (occ_err != cudaSuccess) {
        std::cerr << "ERROR: Dispatch kernel occupancy query failed: "
                  << cudaGetErrorString(occ_err) << std::endl;
        return 1;
      }
      std::cout << "  Dispatch kernel occupancy: " << dispatch_blocks
                << " blocks/SM" << std::endl;

      // Hololink kernel module
      if (!hololink_query_kernel_occupancy()) {
        std::cerr << "ERROR: Hololink kernel occupancy query failed"
                  << std::endl;
        return 1;
      }
    }

    //==========================================================================
    // [4/5] Wire cudaq dispatch kernel to Hololink ring buffers
    //==========================================================================
    std::cout << "\n[4/5] Wiring cudaq dispatch kernel..." << std::endl;

    // Set up function table using library helper (no nvcc kernel launch needed)
    cudaq_function_entry_t *d_function_entries = nullptr;
    CUDA_CHECK(
        cudaMalloc(&d_function_entries, sizeof(cudaq_function_entry_t)));
    cudaq::qec::realtime::setup_mock_decode_function_table(d_function_entries);

    // Allocate control variables (mapped memory for host+device access)
    void *tmp_shutdown = nullptr;
    CUDA_CHECK(cudaHostAlloc(&tmp_shutdown, sizeof(int), cudaHostAllocMapped));
    volatile int *shutdown_flag = static_cast<volatile int *>(tmp_shutdown);
    void *tmp_d_shutdown = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&tmp_d_shutdown, tmp_shutdown, 0));
    volatile int *d_shutdown_flag = static_cast<volatile int *>(tmp_d_shutdown);
    *shutdown_flag = 0;
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(const_cast<int *>(d_shutdown_flag), &zero,
                          sizeof(int), cudaMemcpyHostToDevice));

    // Allocate stats
    uint64_t *d_stats = nullptr;
    CUDA_CHECK(cudaMalloc(&d_stats, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_stats, 0, sizeof(uint64_t)));

    // Host API wiring
    cudaq_dispatch_manager_t *manager = nullptr;
    cudaq_dispatcher_t *dispatcher = nullptr;

    if (cudaq_dispatch_manager_create(&manager) != CUDAQ_OK) {
      std::cerr << "ERROR: Failed to create dispatch manager" << std::endl;
      return 1;
    }

    cudaq_dispatcher_config_t config{};
    config.device_id = args.gpu_id;
    config.num_blocks = 1;
    config.threads_per_block = 32;
    config.num_slots = static_cast<uint32_t>(args.num_pages);
    config.slot_size = static_cast<uint32_t>(args.page_size);
    config.vp_id = 0;
    config.kernel_type = CUDAQ_KERNEL_REGULAR;
    config.dispatch_mode = CUDAQ_DISPATCH_DEVICE_CALL;

    if (cudaq_dispatcher_create(manager, &config, &dispatcher) != CUDAQ_OK) {
      std::cerr << "ERROR: Failed to create dispatcher" << std::endl;
      return 1;
    }

    // Wire the Hololink ring buffers to the dispatch kernel
    // This is the key symmetric wiring:
    //   - rx_flags/rx_data: from Hololink RX kernel (syndromes arrive here)
    //   - tx_flags/tx_data: to Hololink TX kernel (corrections go here)
    cudaq_ringbuffer_t ringbuffer{};
    ringbuffer.rx_flags = reinterpret_cast<volatile uint64_t *>(rx_ring_flag);
    ringbuffer.tx_flags = reinterpret_cast<volatile uint64_t *>(tx_ring_flag);
    ringbuffer.rx_data = rx_ring_data;
    ringbuffer.tx_data = tx_ring_data;
    ringbuffer.rx_stride_sz = args.page_size;
    ringbuffer.tx_stride_sz = args.page_size;

    if (cudaq_dispatcher_set_ringbuffer(dispatcher, &ringbuffer) != CUDAQ_OK) {
      std::cerr << "ERROR: Failed to set ringbuffer" << std::endl;
      return 1;
    }

    cudaq_function_table_t table{};
    table.entries = d_function_entries;
    table.count = 1;
    if (cudaq_dispatcher_set_function_table(dispatcher, &table) != CUDAQ_OK) {
      std::cerr << "ERROR: Failed to set function table" << std::endl;
      return 1;
    }

    if (cudaq_dispatcher_set_control(dispatcher, d_shutdown_flag, d_stats) !=
        CUDAQ_OK) {
      std::cerr << "ERROR: Failed to set control" << std::endl;
      return 1;
    }

    if (cudaq_dispatcher_set_launch_fn(
            dispatcher,
            &cudaq::qec::realtime::mock_decode_launch_dispatch_kernel) !=
        CUDAQ_OK) {
      std::cerr << "ERROR: Failed to set launch function" << std::endl;
      return 1;
    }

    // Start the dispatch kernel
    if (cudaq_dispatcher_start(dispatcher) != CUDAQ_OK) {
      std::cerr << "ERROR: Failed to start dispatcher" << std::endl;
      return 1;
    }
    std::cout << "  Dispatch kernel launched" << std::endl;

    // Check CUDA error state after dispatch kernel launch
    {
      cudaError_t post_dispatch_err = cudaPeekAtLastError();
      std::cout << "  CUDA state after dispatch launch: "
                << cudaGetErrorString(post_dispatch_err) << std::endl;
    }

    //==========================================================================
    // [5/5] Launch Hololink kernels and run
    //==========================================================================
    std::cout << "\n[5/5] Launching Hololink kernels..." << std::endl;

    // blocking_monitor launches the RX and TX GPU kernels and blocks until
    // close() is called. Run it in a separate thread.
    std::thread hololink_thread(
        [transceiver]() { hololink_blocking_monitor(transceiver); });

    // Brief wait for Hololink kernels to start
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    std::cout << "  Hololink RX+TX kernels started" << std::endl;

    // Check CUDA error state after Hololink kernel launches
    {
      cudaError_t post_hololink_err = cudaPeekAtLastError();
      std::cout << "  CUDA state after Hololink launch: "
                << cudaGetErrorString(post_hololink_err) << std::endl;
    }

    // Print QP info for use by the FPGA stimulus tool
    std::cout << "\n=== Bridge Ready ===" << std::endl;
    std::cout << "  QP Number: 0x" << std::hex << our_qp << std::dec
              << std::endl;
    std::cout << "  RKey: " << our_rkey << std::endl;
    std::cout << "  Buffer Addr: 0x" << std::hex << our_buffer << std::dec
              << std::endl;
    std::cout << "\nWaiting for FPGA data (Ctrl+C to stop, timeout="
              << args.timeout_sec << "s)..." << std::endl;

    //==========================================================================
    // Main run loop - monitor progress with diagnostics
    //==========================================================================
    // Create a dedicated CUDA stream for diagnostic memcpy operations.
    // The default (NULL) stream implicitly synchronizes with all other streams,
    // which would hang since the dispatch kernel is an infinite loop.
    cudaStream_t diag_stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&diag_stream,
                                         cudaStreamNonBlocking));

    {
      cudaError_t pre_loop_err = cudaPeekAtLastError();
      FILE *df = fopen("/tmp/bridge_diag.txt", "w");
      if (df) {
        fprintf(df, "ENTERING MAIN LOOP g_shutdown=%d cuda_state=%s\n",
                (int)g_shutdown, cudaGetErrorString(pre_loop_err));
        fflush(df);
        fclose(df);
      }
    }
    auto start_time = std::chrono::steady_clock::now();
    uint64_t last_processed = 0;
    int diag_count = 0;

    while (!g_shutdown) {
      auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                         std::chrono::steady_clock::now() - start_time)
                         .count();
      if (elapsed > args.timeout_sec) {
        std::cout << "\nTimeout reached (" << args.timeout_sec << "s)"
                  << std::endl;
        break;
      }

      // Diagnostic: read GPU ring buffer state
      if (diag_count < 30) {
        diag_count++;

        // Read stats
        uint64_t processed = 0;
        cudaMemcpyAsync(&processed, d_stats, sizeof(uint64_t),
                        cudaMemcpyDeviceToHost, diag_stream);

        // Read first 4 rx_flags and first 4 tx_flags
        uint64_t rx_flags_h[4] = {};
        uint64_t tx_flags_h[4] = {};
        cudaMemcpyAsync(rx_flags_h, rx_ring_flag, 4 * sizeof(uint64_t),
                        cudaMemcpyDeviceToHost, diag_stream);
        cudaMemcpyAsync(tx_flags_h, tx_ring_flag, 4 * sizeof(uint64_t),
                        cudaMemcpyDeviceToHost, diag_stream);

        // Read first 16 bytes of rx_ring_data slot 0
        uint32_t slot0[4] = {};
        cudaMemcpyAsync(slot0, rx_ring_data, 16,
                        cudaMemcpyDeviceToHost, diag_stream);

        cudaStreamSynchronize(diag_stream);

        FILE *df = fopen("/tmp/bridge_diag.txt", "a");
        if (df) {
          fprintf(df, "[DIAG %d] dispatched=%lu"
                      " rx_flags=[%lx %lx %lx %lx]"
                      " tx_flags=[%lx %lx %lx %lx]"
                      " slot0=[%08x %08x %08x %08x]\n",
                  diag_count, (unsigned long)processed,
                  (unsigned long)rx_flags_h[0], (unsigned long)rx_flags_h[1],
                  (unsigned long)rx_flags_h[2], (unsigned long)rx_flags_h[3],
                  (unsigned long)tx_flags_h[0], (unsigned long)tx_flags_h[1],
                  (unsigned long)tx_flags_h[2], (unsigned long)tx_flags_h[3],
                  slot0[0], slot0[1], slot0[2], slot0[3]);
          fflush(df);
          fclose(df);
        }
      }

      // Print progress every 5 seconds (use async memcpy to avoid hang)
      if (elapsed > 0 && elapsed % 5 == 0) {
        uint64_t processed = 0;
        cudaMemcpyAsync(&processed, d_stats, sizeof(uint64_t),
                        cudaMemcpyDeviceToHost, diag_stream);
        cudaStreamSynchronize(diag_stream);
        if (processed != last_processed) {
          std::cout << "  [" << elapsed << "s] Processed " << processed
                    << " packets" << std::endl;
          last_processed = processed;
        }
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    //==========================================================================
    // Shutdown
    //==========================================================================
    std::cout << "\n=== Shutting down ===" << std::endl;

    // Destroy diag stream before stopping dispatcher
    if (diag_stream) {
      cudaStreamDestroy(diag_stream);
      diag_stream = nullptr;
    }

    // Signal dispatch kernel to stop
    *shutdown_flag = 1;
    __sync_synchronize();
    cudaq_dispatcher_stop(dispatcher);

    // Get final stats (safe now that dispatch kernel is stopped)
    uint64_t total_processed = 0;
    cudaq_dispatcher_get_processed(dispatcher, &total_processed);
    std::cout << "  Total packets processed: " << total_processed << std::endl;

    // Close Hololink (signals blocking_monitor to exit)
    hololink_close(transceiver);

    // Wait for Hololink thread
    if (hololink_thread.joinable()) {
      hololink_thread.join();
    }

    // Cleanup
    cudaq_dispatcher_destroy(dispatcher);
    cudaq_dispatch_manager_destroy(manager);
    hololink_destroy_transceiver(transceiver);

    if (shutdown_flag)
      cudaFreeHost(const_cast<int *>(shutdown_flag));
    if (d_stats)
      cudaFree(d_stats);
    if (d_function_entries)
      cudaFree(d_function_entries);

    gpu_resources.cleanup();

    std::cout << "\n*** Bridge shutdown complete ***" << std::endl;
    return 0;

  } catch (const std::exception &e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
    return 1;
  }
}
