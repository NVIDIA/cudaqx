/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file hololink_mock_decoder_bridge.cpp
/// @brief Mock decoder bridge adapter using the generic Hololink bridge
///        skeleton from cuda-quantum.
///
/// This thin adapter:
///   1. Parses --config and --syndromes arguments
///   2. Loads the mock decoder lookup table onto the GPU
///   3. Registers mock_decode_rpc in the dispatch function table
///   4. Delegates all Hololink / dispatch-kernel plumbing to bridge_run()
///
/// Usage:
///   ./hololink_mock_decoder_bridge \
///       --device=rocep1s0f0 --peer-ip=10.0.0.2 --remote-qp=0x2a8 \
///       --config=path/to/config.yml --syndromes=path/to/syndromes.txt

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

// Generic bridge skeleton (Hololink setup, dispatch wiring, main loop)
#include "cudaq/nvqlink/hololink_bridge_common.h"

// Mock decoder host-side helpers (config parsing, syndrome loading, GPU setup)
#include "cudaq/qec/realtime/mock_decode_setup.h"

// Mock decoder device code declarations (setup_mock_decode_function_table)
#include "cudaq/qec/realtime/mock_decode_handler.cuh"

int main(int argc, char *argv[]) {
  // ---- Parse tool-specific arguments ----
  std::string config_path;
  std::string syndrome_path;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg.find("--config=") == 0)
      config_path = arg.substr(9);
    else if (arg.find("--syndromes=") == 0)
      syndrome_path = arg.substr(12);
    else if (arg == "--help" || arg == "-h") {
      std::cout
          << "Usage: " << argv[0] << " [options]\n\n"
          << "Mock decoder bridge: Hololink GPU-RoCE <-> cudaq dispatch "
             "kernel.\n\n"
          << "Decoder options:\n"
          << "  --config=PATH         Path to config YAML (required)\n"
          << "  --syndromes=PATH      Path to syndromes text file (required)\n\n"
          << "Bridge options (passed to generic skeleton):\n"
          << "  --device=NAME         IB device (default: rocep1s0f0)\n"
          << "  --peer-ip=ADDR        FPGA/emulator IP (default: 10.0.0.2)\n"
          << "  --remote-qp=N         Remote QP number (default: 0x2)\n"
          << "  --gpu=N               GPU device ID (default: 0)\n"
          << "  --timeout=N           Timeout in seconds (default: 60)\n"
          << "  --page-size=N         Ring buffer slot size (default: 384)\n"
          << "  --num-pages=N         Ring buffer slots (default: 64)\n"
          << "  --exchange-qp         Enable QP exchange (emulator mode)\n"
          << "  --exchange-port=N     QP exchange TCP port (default: 12345)\n";
      return 0;
    }
  }

  if (config_path.empty()) {
    std::cerr << "ERROR: --config=PATH is required" << std::endl;
    return 1;
  }
  if (syndrome_path.empty()) {
    std::cerr << "ERROR: --syndromes=PATH is required" << std::endl;
    return 1;
  }

  // ---- Load config and syndromes ----
  std::cout << "=== Hololink Mock Decoder Bridge ===" << std::endl;

  std::ifstream config_file(config_path);
  if (!config_file.good()) {
    std::cerr << "ERROR: Cannot open config: " << config_path << std::endl;
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

  auto syndromes =
      cudaq::qec::realtime::load_syndromes(syndrome_path, syndrome_size);
  if (syndromes.empty()) {
    std::cerr << "ERROR: No syndromes loaded from: " << syndrome_path
              << std::endl;
    return 1;
  }
  std::cout << "  Loaded " << syndromes.size() << " syndrome entries"
            << std::endl;

  // ---- Populate BridgeConfig ----
  cudaq::nvqlink::BridgeConfig config;
  cudaq::nvqlink::parse_bridge_args(argc, argv, config);

  // Frame size = RPCHeader + syndrome data (minimum 256)
  config.frame_size =
      sizeof(cudaq::nvqlink::RPCHeader) +
      std::max(syndrome_size, static_cast<std::size_t>(256));

  // Ensure page_size fits the frame
  if (config.page_size < config.frame_size)
    config.page_size = config.frame_size;

  // ---- Set up mock decoder on GPU ----
  // (Must happen after CUDA init, but parse_bridge_args doesn't init CUDA.
  //  bridge_run() calls cudaSetDevice. We need to init before bridge_run
  //  because the function table setup requires GPU allocation.)
  BRIDGE_CUDA_CHECK(cudaSetDevice(config.gpu_id));

  cudaq::qec::realtime::MockDecoderGpuResources gpu_resources;
  cudaError_t ctx_err =
      cudaq::qec::realtime::setup_mock_decoder_from_syndromes(
          syndromes, syndrome_size, gpu_resources);
  if (ctx_err != cudaSuccess) {
    std::cerr << "ERROR: Mock decoder GPU setup failed: "
              << cudaGetErrorString(ctx_err) << std::endl;
    return 1;
  }
  std::cout << "  Mock decoder context initialized" << std::endl;

  // Register mock_decode_rpc in dispatch function table
  cudaq_function_entry_t *d_function_entries = nullptr;
  BRIDGE_CUDA_CHECK(
      cudaMalloc(&d_function_entries, sizeof(cudaq_function_entry_t)));
  cudaq::qec::realtime::setup_mock_decode_function_table(d_function_entries);

  config.d_function_entries = d_function_entries;
  config.func_count = 1;
  config.launch_fn = &cudaq::nvqlink::bridge_launch_dispatch_kernel;
  config.cleanup_fn = [d_function_entries, &gpu_resources]() {
    cudaFree(d_function_entries);
    gpu_resources.cleanup();
  };

  // ---- Run the bridge ----
  return cudaq::nvqlink::bridge_run(config);
}
