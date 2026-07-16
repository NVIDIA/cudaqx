/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025-2026 NVIDIA Corporation & Affiliates.                    *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file hololink_qldpc_graph_decoder_bridge.cpp
/// @brief QLDPC BP decoder bridge adapter using CPU-launched CUDA graph
///        dispatch (HOST_LOOP) with the generic Hololink bridge skeleton.
///
/// This thin adapter:
///   1. Parses --config argument (Relay BP config YAML)
///   2. Loads the decoder config, builds the H tensor, creates the decoder
///   3. Calls capture_decode_graph() to get a CUDA graph + mailbox
///   4. Builds a cudaq_function_entry_t with the graph_exec
///   5. Configures BridgeConfig for HOST_LOOP backend
///   6. Delegates all Hololink / dispatcher plumbing to bridge_run()
///
/// The HOST_LOOP dispatcher (CPU thread) polls Hololink ring flags, then
/// launches the CUDA graph for each incoming RPC request.  This avoids
/// the 120-outstanding-graph limit of device-side cudaGraphLaunch.
///
/// Requires a Grace-based system (DGX Spark / GB200) where GPU memory
/// is CPU-accessible via NVLink-C2C.

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cudaq/realtime/hololink_bridge_common.h"

#include "cudaq/qec/decoder.h"
#include "cudaq/qec/realtime/decoding_config.h"
#include "cudaq/qec/realtime/graph_resources.h"
#include "cudaq/qec/realtime/sparse_to_csr.h"

static std::string read_file(const std::string &path) {
  std::ifstream f(path);
  if (!f.is_open()) {
    std::cerr << "ERROR: Cannot open file: " << path << std::endl;
    return {};
  }
  return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

int main(int argc, char *argv[]) {
  std::string config_path;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg.find("--config=") == 0)
      config_path = arg.substr(9);
    else if (arg == "--help" || arg == "-h") {
      std::cout
          << "Usage: " << argv[0] << " [options]\n\n"
          << "QLDPC BP decoder bridge: Hololink GPU-RoCE <-> HOST_LOOP "
             "graph dispatch.\n\n"
          << "Decoder options:\n"
          << "  --config=PATH         Path to Relay BP config YAML "
             "(required)\n\n"
          << "Bridge options (passed to generic skeleton):\n"
          << "  --device=NAME         IB device (default: rocep1s0f0)\n"
          << "  --peer-ip=ADDR        FPGA/emulator IP (default: 10.0.0.2)\n"
          << "  --remote-qp=N         Remote QP number (default: 0x2)\n"
          << "  --gpu=N               GPU device ID (default: 0)\n"
          << "  --timeout=N           Timeout in seconds (default: 60)\n"
          << "  --page-size=N         Ring buffer slot size (default: 384)\n"
          << "  --num-pages=N         Ring buffer slots (default: 64)\n"
          << "  --exchange-qp         Enable QP exchange (emulator mode)\n"
          << "  --exchange-port=N     QP exchange TCP port (default: "
             "12345)\n";
      return 0;
    }
  }

  if (config_path.empty()) {
    std::cerr << "ERROR: --config=PATH is required" << std::endl;
    return 1;
  }

  std::cout << "=== Hololink QLDPC BP Decoder Bridge (Graph Launch) ==="
            << std::endl;

  // ---- Load decoder config ----
  std::string yaml_str = read_file(config_path);
  if (yaml_str.empty())
    return 1;

  auto mdc = cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
      yaml_str);
  if (mdc.decoders.empty()) {
    std::cerr << "ERROR: No decoders found in config" << std::endl;
    return 1;
  }
  auto &dec = mdc.decoders[0];

  // ---- Build H tensor from sparse representation ----
  std::vector<uint32_t> h_row_ptr, h_col_idx;
  std::size_t h_rows = cudaq::qec::realtime::sparse_vec_to_csr(
      dec.H_sparse, h_row_ptr, h_col_idx);
  std::size_t bs = dec.block_size;
  std::size_t ss = dec.syndrome_size;

  cudaqx::tensor<uint8_t> H_tensor({ss, bs});
  for (std::size_t r = 0; r < ss; ++r)
    for (uint32_t j = h_row_ptr[r]; j < h_row_ptr[r + 1]; ++j)
      H_tensor.at({r, static_cast<std::size_t>(h_col_idx[j])}) = 1;

  // ---- Create decoder ----
  auto params = dec.decoder_custom_args_to_heterogeneous_map();
  auto decoder = cudaq::qec::decoder::get("nv-qldpc-decoder", H_tensor, params);
  if (!decoder) {
    std::cerr << "ERROR: Failed to create nv-qldpc-decoder" << std::endl;
    return 1;
  }

  decoder->set_D_sparse(dec.D_sparse);
  decoder->set_O_sparse(dec.O_sparse);

  // Derive num_measurements and num_observables for frame size calculation
  std::vector<uint32_t> d_rp, d_ci;
  cudaq::qec::realtime::sparse_vec_to_csr(dec.D_sparse, d_rp, d_ci);
  std::size_t num_measurements = 0;
  for (auto c : d_ci)
    num_measurements =
        std::max(num_measurements, static_cast<std::size_t>(c + 1));
  std::vector<uint32_t> o_rp, o_ci;
  std::size_t num_observables =
      cudaq::qec::realtime::sparse_vec_to_csr(dec.O_sparse, o_rp, o_ci);

  std::cout << "  block_size: " << bs << std::endl;
  std::cout << "  syndrome_size: " << ss << std::endl;
  std::cout << "  num_measurements: " << num_measurements << std::endl;
  std::cout << "  num_observables: " << num_observables << std::endl;
  (void)h_rows;

  // ---- Capture CUDA graph ----
  if (!decoder->supports_graph_dispatch()) {
    std::cerr << "ERROR: nv-qldpc-decoder does not support graph dispatch"
              << std::endl;
    return 1;
  }

  cudaq::realtime::BridgeConfig config;
  cudaq::realtime::parse_bridge_args(argc, argv, config);

  BRIDGE_CUDA_CHECK(cudaSetDevice(config.gpu_id));

  void *raw_res = decoder->capture_decode_graph(/*reserved_sms=*/2);
  if (!raw_res) {
    std::cerr << "ERROR: capture_decode_graph() returned null" << std::endl;
    return 1;
  }
  auto *graph_res =
      static_cast<cudaq::qec::realtime::graph_resources *>(raw_res);

  std::cout << "  Graph captured: function_id=0x" << std::hex
            << graph_res->function_id << std::dec << std::endl;

  // ---- Configure HOST_LOOP bridge ----
  cudaq_function_entry_t entry{};
  entry.handler.graph_exec = graph_res->graph_exec;
  entry.function_id = graph_res->function_id;
  entry.dispatch_mode = CUDAQ_DISPATCH_GRAPH_LAUNCH;

  config.dispatch_path = CUDAQ_DISPATCH_PATH_HOST;
  config.h_function_entries = &entry;
  config.h_func_count = 1;
  config.h_mailbox = graph_res->h_mailbox;
  config.d_mailbox = graph_res->d_mailbox;

  config.frame_size = sizeof(cudaq::realtime::RPCHeader) +
                      std::max(num_measurements, num_observables);
  if (config.page_size < config.frame_size)
    config.page_size = config.frame_size;

  auto *decoder_ptr = decoder.get();
  config.cleanup_fn = [decoder_ptr, raw_res]() {
    decoder_ptr->release_decode_graph(raw_res);
  };

  return cudaq::realtime::bridge_run(config);
}
