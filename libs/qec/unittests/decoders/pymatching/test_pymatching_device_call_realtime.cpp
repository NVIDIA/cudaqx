/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq.h"
#include "cudaq/qec/realtime/decoding.h"
#include "cudaq/qec/realtime/decoding_config.h"
#include "cudaq/realtime.h"

#include <cstdint>
#include <exception>
#include <iostream>
#include <variant>
#include <vector>

extern "C" void cudaqx_qec_realtime_device_call_service_force_link();

namespace {

namespace config = cudaq::qec::decoding::config;

config::multi_decoder_config make_config() {
  config::decoder_config decoder_config;
  decoder_config.id = 0;
  decoder_config.type = "pymatching";
  decoder_config.block_size = 3;
  decoder_config.syndrome_size = 3;
  decoder_config.H_sparse = {0, -1, 1, -1, 2, -1};
  decoder_config.O_sparse = {0, -1, 1, -1, 2, -1};
  decoder_config.D_sparse = {0, -1, 1, -1, 2, -1};

  decoder_config.decoder_custom_args = config::pymatching_config();
  auto &pymatching_config =
      std::get<config::pymatching_config>(decoder_config.decoder_custom_args);
  pymatching_config.error_rate_vec = std::vector<double>{0.1, 0.1, 0.1};
  pymatching_config.merge_strategy = "smallest_weight";

  config::multi_decoder_config multi_config;
  multi_config.decoders.push_back(decoder_config);
  return multi_config;
}

struct DecoderGuard {
  bool armed = false;
  ~DecoderGuard() {
    if (armed)
      config::finalize_decoders();
  }
};

struct RealtimeGuard {
  bool armed = false;
  ~RealtimeGuard() {
    if (armed)
      cudaq::realtime::finalize();
  }
};

__qpu__ std::int64_t pymatching_device_call_kernel() {
  cudaq::qec::decoding::reset_decoder(/*decoder_id=*/0);

  std::vector<bool> syndrome(3);
  syndrome[1] = true;
  cudaq::qec::decoding::enqueue_syndromes_test(
      /*decoder_id=*/0, syndrome, /*tag=*/1);

  auto corrections = cudaq::qec::decoding::get_corrections(
      /*decoder_id=*/0, /*return_size=*/3, /*reset=*/true);
  std::int64_t packed = 0;
  for (std::size_t i = 0; i < corrections.size(); ++i)
    if (corrections[i])
      packed |= (std::int64_t{1} << i);
  return packed;
}

} // namespace

int main(int argc, char **argv) {
  try {
    // Keep the service library loaded so CUDA-Q can discover its
    // cudaqGetDeviceCallServicePluginInfo symbol via dlsym(RTLD_DEFAULT).
    cudaqx_qec_realtime_device_call_service_force_link();

    auto decoder_config = make_config();
    if (config::configure_decoders(decoder_config) != 0) {
      std::cerr << "failed to configure PyMatching decoder\n";
      return 1;
    }
    DecoderGuard decoder_guard{true};

    cudaq::realtime::initialize(argc, argv);
    RealtimeGuard realtime_guard{true};

    const auto results = cudaq::run(1, pymatching_device_call_kernel);
    if (results.size() != 1) {
      std::cerr << "expected one result, got " << results.size() << "\n";
      return 1;
    }

    constexpr std::int64_t expected = 0b010;
    if (results[0] != expected) {
      std::cerr << "expected correction " << expected << ", got " << results[0]
                << "\n";
      return 1;
    }

    return 0;
  } catch (const std::exception &ex) {
    std::cerr << ex.what() << "\n";
    return 1;
  }
}
