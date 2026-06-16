/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq.h"
#include "cudaq/qec/realtime/decoding.h"
#include "cudaq/qec/realtime/decoding_config.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <variant>
#include <vector>

namespace {

__qpu__ std::vector<bool>
decode_syndrome_via_qpu_device_call(std::vector<bool> syndrome) {
  cudaq::qec::decoding::enqueue_syndromes_test(/*decoder_id=*/0, syndrome,
                                               /*tag=*/1);
  return cudaq::qec::decoding::get_corrections(/*decoder_id=*/0,
                                               /*return_size=*/3,
                                               /*reset=*/true);
}

__qpu__ std::vector<bool> read_corrections_via_qpu_device_call() {
  return cudaq::qec::decoding::get_corrections(/*decoder_id=*/0,
                                               /*return_size=*/3,
                                               /*reset=*/false);
}

cudaq::qec::decoding::config::multi_decoder_config make_pymatching_config() {
  namespace config = cudaq::qec::decoding::config;

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

} // namespace

TEST(PyMatchingQpuDeviceCall, DecodesViaHostDispatchSharedMemory) {
  namespace config = cudaq::qec::decoding::config;

  auto decoder_config = make_pymatching_config();
  ASSERT_EQ(config::configure_decoders(decoder_config), 0);
  struct DecoderConfigGuard {
    ~DecoderConfigGuard() { cudaq::qec::decoding::config::finalize_decoders(); }
  } decoder_guard;

  char program[] = "test_pymatching_qpu_device_call";
  char channel[] = "--cudaq-device-call=host-dispatch";
  char slots[] = "--cudaq-device-call-slots=4";
  char slot_size[] = "--cudaq-device-call-slot-size=256";
  char *argv[] = {program, channel, slots, slot_size};
  cudaq::realtime::initialize(4, argv);
  struct RealtimeGuard {
    ~RealtimeGuard() { cudaq::realtime::finalize(); }
  } realtime_guard;

  const auto decoded_runs = cudaq::run(1, decode_syndrome_via_qpu_device_call,
                                       std::vector<bool>{false, true, false});
  ASSERT_EQ(decoded_runs.size(), 1);
  EXPECT_EQ(decoded_runs.front(), (std::vector<bool>{false, true, false}));

  const auto reset_runs = cudaq::run(1, read_corrections_via_qpu_device_call);
  ASSERT_EQ(reset_runs.size(), 1);
  EXPECT_EQ(reset_runs.front(), (std::vector<bool>{false, false, false}));
}
