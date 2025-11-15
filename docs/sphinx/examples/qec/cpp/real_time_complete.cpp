/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// Compile and run:
// nvq++ --target=stim -lcudaq-qec -lcudaq-qec-realtime-decoding -lcudaq-qec-realtime-decoding-simulation real_time_complete.cpp
// NOTE: This must be on one line for the CI system to parse it correctly.
// clang-format on

// [Begin Documentation]

// Simple 3-qubit repetition code with real-time decoding
// This is the most basic QEC example possible

#include "cudaq.h"
#include "cudaq/qec/code.h"
#include "cudaq/qec/experiments.h"
#include "cudaq/qec/pcm_utils.h"
#include "cudaq/qec/realtime/decoding.h"
#include "cudaq/qec/realtime/decoding_config.h"
#include <common/NoiseModel.h>
#include <fstream>

// [Begin Save DEM]
// Save decoder configuration to YAML file
void save_dem(const cudaq::qec::detector_error_model &dem,
              const std::string &filename) {
  // Create decoder config
  cudaq::qec::decoding::config::decoder_config config;
  config.id = 0;
  config.type = "multi_error_lut";
  config.block_size = dem.num_error_mechanisms();
  config.syndrome_size = dem.num_detectors();
  config.H_sparse = cudaq::qec::pcm_to_sparse_vec(dem.detector_error_matrix);
  config.O_sparse = cudaq::qec::pcm_to_sparse_vec(dem.observables_flips_matrix);

  // Calculate numRounds from DEM (we send 1 additional round, so add 1)
  uint64_t numSyndromesPerRound = 2; // Z0Z1 and Z1Z2
  auto numRounds = dem.num_detectors() / numSyndromesPerRound + 1;
  config.D_sparse = cudaq::qec::generate_timelike_sparse_detector_matrix(
      numSyndromesPerRound, numRounds, false);

  cudaq::qec::decoding::config::multi_error_lut_config lut_config;
  lut_config.lut_error_depth = 2;
  config.decoder_custom_args = lut_config;

  cudaq::qec::decoding::config::multi_decoder_config multi_config;
  multi_config.decoders.push_back(config);

  std::ofstream file(filename);
  file << multi_config.to_yaml_str(200);
  file.close();
  printf("Saved config to %s\n", filename.c_str());
}
// [End Save DEM]

// [Begin Load DEM]
// Load decoder configuration from YAML file
void load_dem(const std::string &filename) {
  std::ifstream file(filename);
  std::string yaml((std::istreambuf_iterator<char>(file)),
                   std::istreambuf_iterator<char>());
  auto config =
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(yaml);
  cudaq::qec::decoding::config::configure_decoders(config);
  printf("Loaded config from %s\n", filename.c_str());
}
// [End Load DEM]

// Prepare logical |0⟩
__qpu__ void prep0(cudaq::qec::patch logical) {
  for (std::size_t i = 0; i < logical.data.size(); ++i) {
    cudaq::reset(logical.data[i]);
  }
}

// Measure ZZ stabilizers for 3-qubit repetition code
__qpu__ std::vector<cudaq::measure_result>
measure_stabilizers(cudaq::qec::patch logical) {
  for (std::size_t i = 0; i < logical.ancz.size(); ++i) {
    cudaq::reset(logical.ancz[i]);
  }

  // Z0Z1 stabilizer
  cudaq::x<cudaq::ctrl>(logical.data[0], logical.ancz[0]);
  cudaq::x<cudaq::ctrl>(logical.data[1], logical.ancz[0]);

  // Z1Z2 stabilizer
  cudaq::x<cudaq::ctrl>(logical.data[1], logical.ancz[1]);
  cudaq::x<cudaq::ctrl>(logical.data[2], logical.ancz[1]);

  return {mz(logical.ancz[0]), mz(logical.ancz[1])};
}

// [Begin QEC Circuit]
// QEC circuit with real-time decoding
__qpu__ int64_t qec_circuit() {
  cudaq::qec::decoding::reset_decoder(0);

  cudaq::qvector data(3);
  cudaq::qvector ancz(2);
  cudaq::qvector ancx; // Empty for repetition code
  cudaq::qec::patch logical(data, ancx, ancz);

  prep0(logical);

  // 3 rounds of syndrome measurement
  for (int round = 0; round < 3; ++round) {
    auto syndromes = measure_stabilizers(logical);
    cudaq::qec::decoding::enqueue_syndromes(0, syndromes);
  }

  // Get corrections and apply them
  auto corrections = cudaq::qec::decoding::get_corrections(0, 3);
  for (std::size_t i = 0; i < 3; ++i) {
    if (corrections[i])
      cudaq::x(data[i]);
  }

  return cudaq::to_integer(mz(data));
}
// [End QEC Circuit]

int main() {
  auto code = cudaq::qec::get_code("repetition",
                                   cudaqx::heterogeneous_map{{"distance", 3}});

  // [Begin DEM Generation]
  // Step 1: Generate detector error model
  printf("Step 1: Generating DEM...\n");
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("x", cudaq::depolarization2(0.01), 1);

  auto dem = cudaq::qec::z_dem_from_memory_circuit(
      *code, cudaq::qec::operation::prep0, 3, noise);
  // [End DEM Generation]

  save_dem(dem, "config.yaml");

  // Step 2: Load config and run circuit
  printf("\nStep 2: Running circuit with decoding...\n");
  load_dem("config.yaml");

  cudaq::run(10, qec_circuit);
  printf("Ran 10 shots\n");

  cudaq::qec::decoding::config::finalize_decoders();

  printf("\nDone!\n");
  return 0;
}
// [End Documentation]
