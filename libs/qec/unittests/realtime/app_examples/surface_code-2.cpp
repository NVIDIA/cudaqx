/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// For full test script: surface_code-2-test.sh

#include "cudaq.h"
#include "cudaq/qec/code.h"
#include "cudaq/qec/codes/surface_code.h"
#include "cudaq/qec/decoder.h"
#include "cudaq/qec/experiments.h"
#include "cudaq/qec/pcm_utils.h"
#include "cudaq/qec/realtime/decoding.h"
#include "cudaq/qec/realtime/decoding_config.h"
#include <common/CustomOp.h>
#include <common/ExecutionContext.h>
#include <common/NoiseModel.h>
#include <fstream>

// Whether or not to put calls to debug functions in the QIR program. You cannot
// set this to 1 if you are submitting to hardware.
#ifndef PER_SHOT_DEBUG
#define PER_SHOT_DEBUG 0
#endif

// Uncomment this to manually inject errors.
// #define MANUALLY_INJECT_ERRORS

void save_dem_to_file(const cudaq::qec::detector_error_model &dem,
                      const std::vector<int64_t> &det_mat,
                      std::string dem_filename, uint64_t numSyndromesPerRound,
                      uint64_t numRounds, uint64_t numLogical) {
  cudaq::qec::decoding::config::multi_decoder_config multi_config;
  for (uint64_t i = 0; i < numLogical; i++) {
    cudaq::qec::decoding::config::decoder_config config;
    config.id = i;
    config.type = "multi_error_lut";
    config.block_size = dem.num_error_mechanisms();
    config.syndrome_size = dem.num_detectors();
    config.num_syndromes_per_round = numSyndromesPerRound;
    config.H_sparse = cudaq::qec::pcm_to_sparse_vec(dem.detector_error_matrix);
    config.O_sparse =
        cudaq::qec::pcm_to_sparse_vec(dem.observables_flips_matrix);
    config.D_sparse = std::vector<int64_t>(det_mat);
    config.decoder_custom_args =
        cudaq::qec::decoding::config::multi_error_lut_config();
    auto &multi_error_lut_config =
        std::get<cudaq::qec::decoding::config::multi_error_lut_config>(
            config.decoder_custom_args);
    multi_error_lut_config.lut_error_depth = 2;
    multi_config.decoders.push_back(config);
  }
  std::string config_str = multi_config.to_yaml_str(200);
  std::ofstream config_file(dem_filename);
  config_file << config_str;
  config_file.close();
  printf("Saved config to file: %s\n", dem_filename.c_str());
  return;
}

void load_dem_from_file(const std::string &dem_filename,
                        cudaq::qec::detector_error_model &dem,
                        uint64_t numLogical) {
  printf("load_dem_from_file: Loading dem from file: %s\n",
         dem_filename.c_str());
  // Read dem_filename into a std::string
  std::ifstream dem_file(dem_filename);
  std::string dem_str((std::istreambuf_iterator<char>(dem_file)),
                      std::istreambuf_iterator<char>());
  auto config =
      cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
          dem_str);
  if (numLogical != config.decoders.size()) {
    printf("ERROR: numLogical [%ld] !- config.decoders.size() [%ld]\n",
           numLogical, config.decoders.size());
    exit(1);
  }
  auto decoder_config = config.decoders[0];
  auto multi_error_lut_config =
      std::get<cudaq::qec::decoding::config::multi_error_lut_config>(
          decoder_config.decoder_custom_args);
  dem.detector_error_matrix = cudaq::qec::pcm_from_sparse_vec(
      decoder_config.H_sparse, decoder_config.syndrome_size,
      decoder_config.block_size);
  // Count how many rows there are in the O_sparse by counting the number of
  // -1s.
  size_t num_observables = std::count(decoder_config.O_sparse.begin(),
                                      decoder_config.O_sparse.end(), -1);
  dem.observables_flips_matrix = cudaq::qec::pcm_from_sparse_vec(
      decoder_config.O_sparse, num_observables, decoder_config.block_size);
  printf("Loaded dem from file: %s\n", dem_filename.c_str());

  // Now configure the decoders
  cudaq::qec::decoding::config::configure_decoders(config);
}

std::vector<size_t> get_stab_cnot_schedule(char stab_type, int distance) {
  cudaq::qec::surface_code::stabilizer_grid grid(distance);
  if (stab_type != 'X' && stab_type != 'Z') {
    throw std::runtime_error(
        "get_stab_cnot_schedule: Invalid stabilizer type. Must be 'X' or 'Z'.");
  }
  // First get the stabilizers
  auto stabs = grid.get_spin_op_stabilizers();
  cudaq::qec::sortStabilizerOps(stabs);
  std::size_t stab_idx = 0;
  std::vector<size_t> cnot_schedule;
  for (const auto &stab : stabs) {
    auto stab_word = stab.get_pauli_word(distance * distance);
    if (stab_word.find(stab_type) == std::string::npos)
      continue; // None of the desired stabilizers in this row
    for (std::size_t d = 0; d < stab_word.size(); ++d) {
      if (stab_word[d] == stab_type) {
        cnot_schedule.push_back(stab_idx);
        cnot_schedule.push_back(d);
      }
    }
    stab_idx++;
  }
  return cnot_schedule;
}

void debug_print_syndromes(int64_t syndrome_x_int, int64_t syndrome_z_int) {
  printf("syndrome_x_int: %ld, syndrome_z_int: %ld\n", syndrome_x_int,
         syndrome_z_int);
}

void debug_print_applying_correction(int64_t correction) {
  printf("Applying correction: %ld\n", correction);
}

void debug_start_shot() { printf("Starting shot\n"); }

namespace cudaq::qec::qpu {

// Transversal CNOT gate
__qpu__ void logical_cnot(cudaq::qview<> ctrl_data, cudaq::qview<> tgt_data) {
  for (std::size_t i = 0; i < ctrl_data.size(); i++) {
    cudaq::x<cudaq::ctrl>(ctrl_data[i], tgt_data[i]);
  }
}
__qpu__ void spam_error(cudaq::qec::patch logicalQubit, double p_spam_data,
                        double p_spam_ancx, double p_spam_ancz) {
  for (std::size_t i = 0; i < logicalQubit.data.size(); i++) {
    cudaq::apply_noise<cudaq::depolarization1>(p_spam_data,
                                               logicalQubit.data[i]);
  }
  for (std::size_t i = 0; i < logicalQubit.ancx.size(); i++) {
    cudaq::apply_noise<cudaq::depolarization1>(p_spam_ancx,
                                               logicalQubit.ancx[i]);
  }
  for (std::size_t i = 0; i < logicalQubit.ancz.size(); i++) {
    cudaq::apply_noise<cudaq::depolarization1>(p_spam_ancz,
                                               logicalQubit.ancz[i]);
  }
}

__qpu__ std::vector<cudaq::measure_result>
se_z_ft(cudaq::qec::patch logicalQubit,
        const std::vector<std::size_t> &cnot_sched) {
  for (std::size_t i = 0; i < cnot_sched.size(); i += 2) {
    cudaq::x<cudaq::ctrl>(logicalQubit.data[cnot_sched[i + 1]],
                          logicalQubit.ancz[cnot_sched[i]]);
  }
  auto results = mz(logicalQubit.ancz);
  for (std::size_t i = 0; i < logicalQubit.ancz.size(); i++)
    reset(logicalQubit.ancz[i]);
  return results;
}

__qpu__ std::vector<cudaq::measure_result>
se_x_ft(cudaq::qec::patch logicalQubit,
        const std::vector<std::size_t> &cnot_sched) {
  h(logicalQubit.ancx);
  for (std::size_t i = 0; i < cnot_sched.size(); i += 2) {
    cudaq::x<cudaq::ctrl>(logicalQubit.ancx[cnot_sched[i]],
                          logicalQubit.data[cnot_sched[i + 1]]);
  }
  h(logicalQubit.ancx);
  auto results = mz(logicalQubit.ancx);
  for (std::size_t i = 0; i < logicalQubit.ancx.size(); i++)
    reset(logicalQubit.ancx[i]);
  return results;
}

__qpu__ void custom_memory_circuit_stabs(
    cudaq::qview<> data, cudaq::qview<> xstab_anc, cudaq::qview<> zstab_anc,
    std::size_t numRounds, const std::vector<std::size_t> &cnot_schedX_flat,
    const std::vector<std::size_t> &cnot_schedZ_flat, bool enqueue_syndromes,
    bool do_errors_after_non_last_rounds, double p_spam,
    int logical_qubit_idx) {
  // Create the logical patch
  patch logical(data, xstab_anc, zstab_anc);
  std::vector<cudaq::measure_result> combined_syndrome(xstab_anc.size() +
                                                       zstab_anc.size());

  // Generate syndrome data
  for (std::size_t round = 0; round < numRounds; round++) {
    auto syndrome_z = se_z_ft(logical, cnot_schedZ_flat);
    auto syndrome_x = se_x_ft(logical, cnot_schedX_flat);
    int i = 0;
    for (auto s : syndrome_z)
      combined_syndrome[i++] = s;
    for (auto s : syndrome_x)
      combined_syndrome[i++] = s;
    if (enqueue_syndromes) {
      cudaq::qec::decoding::enqueue_syndromes(
          /*decoder_id=*/logical_qubit_idx, combined_syndrome);
    }
#if PER_SHOT_DEBUG
    debug_print_syndromes(syndrome_x_int, syndrome_z_int);
#endif
    if (do_errors_after_non_last_rounds && round < numRounds - 1) {
      // spam_error(logical, p_spam, p_spam, p_spam);
      spam_error(logical, p_spam, 0.0, 0.0);
      // Uncomment the following to force a single error that should likely be
      // correctable.
#if MANUALLY_INJECT_ERRORS
      if (round == 0) {
        // Inject a single error
        cudaq::x(logical.data[3]);
      }
#endif
    }
  }
}

__qpu__ std::int64_t
demo_circuit_qpu(bool allow_device_calls,
                 const cudaq::qec::code::one_qubit_encoding &statePrep,
                 std::size_t numData, std::size_t numAncx, std::size_t numAncz,
                 std::size_t numRounds, std::size_t numLogical,
                 const std::vector<std::size_t> &cnot_schedX_flat,
                 const std::vector<std::size_t> &cnot_schedZ_flat,
                 double p_spam, bool apply_corrections) {
#if PER_SHOT_DEBUG
  debug_start_shot();
#endif
  std::uint64_t num_corrections = 0;

  // Reset the decoder
  if (allow_device_calls) {
    for (int i = 0; i < numLogical; i++) {
      cudaq::qec::decoding::reset_decoder(/*decoder_id=*/i);
    }
  }

  // Allocate the data and ancilla qubits
  cudaq::qvector data(numLogical * numData), xstab_anc(numLogical * numAncx),
      zstab_anc(numLogical * numAncz);

  // Call state prep
  for (int i = 0; i < numLogical; i++) {
    auto subData = data.slice(i * numData, numData);
    auto subXstab_anc = xstab_anc.slice(i * numAncx, numAncx);
    auto subZstab_anc = zstab_anc.slice(i * numAncz, numAncz);
    patch logical(subData, subXstab_anc, subZstab_anc);
    statePrep(logical);
  }

  // Do 1 stabilizer round to lock in the stabilizers
  {
    for (int i = 0; i < numLogical; i++) {
      auto subData = data.slice(i * numData, numData);
      auto subXstab_anc = xstab_anc.slice(i * numAncx, numAncx);
      auto subZstab_anc = zstab_anc.slice(i * numAncz, numAncz);

      custom_memory_circuit_stabs(
          subData, subXstab_anc, subZstab_anc,
          /*numRounds=*/1, cnot_schedX_flat, cnot_schedZ_flat,
          /*enqueue_syndromes=*/allow_device_calls,
          /*do_errors_after_non_last_rounds=*/false, p_spam, i);
    }
  }

  // Inject errors
  for (int i = 0; i < numLogical; i++) {
    auto subData = data.slice(i * numData, numData);
    auto subXstab_anc = xstab_anc.slice(i * numAncx, numAncx);
    auto subZstab_anc = zstab_anc.slice(i * numAncz, numAncz);
    patch logical(subData, subXstab_anc, subZstab_anc);
    spam_error(logical, /*p_spam_data=*/p_spam, /*p_spam_ancx=*/0.0,
               /*p_spam_ancz=*/0.0);
  }

  // Do stabilizer rounds
  for (int i = 0; i < numLogical; i++) {
    auto subData = data.slice(i * numData, numData);
    auto subXstab_anc = xstab_anc.slice(i * numAncx, numAncx);
    auto subZstab_anc = zstab_anc.slice(i * numAncz, numAncz);

    custom_memory_circuit_stabs(
        subData, subXstab_anc, subZstab_anc, numRounds, cnot_schedX_flat,
        cnot_schedZ_flat, /*enqueue_syndromes=*/allow_device_calls,
        /*do_errors_after_non_last_rounds=*/true, p_spam, i);
  }

  if (allow_device_calls && apply_corrections) {
    for (int i = 0; i < numLogical; i++) {
      auto subData = data.slice(i * numData, numData);
      auto subXstab_anc = xstab_anc.slice(i * numAncx, numAncx);
      auto subZstab_anc = zstab_anc.slice(i * numAncz, numAncz);
      auto correction_result = cudaq::qec::decoding::get_corrections(
          /*decoder_id=*/i, /*return_size=*/1, /*reset=*/false);
      if (correction_result[0] != 0) {
        num_corrections++;
        // Transversal correction
        cudaq::x(subData);
#if PER_SHOT_DEBUG
        debug_print_applying_correction(correction_result);
#endif
      }
    }
  }

  // Note: this only works up to 64 bits, so a single logical qubit with
  // distance 7.
  std::uint64_t ret = 0;
  for (int i = 0; i < numLogical; i++) {
    if (i > 0)
      ret <<= numData;
    auto subData = data.slice(i * numData, numData);
    auto subMeas = mz(subData);
    ret |= cudaq::to_integer(subMeas);
  }
  // The remaining bits are allocated to the number of corrections.
  ret |= num_corrections << (numData * numLogical);
  return ret;
}
} // namespace cudaq::qec::qpu

void demo_circuit_host(const cudaq::qec::code &code, int distance,
                       double p_spam, cudaq::qec::operation statePrep,
                       std::size_t numShots, std::size_t numRounds,
                       std::size_t numLogical, std::string dem_filename,
                       bool save_dem, bool load_dem) {
  if (!code.contains_operation(statePrep))
    throw std::runtime_error(
        "sample_memory_circuit_error - requested state prep kernel not found.");

  auto &prep =
      code.get_operation<cudaq::qec::code::one_qubit_encoding>(statePrep);

  if (!code.contains_operation(cudaq::qec::operation::stabilizer_round))
    throw std::runtime_error("demo_circuit_host error - no stabilizer "
                             "round kernel for this code.");

  auto &stabRound = code.get_operation<cudaq::qec::code::stabilizer_round>(
      cudaq::qec::operation::stabilizer_round);

  auto numData = code.get_num_data_qubits();
  auto numAncx = code.get_num_ancilla_x_qubits();
  auto numAncz = code.get_num_ancilla_z_qubits();

  auto cnot_schedX_flat = get_stab_cnot_schedule('X', distance);
  auto cnot_schedZ_flat = get_stab_cnot_schedule('Z', distance);

  printf("cnot_schedX_flat: ");
  // Put a comma in between each pair of elements
  for (std::size_t i = 0; i < cnot_schedX_flat.size(); i += 2)
    printf("%lu %lu, ", cnot_schedX_flat[i], cnot_schedX_flat[i + 1]);
  printf("\n");
  printf("cnot_schedZ_flat: ");
  for (std::size_t i = 0; i < cnot_schedZ_flat.size(); i += 2)
    printf("%lu %lu, ", cnot_schedZ_flat[i], cnot_schedZ_flat[i + 1]);
  printf("\n");

  cudaq::noise_model noise;

  // First get the MSM
  cudaq::qec::detector_error_model dem;
  if (load_dem) {
    load_dem_from_file(dem_filename, dem, numLogical);
  } else {
    if (p_spam == 0.0) {
      printf("p_spam is 0.0, cannot get the MSM\n");
      exit(0);
    }
    cudaq::ExecutionContext ctx_msm_size("msm_size");
    ctx_msm_size.noiseModel = &noise;
    auto &platform = cudaq::get_platform();
    platform.set_exec_ctx(&ctx_msm_size);
    // Always use numLogical = 1 for the MSM
    cudaq::qec::qpu::demo_circuit_qpu(
        /*allow_device_calls=*/false, prep, numData, numAncx, numAncz,
        numRounds,
        /*numLogical=*/1, cnot_schedX_flat, cnot_schedZ_flat, p_spam,
        /*apply_corrections=*/false);
    platform.reset_exec_ctx();
    if (!ctx_msm_size.msm_dimensions.has_value()) {
      throw std::runtime_error("No MSM dimensions found");
    }
    if (ctx_msm_size.msm_dimensions.value().second == 0) {
      throw std::runtime_error("No MSM dimensions found");
    }
    cudaq::ExecutionContext ctx_msm("msm");
    ctx_msm.noiseModel = &noise;
    ctx_msm.msm_dimensions = ctx_msm_size.msm_dimensions;
    platform.set_exec_ctx(&ctx_msm);
    // Always use numLogical = 1 for the MSM
    cudaq::qec::qpu::demo_circuit_qpu(
        /*allow_device_calls=*/false, prep, numData, numAncx, numAncz,
        numRounds,
        /*numLogical=*/1, cnot_schedX_flat, cnot_schedZ_flat, p_spam,
        /*apply_corrections=*/false);
    platform.reset_exec_ctx();

    auto msm_as_strings = ctx_msm.result.sequential_data();
    printf("MSM Dimensions: %ld measurements x %ld error mechanisms\n",
           ctx_msm.msm_dimensions.value().first,
           ctx_msm.msm_dimensions.value().second);
    for (std::size_t i = 0; i < ctx_msm.msm_dimensions.value().first; i++) {
      for (std::size_t j = 0; j < ctx_msm.msm_dimensions.value().second; j++) {
        printf("%c", msm_as_strings[j][i] == '1' ? '1' : '.');
      }
      printf("\n");
    }
    // Populate error rates and error IDs
    dem.error_rates = std::move(ctx_msm.msm_probabilities.value());
    dem.error_ids = std::move(ctx_msm.msm_prob_err_id.value());

    cudaqx::tensor<uint8_t> mzTable(msm_as_strings);
    mzTable = mzTable.transpose();
    printf("mzTable:\n");
    mzTable.dump_bits();
    // Subtract the number of data qubits to get the number of syndrome
    // measurements.
    std::size_t totalNumSyndromes = mzTable.shape()[0] - distance * distance;
    std::size_t numNoiseMechs = mzTable.shape()[1];
    std::size_t numSyndromesPerRound = distance * distance - 1;

    std::size_t numRoundsOfSyndromData =
        totalNumSyndromes / numSyndromesPerRound;
    if (numRoundsOfSyndromData != numRounds + 1) {
      throw std::runtime_error("Num rounds of syndrome data [" +
                               std::to_string(numRoundsOfSyndromData) +
                               "] is not equal to the number of rounds + 1[" +
                               std::to_string(numRounds + 1) + "]");
    }

    // There should be (numRounds + 1) rounds of data in MSM.
    // This corresponds to numRounds + measurements during state prep
    // Not every measurement during stateprep is a detector, but some
    // may be.
    // In this Z-basis surface code case, the Z stabs during state prep
    // are detectors.
    // Skip the X stabs during the first round.
    std::size_t numDetectors =
        numSyndromesPerRound * numRounds + numSyndromesPerRound / 2;

    dem.detector_error_matrix =
        cudaqx::tensor<uint8_t>({numDetectors, numNoiseMechs});
    // Grab first half of first "round"
    std::size_t r0_offset = 0;
    for (std::size_t syndrome = 0; syndrome < numSyndromesPerRound / 2;
         syndrome++) {
      for (std::size_t noise_mech = 0; noise_mech < numNoiseMechs;
           noise_mech++) {
        // round 0
        dem.detector_error_matrix.at({r0_offset, noise_mech}) =
            mzTable.at({syndrome, noise_mech});
      }
      r0_offset += 1;
    }

    // Grab all of rounds >=1.
    for (std::size_t round = 0; round < numRounds; round++) {
      for (std::size_t syndrome = 0; syndrome < numSyndromesPerRound;
           syndrome++) {
        for (std::size_t noise_mech = 0; noise_mech < numNoiseMechs;
             noise_mech++) {
          dem.detector_error_matrix.at(
              {round * numSyndromesPerRound + syndrome + r0_offset,
               noise_mech}) =
              mzTable.at(
                  {(round + 0) * numSyndromesPerRound + syndrome, noise_mech}) ^
              mzTable.at(
                  {(round + 1) * numSyndromesPerRound + syndrome, noise_mech});
        }
      }
    }
    auto first_data_row = (numRounds + 1) * numSyndromesPerRound;
    cudaqx::tensor<uint8_t> msm_obs(
        {mzTable.shape()[0] - first_data_row, numNoiseMechs});
    for (std::size_t row = first_data_row; row < mzTable.shape()[0]; row++)
      for (std::size_t col = 0; col < numNoiseMechs; col++)
        msm_obs.at({row - first_data_row, col}) = mzTable.at({row, col});

    // Populate dem.observables_flips_matrix by converting the physical data
    // qubit measurements to logical observables.
    auto obs_matrix = code.get_observables_z();
    printf("obs_matrix:\n");
    obs_matrix.dump_bits();
    dem.observables_flips_matrix = obs_matrix.dot(msm_obs) % 2;
    printf("numSyndromesPerRound: %ld\n", numSyndromesPerRound);
    dem.canonicalize_for_rounds(numSyndromesPerRound);

    printf("dem.detector_error_matrix:\n");
    dem.detector_error_matrix.dump_bits();
    printf("dem.observables_flips_matrix:\n");
    dem.observables_flips_matrix.dump_bits();

    // Prep0 means that first round of Z stabs should be deterministic
    // These are measured first.
    std::vector<int64_t> first_round;
    for (int i = 0; i < numSyndromesPerRound / 2; i++) {
      first_round.push_back(i);
      first_round.push_back(-1);
    }

    // TO DO:
    // Does numRounds include first round?
    std::vector<int64_t> det_mat =
        cudaq::qec::generate_timelike_sparse_detector_matrix(
            numSyndromesPerRound, numRounds + 1, first_round);

    printf("detector_matrix with first round:\n");
    for (int i = 0; i < det_mat.size(); i++) {
      printf("%ld ", det_mat[i]);
    }
    printf("\n");

    if (save_dem) {
      save_dem_to_file(dem, det_mat, dem_filename, numSyndromesPerRound,
                       numRounds, numLogical);
      return;
    }
  }

  size_t numSyndromesPerRound = distance * distance - 1;

  size_t numRoundsOfSyndromData =
      dem.detector_error_matrix.shape()[0] / numSyndromesPerRound;
  if (numRoundsOfSyndromData != numRounds) {
    throw std::runtime_error("Num rounds of syndrome data [" +
                             std::to_string(numRoundsOfSyndromData) +
                             "] is not equal to the number of rounds [" +
                             std::to_string(numRounds) + "]");
  }

  // If this is a remote platform (not local sim nor emulation), don't use the
  // noise model.
  auto run_result =
      cudaq::get_platform().is_remote()
          ? cudaq::run(numShots, cudaq::qec::qpu::demo_circuit_qpu,
                       /*allow_device_calls=*/true, prep, numData, numAncx,
                       numAncz, numRounds, numLogical, cnot_schedX_flat,
                       cnot_schedZ_flat, p_spam, /*apply_corrections=*/true)
          : cudaq::run(numShots, noise, cudaq::qec::qpu::demo_circuit_qpu,
                       /*allow_device_calls=*/true, prep, numData, numAncx,
                       numAncz, numRounds, numLogical, cnot_schedX_flat,
                       cnot_schedZ_flat, p_spam, /*apply_corrections=*/true);
  printf("Result size: %ld\n", run_result.size());
  std::vector<std::vector<uint8_t>> logical_results;
  auto obs_matrix = code.get_observables_z();
  int num_non_zero_values = 0;
  std::int64_t num_corrections = 0;
  for (int i = 0; i < run_result.size(); i++) {
    logical_results.emplace_back();
    num_corrections += (run_result[i] >> (numData * numLogical));
    for (int j = 0; j < numLogical; j++) {
      std::vector<double> result_vec(numData);
      for (int l = j * numData; l < (j + 1) * numData; l++) {
        result_vec[l - j * numData] = (run_result[i] & (1ul << l)) ? 1.0 : 0.0;
      }
      cudaqx::tensor<uint8_t> result_tensor;
      cudaq::qec::convert_vec_soft_to_tensor_hard(result_vec, result_tensor);
      // Calculate the logical observable for each logical qubit
      uint8_t logical_result = (obs_matrix.dot(result_tensor) % 2).at({0});
      logical_results.back().push_back(logical_result);
      if (logical_result != 0)
        num_non_zero_values++;
    }
  }
  printf("Number of non-zero values measured : %d\n", num_non_zero_values);
  printf("Number of corrections decoder found: %ld\n", num_corrections);
}

void show_help() {
  printf("Usage: qec-test4 [options]\n");
  printf("Options:\n");
  printf("  --distance <int>    Distance of the surface code. Default: 5\n");
  printf("  --num_shots <int>   Number of shots. Default: 10\n");
  printf(
      "  --p_spam <double>   SPAM probability. Range[0, 1]. Default: 0.01\n");
  printf("  --num_logical <int> Number of logical qubits. Default: 1\n");
  printf("  --save_dem <string> Save the detector error model to a file.\n");
  printf("  --load_dem <string> Load the detector error model from a file. "
         "(Cannot be used with --save_dem)\n");
  printf("  --help              Show this help message\n");
}

int main(int argc, char **argv) {
  int num_shots = 10;
  int distance = 5;
  double p_spam = 0.01;
  int num_logical = 1;
  bool save_dem = false;
  bool load_dem = false;
  std::string dem_filename;

  // Parse the command line arguments
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--distance") {
      distance = std::stoi(argv[i + 1]);
      i++;
    } else if (arg == "--num_shots") {
      num_shots = std::stoi(argv[i + 1]);
      i++;
    } else if (arg == "--p_spam") {
      p_spam = std::stod(argv[i + 1]);
      i++;
    } else if (arg == "--help" || arg == "-h") {
      show_help();
      return 0;
    } else if (arg == "--num_logical") {
      num_logical = std::stoi(argv[i + 1]);
      i++;
    } else if (arg == "--save_dem") {
      save_dem = true;
      dem_filename = argv[i + 1];
      i++;
    } else if (arg == "--load_dem") {
      load_dem = true;
      dem_filename = argv[i + 1];
      i++;
    } else {
      printf("Unknown argument: %s\n", arg.c_str());
      show_help();
      return 1;
    }
  }

  if (!load_dem && !save_dem) {
    printf("Neither --save_dem nor --load_dem was specified. This is not a "
           "valid use case for this version of this program.\n");
    show_help();
    return 1;
  }

  int num_rounds = distance;
  if (num_logical * distance * distance >= 64) {
    printf("num_logical * distance * distance >= 64 is not supported.\n");
    return 1;
  }

  printf("Running with p_spam = %f, distance = %d, num_shots = %d\n", p_spam,
         distance, num_shots);
  auto code = cudaq::qec::get_code(
      "surface_code", cudaqx::heterogeneous_map{{"distance", distance}});

  demo_circuit_host(*code, distance, p_spam, cudaq::qec::operation::prep0,
                    num_shots, num_rounds, num_logical, dem_filename, save_dem,
                    load_dem);

  // Ensure clean shutdown
  cudaq::qec::decoding::config::finalize_decoders();

  return 0;
}
