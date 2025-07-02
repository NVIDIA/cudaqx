/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/experiments.h"
#include "device/memory_circuit.h"

using namespace cudaqx;

namespace cudaq::qec {

namespace details {
auto __sample_code_capacity(const cudaqx::tensor<uint8_t> &H,
                            std::size_t nShots, double error_probability,
                            unsigned seed) {
  // init RNG
  std::mt19937 rng(seed);
  std::bernoulli_distribution dist(error_probability);

  // Each row is a shot
  // Each row elem is a 1 if error, 0 else.
  cudaqx::tensor<uint8_t> data({nShots, H.shape()[1]});
  cudaqx::tensor<uint8_t> syndromes({nShots, H.shape()[0]});

  std::vector<uint8_t> bits(nShots * H.shape()[1]);
  std::generate(bits.begin(), bits.end(), [&]() { return dist(rng); });

  data.copy(bits.data(), data.shape());

  // Syn = D * H^T
  // [n,s] = [n,d]*[d,s]
  syndromes = data.dot(H.transpose()) % 2;

  return std::make_tuple(syndromes, data);
}
} // namespace details

// Single shot version
cudaqx::tensor<uint8_t> generate_random_bit_flips(size_t numBits,
                                                  double error_probability) {
  // init RNG
  std::random_device rd;
  std::mt19937 rng(rd());
  std::bernoulli_distribution dist(error_probability);

  // Each row is a shot
  // Each row elem is a 1 if error, 0 else.
  cudaqx::tensor<uint8_t> data({numBits});
  std::vector<uint8_t> bits(numBits);
  std::generate(bits.begin(), bits.end(), [&]() { return dist(rng); });

  data.copy(bits.data(), data.shape());
  return data;
}

std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_code_capacity(const cudaqx::tensor<uint8_t> &H, std::size_t nShots,
                     double error_probability, unsigned seed) {
  return details::__sample_code_capacity(H, nShots, error_probability, seed);
}

std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_code_capacity(const cudaqx::tensor<uint8_t> &H, std::size_t nShots,
                     double error_probability) {
  return details::__sample_code_capacity(H, nShots, error_probability,
                                         std::random_device()());
}

std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_code_capacity(const code &code, std::size_t nShots,
                     double error_probability) {
  return sample_code_capacity(code.get_parity(), nShots, error_probability);
}

std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_code_capacity(const code &code, std::size_t nShots,
                     double error_probability, unsigned seed) {
  return sample_code_capacity(code.get_parity(), nShots, error_probability,
                              seed);
}

std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_memory_circuit(const code &code, operation statePrep,
                      std::size_t numShots, std::size_t numRounds,
                      cudaq::noise_model &noise) {
  if (!code.contains_operation(statePrep))
    throw std::runtime_error(
        "sample_memory_circuit_error - requested state prep kernel not found.");

  auto &prep = code.get_operation<code::one_qubit_encoding>(statePrep);

  if (!code.contains_operation(operation::stabilizer_round))
    throw std::runtime_error("sample_memory_circuit error - no stabilizer "
                             "round kernel for this code.");

  auto &stabRound =
      code.get_operation<code::stabilizer_round>(operation::stabilizer_round);

  auto parity_x = code.get_parity_x();
  auto parity_z = code.get_parity_z();
  auto numData = code.get_num_data_qubits();
  auto numAncx = code.get_num_ancilla_x_qubits();
  auto numAncz = code.get_num_ancilla_z_qubits();

  std::vector<std::size_t> xVec(parity_x.data(),
                                parity_x.data() + parity_x.size());
  std::vector<std::size_t> zVec(parity_z.data(),
                                parity_z.data() + parity_z.size());

  std::size_t numRows = numShots * numRounds;
  std::size_t numCols = numAncx + numAncz;

  // Allocate the tensor data for the syndromes and data.
  cudaqx::tensor<uint8_t> syndromeTensor({numShots * numRounds, numCols});
  cudaqx::tensor<uint8_t> dataResults({numShots, numData});

  cudaq::sample_options opts{
      .shots = numShots, .noise = noise, .explicit_measurements = true};

  cudaq::sample_result result;

  // Run the memory circuit experiment
  if (statePrep == operation::prep0 || statePrep == operation::prep1) {
    // run z basis
    result = cudaq::sample(opts, memory_circuit_mz, stabRound, prep, numData,
                           numAncx, numAncz, numRounds, xVec, zVec);
  } else if (statePrep == operation::prepp || statePrep == operation::prepm) {
    // run x basis
    result = cudaq::sample(opts, memory_circuit_mx, stabRound, prep, numData,
                           numAncx, numAncz, numRounds, xVec, zVec);
  } else {
    throw std::runtime_error(
        "sample_memory_circuit_error - invalid requested state prep kernel.");
  }

  cudaqx::tensor<uint8_t> mzTable(result.sequential_data());
  const auto numColsBeforeData = numCols * numRounds;

  // Populate dataResults from mzTable
  for (std::size_t shot = 0; shot < numShots; shot++) {
    uint8_t __restrict__ *dataResultsRow = &dataResults.at({shot, 0});
    uint8_t __restrict__ *mzTableRow = &mzTable.at({shot, 0});
    for (std::size_t d = 0; d < numData; d++)
      dataResultsRow[d] = mzTableRow[numColsBeforeData + d];
  }

  // Now populate syndromeTensor.

  // First round, store bare syndrome measurement
  for (std::size_t shot = 0; shot < numShots; ++shot) {
    std::size_t round = 0;
    std::size_t measIdx = shot * numRounds + round;
    std::uint8_t __restrict__ *syndromeTensorRow =
        &syndromeTensor.at({measIdx, 0});
    std::uint8_t __restrict__ *mzTableRow = &mzTable.at({shot, 0});
    for (std::size_t col = 0; col < numCols; ++col)
      syndromeTensorRow[col] = mzTableRow[col];
  }

  // After first round, store syndrome flips
  for (std::size_t shot = 0; shot < numShots; ++shot) {
    std::uint8_t __restrict__ *mzTableRow = &mzTable.at({shot, 0});
    for (std::size_t round = 1; round < numRounds; ++round) {
      std::size_t measIdx = shot * numRounds + round;
      std::uint8_t __restrict__ *syndromeTensorRow =
          &syndromeTensor.at({measIdx, 0});
      for (std::size_t col = 0; col < numCols; ++col) {
        syndromeTensorRow[col] = mzTableRow[round * numCols + col] ^
                                 mzTableRow[(round - 1) * numCols + col];
      }
    }
  }

  // Return the data.
  return std::make_tuple(syndromeTensor, dataResults);
}

std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_memory_circuit(const code &code, operation op, std::size_t numShots,
                      std::size_t numRounds) {
  cudaq::noise_model noise;
  return sample_memory_circuit(code, op, numShots, numRounds, noise);
}

std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_memory_circuit(const code &code, std::size_t numShots,
                      std::size_t numRounds) {
  return sample_memory_circuit(code, operation::prep0, numShots, numRounds);
}

std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_memory_circuit(const code &code, std::size_t numShots,
                      std::size_t numRounds, cudaq::noise_model &noise) {
  return sample_memory_circuit(code, operation::prep0, numShots, numRounds,
                               noise);
}

namespace details {
/// @brief Given a memory circuit setup, generate a DEM. This is the main driver
/// function that all of the function overloads invoke. Hence, it is kept in the
/// details namespace.
cudaq::qec::detector_error_model dem_from_memory_circuit(
    const code &code, operation statePrep, std::size_t numRounds,
    cudaq::noise_model &noise, const cudaqx::tensor<uint8_t> &obs_matrix,
    bool run_mz_circuit, bool keep_x_stabilizers, bool keep_z_stabilizers) {
  if (!code.contains_operation(statePrep))
    throw std::runtime_error("dem_from_memory_circuit error - requested state "
                             "prep kernel not found.");

  if (!keep_x_stabilizers && !keep_z_stabilizers)
    throw std::runtime_error("dem_from_memory_circuit error - no stabilizers "
                             "to keep.");

  detector_error_model dem; // DEM to return

  auto &prep = code.get_operation<code::one_qubit_encoding>(statePrep);

  if (!code.contains_operation(operation::stabilizer_round))
    throw std::runtime_error("sample_memory_circuit error - no stabilizer "
                             "round kernel for this code.");

  auto &stabRound =
      code.get_operation<code::stabilizer_round>(operation::stabilizer_round);

  auto parity_x = code.get_parity_x();
  auto parity_z = code.get_parity_z();
  auto numData = code.get_num_data_qubits();
  auto numAncx = code.get_num_ancilla_x_qubits();
  auto numAncz = code.get_num_ancilla_z_qubits();

  std::vector<std::size_t> xVec(parity_x.data(),
                                parity_x.data() + parity_x.size());
  std::vector<std::size_t> zVec(parity_z.data(),
                                parity_z.data() + parity_z.size());

  std::size_t numCols = numAncx + numAncz;

  cudaq::ExecutionContext ctx_msm_size("msm_size");
  ctx_msm_size.noiseModel = &noise;
  auto &platform = cudaq::get_platform();
  platform.set_exec_ctx(&ctx_msm_size);

  // Run the memory circuit experiment
  if (run_mz_circuit) {
    memory_circuit_mz(stabRound, prep, numData, numAncx, numAncz, numRounds,
                      xVec, zVec);
  } else {
    memory_circuit_mx(stabRound, prep, numData, numAncx, numAncz, numRounds,
                      xVec, zVec);
  }

  platform.reset_exec_ctx();

  if (!ctx_msm_size.msm_dimensions.has_value()) {
    throw std::runtime_error(
        "dem_from_memory_circuit error: no MSM dimensions found");
  }
  if (ctx_msm_size.msm_dimensions.value().second == 0) {
    throw std::runtime_error(
        "dem_from_memory_circuit error: no noise mechanisms found in circuit. "
        "Cannot generate a DEM. Did you forget to enable noise?");
  }

  cudaq::ExecutionContext ctx_msm("msm");
  ctx_msm.noiseModel = &noise;
  ctx_msm.msm_dimensions = ctx_msm_size.msm_dimensions;
  platform.set_exec_ctx(&ctx_msm);

  // Run the memory circuit experiment
  if (run_mz_circuit) {
    memory_circuit_mz(stabRound, prep, numData, numAncx, numAncz, numRounds,
                      xVec, zVec);
  } else {
    memory_circuit_mx(stabRound, prep, numData, numAncx, numAncz, numRounds,
                      xVec, zVec);
  }

  platform.reset_exec_ctx();

  // Populate error rates and error IDs
  dem.error_rates = std::move(ctx_msm.msm_probabilities.value());
  dem.error_ids = std::move(ctx_msm.msm_prob_err_id.value());

  auto msm_as_strings = ctx_msm.result.sequential_data();
  cudaqx::tensor<uint8_t> msm_data(
      std::vector<std::size_t>({ctx_msm_size.msm_dimensions->first,
                                ctx_msm_size.msm_dimensions->second}));
  cudaqx::tensor<uint8_t> mzTable(msm_as_strings);
  mzTable = mzTable.transpose();
  std::size_t numNoiseMechs = mzTable.shape()[1];
  std::size_t numSyndromesPerRound = numCols;

  // Populate dem.detector_error_matrix by XORing consecutive rounds. Generally
  // speaking, this is calculating H = D*Ω, where H is the Detector Error
  // Matrix, D is the Detector Matrix, and Ω is Measurement Syndrome Matrix.
  // However, D is very sparse, and is it represents simple XORs of a syndrome
  // with the prior round's syndrome.
  // Reference: https://arxiv.org/pdf/2407.13826
  auto numReturnSynPerRound = keep_x_stabilizers && keep_z_stabilizers
                                  ? numSyndromesPerRound
                                  : numSyndromesPerRound / 2;
  // If we are returning only x-stabilizers, we need to offset the syndrome
  // indices of mzTable by numSyndromesPerRound / 2.
  auto offset =
      keep_x_stabilizers && !keep_z_stabilizers ? numSyndromesPerRound / 2 : 0;
  dem.detector_error_matrix = cudaqx::tensor<uint8_t>(
      {numRounds * numReturnSynPerRound, numNoiseMechs});
  for (std::size_t round = 0; round < numRounds; round++) {
    if (round == 0) {
      for (std::size_t syndrome = 0; syndrome < numReturnSynPerRound;
           syndrome++) {
        for (std::size_t noise_mech = 0; noise_mech < numNoiseMechs;
             noise_mech++) {
          dem.detector_error_matrix.at(
              {round * numReturnSynPerRound + syndrome, noise_mech}) =
              mzTable.at({round * numSyndromesPerRound + syndrome + offset,
                          noise_mech});
        }
      }
    } else {
      for (std::size_t syndrome = 0; syndrome < numReturnSynPerRound;
           syndrome++) {
        for (std::size_t noise_mech = 0; noise_mech < numNoiseMechs;
             noise_mech++) {
          dem.detector_error_matrix.at(
              {round * numReturnSynPerRound + syndrome, noise_mech}) =
              mzTable.at({round * numSyndromesPerRound + syndrome + offset,
                          noise_mech}) ^
              mzTable.at(
                  {(round - 1) * numSyndromesPerRound + syndrome + offset,
                   noise_mech});
        }
      }
    }
  }

  // Uncomment for debugging:
  // printf("dem.detector_error_matrix:\n");
  // dem.detector_error_matrix.dump_bits();

  // Populate dem.observables_flips_matrix by converting the physical data qubit
  // measurements to logical observables.
  auto first_data_row = numRounds * numSyndromesPerRound;
  assert(first_data_row < mzTable.shape()[0]);

  cudaqx::tensor<uint8_t> msm_obs(
      {mzTable.shape()[0] - first_data_row, numNoiseMechs});
  for (std::size_t row = first_data_row; row < mzTable.shape()[0]; row++)
    for (std::size_t col = 0; col < numNoiseMechs; col++)
      msm_obs.at({row - first_data_row, col}) = mzTable.at({row, col});

  // Populate dem.observables_flips_matrix by converting the physical data qubit
  // measurements to logical observables.
  dem.observables_flips_matrix = obs_matrix.dot(msm_obs) % 2;

  // Uncomment print statements for debugging:
  // printf("dem.detector_error_matrix Before canonicalization:\n");
  // dem.detector_error_matrix.dump_bits();
  // printf("dem.observables_flips_matrix Before canonicalization:\n");
  // dem.observables_flips_matrix.dump_bits();
  dem.canonicalize_for_rounds(numReturnSynPerRound);
  // printf("dem.detector_error_matrix After canonicalization:\n");
  // dem.detector_error_matrix.dump_bits();
  // printf("dem.observables_flips_matrix After canonicalization:\n");
  // dem.observables_flips_matrix.dump_bits();

  return dem;
}
} // namespace details

/// @brief Given a memory circuit setup, generate a DEM
cudaq::qec::detector_error_model
dem_from_memory_circuit(const code &code, operation statePrep,
                        std::size_t numRounds, cudaq::noise_model &noise) {
  constexpr bool keep_x_stabilizers = true;
  constexpr bool keep_z_stabilizers = true;
  const bool is_z =
      statePrep == operation::prep0 || statePrep == operation::prep1;
  const auto obs_matrix =
      is_z ? code.get_observables_z() : code.get_observables_x();
  const bool run_mz_circuit = is_z;
  return details::dem_from_memory_circuit(
      code, statePrep, numRounds, noise, obs_matrix, run_mz_circuit,
      keep_x_stabilizers, keep_z_stabilizers);
}

// For CSS codes, may want to partition x vs z decoding
detector_error_model x_dem_from_memory_circuit(const code &code,
                                               operation statePrep,
                                               std::size_t numRounds,
                                               cudaq::noise_model &noise) {
  constexpr bool keep_x_stabilizers = true;
  constexpr bool keep_z_stabilizers = false;
  bool is_z = statePrep == operation::prep0 || statePrep == operation::prep1;
  auto obs_matrix = is_z ? code.get_observables_z() : code.get_observables_x();
  return details::dem_from_memory_circuit(code, statePrep, numRounds, noise,
                                          obs_matrix, is_z, keep_x_stabilizers,
                                          keep_z_stabilizers);
}

detector_error_model z_dem_from_memory_circuit(const code &code,
                                               operation statePrep,
                                               std::size_t numRounds,
                                               cudaq::noise_model &noise) {
  constexpr bool keep_x_stabilizers = false;
  constexpr bool keep_z_stabilizers = true;
  bool is_z = statePrep == operation::prep0 || statePrep == operation::prep1;
  auto obs_matrix = is_z ? code.get_observables_z() : code.get_observables_x();
  return details::dem_from_memory_circuit(code, statePrep, numRounds, noise,
                                          obs_matrix, is_z, keep_x_stabilizers,
                                          keep_z_stabilizers);
}

} // namespace cudaq::qec
