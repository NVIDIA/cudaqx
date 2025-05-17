/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/experiments.h"
#include "device/memory_circuit.h"
#include "cudaq/qec/pcm_utils.h"
#include <common/MeasureCounts.h>

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

std::tuple<cudaqx::tensor<uint8_t>, std::vector<double>,
           cudaqx::tensor<uint8_t>, std::vector<double>>
pcm_memory_circuit(const code &code, operation statePrep, std::size_t numShots,
                   std::size_t numRounds, cudaq::noise_model &noise) {
  if (!code.contains_operation(statePrep))
    throw std::runtime_error(
        "pcm_memory_circuit_error - requested state prep kernel not found.");

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
  // cudaqx::tensor<uint8_t> syndromeTensor({numShots * numRounds, numCols});
  // cudaqx::tensor<uint8_t> dataResults({numShots, numData});

  cudaq::sample_options opts{
      .shots = numShots, .noise = noise, .explicit_measurements = true};

  cudaq::sample_result result;

  cudaq::ExecutionContext ctx_pcm_size("pcm_size");
  ctx_pcm_size.noiseModel = &noise;
  ctx_pcm_size.shots = numShots;
  auto &platform = cudaq::get_platform();
  platform.set_exec_ctx(&ctx_pcm_size);

  // Run the memory circuit experiment
  if (statePrep == operation::prep0 || statePrep == operation::prep1) {
    // run z basis
    memory_circuit_mz(stabRound, prep, numData, numAncx, numAncz, numRounds,
                      xVec, zVec);
  } else if (statePrep == operation::prepp || statePrep == operation::prepm) {
    // run x basis
    memory_circuit_mx(stabRound, prep, numData, numAncx, numAncz, numRounds,
                      xVec, zVec);
  } else {
    throw std::runtime_error(
        "sample_memory_circuit_error - invalid requested state prep kernel.");
  }

  platform.reset_exec_ctx();

  cudaq::ExecutionContext ctx_pcm("pcm");
  ctx_pcm.noiseModel = &noise;
  ctx_pcm.pcm_dimensions = ctx_pcm_size.pcm_dimensions;
  platform.set_exec_ctx(&ctx_pcm);

  // Run the memory circuit experiment
  if (statePrep == operation::prep0 || statePrep == operation::prep1) {
    // run z basis
    memory_circuit_mz(stabRound, prep, numData, numAncx, numAncz, numRounds,
                      xVec, zVec);
  } else if (statePrep == operation::prepp || statePrep == operation::prepm) {
    // run x basis
    memory_circuit_mx(stabRound, prep, numData, numAncx, numAncz, numRounds,
                      xVec, zVec);
  } else {
    throw std::runtime_error(
        "sample_memory_circuit_error - invalid requested state prep kernel.");
  }

  platform.reset_exec_ctx();

  // TODO, split out X and Z PCMs
  auto pcm_as_strings = ctx_pcm.result.sequential_data();
  std::vector<double> pcm_probabilities = ctx_pcm.pcm_probabilities.value();
  cudaqx::tensor<uint8_t> pcm_data(
      std::vector<std::size_t>({ctx_pcm_size.pcm_dimensions->first,
                                ctx_pcm_size.pcm_dimensions->second}));
  cudaqx::tensor<uint8_t> mzTable(pcm_as_strings);
  mzTable = mzTable.transpose();
  printf("BMH GOT HERE %s:%d\n", __FILE__, __LINE__);
  // Print dimensions
  printf("mzTable dimensions: %zu x %zu\n", mzTable.shape()[0],
         mzTable.shape()[1]);
  mzTable.dump_bits();
  std::size_t numNoiseMechs = mzTable.shape()[1];
  std::size_t numSyndromesPerRound = numCols;

  // Split mzTable into X and Z
  cudaqx::tensor<uint8_t> mzTableX(std::vector<std::size_t>(
      {numRounds * numSyndromesPerRound / 2, numNoiseMechs}));
  cudaqx::tensor<uint8_t> mzTableZ(std::vector<std::size_t>(
      {numRounds * numSyndromesPerRound / 2, numNoiseMechs}));

  for (std::size_t i = 0, j = 0, k = 0; i < numRounds * numSyndromesPerRound;
       i++) {
    if ((i / (numSyndromesPerRound / 2)) % 2 == 0) {
      for (std::size_t l = 0; l < numNoiseMechs; l++)
        mzTableX.at({j, l}) = mzTable.at({i, l});
      j++;
    } else {
      for (std::size_t l = 0; l < numNoiseMechs; l++)
        mzTableZ.at({k, l}) = mzTable.at({i, l});
      k++;
    }
  }
  printf("XORing %lu rounds\n", numRounds);
  auto numXZPerRound = numSyndromesPerRound / 2;
  cudaqx::tensor<uint8_t> mzTableXORed_x(
      {numRounds * numXZPerRound, numNoiseMechs});
  cudaqx::tensor<uint8_t> mzTableXORed_z(
      {numRounds * numXZPerRound, numNoiseMechs});
  for (std::size_t round = 0; round < numRounds; round++) {
    printf("Round %zu, numXZPerRound %zu\n", round, numXZPerRound);
    if (round == 0) {
      for (std::size_t syndrome = 0; syndrome < numXZPerRound; syndrome++) {
        for (std::size_t noise_mech = 0; noise_mech < numNoiseMechs;
             noise_mech++) {
          mzTableXORed_x.at({round * numXZPerRound + syndrome, noise_mech}) =
              mzTableX.at({round * numXZPerRound + syndrome, noise_mech});
          mzTableXORed_z.at({round * numXZPerRound + syndrome, noise_mech}) =
              mzTableZ.at({round * numXZPerRound + syndrome, noise_mech});
        }
      }
    } else {
      for (std::size_t syndrome = 0; syndrome < numXZPerRound; syndrome++) {
        for (std::size_t noise_mech = 0; noise_mech < numNoiseMechs;
             noise_mech++) {
          mzTableXORed_x.at({round * numXZPerRound + syndrome, noise_mech}) =
              mzTableX.at({round * numXZPerRound + syndrome, noise_mech}) ^
              mzTableX.at({(round - 1) * numXZPerRound + syndrome, noise_mech});
          mzTableXORed_z.at({round * numXZPerRound + syndrome, noise_mech}) =
              mzTableZ.at({round * numXZPerRound + syndrome, noise_mech}) ^
              mzTableZ.at({(round - 1) * numXZPerRound + syndrome, noise_mech});
        }
      }
    }
  }
  printf("BMH GOT HERE %s:%d\n", __FILE__, __LINE__);
  printf("mzTableXORed_x\n");
  mzTableXORed_x.dump_bits();
  printf("mzTableXORed_z\n");
  mzTableXORed_z.dump_bits();
  auto [simplified_pcm_x, simplified_pcm_probabilities_x] =
      simplify_pcm(mzTableXORed_x, pcm_probabilities, numSyndromesPerRound);
  auto [simplified_pcm_z, simplified_pcm_probabilities_z] =
      simplify_pcm(mzTableXORed_z, pcm_probabilities, numSyndromesPerRound);

  return std::make_tuple(simplified_pcm_x, simplified_pcm_probabilities_x,
                         simplified_pcm_z, simplified_pcm_probabilities_z);
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
  cudaq::noise_model noise; // empty noise model
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

} // namespace cudaq::qec
