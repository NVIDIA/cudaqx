/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
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

  cudaq::ExecutionContext ctx("");
  ctx.noiseModel = &noise;
  auto &platform = cudaq::get_platform();

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

  // Run the memory circuit experiment
  if (statePrep == operation::prep0 || statePrep == operation::prep1) {
    // run z basis
    for (std::size_t shot = 0; shot < numShots; shot++) {
      platform.set_exec_ctx(&ctx);
      memory_circuit_mz(stabRound, prep, numData, numAncx, numAncz, numRounds,
                        xVec, zVec);
      platform.reset_exec_ctx();
    }
  } else if (statePrep == operation::prepp || statePrep == operation::prepm) {
    // run z basis
    for (std::size_t shot = 0; shot < numShots; shot++) {
      platform.set_exec_ctx(&ctx);
      memory_circuit_mx(stabRound, prep, numData, numAncx, numAncz, numRounds,
                        xVec, zVec);
      platform.reset_exec_ctx();
    }
  } else {
    throw std::runtime_error(
        "sample_memory_circuit_error - invalid requested state prep kernel.");
  }

  auto &dataMeasures = getMemoryCircuitDataMeasurements();
  dataResults.copy(dataMeasures.data());

  // Get the raw ancilla measurments
  auto &measurements = getMemoryCircuitAncillaMeasurements();

  std::size_t numMeasRows = numShots * numRounds;
  cudaqx::tensor<uint8_t> measuresTensor({numMeasRows, numCols});
  measuresTensor.borrow(measurements.data());


  // First round, store bare syndrome measurement
  for (std::size_t col = 0; col < numCols; ++col) {
    std::size_t shot = 0;
    std::size_t round = 0;
    std::size_t measIdx = shot * numRounds + round;
    syndromeTensor.at({measIdx, col}) =
      measuresTensor.at({measIdx, col});
  }

  // After first round, store syndrome flips
  // #pragma omp parallel for collapse(2)
  for (std::size_t shot = 0; shot < numShots; ++shot)
    for (std::size_t round = 1; round < numRounds; ++round)
      for (std::size_t col = 0; col < numCols; ++col) {
        std::size_t measIdx = shot * numRounds + round;
        std::size_t prevMeasIdx = shot * numRounds + (round - 1);
        syndromeTensor.at({measIdx, col}) =
            measuresTensor.at({measIdx, col}) ^
            measuresTensor.at({prevMeasIdx, col});
      }

  clearRawMeasurements();

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

} // namespace cudaq::qec
