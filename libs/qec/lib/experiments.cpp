/*******************************************************************************
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/experiments.h"
#include "device/memory_circuit.h"
#include "cudaq/algorithms/dem.h"
#include "cudaq/qec/dem_sampling.h"
#include "cudaq/qec/pcm_utils.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>

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

} // namespace cudaq::qec

namespace cudaq::qec::dem_sampler::cpu {

std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_dem(const cudaqx::tensor<uint8_t> &check_matrix, std::size_t numShots,
           const std::vector<double> &error_probabilities, unsigned seed) {
  if (check_matrix.rank() != 2)
    throw std::invalid_argument("check_matrix must be rank-2");

  size_t num_checks = check_matrix.shape()[0];
  size_t num_error_mechanisms = check_matrix.shape()[1];

  if (error_probabilities.size() != num_error_mechanisms)
    throw std::invalid_argument(
        "error_probabilities size must match number of error mechanisms");

  for (double p : error_probabilities) {
    if (!std::isfinite(p) || p < 0.0 || p > 1.0)
      throw std::invalid_argument(
          "error_probabilities entries must be finite values in [0, 1]");
  }

  if (numShots == 0) {
    cudaqx::tensor<uint8_t> errors({0, num_error_mechanisms});
    cudaqx::tensor<uint8_t> checks({0, num_checks});
    return std::make_tuple(checks, errors);
  }

  std::mt19937 rng(seed);
  std::vector<std::bernoulli_distribution> distributions;
  distributions.reserve(num_error_mechanisms);
  for (double p : error_probabilities)
    distributions.emplace_back(p);

  cudaqx::tensor<uint8_t> H_binary({num_checks, num_error_mechanisms});
  std::vector<uint8_t> h_bin(num_checks * num_error_mechanisms);
  for (size_t i = 0; i < num_checks; ++i)
    for (size_t j = 0; j < num_error_mechanisms; ++j)
      h_bin[i * num_error_mechanisms + j] =
          check_matrix.at({i, j}) & static_cast<uint8_t>(1);
  H_binary.copy(h_bin.data(), H_binary.shape());

  cudaqx::tensor<uint8_t> errors({numShots, num_error_mechanisms});
  cudaqx::tensor<uint8_t> checks({numShots, num_checks});

  std::vector<uint8_t> error_bits(numShots * num_error_mechanisms);
  for (size_t shot = 0; shot < numShots; ++shot) {
    for (size_t err = 0; err < num_error_mechanisms; ++err) {
      error_bits[shot * num_error_mechanisms + err] = distributions[err](rng);
    }
  }

  errors.copy(error_bits.data(), errors.shape());
  checks = errors.dot(H_binary.transpose()) % 2;

  return std::make_tuple(checks, errors);
}

std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_dem(const cudaqx::tensor<uint8_t> &check_matrix, std::size_t numShots,
           const std::vector<double> &error_probabilities) {
  return sample_dem(check_matrix, numShots, error_probabilities,
                    std::random_device()());
}

} // namespace cudaq::qec::dem_sampler::cpu

namespace cudaq::qec {

std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
dem_sampling(const cudaqx::tensor<uint8_t> &check_matrix, std::size_t nShots,
             const std::vector<double> &error_probabilities) {
  return dem_sampler::cpu::sample_dem(check_matrix, nShots,
                                      error_probabilities);
}

std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
dem_sampling(const cudaqx::tensor<uint8_t> &check_matrix, std::size_t nShots,
             const std::vector<double> &error_probabilities, unsigned seed) {
  return dem_sampler::cpu::sample_dem(check_matrix, nShots, error_probabilities,
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
  if (!(statePrep == operation::prep0 || statePrep == operation::prep1 ||
        statePrep == operation::prepp || statePrep == operation::prepm))
    throw std::runtime_error(
        "sample_memory_circuit_error - invalid requested state prep kernel.");

  bool is_z_prep =
      statePrep == operation::prep0 || statePrep == operation::prep1;

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
  auto logical_obs =
      is_z_prep ? code.get_observables_z() : code.get_observables_x();
  const std::size_t num_obs = logical_obs.shape()[0];
  std::vector<std::size_t> obs_flat(logical_obs.data(),
                                    logical_obs.data() + logical_obs.size());

  const std::size_t numCols = numAncx + numAncz;

  // Obtain the Measurement-to-Detector (M2D) sparse matrix.
  // m2d.rows[d] = set of chronological measurement indices whose XOR = detector
  // d.
  cudaq::M2DSparseMatrix m2d;
  cudaq::detail::runDemFromKernel(
      cudaq::getKernelName(memory_circuit), cudaq::get_platform(), &noise,
      [&] {
        memory_circuit(stabRound, prep, numData, numAncx, numAncz, numRounds,
                       xVec, zVec, obs_flat, num_obs, !is_z_prep);
      },
      /*options=*/{}, /*plugin_name=*/"stim", &m2d);

  // Sample the memory circuit and collect all raw measurements.
  cudaq::sample_options opts{
      .shots = numShots, .noise = noise, .explicit_measurements = true};
  auto result = cudaq::sample(opts, memory_circuit, stabRound, prep, numData,
                              numAncx, numAncz, numRounds, xVec, zVec, obs_flat,
                              num_obs, !is_z_prep);

  // mzTable[shot, meas_idx] = raw 0/1 outcome; shape (numShots,
  // numMeasPerShot). Measurement layout per shot: numRounds*numCols ancilla,
  // then numData qubits.
  cudaqx::tensor<uint8_t> mzTable(result.sequential_data());

  // Data results: tail numData measurements of each shot.
  cudaqx::tensor<uint8_t> dataResults({numShots, numData});
  for (std::size_t shot = 0; shot < numShots; ++shot)
    std::memcpy(&dataResults.at({shot, 0}),
                &mzTable.at({shot, numCols * numRounds}), numData);

  // syndromeTensor = M2D @ mzTable   (per shot).
  // Output shape: (numShots, k) where k = number of detectors
  std::size_t k = m2d.rows.size();

  cudaqx::tensor<uint8_t> syndromeTensor({numShots, k});
  for (std::size_t shot = 0; shot < numShots; ++shot) {
    const uint8_t *measRow = &mzTable.at({shot, 0});
    std::size_t d = 0;
    for (const auto &rowIdx : m2d.rows) {
      uint8_t val = 0;
      for (std::size_t m : rowIdx) {
        val ^= measRow[m];
      }
      syndromeTensor.at({shot, d++}) = val;
    }
  }

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
/// function that all of the function overloads invoke.
cudaq::qec::detector_error_model
dem_from_memory_circuit(const code &code, operation statePrep,
                        std::size_t numRounds, cudaq::noise_model &noise,
                        bool keep_x_stabilizers, bool keep_z_stabilizers,
                        bool decompose_errors) {
  if (!keep_x_stabilizers && !keep_z_stabilizers)
    throw std::runtime_error("dem_from_memory_circuit error - no stabilizers "
                             "to keep.");
  if (!code.contains_operation(statePrep))
    throw std::runtime_error("dem_from_memory_circuit error - requested state "
                             "prep kernel not found.");
  if (!code.contains_operation(operation::stabilizer_round))
    throw std::runtime_error("dem_from_memory_circuit error - no stabilizer "
                             "round kernel for this code.");
  if (!(statePrep == operation::prep0 || statePrep == operation::prep1 ||
        statePrep == operation::prepp || statePrep == operation::prepm))
    throw std::runtime_error(
        "dem_from_memory_circuit - invalid requested state prep kernel.");

  auto &prep = code.get_operation<code::one_qubit_encoding>(statePrep);
  auto &stabRound =
      code.get_operation<code::stabilizer_round>(operation::stabilizer_round);
  auto parity_x = code.get_parity_x();
  auto parity_z = code.get_parity_z();
  auto numData = code.get_num_data_qubits();
  auto numAncx = code.get_num_ancilla_x_qubits();
  auto numAncz = code.get_num_ancilla_z_qubits();
  bool is_z_prep =
      statePrep == operation::prep0 || statePrep == operation::prep1;
  std::vector<std::size_t> xVec(parity_x.data(),
                                parity_x.data() + parity_x.size());
  std::vector<std::size_t> zVec(parity_z.data(),
                                parity_z.data() + parity_z.size());

  auto logical_obs =
      is_z_prep ? code.get_observables_z() : code.get_observables_x();
  const std::size_t num_obs = logical_obs.shape()[0];
  std::vector<std::size_t> obs_flat(logical_obs.data(),
                                    logical_obs.data() + logical_obs.size());

  cudaq::dem_options dem_opts;
  dem_opts.decompose_errors = decompose_errors;
  auto dem_text = cudaq::dem_from_kernel(
      memory_circuit, &noise, dem_opts, stabRound, prep, numData, numAncx,
      numAncz, numRounds, xVec, zVec, obs_flat, num_obs, !is_z_prep);
  auto dem = cudaq::qec::dem_from_stim_text(dem_text, decompose_errors);

  const auto numXStabs = code.get_num_x_stabilizers();
  const auto numZStabs = code.get_num_z_stabilizers();
  const auto numSyndromesPerRound = numXStabs + numZStabs;

  std::size_t numReturnSynPerRound = numSyndromesPerRound;
  if (keep_x_stabilizers && !keep_z_stabilizers) {
    numReturnSynPerRound = numXStabs;
  } else if (!keep_x_stabilizers && keep_z_stabilizers) {
    numReturnSynPerRound = numZStabs;
  }

  if (keep_x_stabilizers != keep_z_stabilizers) {
    const auto numNoiseMechs = dem.num_error_mechanisms();
    const auto rowOffset = keep_x_stabilizers ? numZStabs : 0;
    std::vector<std::uint32_t> all_columns(numNoiseMechs);
    std::iota(all_columns.begin(), all_columns.end(), 0);
    cudaqx::tensor<uint8_t> sliced(std::vector<std::size_t>{
        numRounds * numReturnSynPerRound, numNoiseMechs});

    for (std::size_t round = 0; round < numRounds; ++round) {
      const auto src_row_begin =
          static_cast<std::uint32_t>(round * numSyndromesPerRound + rowOffset);
      const auto src_row_end =
          src_row_begin + static_cast<std::uint32_t>(numReturnSynPerRound) - 1;
      auto round_block = cudaq::qec::reorder_pcm_columns(
          dem.detector_error_matrix, all_columns, src_row_begin, src_row_end);

      auto *dst = sliced.data() + round * numReturnSynPerRound * numNoiseMechs;
      std::memcpy(dst, round_block.data(),
                  numReturnSynPerRound * numNoiseMechs * sizeof(uint8_t));
    }
    dem.detector_error_matrix = std::move(sliced);
    dem.canonicalize_for_rounds(numReturnSynPerRound,
                                /*remove_zero_syndrome_errors=*/true);
  }

  return dem;
}
} // namespace details

/// @brief Given a memory circuit setup, generate a DEM
cudaq::qec::detector_error_model
dem_from_memory_circuit(const code &code, operation statePrep,
                        std::size_t numRounds, cudaq::noise_model &noise,
                        bool decompose_errors) {
  return details::dem_from_memory_circuit(code, statePrep, numRounds, noise,
                                          /*keep_x_stabilizers=*/true,
                                          /*keep_z_stabilizers=*/true,
                                          decompose_errors);
}

// For CSS codes, may want to partition x vs z decoding
detector_error_model x_dem_from_memory_circuit(const code &code,
                                               operation statePrep,
                                               std::size_t numRounds,
                                               cudaq::noise_model &noise,
                                               bool decompose_errors) {
  return details::dem_from_memory_circuit(code, statePrep, numRounds, noise,
                                          /*keep_x_stabilizers=*/true,
                                          /*keep_z_stabilizers=*/false,
                                          decompose_errors);
}

detector_error_model z_dem_from_memory_circuit(const code &code,
                                               operation statePrep,
                                               std::size_t numRounds,
                                               cudaq::noise_model &noise,
                                               bool decompose_errors) {
  return details::dem_from_memory_circuit(code, statePrep, numRounds, noise,
                                          /*keep_x_stabilizers=*/false,
                                          /*keep_z_stabilizers=*/true,
                                          decompose_errors);
}

} // namespace cudaq::qec
