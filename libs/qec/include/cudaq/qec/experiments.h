/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq/qec/code.h"
#include "cudaq/qec/detector_error_model.h"
#include <tuple>
#include <vector>

namespace cudaq::qec {

/// @brief Generate rank-1 tensor of random bit flips
/// @param numBits Number of bits in tensor
/// @param error_probability Probability of bit flip on data
/// @return Tensor of randomly flipped bits
cudaqx::tensor<uint8_t> generate_random_bit_flips(size_t numBits,
                                                  double error_probability);

/// @brief Sample syndrome measurements with code capacity noise
/// @param H Parity check matrix of a QEC code
/// @param numShots Number of measurement shots
/// @param error_probability Probability of bit flip on data
/// @return Tuple containing syndrome measurements and data qubit
/// measurements
std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_code_capacity(const cudaqx::tensor<uint8_t> &H, std::size_t numShots,
                     double error_probability);

/// @brief Sample syndrome measurements with code capacity noise
/// @param H Parity check matrix of a QEC code
/// @param numShots Number of measurement shots
/// @param error_probability Probability of bit flip on data
/// @param seed RNG seed for reproducible experiments
/// @return Tuple containing syndrome measurements and data qubit
/// measurements
std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_code_capacity(const cudaqx::tensor<uint8_t> &H, std::size_t numShots,
                     double error_probability, unsigned seed);

/// @brief Sample syndrome measurements with code capacity noise
/// @param code QEC Code to sample
/// @param numShots Number of measurement shots
/// @param error_probability Probability of bit flip on data
/// @param seed RNG seed for reproducible experiments
/// @return Tuple containing syndrome measurements and data qubit
/// measurements
std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_code_capacity(const code &code, std::size_t numShots,
                     double error_probability, unsigned seed);

/// @brief Sample syndrome measurements with code capacity noise
/// @param code QEC Code to sample
/// @param numShots Number of measurement shots
/// @param error_probability Probability of bit flip on data
/// @return Tuple containing syndrome measurements and data qubit
/// measurements
std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_code_capacity(const code &code, std::size_t numShots,
                     double error_probability);

/// @brief Sample a detector error model on CPU (legacy API).
/// @param check_matrix Binary matrix [num_checks x num_error_mechanisms]
/// @param numShots Number of independent Monte-Carlo shots
/// @param error_probabilities Per-error-mechanism Bernoulli probabilities
/// @return Tuple of (checks, errors)
std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
dem_sampling(const cudaqx::tensor<uint8_t> &check_matrix, std::size_t numShots,
             const std::vector<double> &error_probabilities);

/// @brief Sample a detector error model on CPU (legacy API, seeded).
/// @param check_matrix Binary matrix [num_checks x num_error_mechanisms]
/// @param numShots Number of independent Monte-Carlo shots
/// @param error_probabilities Per-error-mechanism Bernoulli probabilities
/// @param seed RNG seed for reproducibility
/// @return Tuple of (checks, errors)
std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
dem_sampling(const cudaqx::tensor<uint8_t> &check_matrix, std::size_t numShots,
             const std::vector<double> &error_probabilities, unsigned seed);

/// @brief Sample syndrome measurements with circuit-level noise
/// @param code QEC Code to sample
/// @param statePrep Initial state preparation operation
/// @param numShots Number of measurement shots
/// @param numRounds Number of stabilizer measurement rounds
/// @param noise Noise model to apply
/// @return Tuple of (syndromeTensor, dataResults). `syndromeTensor` has shape
/// `(numShots, numDetectors)`: `numFixed` boundary detectors (the stabilizer
/// type matching `statePrep`'s basis, since only that type is deterministic
/// at the circuit's endpoints), then one detector block per each of the
/// `numRounds - 1` inter-round transitions, then `numFixed` more boundary
/// detectors. `dataResults` has shape `(numShots, numDataQubits)`.
std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_memory_circuit(const code &code, operation statePrep,
                      std::size_t numShots, std::size_t numRounds,
                      cudaq::noise_model &noise);

/// @brief Sample syndrome measurements with circuit-level noise, keeping only
/// the X stabilizer syndromes.
/// @param code QEC Code to sample
/// @param statePrep Initial state preparation operation
/// @param numShots Number of measurement shots
/// @param numRounds Number of stabilizer measurement rounds
/// @param noise Noise model to apply
/// @return Tuple of (syndromeTensor, dataResults), same layout as
/// `sample_memory_circuit` but restricted to X-stabilizer detector values.
std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
x_sample_memory_circuit(const code &code, operation statePrep,
                        std::size_t numShots, std::size_t numRounds,
                        cudaq::noise_model &noise);

/// @brief Sample syndrome measurements with circuit-level noise, keeping only
/// the Z stabilizer syndromes.
/// @param code QEC Code to sample
/// @param statePrep Initial state preparation operation
/// @param numShots Number of measurement shots
/// @param numRounds Number of stabilizer measurement rounds
/// @param noise Noise model to apply
/// @return Tuple of (syndromeTensor, dataResults), same layout as
/// `sample_memory_circuit` but restricted to Z-stabilizer detector values.
std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
z_sample_memory_circuit(const code &code, operation statePrep,
                        std::size_t numShots, std::size_t numRounds,
                        cudaq::noise_model &noise);

/// @brief Sample syndrome measurements from the memory circuit
/// @param code QEC Code to sample
/// @param statePrep Initial state preparation operation
/// @param numShots Number of measurement shots
/// @param numRounds Number of stabilizer measurement rounds
/// @return Tuple of (syndromeTensor, dataResults), equivalent to the
/// `operation::prep0` overload of `sample_memory_circuit`.
std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_memory_circuit(const code &code, operation statePrep,
                      std::size_t numShots, std::size_t numRounds = 1);

/// @brief Sample syndrome measurements starting from |0⟩ state
/// @param code QEC Code to sample
/// @param numShots Number of measurement shots
/// @param numRounds Number of stabilizer measurement rounds
/// @return Tuple of (syndromeTensor, dataResults), equivalent to the
/// `operation::prep0` overload of `sample_memory_circuit`.
std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_memory_circuit(const code &code, std::size_t numShots,
                      std::size_t numRounds = 1);

/// @brief Sample syndrome measurements from |0⟩ state with noise
/// @param code QEC Code to sample
/// @param numShots Number of measurement shots
/// @param numRounds Number of stabilizer measurement rounds
/// @param noise Noise model to apply
/// @return Tuple of (syndromeTensor, dataResults), equivalent to the
/// `operation::prep0` overload of `sample_memory_circuit`.
std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_memory_circuit(const code &code, std::size_t numShots,
                      std::size_t numRounds, cudaq::noise_model &noise);

/// @brief Given a memory circuit setup, generate a DEM
/// @param code QEC Code to sample
/// @param statePrep Initial state preparation operation
/// @param numRounds Number of stabilizer measurement rounds
/// @param noise Noise model to apply
/// @param decompose_errors If true, hyperedge error mechanisms are decomposed
///        into pairs of two-detector edges by Stim before returning.
/// @return Detector error model
cudaq::qec::detector_error_model
dem_from_memory_circuit(const code &code, operation statePrep,
                        std::size_t numRounds, cudaq::noise_model &noise,
                        bool decompose_errors = false);

/// @brief Given a memory circuit setup, generate a DEM for X stabilizers.
/// @param code QEC Code to sample
/// @param statePrep Initial state preparation operation
/// @param numRounds Number of stabilizer measurement rounds
/// @param noise Noise model to apply
/// @param decompose_errors If true, hyperedge error mechanisms are decomposed
///        into pairs of two-detector edges by Stim before returning.
/// @return Detector error model
detector_error_model x_dem_from_memory_circuit(const code &code,
                                               operation statePrep,
                                               std::size_t numRounds,
                                               cudaq::noise_model &noise,
                                               bool decompose_errors = false);

/// @brief Given a memory circuit setup, generate a DEM for Z stabilizers.
/// @param code QEC Code to sample
/// @param statePrep Initial state preparation operation
/// @param numRounds Number of stabilizer measurement rounds
/// @param noise Noise model to apply
/// @param decompose_errors If true, hyperedge error mechanisms are decomposed
///        into pairs of two-detector edges by Stim before returning.
/// @return Detector error model
detector_error_model z_dem_from_memory_circuit(const code &code,
                                               operation statePrep,
                                               std::size_t numRounds,
                                               cudaq::noise_model &noise,
                                               bool decompose_errors = false);
} // namespace cudaq::qec
