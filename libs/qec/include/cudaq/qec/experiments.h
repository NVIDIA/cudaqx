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

/// @brief Sample syndrome measurements with circuit-level noise
/// @param statePrep Initial state preparation operation
/// @param numShots Number of measurement shots
/// @param numRounds Number of stabilizer measurement rounds
/// @param noise Noise model to apply
/// @return Tuple containing syndrome measurements and data qubit
/// measurements (mz for z basis state prep, mx for x basis)
std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_memory_circuit(const code &code, operation statePrep,
                      std::size_t numShots, std::size_t numRounds,
                      cudaq::noise_model &noise);

/// @brief Sample syndrome measurements from the memory circuit
/// @param statePrep Initial state preparation operation
/// @param numShots Number of measurement shots
/// @param numRounds Number of stabilizer measurement rounds
/// @return Tuple containing syndrome measurements and data qubit
/// measurements (mz for z basis state prep, mx for x basis)
std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_memory_circuit(const code &code, operation statePrep,
                      std::size_t numShots, std::size_t numRounds = 1);

/// @brief Sample syndrome measurements starting from |0⟩ state
/// @param numShots Number of measurement shots
/// @param numRounds Number of stabilizer measurement rounds
/// @return Tuple containing syndrome measurements and data qubit
/// measurements (mz for z basis state prep, mx for x basis)
std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_memory_circuit(const code &code, std::size_t numShots,
                      std::size_t numRounds = 1);

/// @brief Sample syndrome measurements from |0⟩ state with noise
/// @param numShots Number of measurement shots
/// @param numRounds Number of stabilizer measurement rounds
/// @param noise Noise model to apply
/// @return Tuple containing syndrome measurements and data qubit
/// measurements (mz for z basis state prep, mx for x basis)
std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_memory_circuit(const code &code, std::size_t numShots,
                      std::size_t numRounds, cudaq::noise_model &noise);

/// @brief Given a memory circuit setup, generate a DEM
/// @param code QEC Code to sample
/// @param statePrep Initial state preparation operation
/// @param numRounds Number of stabilizer measurement rounds
/// @param noise Noise model to apply
/// @return Detector error model
cudaq::qec::detector_error_model
dem_from_memory_circuit(const code &code, operation statePrep,
                        std::size_t numRounds, cudaq::noise_model &noise);

/// @brief Given a memory circuit setup, generate a DEM for X stabilizers.
/// @param code QEC Code to sample
/// @param statePrep Initial state preparation operation
/// @param numRounds Number of stabilizer measurement rounds
/// @param noise Noise model to apply
/// @return Detector error model
detector_error_model x_dem_from_memory_circuit(const code &code,
                                               operation statePrep,
                                               std::size_t numRounds,
                                               cudaq::noise_model &noise);

/// @brief Given a memory circuit setup, generate a DEM for Z stabilizers.
/// @param code QEC Code to sample
/// @param statePrep Initial state preparation operation
/// @param numRounds Number of stabilizer measurement rounds
/// @param noise Noise model to apply
/// @return Detector error model
detector_error_model z_dem_from_memory_circuit(const code &code,
                                               operation statePrep,
                                               std::size_t numRounds,
                                               cudaq::noise_model &noise);
} // namespace cudaq::qec
