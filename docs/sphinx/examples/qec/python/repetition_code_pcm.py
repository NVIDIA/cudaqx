# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# [Begin Documentation]
import cudaq
import cudaq_qec as qec
import numpy as np

nRounds = 3
nShots = 500

# Retrieve an instance of the three-qubit repetition code for 3 rounds
three_qubit_repetition_code = qec.get_code("repetition", distance=3)
xn = three_qubit_repetition_code.get_num_x_stabilizers()
# Retrieve the state preparation kernel
statePrep = qec.operation.prep1

# Create a noise model
noise_model = cudaq.NoiseModel()

# Physical error rate
p = 0.001

# Add measurement noise
p_per_mz = 0.01


noise_model.add_all_qubit_channel("x", cudaq.Depolarization2(p), 1)
noise_model.add_all_qubit_channel("mz", cudaq.BitFlipChannel(p_per_mz))

# The following section will demonstrate how to decode the results
# Get the parity check matrix for n-rounds of the repetition code

dem_rep = qec.dem_from_memory_circuit(three_qubit_repetition_code, statePrep, nRounds, noise_model)
H_pcm_from_dem = dem_rep.detector_error_matrix

# Get observables
# such as surface codes or other stabilizer codes.

Lz_nround_mz = dem_rep.observables_flips_matrix

# Get a decoder
decoder = qec.get_decoder("single_error_lut", H_pcm_from_dem)
nLogicalErrors = 0

# initialize a Pauli frame to track logical flips
# through the stabilizer rounds. Only need the Z component for the repetition code.
pauli_frame = np.array([0, 0], dtype=np.uint8)
expected_value = 1
for shot, outcome in enumerate(result.get_sequential_data()):
    outcome_array = np.array([int(bit) for bit in outcome], dtype=np.uint8)
    syndrome = outcome_array[:len(outcome_array) - 3]
    data = outcome_array[len(outcome_array) - 3:]
    print("\nshot:", shot)
    print("syndrome:", syndrome)

    # Decode the syndrome
    results = decoder.decode(syndrome)
    convergence = results.converged
    result = results.result
    data_prediction = np.array(result, dtype=np.uint8)

    # See if the decoded result anti-commutes with the observables
    print("decode result:", data_prediction)
    decoded_observables = (Lz_nround_mz @ data_prediction) % 2
    print("decoded_observables:", decoded_observables)

    # update pauli frame
    pauli_frame[0] = (pauli_frame[0] + decoded_observables) % 2
    print("pauli frame:", pauli_frame)

    logical_measurements = (Lz @ data.transpose()) % 2
    print("LMz:", logical_measurements)

    corrected_mz = (logical_measurements + pauli_frame[0]) % 2
    print("Expected value:", expected_value)
    print("Corrected value:", corrected_mz)
    if (corrected_mz != expected_value):
        nLogicalErrors += 1

# Count how many shots the decoder failed to correct the errors
print("\nNumber of logical errors:", nLogicalErrors)
