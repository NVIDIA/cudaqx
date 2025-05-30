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
# Physical error rate
p_per_round = 0.01
p_per_mz = 0.01


# Construct the measurement error syndrome matrix based on the distance and number of rounds
def construct_measurement_error_syndrome(distance, nRounds):
    num_stabilizers = distance - 1
    num_mea_q = num_stabilizers * nRounds

    syndrome_rows = []

    # In this scheme, need two rounds for each measurement syndrome
    for i in range(nRounds - 1):
        for j in range(num_stabilizers):
            syndrome = np.zeros((num_mea_q,), dtype=np.uint8)

            # The error on ancilla (j) in round (i) affects stabilizer checks at two positions:
            # First occurrence in round i
            pos1 = i * num_stabilizers + j
            # Second occurrence in round i+1
            pos2 = (i + 1) * num_stabilizers + j

            # Mark the syndrome
            syndrome[pos1] = 1
            syndrome[pos2] = 1

            syndrome_rows.append(syndrome)

    return np.array(syndrome_rows).T


# Generate the parity check matrix for n-rounds by duplicating the input parity check matrix Hz
# and appending the measurement error syndrome matrix.
def get_circuit_level_pcm(distance, nRounds, Hz):
    if nRounds < 2:
        raise ValueError("nRounds must be greater than or equal to 2")
    if distance < 3:
        raise ValueError("distance must be greater than or equal to 3")

    # Parity check matrix for a single round
    H = np.array(Hz)

    # Extends H to nRounds
    rows, cols = H.shape
    H_nrounds = np.zeros((rows * nRounds, cols * nRounds), dtype=np.uint8)
    for i in range(nRounds):
        H_nrounds[i * rows:(i + 1) * rows, i * cols:(i + 1) * cols] = H
    print("H_nrounds\n", H_nrounds)

    # Construct the measurement error syndrome matrix for Z errors
    H_mz = construct_measurement_error_syndrome(distance, nRounds)
    print("H_mz\n", H_mz)
    assert H_nrounds.shape[0] == H_mz.shape[
        0], "Dimensions of H_nrounds and H_mz do not match"

    # Append columns for measurement errors to H
    H_pcm = np.concatenate((H_nrounds, H_mz), axis=1)
    print(f"H_pcm:\n{H_pcm}")

    return H_pcm


# Example of how to construct a repetition code with a distance of 3 and random
# bit flip errors applied to the data qubits
@cudaq.kernel
def three_qubit_repetition_code():
    data_qubits = cudaq.qvector(3)
    ancilla_qubits = cudaq.qvector(2)

    # Initialize the logical |1> state as |111>
    x(data_qubits)

    for i in range(nRounds):
        # Random Bit Flip Errors
        for j in range(3):
            cudaq.apply_noise(cudaq.XError, p_per_round, data_qubits[j])

        # Extract Syndromes
        h(ancilla_qubits)

        # First Parity Check
        z.ctrl(ancilla_qubits[0], data_qubits[0])
        z.ctrl(ancilla_qubits[0], data_qubits[1])

        # Second Parity Check
        z.ctrl(ancilla_qubits[1], data_qubits[1])
        z.ctrl(ancilla_qubits[1], data_qubits[2])

        h(ancilla_qubits)

        # Measure the ancilla qubits
        s0 = mz(ancilla_qubits[0])
        s1 = mz(ancilla_qubits[1])
        reset(ancilla_qubits[0])
        reset(ancilla_qubits[1])

    # Final measurement to get the data qubits
    mz(data_qubits)


# Create a noise model
noise_model = cudaq.NoiseModel()
# Add measurement noise
noise_model.add_all_qubit_channel("mz", cudaq.BitFlipChannel(p_per_mz))

# Run the kernel and observe results
# The percent of samples that are 000 corresponds to the logical error rate
cudaq.set_target("stim")
result = cudaq.sample(three_qubit_repetition_code,
                      shots_count=nShots,
                      noise_model=noise_model,
                      explicit_measurements=True)

# The following section will demonstrate how to decode the results
# Get the parity check matrix for n-rounds of the repetition code
Hz = [[1, 1, 0], [0, 1, 1]]  # Parity check matrix for 1 round
H_pcm = get_circuit_level_pcm(3, nRounds, Hz)

# Get observables
observables = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
Lz = np.array([1, 0, 0], dtype=np.uint8)
print(f"observables:\n{observables}")
print(f"Lz:\n{Lz}")
# Pad the observables to be the same dimension as the decoded observable
Lz_nrounds = np.tile(Lz, nRounds)
pad_size = max(0, H_pcm.shape[1] - Lz_nrounds.shape[0])
Lz_nround_mz = np.pad(Lz_nrounds, (0, pad_size), mode='constant')
print(f"Lz_nround_mz\n{Lz_nround_mz}")

# Get a decoder
decoder = qec.get_decoder("single_error_lut", H_pcm)
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
    convergence, result, opt = decoder.decode(syndrome)
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
