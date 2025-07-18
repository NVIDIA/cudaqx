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

# Set target simulator (Stim) for fast stabilizer circuit simulation
cudaq.set_target("stim")

distance = 3         # Code distance (number of physical qubits for repetition code)
nRounds = 4          # Number of syndrome measurement rounds
nShots = 1000        # Number of circuit samples to run

# Retrieve a 3-qubit repetition code instance
three_qubit_repetition_code = qec.get_code("repetition", distance=distance)

# Z logical observable (for repetition codes, only Z matters)
logical_single_round = three_qubit_repetition_code.get_observables_z()

# Use predefined state preparation (|1⟩ for logical '1')
statePrep = qec.operation.prep1

# Create a noise model instance
noise_model = cudaq.NoiseModel()

# Define physical gate error probability
p = 0.01
# Define measurement error probability (not activated by default)
p_per_mz = 0.001

# Inject depolarizing noise on CX gates
noise_model.add_all_qubit_channel("x", cudaq.Depolarization2(p), 1)
# noise_model.add_all_qubit_channel("mz", cudaq.BitFlipChannel(p_per_mz))  # Optional: measurement noise

# === Decoder Setup ===

# Generate full detector error model (DEM), tracking all observables
dem_rep_full = qec.dem_from_memory_circuit(three_qubit_repetition_code, statePrep, nRounds, noise_model)

# Generate Z-only detector error model (sufficient for repetition code)
dem_rep_z = qec.z_dem_from_memory_circuit(three_qubit_repetition_code, statePrep, nRounds, noise_model)

# Extract multi-round parity check matrix (H matrix)
H_pcm_from_dem_full = dem_rep_full.detector_error_matrix
H_pcm_from_dem_z = dem_rep_z.detector_error_matrix

# Sanity check: for repetition codes, full and Z-only matrices should match
assert (H_pcm_from_dem_z == H_pcm_from_dem_full).all()

# Retrieve observable flips matrix: maps physical errors to logical flips
Lz_observables_flips_matrix = dem_rep_z.observables_flips_matrix

# Instantiate a decoder: single-error lookup table (fast and sufficient for small codes)
decoder = qec.get_decoder("single_error_lut", H_pcm_from_dem_z)

# === Simulation ===

# Sample noisy executions of the code circuit
syndromes, data = qec.sample_memory_circuit(
    three_qubit_repetition_code, statePrep, nShots, nRounds, noise_model)

# Initialize Pauli frame (Z only for repetition code)
pauli_frame = np.array([0, 0], dtype=np.uint8)

# Expected logical measurement (we prepared |1⟩)
expected_value = 1

# Counters for statistics
nLogicalErrorsWithoutDecoding = 0
nLogicalErrorsWDecoding = 0
nCorrections = 0

# === Loop over shots ===
for i in range(nShots):
    print(f"shot: {i}")

    data_i = data[i]  # Final data measurement
    print(f"data: {data_i}") 

    # Construct multi-round syndrome vector
    multi_round_syndromes = []
    previous_syndrome = [0] * len(syndromes[0])
    for j in range(nRounds):
        syndrome_i_j = syndromes[i + j]
        print(f"syndrome: {syndrome_i_j}")

        # Compute syndrome difference (flip tracking)
        multi_round_syndromes += list(np.bitwise_xor(previous_syndrome, syndrome_i_j))
        previous_syndrome = syndrome_i_j

    # Decode syndrome into predicted error pattern
    results = decoder.decode(np.array(multi_round_syndromes))
    convergence = results.converged
    result = results.result
    error_prediction = np.array(result, dtype=np.uint8)
    print(f"error_prediction: {error_prediction}")

    # Apply prediction to infer logical flip
    predicted_observable_flip = Lz_observables_flips_matrix @ error_prediction % 2
    print(f"predicted_observable_flip: {predicted_observable_flip}")

    # Check what the circuit actually measured
    measured_obserable = logical_single_round @ data_i % 2
    print(f"measured_obserable: {measured_obserable}")

    # Count error without decoding
    if (measured_obserable != expected_value):
        nLogicalErrorsWithoutDecoding += 1

    # Correct prediction by combining predicted flip with actual measurement
    predicted_observable = predicted_observable_flip ^ measured_obserable
    print(f"predicted_observable: {predicted_observable}")

    # Count logical error after decoding
    if(predicted_observable != expected_value):
        nLogicalErrorsWDecoding += 1 

    # Track how many corrections were made
    nCorrections += int(predicted_observable_flip[0])

# === Summary statistics ===
print(f"{nLogicalErrorsWithoutDecoding} logical errors without decoding in {nShots} shots\n")
print(f"{nLogicalErrorsWDecoding} logical errors with decoding in {nShots} shots\n")
print(f"{nCorrections} corrections applied in {nShots} shots\n")
