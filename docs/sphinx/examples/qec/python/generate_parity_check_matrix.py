import cudaq
import cudaq_qec as qec
import numpy as np

# Set the number of shots and rounds
nShots = 30
nRounds = 5
distance = 4
cudaq.set_target("stim")


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


def get_code_and_pcm(distance, nRounds):
    if nRounds < 2:
        raise ValueError("nRounds must be greater than or equal to 2")
    if distance < 3:
        raise ValueError("distance must be greater than or equal to 3")

    # Create distance-3 repetition code
    code = qec.get_code("repetition", distance=distance)

    # Get the parity check matrix for a single round
    Hz = code.get_parity_z()
    H = np.array(Hz)

    # Extends H to nRounds
    rows, cols = H.shape
    H_nrounds = np.zeros((rows * nRounds, cols * nRounds), dtype=np.uint8)
    for i in range(nRounds):
        H_nrounds[i * rows:(i + 1) * rows, i * cols:(i + 1) * cols] = H
    print("H_nrounds\n", H_nrounds)

    # Append columns for measurement errors to H
    Mz = construct_measurement_error_syndrome(distance, nRounds)
    print("Mz\n", Mz)
    assert H_nrounds.shape[0] == Mz.shape[
        0], "Dimensions of H_nrounds and Mz do not match"
    H_pcm = np.concatenate((H_nrounds, Mz), axis=1)
    print(f"H_pcm:\n{H_pcm}")

    return code, H_pcm, Mz


code, H_pcm, Mz = get_code_and_pcm(distance, nRounds)

# Define the error probability
p = 0.01
noise = cudaq.NoiseModel()
noise.add_all_qubit_channel("x", cudaq.Depolarization2(p), 1)
noise.add_all_qubit_channel("mz", cudaq.BitFlipChannel(p))

# Prepare the logical |0> state
statePrep = qec.operation.prep0
expected_value = 0

# Sample the memory circuit with noise
syndromes, data = qec.sample_memory_circuit(code, statePrep, nShots, nRounds,
                                            noise)
print("From sample function:\n")
print("syndromes:\n", syndromes)
print("data:\n", data)

# Combine nRounds of syndromes per shot into one vector
syndromes_nrounds = np.array(syndromes).reshape(
    nShots, nRounds * len(code.get_stabilizers()))
print(f"syndromes:\n{syndromes_nrounds}")

# Get observables
observables = code.get_pauli_observables_matrix()
Lz = code.get_observables_z()
print(f"observables:\n{observables}")
print(f"Lz:\n{Lz}")
# Pad the observables to be the same dimension as the decoded observable
Lz_nrounds = np.tile(Lz[0], nRounds)
Lz_nround_mz = np.pad(Lz_nrounds, (0, Mz.shape[1]), mode='constant')
print(f"Lz_nround_mz\n{Lz_nround_mz}")

# Get a decoder
decoder = qec.get_decoder("single_error_lut", H_pcm)
nLogicalErrors = 0

# Logical Mz each shot (use Lx if preparing in X-basis)
logical_measurements = (Lz @ data.transpose()) % 2
# only one logical qubit, so do not need the second axis
logical_measurements = logical_measurements.flatten()
print("LMz:\n", logical_measurements)

# initialize a Pauli frame to track logical flips
# through the stabilizer rounds. Only need the Z component for the repetition code.
pauli_frame = np.array([0, 0], dtype=np.uint8)

for shot, syndrome in enumerate(syndromes_nrounds):
    print("\nshot:", shot)
    print("syndrome:", syndrome)

    # Decode the syndrome
    convergence, result = decoder.decode(syndrome)
    data_prediction = np.array(result, dtype=np.uint8)

    # See if the decoded result anti-commutes with the observables
    print("decode result:", data_prediction)
    decoded_observables = (Lz_nround_mz @ data_prediction) % 2
    print("decoded_observables:", decoded_observables)

    # update pauli frame
    pauli_frame[0] = (pauli_frame[0] + decoded_observables) % 2
    print("pauli frame:", pauli_frame)

    corrected_mz = (logical_measurements[shot] + pauli_frame[0]) % 2
    print("Expected value:", expected_value)
    print("Corrected value:", corrected_mz)
    if (corrected_mz != expected_value):
        nLogicalErrors += 1

# Count how many shots the decoder failed to correct the errors
print("\nNumber of logical errors:", nLogicalErrors)
