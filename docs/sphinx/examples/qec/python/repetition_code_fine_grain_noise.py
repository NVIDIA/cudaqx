import cudaq
import cudaq_qec as qec
import numpy as np
from generate_parity_check_matrix import get_code_and_pcm

nRounds = 3
nShots = 100
p = 0.1


@cudaq.kernel
def three_qubit_repetition_code():
    data_qubits = cudaq.qvector(3)
    ancilla_qubits = cudaq.qvector(2)

    # Initialize the logical |1> state as |111>
    x(data_qubits)

    for i in range(nRounds):
        # Random Bit Flip Errors
        for j in range(3):
            cudaq.apply_noise(cudaq.XError, p, data_qubits[j])

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


noise_model = cudaq.NoiseModel()
# Add measurement noise
noise_model.add_all_qubit_channel("mz", cudaq.BitFlipChannel(0.01))

# Run the kernel and observe results
# The percent of samples that are 000 corresponds to the logical error rate
cudaq.set_target("stim")
result = cudaq.sample(three_qubit_repetition_code,
                      shots_count=nShots,
                      noise_model=noise_model,
                      explicit_measurements=True)

# Get the parity check matrix of the repetition code
Hz = [[1, 1, 0], [0, 1, 1]]
code, H_pcm, Mz = get_code_and_pcm(3, nRounds, Hz)

# Get observables
observables = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
Lz = np.array([1, 0, 0], dtype=np.uint8)
print(f"observables:\n{observables}")
print(f"Lz:\n{Lz}")
# Pad the observables to be the same dimension as the decoded observable
Lz_nrounds = np.tile(Lz, nRounds)
Lz_nround_mz = np.pad(Lz_nrounds, (0, Mz.shape[1]), mode='constant')
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
    convergence, result = decoder.decode(syndrome)
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
