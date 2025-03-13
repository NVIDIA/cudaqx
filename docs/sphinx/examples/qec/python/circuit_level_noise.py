import numpy as np
import cudaq
import cudaq_qec as qec

# Get a QEC code
cudaq.set_target("stim")
steane = qec.get_code("steane")

# Get the parity check matrix of a code
# Can get the full code, or for CSS codes
# just the X or Z component
H = steane.get_parity()
print(f"H:\n{H}")
observables = steane.get_pauli_observables_matrix()
Lz = steane.get_observables_z()
print(f"observables:\n{observables}")
print(f"Lz:\n{Lz}")

nShots = 3
nRounds = 4

# error probabily
p = 0.01
noise = cudaq.NoiseModel()
noise.add_all_qubit_channel("x", cudaq.Depolarization2(p), 1)

# prepare logical |0> state, tells the sampler to do z-basis experiment
statePrep = qec.operation.prep0
# our expected measurement in this state is 0
expected_value = 0

# sample the steane memory circuit with noise on each cx gate
# reading out the syndromes after each stabilizer round (xor'd against the previous)
# and readout out the data qubits at the end of the experiment
syndromes, data = qec.sample_memory_circuit(steane, statePrep, nShots, nRounds,
                                            noise)
print("From sample function:\n")
print("syndromes:\n", syndromes)
print("data:\n", data)

# Get a decoder
decoder = qec.get_decoder("single_error_lut", H)
nLogicalErrors = 0

# Logical Mz each shot (use Lx if preparing in X-basis)
logical_measurements = (Lz @ data.transpose()) % 2
# only one logical qubit, so do not need the second axis
logical_measurements = logical_measurements.flatten()
print("LMz:\n", logical_measurements)

# initialize a Pauli frame to track logical flips
# through the stabilizer rounds
pauli_frame = np.array([0, 0], dtype=np.uint8)
for shot in range(0, nShots):
    print("shot:", shot)
    for syndrome in syndromes:
        print("syndrome:", syndrome)
        # decode the syndrome
        convergence, result = decoder.decode(syndrome)
        data_prediction = np.array(result, dtype=np.uint8)

        # see if the decoded result anti-commutes with the observables
        print("decode result:", data_prediction)
        decoded_observables = (observables @ data_prediction) % 2
        print("decoded_observables:", decoded_observables)

        # update pauli frame
        pauli_frame = (pauli_frame + decoded_observables) % 2
        print("pauli frame:", pauli_frame)

    # after pauli frame has tracked corrections through the rounds
    # apply the pauli frame correction to the measurement, and see
    # if this matches the state we intended to prepare
    # We prepared |0>, so we check if logical measurement Mz + Pf_X = 0
    corrected_mz = (logical_measurements[shot] + pauli_frame[0]) % 2
    print("Expected value:", expected_value)
    print("Corrected value:", corrected_mz)
    if (corrected_mz != expected_value):
        nLogicalErrors += 1

# Count how many shots the decoder failed to correct the errors
print("Number of logical errors:", nLogicalErrors)
