import cudaq
import numpy as np


@cudaq.kernel
def three_qubit_repetition_code():
    data_qubits = cudaq.qvector(3)
    ancilla_qubits = cudaq.qvector(2)

    # Initialize the logical |1> state as |111>
    x(data_qubits)

    # Random Bit Flip Errors
    for i in range(3):
        cudaq.apply_noise(cudaq.XError, 0.1, data_qubits[i])

    # Extract Syndromes
    h(ancilla_qubits)

    # First Parity Check
    z.ctrl(ancilla_qubits[0], data_qubits[0])
    z.ctrl(ancilla_qubits[0], data_qubits[1])

    # Second Parity Check
    z.ctrl(ancilla_qubits[1], data_qubits[1])
    z.ctrl(ancilla_qubits[1], data_qubits[2])

    h(ancilla_qubits)

    s0 = mz(ancilla_qubits[0])
    s1 = mz(ancilla_qubits[1])

    # Correct errors based on syndromes
    if s0 and s1:
        x(data_qubits[1])
    elif s0:
        x(data_qubits[0])
    elif s1:
        x(data_qubits[2])

    mz(data_qubits)


noise_model = cudaq.NoiseModel()
# Add measurement noise
noise_model.add_all_qubit_channel("mz", cudaq.BitFlipChannel(0.01))

# Run the kernel and observe results
# The percent of samples that are 000 corresponds to the logical error rate
cudaq.set_target("stim")
result = cudaq.sample(three_qubit_repetition_code, noise_model=noise_model)
print(result)
