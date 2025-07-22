import cudaq
import cudaq_qec as qec
from cudaq_qec import patch
import numpy as np

# Set the target. STIM is a great choice for stabilizer simulations.
cudaq.set_target("stim")

# Define code parameters
distance = 3
nRounds = 3
nShots = 10000


# Define the logical operations as CUDA-Q kernels
@cudaq.kernel
def x_logical(logicalQubit: patch):
    # For a repetition code, the logical X is applying X to all data qubits.
    for i in range(len(logicalQubit.data)):
        x(logicalQubit.data[i])


@cudaq.kernel
def prep0(logicalQubit: patch):
    # Reset all qubits to the |0> state.
    reset(logicalQubit.data)
    reset(logicalQubit.ancz)


@cudaq.kernel
def prep1(logicalQubit: patch):
    prep0(logicalQubit)
    x_logical(logicalQubit)


@cudaq.kernel
def stabilizer_round(logicalQubit: patch) -> list[bool]:
    # For a Z-basis repetition code, we measure ZZ stabilizers.
    num_ancilla = len(logicalQubit.ancz)
    num_data = len(logicalQubit.data)

    for i in range(num_data):
        # apply noise
        cudaq.apply_noise(cudaq.DepolarizationChannel, 0.1,
                          logicalQubit.data[i])
        # cudaq.apply_noise(cudaq.Pauli1, 0.1, 0.1, 0.1, q[2]) # in order pX, pY, and pZ errors.

    # Measure each Z stabilizer: Z_i * Z_{i+1}
    for i in range(num_ancilla):
        # Use CNOTs to measure the parity between data qubits i and i+1
        x.ctrl(logicalQubit.data[i], logicalQubit.ancz[i])
        x.ctrl(logicalQubit.data[i + 1], logicalQubit.ancz[i])

    # Measure the ancilla qubits to get the syndrome
    measurements = mz(logicalQubit.ancz)

    # Reset ancilla qubits for the next round
    reset(logicalQubit.ancz)

    return measurements


# Use the decorator to register the Python class with the C++ backend.
# Note: The class does need to inherit from qec.Code.
@qec.code('custom_repetition_code')
class MyRepetitionCode:

    def __init__(self, **kwargs):
        qec.Code.__init__(self)
        self.distance = kwargs.get("distance", 3)

        # Create stabilizer strings for the repetition code
        stabilizers_str = self.__make_stabilizers_strings()
        self.stabilizers = [
            cudaq.SpinOperator.from_word(s) for s in stabilizers_str
        ]

        # Register the CUDA-Q kernels for logical operations
        self.operation_encodings = {
            qec.operation.prep0: prep0,
            qec.operation.prep1: prep1,
            qec.operation.x: x_logical,
            qec.operation.stabilizer_round: stabilizer_round
        }

    def __make_stabilizers_strings(self):
        d = self.distance
        # Z_i * Z_{i+1} stabilizers
        return ['I' * i + 'ZZ' + 'I' * (d - i - 2) for i in range(d - 1)]

    # These methods are required by the C++ backend.
    def get_num_data_qubits(self):
        return self.distance

    def get_num_ancilla_x_qubits(self):
        return 0

    def get_num_ancilla_z_qubits(self):
        return self.distance - 1

    def get_num_ancilla_qubits(self):
        return self.get_num_ancilla_z_qubits() + self.get_num_ancilla_x_qubits()

    def get_num_x_stabilizers(self):
        return 0

    def get_num_z_stabilizers(self):
        return self.distance - 1


# Now, this call will succeed without hanging.
my_repetition_code = qec.get_code("custom_repetition_code", distance=distance)
print(f"Successfully created code with distance {my_repetition_code}.")

# This still won't show your custom code, and that's okay!
available_codes = qec.get_available_codes()
print("Available built-in C++ codes:", available_codes)

# Get stabilizer generators
stabilizers = my_repetition_code.get_stabilizers()
print(f"Code has {len(stabilizers)} stabilizers:")
for s in stabilizers:
    print(s)

# === You can now proceed with your simulation ===
# Create a noise model instance
noise_model = cudaq.NoiseModel()

# Define physical gate error probability
p = 0.01
# Inject depolarizing noise on CX gates
noise_model.add_all_qubit_channel("x", cudaq.Depolarization2(p), 1)
# Define the initial state for the memory experiment
statePrep = qec.operation.prep0

# Generate full detector error model (DEM), tracking all observables
dem_rep = qec.dem_from_memory_circuit(my_repetition_code, statePrep, nRounds,
                                      noise_model)

# Extract multi-round parity check matrix (H matrix)
H_pcm = dem_rep.detector_error_matrix

# Retrieve observable flips matrix: maps physical errors to logical flips
Lz_observables_flips_matrix = dem_rep.observables_flips_matrix

# === Simulation ===

# Sample noisy executions of the code circuit
syndromes, data = qec.sample_memory_circuit(my_repetition_code, statePrep,
                                            nShots, nRounds, noise_model)

syndromes = syndromes.reshape((nShots, nRounds, -1))
syndromes = syndromes.reshape((nShots, -1))

ccc = 0
