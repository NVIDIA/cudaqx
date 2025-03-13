Quantum Error Correction with Circuit-level Noise Modeling
----------------------------------------------------------
This example builds upon the previous code-capacity noise model example.
In the circuit-level noise modeling experiment, we have many of the same components from the CUDA-Q QEC library: QEC codes, decoders, and noisy data.
The primary difference here, is that we can begin to run CUDA-Q kernels to generate noisy data, rather than just generating random bitstring to represent our errors.

Along with the stabilizers, parity check matrices, and logical observables, the QEC code type also has an encoding map.
This map allows codes to define logical gates in terms of gates on the underlying physical qubits.
These encodings operate on the `qec.patch` type, which represents three registers of physical qubits making up a logical qubit.
A data qubit register, an X-stabilizer ancilla register, and a Z-stabilizer ancilla register.

The most notable encoding stored in the QEC map, is how the `qec.operation.stabilizer_round`, which encodes a `cudaq.kernel` which stores the gate-level information for how to do a stabilizer measurement.
These stabilizer rounds are the gate-level way to encode the parity check matrix of a QEC code into quantum circuits.

This example walks through how to use the CUDA-Q QEC library to perform a quantum memory experiment simulation.
These experiments model how well QEC cycles, or rounds of stabilizer measuments, can protect the information encoded in a logical qubit.
If noise is turned off, then the information is protected indefinitely.
Here, we will model depolarization noise after each CX gate, and track how many logical errors occur.


CUDA-Q QEC Implementation
+++++++++++++++++++++++++++++
Here's how to use CUDA-Q QEC to perform a circuit-level noise model experiment in both Python and C++:

.. tab:: Python

   .. literalinclude:: ../../examples/qec/python/circuit_level_noise.py
      :language: python

.. tab:: C++

   .. literalinclude:: ../../examples/qec/cpp/circuit_level_noise.cpp
      :language: cpp

   Compile and run with

   .. code-block:: bash

      nvq++ --enable-mlir --target=stim -lcudaq-qec circuit_level_noise.cpp -o circuit_level_noise
      ./circuit_level_noise


1. QEC Code and Decoder types:
    - As in the code capacity example, our central objects are the `qec.code` and `qec.decoder` types.

2. Clifford simulation backend:
    - As the size of QEC circuits can grow quite large, Clifford simulation is often the best tool for these simulations.
    - `cudaq.set_target("stim")` selects the highly performant Stim simulator as the simulation backend.

3. Noise model:
    - To add noisy gates we use the `cudaq.NoiseModel` type.
    - CUDA-Q supports the generation of arbitrary noise channels. Here we use a `cudaq.Depolarization2` channel to add a depolarization channel.
    - This is added to the `CX` gate by adding it to the `X` gate with 1 control.
    - This noisy gate is added to every qubit via that `noise.add_all_qubit_channel` function.

4. Getting circuit-level noisy data:
    - The `qec.code` is the first input parameter here, as the code's `stabilizer_round` determines the circuits executed.
    - Each memory circuit runs for an input number of `nRounds`, which specifies how many `stabilizer_round` kernels are ran.
    - After `nRounds` the data qubits are measured and the run is over.
    - This is performed `nShots` number of times.
    - During a shot, each stabilizer round's syndrome is `xor`'d against the preceding syndrome, so that we can track a sparser flow of data showing which round each parity check was violated.
    - The first round returns the syndrome as is, as there is nothing preceding to `xor` against.

5. Data qubit measurements:
    - The data qubits are only read out after the end of each shot, so there are `nShots` worth of data readouts.
    - The basis of the data qubit measurements depends on the state preparation used.
    - Z-basis readout when preparing the logical `|0>` or logical `|1>` state with the `qec.operation.prep0` or `qec.operation.prep1` kernels.
    - X-basis readout when preparing the logical `|+>` or logical `|->` state with the `qec.operation.prepp` or `qec.operation.prepm` kernels.

6. Logical Errors:
    - From here, the decoding procedure is again similar to the code capacity case, expect for we use a pauli frame to track errors that happen each QEC cycle.
    - The final values of the pauli frame tell us how our logical state flipped during the experiment, and what needs to be done to correct it.
    - We compare our known initial state (corrected by the Pauli frame), against our measured data qubits to determine if a logical error occurred.


The CUDA-Q QEC library thus provides a platform for numerical QEC experiments. The `qec.code` can be used to analyze a variety of QEC codes (both library or user provided), with a variety of decoders (both library or user provided).
The CUDA-Q QEC library also provides tools to speed up the automation of generating noisy data and syndromes.

Addtionally, here's how to use CUDA-Q QEC to construct a multi-round parity check matrix and a custom error correction code for the circuit-level noise model experiment in Python:

.. tab:: Python

   .. literalinclude:: ../../examples/qec/python/repetition_code_fine_grain_noise.py
      :language: python

This example illustrates how to:

1. Construct a multi-round parity check matrix – Users can extend a single-round parity check matrix across multiple rounds, 
incorporating measurement errors to track syndrome evolution over time. This enables more accurate circuit-level noise modeling for decoders.

2. Define custom error correction circuits with precise noise injection – Using `cudaq.apply_noise`, users can introduce specific error channels 
at targeted locations within the QEC circuit. This fine-grained control allows for precise testing of how different noise sources affect logical error rates.

In the previous example, we demonstrated how to introduce random X errors into each data qubit using `cudaq.apply_noise` during each round of syndrome extraction. 
CUDA-Q allows users to inject a variety of error channels at different locations within their circuits, enabling fine-grained noise modeling. The example below showcases 
additional ways to introduce errors into a quantum kernel:

   .. code-block:: python

        @cudaq.kernel
        def inject_noise_example():
            q = cudaq.qvector(3)

            # Apply depolarization noise to the first qubit
            cudaq.apply_noise(cudaq.DepolarizationChannel, 0.1, q[0])

            # Perform gate operations
            h(q[1])
            x.ctrl(q[1], q[2])

            # Inject a Y error into the second qubit
            cudaq.apply_noise(cudaq.YError, 0.1, q[1])

            # Apply a general Pauli noise channel to the third qubit, where the 3 values indicate the probability of X, Y, and Z errors.
            cudaq.apply_noise(cudaq.Pauli1, 0.1, 0.1, 0.1, q[2])

        # Define and apply a noise model
        noise = cudaq.NoiseModel()
        counts = cudaq.sample(inject_noise_example, noise_model=noise)

For a full list of supported noise models and their parameters, refer to the `CUDA-Q documentation <https://nvidia.github.io/cuda-quantum/latest/index.html>`_.

