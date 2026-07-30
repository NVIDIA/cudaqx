Modeling Noise in QEC
=====================

CUDA-Q QEC supports two noise-modeling regimes for numerical experiments: an abstract
code-capacity model that applies errors directly to data qubits, and a circuit-level model
that generates errors by running the underlying stabilizer-measurement circuits.

Quantum Error Correction with Code-Capacity Noise Modeling
----------------------------------------------------------

Quantum error correction (QEC) describes a set of tools used to detect and correct errors which occur to qubits on quantum computers.
This example will walk through how the CUDA-Q QEC library handles two of the most common objects in QEC: stabilizer codes, and decoders.
A stabilizer code is the quantum generalization of linear codes in classical error correction, which use parity checks to detect errors on noisy bits.
In QEC, we'll perform stabilizer measurements on ancilla qubits to check the parity of our data qubits.
These stabilizer measurements are non-destructive, and thus allow us to check the relative parity of qubits without destroying our quantum information.

For example, if we prepare two qubits in the state `\Psi = a|00> + b|11>`, we may want to check if a bit-flip error happened.
We can measure the stabilizer `ZZ`, which will return 0 if there are no errors or an even number of errors, but will return 1 if either has flipped.
This is how we can perform parity checks in quantum computing, without performing destructive measurements which collapse our superposition.
How these measurements are physically performed can be seen in the circuit-level noise QEC example.

We can specify a stabilizer code with either a list of stabilizer operators (like `ZZ` above), or equivalently, a parity check matrix.
We can think of the columns of a parity check matrix as the types of errors that can occur. In this case, each qubit can experience a bit flip `X` or a phase flip `Z` error, so the parity check matrix will have 2N columns where N is the number of data qubits.
Each row represents a stabilizer, or a parity check.
The values are either 0 or 1, where a 1 means that the corresponding column does participate in the parity check, and a 0 means it does not.
Therefore, if a single `X/Z` error happens to a qubit, the supported rows of the parity check matrix will trigger.
This is called the syndrome, a string of 0's and 1's corresponding to which parity checks were violated.
A special class of stabilizer codes are called CSS (Calderbank-Shor-Steane) codes, which means the `X` and `Z` components of their parity check matrix can be separated.

This brings us to decoding. Decoding is the act of solving the problem: given a syndrome, which underlying errors are most likely?
There are many decoding algorithms, but this example will use a simple single-error look-up table.
This means that the decoder will enumerate for each single error bit string, what the resulting syndromes are.
Then given a syndrome, it will look up the error string and return that as a result.

The last thing we need, is a way to generate errors.
This example will go through a code capacity noise model where we have an independent and identical chance that an `X` or `Z` error happens on each qubit with some probability `p`.

CUDA-Q QEC Implementation
+++++++++++++++++++++++++++++
Here's how to use CUDA-Q QEC to perform a code capacity noise model experiment in both Python and C++:

.. tab:: Python

   .. literalinclude:: ../../examples/qec/python/code_capacity_noise.py
      :language: python
      :start-after: [Begin Documentation]

.. tab:: C++

   .. literalinclude:: ../../examples/qec/cpp/code_capacity_noise.cpp
      :language: cpp
      :start-after: [Begin Documentation]

   Compile and run with

   .. code-block:: bash

      nvq++ --target=stim -lcudaq-qec -lcudaq-qec-decoders code_capacity_noise.cpp -o code_capacity_noise
      ./code_capacity_noise


Code Explanation
++++++++++++++++

1. QEC Code type:
    - CUDA-Q QEC centers around the `qec.code` type, which contains the data relevant for a given code.
    - In particular, this represents a collection of qubits which represent a single logical qubit.
    - Here we get one of the most well known QEC codes, the Steane code, with the `qec.get_code` function.
    - We can get the stabilizers from a code with the `code.get_stabilizers()` function.
    - In this example, we get the parity check matrix of the code. Because the Steane code is a CSS code, we can extract just the `Z` components of the parity check matrix.
    - Here, we see this matrix has 3 rows and 7 columns, which means there are 7 data qubits (7 possible single bit-flip errors) and 3 Z-stabilizers (parity checks). Note that `Z` stabilizers check for `X` type errors.
    - Lastly, we get the logical `Z` observable for the code. This will allow us to see if the `Z` observable of our logical qubit has flipped.

2. Decoder type:
    - A single-error look-up table (LUT) decoder can be acquired with the `qec.get_decoder` call.
    - Passing in the parity check matrix gives the decoder the required information to associated syndromes with underlying error mechanisms.
    - Once the decode has been constructed, the `decoder.decode(syndrome)` member function is called, which returns a predicted error given the syndrome.

3. Noise model:
    - To generate noisy data, we call `qec.generate_random_bit_flips(nBits, p)` which will return an array of bits, where each bit has probability `p` to have been flipped into 1, and a `1-p` chance to have remained 0.
    - Since we are using the `Z` parity check matrix `H_Z`, we want to simulate random `X` errors on our 7 data qubits.

4. Logical Errors:
    - Once we have noisy data, we see what the resulting syndromes are by multiplying our noisy data vector with our parity check matrix (mod 2).
    - From this syndrome, we see what errors the decoder predicts occurred in the data.
    - To classify as a logical error, the decoder does not need to exactly identify what happened to the data, but only whether there was a flip in the logical observable.
    - If the decoder guesses this successfully, we have corrected the quantum error. If not, we have incurred a logical error.

5. Further automation:
    - While this workflow is nice for seeing things step by step, the `qec.sample_code_capacity` API is provided to generate a batch of noisy data and their corresponding syndromes.

Quantum Error Correction with Circuit-level Noise Modeling
----------------------------------------------------------
This example builds upon the previous code-capacity noise model example.
In the circuit-level noise modeling experiment, we have many of the same components from the CUDA-Q QEC library: QEC codes, decoders, and noisy data.
The primary difference here, is that we can begin to run CUDA-Q kernels to generate noisy data, rather than just generating a random bit string to represent our errors.

Along with the stabilizers, parity check matrices, and logical observables, the QEC code type also has an encoding map.
This map allows codes to define logical gates in terms of gates on the underlying physical qubits.
These encodings operate on the `qec.patch` type, which represents three registers of physical qubits making up a logical qubit.
A data qubit register, an X-stabilizer ancilla register, and a Z-stabilizer ancilla register.

The most notable encoding stored in the QEC map is the `qec.operation.stabilizer_round`, which encodes a `cudaq.kernel` that stores the gate-level information for performing a stabilizer measurement.
These stabilizer rounds are the gate-level way to encode the parity check matrix of a QEC code into quantum circuits.

This example walks through how to use the CUDA-Q QEC library to perform a quantum memory experiment simulation.
These experiments model how well QEC cycles, or rounds of stabilizer measurements, can protect the information encoded in a logical qubit.
If noise is turned off, then the information is protected indefinitely.
Here, we will model depolarization noise after each CX gate, and track how many logical errors occur.


CUDA-Q QEC Implementation
+++++++++++++++++++++++++++++
Here's how to use CUDA-Q QEC to perform a circuit-level noise model experiment in both Python and C++:

.. tab:: Python

   .. literalinclude:: ../../examples/qec/python/circuit_level_noise.py
      :language: python
      :start-after: [Begin Documentation]

.. tab:: C++

   .. literalinclude:: ../../examples/qec/cpp/circuit_level_noise.cpp
      :language: cpp
      :start-after: [Begin Documentation]

   Compile and run with

   .. code-block:: bash

      nvq++ --target=stim -lcudaq-qec -lcudaq-qec-decoders circuit_level_noise.cpp -o circuit_level_noise
      ./circuit_level_noise


Code Explanation
++++++++++++++++

1. QEC Code and Decoder types:
    - As in the code capacity example, our central objects are the `qec.code` and `qec.decoder` types.

2. Clifford simulation backend:
    - As the size of QEC circuits can grow quite large, Clifford simulation is often the best tool for these simulations.
    - `cudaq.set_target("stim")` selects the highly performant Stim simulator as the simulation backend.

3. Noise model:
    - To add noisy gates we use the `cudaq.NoiseModel` type.
    - CUDA-Q supports the generation of arbitrary noise channels. Here we use a `cudaq.Depolarization2` channel to add a depolarization channel.
    - This is added to the `CX` gate by adding it to the `X` gate with 1 control.
    - This noisy gate is added to every qubit via the `noise.add_all_qubit_channel` function.

4. Getting circuit-level noisy data:
    - The `qec.code` is the first input parameter here, as the code's `stabilizer_round` determines the circuits executed.
    - Each memory circuit runs for an input number of `nRounds`, which specifies how many `stabilizer_round` kernels are run.
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
    - From here, the decoding procedure is again similar to the code capacity case, except that we use a Pauli frame to track errors that happen each QEC cycle.
    - The final values of the Pauli frame tell us how our logical state flipped during the experiment, and what needs to be done to correct it.
    - We compare our known initial state (corrected by the Pauli frame), against our measured data qubits to determine if a logical error occurred.


The CUDA-Q QEC library thus provides a platform for numerical QEC experiments. The `qec.code` can be used to analyze a variety of QEC codes (both library or user provided), with a variety of decoders (both library or user provided).
The CUDA-Q QEC library also provides tools to speed up the automation of generating noisy data and syndromes.
