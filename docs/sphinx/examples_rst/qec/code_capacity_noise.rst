Quantum Error Correction with Code-Capacity Noise Modeling
----------------------------------------------------------

Quantum error correction (QEC) describes a set of tools used to detect and correct errors which occur to qubits on quantum computers.
This example will walk through how the CUDA-Q QEC library handles two of the most common objects in QEC: stabilizer codes, and decoders.
A stabilizer code is the quantum generalization of linear codes in classical error correction, which use parity checks to detect errors on noise bits.
In QEC, we'll perform stabilizer measurements on ancilla qubits to check the parity of our data qubits.
These stabilizer measurements are non-destructive, and thus allow us to check the relative parity of qubits without destroying our quantum information.

For example, if we prepare two qubits in the state `\Psi = a|00> + b|11>`, we maybe want to check if a bit-flip error happened.
We can measure the stabilizer `ZZ`, which will return 0 if there are no errors or even number of errors, but will return 1 if either has flipped.
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

.. tab:: C++

   .. literalinclude:: ../../examples/qec/cpp/code_capacity_noise.cpp
      :language: cpp

   Compile and run with

   .. code-block:: bash

      nvq++ --enable-mlir --target=stim -lcudaq-qec code_capacity_noise.cpp -o code_capacity_noise
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
    - Once we have noisy data, we see what the resuling syndromes are by multiplying our noisy data vector with our parity check matrix (mod 2).
    - From this syndrome, we see what the decoder predicts what errors occurred in the data.
    - To classify as a logical error, the decoder does not need to exactly guess what happened to the data, but if there was a flip in the logical observable or not.
    - If the decoder guesses this successfully, we have corrected the quantum error. If not, we have incurred a logical error.

5. Further automation:
    - While this workflow is nice for seeing things step by step, the `qec.sample_code_capacity` API is provided to generate a batch of noisy data and their corresponding syndromes.

The CUDA-Q QEC library thus provides a platform for numerical QEC experiments. The `qec.code` can be used to analyze a variety of QEC codes (both library or user provided), with a variety of decoders (both library or user provided).
The CUDA-Q QEC library also provides tools to speed up the automation of generating noisy data and syndromes.
