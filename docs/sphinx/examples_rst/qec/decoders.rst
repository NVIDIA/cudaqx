Decoders
--------

In quantum error correction, decoders are responsible for interpreting measurement outcomes (syndromes) to identify and correct quantum errors. 
We measure a set of stabilizers that give us information about what errors might have happened. The pattern of these measurements is called a syndrome, 
and the decoder's task is to determine what errors most likely caused that syndrome.

The relationship between errors and syndromes is captured mathematically by the parity check matrix. Each row of this matrix represents a 
stabilizer measurement, while each column represents a possible error. When we multiply an error pattern by this matrix, we get the syndrome 
that would result from those errors.

Creating a Multi-Round Parity Check Matrix
++++++++++++++++++++++++++++++++++++++++++

Below, we'll show how to use CUDA-Q QEC to construct the multi-round parity check matrix of an error correction code for the circuit-level noise model experiment in Python:

.. tab:: Python

   .. literalinclude:: ../../examples/qec/python/repetition_code_pcm.py
      :language: python
      :start-after: [Begin Documentation]

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


Detector Error Model
+++++++++++++++++++++

In the previous example, we showed how to create a multi-round parity check matrix manually. Here we introduce the `cudaq.qec.detector_error_model` type, which 
allows us to create a detector error model (DEM) from a QEC circuit and noise model.

The DEM can be generated from a QEC circuit and noise model using functions like `dem_from_memory_circuit()`. For circuit-level noise, the DEM can be put into a 
canonical form that's organized by measurement rounds, making it suitable for multi-round decoding.

For a complete example of using the surface code with DEM to generate parity check matrices and perform decoding, see the :doc:`circuit level noise example <circuit_level_noise>`.

Getting Started with the NVIDIA QLDPC Decoder
+++++++++++++++++++++++++++++++++++++++++++++

Starting with CUDA-Q QEC v0.2, a GPU-accelerated decoder is included with the
CUDA-Q QEC library. The library follows the CUDA-Q decoder Python and C++ interfaces
(namely :class:`cudaq_qec.Decoder` for Python and
:cpp:class:`cudaq::qec::decoder` for C++), but as documented in the API sections
(:ref:`nv_qldpc_decoder_api_python` for Python and
:ref:`nv_qldpc_decoder_api_cpp` for C++), there are many configuration options
that can be passed to the constructor. The following example shows how to
exercise the decoder using non-trivial pre-generated test data. The test data
was generated using scripts originating from the GitHub repo for
`BivariateBicycleCodes
<https://github.com/sbravyi/BivariateBicycleCodes>`_ [#f1]_; it includes parity
check matrices (PCMs) and test syndromes to exercise a decoder.

.. literalinclude:: ../../examples/qec/python/nv-qldpc-decoder.py
    :language: python
    :start-after: [Begin Documentation]

.. rubric:: Footnotes

.. [#f1] [BCGMRY] Sergey Bravyi, Andrew Cross, Jay Gambetta, Dmitri Maslov, Patrick Rall, Theodore Yoder, High-threshold and low-overhead fault-tolerant quantum memory https://arxiv.org/abs/2308.07915
