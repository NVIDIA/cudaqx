Decoders
--------

In quantum error correction, decoders are responsible for interpreting measurement outcomes (syndromes) to identify and correct quantum errors. 
We measure a set of stabilizers that give us information about what errors might have happened. The pattern of these measurements is called a syndrome, 
and the decoder's task is to determine what errors most likely caused that syndrome.

The relationship between errors and syndromes is captured mathematically by the parity check matrix. Each row of this matrix represents a 
stabilizer measurement, while each column represents a possible error. When we multiply an error pattern by this matrix, we get the syndrome 
that would result from those errors.

Detector Error Model
+++++++++++++++++++++

Here we introduce the `cudaq.qec.detector_error_model` type, which allows us to create a detector error model (DEM) from a QEC circuit and noise model.

The DEM can be generated from a QEC circuit and noise model using functions like `dem_from_memory_circuit()`. For circuit-level noise, the DEM can be put into a 
canonical form that's organized by measurement rounds, making it suitable for multi-round decoding.

For a complete example of using the surface code with DEM to generate parity check matrices and perform decoding, see the :doc:`circuit level noise example <circuit_level_noise>`.

Generating a Multi-Round Parity Check Matrix
++++++++++++++++++++++++++++++++++++++++++++

Below, we demonstrate how to use CUDA-Q QEC to construct a multi-round parity check matrix for an error correction code under a circuit-level noise model in Python:

.. tab:: Python

   .. literalinclude:: ../../examples/qec/python/repetition_code_pcm.py
      :language: python
      :start-after: [Begin Documentation]

This example illustrates how to:

* Retrieve and configure an error correction code  
  Load a repetition code using ``qec.get_code(...)`` from the CUDA-Q QEC library, and define a custom circuit-level noise model using ``.add_all_qubit_channel(...)``.

* Generate a multi-round parity check matrix  
  Extend a single-round detector error model (DEM) across multiple rounds using ``qec.dem_from_memory_circuit(...)``. This captures syndrome evolution over time, including measurement noise, and provides:
  
  * ``detector_error_matrix`` – the multi-round parity check matrix
  * ``observables_flips_matrix`` – used to identify logical flips due to physical errors

* Simulate circuit-level noise and collect data  
  Run multiple shots of the memory experiment using ``qec.sample_memory_circuit(...)`` to sample both the data and syndrome measurements from noisy executions. The resulting bitstrings can be used for decoding and performance evaluation of the error correction scheme.

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
