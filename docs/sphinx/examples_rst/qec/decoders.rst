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

Creating New QEC codes
++++++++++++++++++++++++++++++++++++++++++++

Below, we demonstrate how to use CUDA-Q QEC to define a new QEC code entirely in Python. This powerful feature allows for rapid prototyping and testing of custom error correction schemes.

.. tab:: Python

   .. literalinclude:: ../../examples/qec/python/custom_repetition_code_fine_grain_noise.py
      :language: python
      :start-after: [Begin Documentation]

This example illustrates several key concepts for defining custom codes:

* **Define a Code Class**: A new code is defined by creating a Python class decorated with ``@qec.code(...)``, which registers it with the CUDA-Q QEC runtime. The class must inherit from ``qec.Code``.
* **Implement Required Methods**: The class must implement methods that describe the code's structure, such as ``get_num_data_qubits()`` and ``get_num_ancilla_qubits()``.
* **Define Logical Operations as Kernels**: Quantum operations like state preparation (``prep0``, ``prep1``), logical gates (``x_logical``), and stabilizer measurements (``stabilizer_round``) are implemented as standard CUDA-Q kernels.
* **Map Operations to Kernels**: The ``operation_encodings`` dictionary links abstract QEC operations (e.g., ``qec.operation.prep0``) to the concrete CUDA-Q kernels that implement them.
* **Provide Stabilizers and Observables**: The code's stabilizer generators and logical observables must be defined. This is typically done by creating lists of ``cudaq.SpinOperator`` objects representing the Pauli strings for the stabilizers (e.g., "ZZI") and logical operators (e.g., "ZZZ").
* **Specify Fine-Grained Noise**: This example demonstrates applying noise at a specific point within a kernel. Inside ``stabilizer_round``, ``cudaq.apply_noise`` is called on each data qubit, offering precise control over the noise model, in contrast to applying noise globally to all gates of a certain type.

Once defined, the custom code can be instantiated with ``qec.get_code()`` and used with all standard CUDA-Q QEC tools, including ``qec.dem_from_memory_circuit()`` and ``qec.sample_memory_circuit()``.

Getting Started with the NVIDIA QLDPC Decoder
+++++++++++++++++++++++++++++++++++++++++++++

Starting with CUDA-Q QEC v0.2, a GPU-accelerated decoder is included with the
CUDA-Q QEC library. The library follows the CUDA-Q decoder Python and C++ interfaces
(namely :class:`cudaq_qec.Decoder` for Python and
:cpp:class:`cudaq::qec::decoder` for C++), but as documented in the API sections
(:ref:`nv_qldpc_decoder_api_python` for Python and
:ref:`nv_qldpc_decoder_api_cpp` for C++), there are many configuration options
that can be passed to the constructor.

Belief Propagation Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``nv-qldpc-decoder`` supports multiple belief propagation (BP) algorithms, each with different trade-offs 
between accuracy, convergence, and speed:

* **Sum-Product BP** (``bp_method=0``): The standard BP algorithm. Good baseline performance.
* **Min-Sum BP** (``bp_method=1``): Faster approximation to sum-product. Can be tuned with ``scale_factor``.
* **Memory-based BP** (``bp_method=2``): Adds uniform memory (``gamma0``) to help escape local minima. Useful when standard BP fails to converge.
* **Disordered Memory BP** (``bp_method=3``): Uses per-variable memory strengths for better adaptability to code structure.
* **Sequential Relay BP** (``composition=1``): Advanced method that runs multiple "relay legs" with different gamma configurations. See examples below for configuration.

Usage Example
~~~~~~~~~~~~~

The following example shows how to exercise the decoder using non-trivial pre-generated test data. 
The test data was generated using scripts originating from the GitHub repo for
`BivariateBicycleCodes <https://github.com/sbravyi/BivariateBicycleCodes>`_ [#f1]_; 
it includes parity check matrices (PCMs) and test syndromes to exercise a decoder.

The example demonstrates:

1. **Basic decoder configuration** with OSD post-processing
2. **All BP methods** including Sequential Relay BP
3. **Batched decoding** for improved performance

.. literalinclude:: ../../examples/qec/python/nv-qldpc-decoder.py
    :language: python
    :start-after: [Begin Documentation]

.. rubric:: Footnotes

.. [#f1] [BCGMRY] Sergey Bravyi, Andrew Cross, Jay Gambetta, Dmitri Maslov, Patrick Rall, Theodore Yoder, High-threshold and low-overhead fault-tolerant quantum memory https://arxiv.org/abs/2308.07915

Exact Maximum Likelihood Decoding with NVIDIA Tensor Network Decoder
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


Starting with CUDA-Q QEC v0.4.0, a GPU-accelerated Maximum Likelihood Decoder is included with the
CUDA-Q QEC library. The library follows the CUDA-Q decoder Python interface, namely :class:`cudaq_qec.Decoder`.
At this time, we only support the Python interface for the decoder, which is
available at :class:`cudaq_qec.plugins.decoders.tensor_network_decoder.TensorNetworkDecoder`.
As documented in the API sections :ref:`tensor_network_decoder_api_python`, there are many configuration options
that can be passed to the constructor. The decoder requires Python 3.11 or higher.

In the following example, we show how to use the `TensorNetworkDecoder` class from the `cudaq_qec` library to decode a circuit-level noise problem derived from a Stim surface code circuit.

.. literalinclude:: ../../examples/qec/python/tensor_network_decoder.py
    :language: python
    :start-after: [Begin Documentation]

Output:

The decoder returns the probability that the logical observable has flipped for each syndrome. This can be used to assess the performance of the code and the decoder under different error scenarios.

See Also:

- ``cudaq_qec.plugins.decoders.tensor_network_decoder``

.. _deploying-ai-decoders:

Deploying AI Decoders with TensorRT
+++++++++++++++++++++++++++++++++++++++++++++++++

Starting with CUDA-Q QEC v0.5.0, a GPU-accelerated TensorRT-based decoder is included with the
CUDA-Q QEC library. The TensorRT decoder (``trt_decoder``) enables users to leverage custom AI
models for quantum error correction, providing a flexible framework for deploying trained models
with optimized inference performance on NVIDIA GPUs.

Unlike traditional algorithmic decoders, neural network decoders can be trained on specific error
models and code structures, potentially achieving superior performance for certain noise regimes.
The TensorRT decoder supports loading models in ONNX format and provides configurable precision
modes (fp16, bf16, int8, fp8, tf32) to balance accuracy and inference speed.

This tutorial demonstrates the complete workflow for training a simple multi-layer perceptron (MLP)
to decode surface code syndromes using PyTorch and Stim, exporting the model to ONNX format, and
deploying it with the TensorRT decoder for accelerated inference.

Overview of the Training-to-Deployment Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The workflow consists of three main stages:

1. **Data Generation**: Use Stim to generate synthetic quantum error correction data by simulating
   surface code circuits with realistic noise models. This produces detector measurements (syndromes)
   and observable flips (logical errors) that serve as training data.

2. **Model Training**: Train a neural network (in this case, an MLP) using PyTorch to learn the
   mapping from syndromes to logical error predictions. The model is trained with standard deep
   learning techniques including dropout regularization, learning rate scheduling, and validation monitoring.

3. **ONNX Export and Deployment**: Export the trained PyTorch model to ONNX format, which can then
   be loaded by the TensorRT decoder for optimized GPU inference in production QEC workflows.

Training a Neural Network Decoder with PyTorch and Stim
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following example shows how to generate training data using Stim's built-in surface code
generator, train an MLP decoder with PyTorch, and export the model to ONNX format.
For instructions on installing PyTorch, see :ref:`Installing PyTorch <installing-pytorch>`.

.. literalinclude:: ../../examples/qec/python/train_mlp_decoder.py
   :language: python
   :start-after: [Begin Documentation]

Using the TensorRT Decoder in CUDA-Q QEC
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once you have a trained ONNX model, you can load it with the TensorRT decoder for accelerated
inference. The decoder can be used in both C++ and Python workflows.

**Loading from ONNX (with automatic TensorRT optimization)**:

.. tab:: Python

   .. code-block:: python

      import cudaq_qec as qec
      import numpy as np

      # Note: The AI decoder doesn't use the parity check matrix.
      # A placeholder matrix is provided here to satisfy the API.
      H = np.array([[1, 0, 0, 1, 0, 1, 1],
                    [0, 1, 0, 1, 1, 0, 1],
                    [0, 0, 1, 0, 1, 1, 1]], dtype=np.uint8)

      # Create TensorRT decoder from ONNX model
      decoder = qec.get_decoder("trt_decoder", H,
                                onnx_load_path="ai_decoder.onnx")

      # Decode a syndrome
      syndrome = np.array([1.0, 0.0, 1.0], dtype=np.float32)
      result = decoder.decode(syndrome)
      print(f"Predicted error: {result}")

.. tab:: C++

   .. code-block:: cpp

      #include "cudaq/qec/decoder.h"
      #include "cuda-qx/core/tensor.h"
      #include "cuda-qx/core/heterogeneous_map.h"

      int main() {
          // Note: The AI decoder doesn't use the parity check matrix.
          // A placeholder matrix is provided here to satisfy the API.
          std::vector<std::vector<uint8_t>> H_vec = {
              {1, 0, 0, 1, 0, 1, 1},
              {0, 1, 0, 1, 1, 0, 1},
              {0, 0, 1, 0, 1, 1, 1}
          };
          
          // Convert to tensor
          cudaqx::tensor<uint8_t> H({3, 7});
          for (size_t i = 0; i < 3; ++i) {
              for (size_t j = 0; j < 7; ++j) {
                  H.at({i, j}) = H_vec[i][j];
              }
          }

          // Create decoder parameters
          cudaqx::heterogeneous_map params;
          params.insert("onnx_load_path", "ai_decoder.onnx");
          params.insert("precision", "fp16");

          // Create TensorRT decoder
          auto decoder = cudaq::qec::get_decoder("trt_decoder", H, params);

          // Decode syndrome
          std::vector<cudaq::qec::float_t> syndrome = {1.0, 0.0, 1.0};
          auto result = decoder->decode(syndrome);

          return 0;
      }

**Loading a pre-built TensorRT engine (for fastest initialization)**:

If you've already converted your ONNX model to a TensorRT engine using the provided utility script,
you can load it directly:

.. tab:: Python

   .. code-block:: python

      decoder = qec.get_decoder("trt_decoder", H,
                                engine_load_path="surface_code_decoder.trt")

Converting ONNX Models to TensorRT Engines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For production deployments where initialization time is critical, you can pre-build a TensorRT
engine from your ONNX model using the ``trtexec`` command-line tool that comes with TensorRT:

.. code-block:: bash

   # Build with FP16 precision
   trtexec --onnx=surface_code_decoder.onnx \
           --saveEngine=surface_code_decoder.trt \
           --fp16

   # Build with best precision for your GPU
   trtexec --onnx=surface_code_decoder.onnx \
           --saveEngine=surface_code_decoder.trt \
           --best

   # Build with specific input shape (optional, for optimization)
   trtexec --onnx=surface_code_decoder.onnx \
           --saveEngine=surface_code_decoder.trt \
           --fp16 \
           --shapes=detectors:1x24

Pre-built engines offer several advantages:

- **Faster initialization**: Engine loading is significantly faster than ONNX parsing and optimization
- **Reproducible optimization**: The same optimization decisions are made every time
- **Version control**: Engines can be versioned alongside code for reproducible deployments


Dependencies and Requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The TensorRT decoder requires:

- **TensorRT**: Version 10.13.3.9 or higher
- **CUDA**: Version 12.0 or higher for x86 and 13.0 for ARM.
- **GPU**: NVIDIA GPU with compute capability 6.0+ (Pascal architecture or newer)

For training:

- **PyTorch**: Version 2.0+ recommended
- **Stim**: For quantum circuit simulation and data generation

See Also
^^^^^^^^

- :class:`cudaq_qec.Decoder` - Base decoder interface
- `ONNX <https://onnx.ai/>`_ - Open Neural Network Exchange format
- `TensorRT Documentation <https://docs.nvidia.com/deeplearning/tensorrt/>`_ - NVIDIA TensorRT
- `Stim Documentation <https://github.com/quantumlib/Stim>`_ - Fast stabilizer circuit simulator
