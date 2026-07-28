Realtime Decoding
==================

Realtime decoding enables CUDA-Q QEC decoders to operate in low-latency, online environments where decoders run concurrently with quantum computations. This capability is essential for quantum error correction on real quantum hardware, where corrections must be calculated and applied within qubit coherence times.

The realtime decoding framework supports two primary deployment scenarios:

1. **Hardware Integration**: Decoders running on classical computers connected to real quantum processing units (QPUs) via low-latency networks
2. **Simulation Mode**: Decoders operating in simulated environments for testing and development on local systems

Realtime decoding integrates seamlessly into quantum error correction pipelines through a carefully designed four-stage workflow. This workflow separates the computationally intensive characterization phase from the latency-critical runtime phase, ensuring that decoders can operate efficiently during quantum circuit execution.

The workflow consists of four stages:

1. **Detector Error Model (DEM) Generation**: Before running a quantum program, the user first characterizes how errors propagate through the quantum circuit. The library internally uses Memory Syndrome Matrix (MSM) representations to track error propagation, but this complexity is abstracted through helper functions like ``z_dem_from_memory_circuit``. The user simply provides a quantum code, noise model, and circuit parameters, and receives a complete detector error model that maps error mechanisms to syndrome patterns. This step is performed once during development.

2. **Decoder Configuration and Saving**: Using the DEM, the user configures decoder instances with the specific error model data. This includes converting parity check matrices to sparse format, setting decoder-specific parameters (like lookup table depth or BP iterations), and assigning unique IDs to each logical qubit's decoder. The configuration is then saved to a YAML file, capturing all the information decoders need to interpret syndrome measurements correctly. This creates a portable, reusable configuration that separates characterization from execution.

3. **Decoder Loading and Initialization**: Just before circuit execution, the user loads the saved YAML configuration file. The library parses the configuration, instantiates the appropriate decoder implementations, initializes internal data structures, and registers the decoders with the CUDA-Q runtime. For GPU-based decoders, matrices are transferred to device memory; for lookup table decoders, syndrome-to-correction mappings are constructed. This initialization takes milliseconds to seconds depending on code size and happens before quantum operations begin.

4. **Realtime Decoding**: During quantum circuit execution, the decoding API is used within quantum kernels to interact with decoders. As the circuit measures stabilizers, syndromes are enqueued to the decoder, which processes them concurrently. When corrections are needed, the decoder is queried and the suggested operations are applied to the logical qubits. This entire process happens within the coherence time constraints of the quantum hardware.

The examples below cover realtime decoding end to end — start with Getting Started, then explore the specialized predecoding and decoding workloads:

.. toctree::
   :maxdepth: 2

   Getting Started with Realtime Decoding <getting_started_realtime_decoding>
   AI Predecoder with CUDA-Q Realtime <ai_predecoder>
   AI Predecoder with CUDA-Q Realtime (with FPGA Data Injection) <ai_predecoder_fpga>
   Relay BP Decoding with CUDA-Q Realtime <realtime_relay_bp>

See Also
--------

* :doc:`/api/qec/cpp_api` - C++ API Reference (includes Realtime Decoding)
* :doc:`/api/qec/python_api` - Python API Reference (includes Realtime Decoding)
* Example source code: ``libs/qec/unittests/realtime/app_examples/``
