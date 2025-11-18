Real-Time Decoding
==================

Real-time decoding enables CUDA-Q QEC decoders to operate in low-latency, online environments where decoders run concurrently with quantum computations. This capability is essential for quantum error correction on real quantum hardware, where corrections must be calculated and applied within qubit coherence times.

The real-time decoding framework supports two primary deployment scenarios:

1. **Hardware Integration**: Decoders running on classical computers connected to real quantum processing units (QPUs) via low-latency networks
2. **Simulation Mode**: Decoders operating in simulated environments for testing and development on local systems

Workflow Overview
-----------------

Real-time decoding integrates seamlessly into quantum error correction pipelines through a carefully designed four-stage workflow. This workflow separates the computationally intensive characterization phase from the latency-critical runtime phase, ensuring that decoders can operate efficiently during quantum circuit execution.

The workflow consists of four stages:

1. **Detector Error Model (DEM) Generation**: Before running a quantum program, the user first characterizes how errors propagate through the quantum circuit. The library internally uses Memory Syndrome Matrix (MSM) representations to track error propagation, but this complexity is abstracted through helper functions like ``z_dem_from_memory_circuit``. The user simply provides a quantum code, noise model, and circuit parameters, and receives a complete detector error model that maps error mechanisms to syndrome patterns. This step is performed once during development.

2. **Decoder Configuration and Saving**: Using the DEM, the user configures decoder instances with the specific error model data. This includes converting parity check matrices to sparse format, setting decoder-specific parameters (like lookup table depth or BP iterations), and assigning unique IDs to each logical qubit's decoder. The configuration is then saved to a YAML file, capturing all the information decoders need to interpret syndrome measurements correctly. This creates a portable, reusable configuration that separates characterization from execution.

3. **Decoder Loading and Initialization**: Just before circuit execution, the user loads the saved YAML configuration file. The library parses the configuration, instantiates the appropriate decoder implementations, initializes internal data structures, and registers the decoders with the CUDA-Q runtime. For GPU-based decoders, matrices are transferred to device memory; for lookup table decoders, syndrome-to-correction mappings are constructed. This initialization takes milliseconds to seconds depending on code size and happens before quantum operations begin.

4. **Real-Time Decoding**: During quantum circuit execution, the decoding API is used within quantum kernels to interact with decoders. As the circuit measures stabilizers, syndromes are enqueued to the decoder, which processes them concurrently. When corrections are needed, the decoder is queried and the suggested operations are applied to the logical qubits. This entire process happens within the coherence time constraints of the quantum hardware.

Real-Time Decoding Example
----------------

Here are two examples demonstrating real-time decoding in Python and C++:

.. tab:: Python

   .. literalinclude:: ../../examples/qec/python/real_time_complete.py
      :language: python
      :start-after: # [Begin Documentation]

.. tab:: C++

   .. literalinclude:: ../../examples/qec/cpp/real_time_complete.cpp
      :language: cpp
      :start-after: // [Begin Documentation]

The examples above showcase the main components of the real-time decoding workflow:

- Decoder configuration file: Initializes and configures the decoders before circuit execution.

- Quantum kernel: Uses the real-time decoding API to interact with the decoders, primarily through reset_decoder, enqueue_syndromes, and get_corrections.

- Syndrome extraction: Measures the stabilizers of the logical qubits.

- Correction application: Applies the corrections to the logical qubits.

- Logical observable measurement: Measures the logical observables of the logical qubits.

- Decoder finalization: Frees up resources after circuit execution.

The API is designed to be called from within quantum kernels (marked with ``@cudaq.kernel`` in Python or ``__qpu__``  in C++). The runtime automatically routes these calls to the appropriate backend—whether a simulation environment on the local machine or a low-latency connection to quantum hardware. The API is device-agnostic, so the same kernel code works across different deployment scenarios.

The user is required to provide a configuration file or generate one if it is not present. The generation process depends on the decoder type and the detector error model studied in other sections of the documentation. Moreover, the user must write an appropriate kernel that describes the correct syndrome extraction and correction application logic.

The next section provides instructions to generate a configuration file, write a quantum kernel, and compile and run the examples correctly.


Configuration
-------------

The configuration process transforms a quantum circuit's error characteristics into a format that decoders can efficiently process. This section walks through each step in detail, showing how to go from circuit simulation to a fully configured real-time decoder.

Step 1: Generate Detector Error Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first step is to characterize the quantum circuit's behavior under noise. 
A detector error model (DEM) captures the relationship between physical errors and the syndrome patterns they produce. 
This characterization is circuit-specific and depends on the code structure, noise model, and measurement schedule.

Under the hood, the CUDA-Q QEC library uses the Memory Syndrome Matrix (MSM) representation to efficiently encode error propagation information. The MSM captures all possible error chains and their syndrome signatures, tracking how errors propagate through the circuit over time. However, this complexity is abstracted away from the user through convenient helper functions.

The library provides a family of ``dem_from_memory_circuit`` functions that automatically handle the MSM generation and processing:

* ``z_dem_from_memory_circuit``: For circuits measuring Z-basis stabilizers (used in the example below)
* ``x_dem_from_memory_circuit``: For circuits measuring X-basis stabilizers
* ``dem_from_memory_circuit``: General-purpose function for arbitrary stabilizer measurements

These functions take a quantum code, an initial state preparation operation, the number of measurement rounds, and a noise model, then return a complete detector error model ready for decoder configuration. The user simply needs to configure the noise model and specify the circuit structure—the library handles all the error tracking and matrix construction automatically.

Here is how to generate a DEM for a circuit:

.. tab:: Python

   .. literalinclude:: ../../examples/qec/python/real_time_complete.py
      :language: python
      :start-after: # [Begin DEM Generation]
      :end-before: # [End DEM Generation]

.. tab:: C++

   .. literalinclude:: ../../examples/qec/cpp/real_time_complete.cpp
      :language: cpp
      :start-after: // [Begin DEM Generation]
      :end-before: // [End DEM Generation]

Step 2: Configure and Save Decoder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once a DEM has been generated, the next step is to package this information into a decoder configuration and save it to a YAML file. 
The configuration structure holds all the parameters a decoder needs: the parity check matrix (H_sparse), 
the observable flip matrix (O_sparse), the detector error matrix (D_sparse), 
and decoder-specific tuning parameters. 

These matrices are generated in sparse matrix format, which is crucial for performance. 
They can be large considering error correcting codes with large number of physical qubits, and moreover, 
real-time decoders process thousands of syndrome measurements per second, and take decision based on these matrices, so compact representations are essential.
The helper function ``pcm_to_sparse_vec`` is used to convert the dense binary matrices into a space-efficient format where -1 marks row boundaries and integers represent column indices of non-zero elements.

Each decoder type has its own configuration structure with specific parameters. 
For lookup table decoders, the user specifies how many simultaneous errors to consider. 
For belief propagation decoders, the user sets iteration limits and convergence criteria. 
The configuration API provides type-safe structures for each decoder, ensuring that all required parameters are included.

The configuration is then saved to a YAML file for reuse. The YAML format is human-readable, making it easy to inspect, modify, and share configurations across different execution environments.

Here is how to create and save a decoder configuration:

.. tab:: Python

   .. literalinclude:: ../../examples/qec/python/real_time_complete.py
      :language: python
      :start-after: # [Begin Save DEM]
      :end-before: # [End Save DEM]

.. tab:: C++

   .. literalinclude:: ../../examples/qec/cpp/real_time_complete.cpp
      :language: cpp
      :start-after: // [Begin Save DEM]
      :end-before: // [End Save DEM]

Step 3: Load Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before running quantum circuits with real-time decoding, the saved decoder configuration must be loaded and initialized. 
This step bridges the gap between the offline characterization phase (Steps 1-2) and the online execution phase (Step 4), 
preparing the decoder instances for real-time operation.

The configuration loading process performs several important operations:

1. **YAML Parsing**: The configuration file is parsed and validated to ensure all required fields are present and properly formatted. This includes checking matrix dimensions, decoder parameters, and metadata.

2. **Decoder Instantiation**: Based on the decoder type specified in the configuration (e.g., ``multi_error_lut``, ``nv-qldpc-decoder``), the appropriate decoder implementation is instantiated and allocated resources on the GPU or CPU.

3. **Matrix Initialization**: The sparse matrices (H_sparse, O_sparse, D_sparse) are loaded into the decoder's internal data structures. For GPU-based decoders, this includes transferring data to device memory.

4. **Decoder-Specific Initialization**: Each decoder type performs its own preparation: lookup table decoders build syndrome-to-correction mappings, belief propagation decoders initialize message-passing structures, and sliding window decoders configure their buffering mechanisms.

5. **Backend Registration**: The decoder instances are registered with the CUDA-Q runtime so they can be accessed from quantum kernels using their unique IDs.

This initialization happens quickly, typically only a few milliseconds for small codes and up to a few seconds for large distance codes with complex decoders. Since it occurs before quantum circuit execution, it does not impact the latency-critical decoding operations.

The separation of configuration from execution provides significant benefits: users can maintain a library of configurations for different code distances, noise levels, and decoder types, then simply load the appropriate one when running experiments. Configurations can be version-controlled alongside code, shared across research teams, and validated offline before deployment to quantum hardware.

Here is how to load a decoder configuration:

.. tab:: Python

   .. literalinclude:: ../../examples/qec/python/real_time_complete.py
      :language: python
      :start-after: # [Begin Load DEM]
      :end-before: # [End Load DEM]

.. tab:: C++

   .. literalinclude:: ../../examples/qec/cpp/real_time_complete.cpp
      :language: cpp
      :start-after: // [Begin Load DEM]
      :end-before: // [End Load DEM]

Step 4: Use in Quantum Kernels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With decoders configured and initialized, they can be used within quantum kernels. The real-time decoding API provides three key functions that integrate seamlessly with CUDA-Q's quantum programming model: ``reset_decoder`` prepares a decoder for a new shot, ``enqueue_syndromes`` sends syndrome measurements to the decoder for processing, and ``get_corrections`` retrieves the decoder's recommended corrections.

These functions are designed to be called from within quantum kernels (marked with ``@cudaq.kernel`` in Python or ``__qpu__`` in C++). The runtime automatically routes these calls to the appropriate backend - whether that is a simulation environment on the local machine or a low-latency connection to quantum hardware. The API is device-agnostic, so the same kernel code works across different deployment scenarios.

The typical usage pattern is: reset the decoder at the start of each shot, enqueue
syndromes after each stabilizer measurement round, then get corrections before
measuring the logical observables. Decoders process syndromes asynchronously, so
by the time ``get_corrections`` is called, the decoder has usually finished its
analysis. If decoding takes longer than expected, ``get_corrections`` will block
until results are available.

.. note::
   While resetting the decoder at the beginning of each shot isn't strictly
   required, it is **strongly** recommended to ensure that when running on a
   remote QPU, any potential errors encountered in one shot do not affect future
   shot results.

Here is how to use the real-time decoding API in quantum kernels:

.. tab:: Python

   .. literalinclude:: ../../examples/qec/python/real_time_complete.py
      :language: python
      :start-after: # [Begin QEC Circuit]
      :end-before: # [End QEC Circuit]

.. tab:: C++

   .. literalinclude:: ../../examples/qec/cpp/real_time_complete.cpp
      :language: cpp
      :start-after: // [Begin QEC Circuit]
      :end-before: // [End QEC Circuit]

Backend Selection
-----------------

CUDA-Q QEC's real-time decoding system is designed to work seamlessly across different execution environments. The backend selection determines where quantum circuits run and how decoders communicate with the quantum processor. Understanding the differences between simulation and hardware backends helps the user develop efficiently and deploy confidently.

Simulation Backend
^^^^^^^^^^^^^^^^^^

The simulation backend is the primary tool during development, testing, and
algorithm validation. It runs entirely on the local machine, using quantum
simulators like Stim to execute circuits while decoders process syndromes and
calculation corrections. This setup is ideal for rapid iteration: the user can
test decoder configurations, validate circuit logic, and debug syndrome
processing without waiting for hardware access or paying for compute time.

The simulation backend mimics real-time decoding's concurrent operation by
running the decoder(s) within the same process as the simulator. This means that
other than GPU hardware differences between the local environment and the remote
NVQLink decoders, the decoders behave the same way whether testing locally or
running on a quantum computer. The main difference is that simulation does not
have the same strict latency constraints, making it easier to experiment with
complex decoder configurations.

Use the simulation backend for local development and testing:

.. tab:: Python

   .. code-block:: python

      import cudaq
      import cudaq_qec as qec
      
      cudaq.set_target("stim")  # Or other simulator
      qec.configure_decoders_from_file("config.yaml")
      
      # Run circuit with noise model
      results = cudaq.run(my_circuit, shots_count=100, 
                         noise_model=cudaq.NoiseModel())

.. tab:: C++

   .. code-block:: bash

      # Compile with simulation support
      nvq++ -std=c++20 my_circuit.cpp -lcudaq-qec \
            -lcudaq-qec-realtime-decoding \
            -lcudaq-qec-realtime-decoding-simulation
      
      ./a.out

Quantinuum Hardware Backend
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Quantinuum hardware backend connects quantum circuits to real ion-trap quantum computers. Unlike the simulation backend where decoders run on the local machine, **the Quantinuum backend uploads the decoder configuration to Quantinuum's infrastructure**, where decoders run on GPU-equipped servers co-located with the quantum hardware. This architecture minimizes latency between syndrome measurements and correction application.

**Important Setup Requirements:**

1. **Configuration Upload**: When ``configure_decoders_from_file()`` or ``configure_decoders()`` is called, the decoder configuration is automatically base64-encoded and uploaded to Quantinuum's REST API (``api/gpu_decoder_configs/v1beta/``). This happens before job submission. The configuration includes all decoder parameters, error models, and sparse matrices.

2. **Extra Payload Provider**: The user **must** specify ``extra_payload_provider="decoder"`` when setting the target. This registers a payload provider that injects the decoder configuration UUID into each job request, telling Quantinuum which decoder configuration to use for the circuit.

3. **Backend Compilation**: For C++, the user must link against ``-lcudaq-qec-realtime-decoding-quantinuum`` instead of the simulation library. This library implements the Quantinuum-specific communication protocol for syndrome transmission.

4. **Configuration Lifetime**: Decoder configurations persist on Quantinuum's servers and are referenced by UUID. If the configuration is modified, it must be uploaded again - the system will generate a new UUID and use the new configuration for subsequent jobs.

Note: The real-time decoding interfaces are experimental, and subject to change. Real-time decoding on Quantinuum's Helios-1 device is currently only available to partners and collaborators. Please email QCSupport@quantinuum.com for more information.

**Emulation vs. Hardware Modes:**

Emulation mode (``emulate=True``) is particularly valuable for testing the deployment setup without consuming hardware credits.
However, the Quantinuum Helios-1 emulator does not support real-time decoder calls at this time.

Use the Quantinuum backend for hardware or emulation:

.. tab:: Python

   .. code-block:: python

      cudaq.set_target("quantinuum",
                       emulate=False,  # True for emulation
                       machine="Helios-1",
                       extra_payload_provider="decoder")
      
      qec.configure_decoders_from_file("config.yaml")
      results = cudaq.run(my_circuit, shots_count=100)

.. tab:: C++

   .. code-block:: bash

      # Compile for Quantinuum
      nvq++ --target quantinuum --quantinuum-machine Helios-1 \
            my_circuit.cpp -lcudaq-qec \
            -lcudaq-qec-realtime-decoding \
            -lcudaq-qec-realtime-decoding-quantinuum
      
      ./a.out

Compilation and Execution Examples
-----------------------------------

This section provides **complete, tested compilation and execution commands** for both simulation and hardware backends, extracted from the CUDA-Q QEC test infrastructure. The section begins with common usage patterns that guide decoder and compilation choices, then provides the specific commands needed for each backend.

Common Use Cases
^^^^^^^^^^^^^^^^^^^^^^

Before diving into compilation details, it is helpful to understand the typical scenarios and how they map to decoder choices and workflow parameters. 
A full set of common examples is provided to guide development.
These examples describe the complete workflow for developing an application that uses real-time decoding in a single file.
The relevant C++ and Python examples can be found at the following path:
`libs/qec/unittests/realtime/app_examples <https://github.com/NVIDIA/cudaqx/tree/main/libs/qec/unittests/realtime/app_examples>`_.
The files have names like ``surface_code-1.cpp`` and ``surface_code_1.py``. The rest of this section shows how to compile and run these 2 examples.

These examples provide comprehensive support for application development with real-time decoding.
The subsequent step, once the user has chosen the appropriate decoder and the appropriate backend, is to compile and execute the application.
Instructions are provided below for both the simulation and the hardware backends.

C++ Compilation
^^^^^^^^^^^^^^^

**Simulation Backend (Stim)**

Compile with the simulation backend for local testing:

.. code-block:: bash

   nvq++ --target stim surface_code-1.cpp         \
         -lcudaq-qec                              \
         -lcudaq-qec-realtime-decoding            \
         -lcudaq-qec-realtime-decoding-simulation \
         -o surface_code-1

   # Execute
   ./surface_code-1 --distance 3 --num_shots 1000 --save_dem config.yaml

**Key Points:**

- ``--target stim``: Use the Stim quantum simulator
- ``-lcudaq-qec``: Core QEC library with codes and experiments
- ``-lcudaq-qec-realtime-decoding``: Real-time decoding core API
- ``-lcudaq-qec-realtime-decoding-simulation``: Simulation-specific decoder backend

**Quantinuum Backend (Hardware)**

Compile for actual Quantinuum hardware:

.. code-block:: bash

   nvq++ --target quantinuum                         \
         --quantinuum-machine Helios-1               \
         --quantinuum-extra-payload-provider decoder \
         surface_code-1.cpp                          \
         -lcudaq-qec                                 \
         -lcudaq-qec-realtime-decoding               \
         -lcudaq-qec-realtime-decoding-quantinuum    \
         -Wl,--export-dynamic                        \
         -o surface_code-1-quantinuum-hardware

   # Execute
   export CUDAQ_QUANTINUUM_CREDENTIALS=<credentials_file_path>
   ./surface_code-1-quantinuum-hardware --distance 3 --num_shots 100 --load_dem config.yaml

**Key Points:**

- Use Quantinuum target names: ``Helios-1``, ``Helios-1E``, ``Helios-1SC``, etc.
- Currently only ``Helios-1`` will run the GPU decoders. The ``Helios-1E`` emulator will not run the GPU decoders.
- Set ``CUDAQ_QUANTINUUM_CREDENTIALS`` environment variable with the user's credentials.
  Check out the `Quantinuum hardware backend documentation <https://nvidia.github.io/cuda-quantum/latest/using/backends/hardware/iontrap.html#quantinuum>`_ for more information.

**Emulated Quantinuum Compilation Workflow**

Compile for Quantinuum emulation mode:

.. code-block:: bash

   nvq++ --target quantinuum --emulate            \
         --quantinuum-machine Helios-Fake         \
         surface_code-1.cpp                       \
         -lcudaq-qec                              \
         -lcudaq-qec-realtime-decoding            \
         -lcudaq-qec-realtime-decoding-quantinuum \
         -Wl,--export-dynamic                     \
         -o surface_code-1-quantinuum-emulate

   # Execute
   ./surface_code-1-quantinuum-emulate --distance 3 --num_shots 1000 --load_dem config.yaml

**Key Points:**

- ``--target quantinuum --emulate``: Emulate Quantinuum compilation path
- ``--quantinuum-machine Helios-Fake``: Specify machine (``Helios-Fake`` for emulation)
- ``-lcudaq-qec-realtime-decoding-quantinuum``: Quantinuum-specific decoder backend (replaces ``-simulation``)
- ``-Wl,--export-dynamic``: **Required** linker flag for dynamic symbol resolution

.. note::
  When running with `--emulate`, there is no noise being applied because there
  is currently no way to express noise in target-specific QIR. Therefore, when
  running with emulation, users will see noise-free sample data.

Python Execution
^^^^^^^^^^^^^^^^

**Simulation Backend (Stim)**

.. code-block:: bash

   # Generate a decoder configuration file
   python3 surface_code-1.py --distance 3 --save_dem config.yaml
   # Run the circuit with the decoder configuration
   python3 surface_code-1.py --distance 3 --load_dem config.yaml --num_shots 1000


**Quantinuum Backend (Hardware)**

.. code-block:: bash

   python3 surface_code-1.py --distance 3 --load_dem config.yaml --num_shots 1000 --target quantinuum --machine-name Helios-1

**Key Points:**

- Use real machine names (check Quantinuum portal for available machines)
- Reduce shot count for hardware experiments (hardware time is expensive)

**Emulated Quantinuum Compilation Workflow**

.. code-block:: bash

   python3 surface_code-1.py --distance 3 --load_dem config.yaml --num_shots 1000 --target quantinuum --emulate
**Key Points:**

- ``emulate=True``: Emulate Quantinuum compilation path
- Decoder config is automatically uploaded to Quantinuum's servers when
  :py:func:`cudaq_qec.configure_decoders_from_file` (Python) or
  :cpp:func:`cudaq::qec::decoding::config::configure_decoders_from_file` (C++) is called

Complete Workflow Example
^^^^^^^^^^^^^^^^^^^^^^^^^^
Given that the user follows the structure of the examples provided, where each executable takes terminal arguments to configure the application, the following workflow can be used to compile and execute the application.


.. code-block:: bash

   # Phase 1: Generate Detector Error Model (DEM)
   # This is done once per code/distance/noise configuration
   
   ## C++
   ./surface_code-1 --distance 3 --num_shots 1000 --p_spam 0.01 \
                    --save_dem config_d3.yaml --num_rounds 12 --decoder_window 6
   
   ## Python
   python surface_code-1.py --distance 3 --num_shots 1000 --p_spam 0.01 \
                            --save_dem config_d3.yaml --num_rounds 12 --decoder_window 6
   
   # Phase 2: Run with Real-Time Decoding
   # Use the saved DEM configuration
   
   ## Simulation
   ./surface_code-1 --distance 3 --num_shots 1000 --load_dem config_d3.yaml \
                    --num_rounds 12 --decoder_window 6
   
   ## Quantinuum Emulation
   ./surface_code-1-quantinuum-emulate --distance 3 --num_shots 1000 --load_dem config_d3.yaml \
                           --num_rounds 12 --decoder_window 6
   
   ## Quantinuum Hardware
   export CUDAQ_QUANTINUUM_CREDENTIALS=credentials.json
   ./surface_code-1-quantinuum-hardware --distance 3 --num_shots 100 --load_dem config_d3.yaml \
                         --num_rounds 12 --decoder_window 6

**Application Parameters:**

- ``--distance``: Code distance (3, 5, 7, etc.)
- ``--num_shots``: Number of circuit repetitions
- ``--p_spam``: Physical error rate for noise model (DEM generation only)
- ``--save_dem``: Generate and save DEM configuration to file
- ``--load_dem``: Load existing DEM configuration from file
- ``--num_rounds``: Total number of syndrome measurement rounds
- ``--decoder_window``: Number of rounds processed per decoding window

Debugging and Environment Variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 
**Useful Environment Variables:**

.. code-block:: bash

   # Enable decoder configuration debugging
   export CUDAQ_QEC_DEBUG_DECODER=1
   
   # Set default simulator
   export CUDAQ_DEFAULT_SIMULATOR=stim
   
   # Dump JIT IR for debugging compilation issues
   export CUDAQ_DUMP_JIT_IR=1
   
   # Set Quantinuum credentials file
   export CUDAQ_QUANTINUUM_CREDENTIALS=/path/to/credentials.json

The variables can be set in the user's environment or in a script.
They are valid both for python and C++ applications, however, they must be set before importing the cudaq or cudaq_qec libraries.

**Common Compilation Issues:**

1. **Missing libraries**: Ensure all ``-lcudaq-qec-*`` libraries are linked
2. **Wrong backend library**: Use ``-simulation`` for Stim, ``-quantinuum`` for Quantinuum
3. **Missing** ``--export-dynamic`` **flag**: Required for Quantinuum targets
4. **Wrong target flags**: ``--emulate`` with ``Helios-Fake`` for emulation, remove for hardware

**Common Runtime Issues:**

1. **"Decoder X not found"**: Call ``configure_decoders_from_file()`` before circuit execution
2. **"Configuration upload failed"**: Check network connectivity and Quantinuum credentials
3. **Dimension mismatch errors**: Verify DEM dimensions match the circuit's syndrome count
4. **High error rates**: Check decoder window size matches DEM generation window


Decoder Selection
^^^^^^^^^^^^^^^^^
The page `CUDA-Q QEC Decoders <https://nvidia.github.io/cudaqx/components/qec/introduction.html#pre-built-qec-decoders>`_ provides information about which decoders are compatible with real-time decoding.

Troubleshooting
---------------

Even with careful configuration, issues may be encountered during real-time decoding. This section covers the most common problems and their solutions, organized by symptom. When troubleshooting, start by isolating whether the issue is in DEM generation, decoder configuration, or runtime execution.

Configuration Upload Failures (Quantinuum Backend)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When using the Quantinuum backend, the decoder configuration must be uploaded to their REST API before job submission. Upload failures prevent the quantum program from running and can be difficult to diagnose without knowing what to look for.

**Possible Issues:**

* **Network connectivity problems**: Connection to Quantinuum's servers is interrupted or unstable
* **Configuration too large**: Decoder configuration exceeds Quantinuum's upload size limits (typically happens with large distance codes and lookup tables)
* **Invalid credentials**: API authentication fails due to expired or incorrect credentials
* **Malformed configuration**: YAML structure is invalid or contains unsupported parameters

**Solutions**:

* **Enable debug logging**: Set ``CUDAQ_QEC_DEBUG_DECODER=1`` environment variable to see the exact configuration being uploaded and any error messages from the REST API
* **Check network**: Verify that Quantinuum's API endpoints can be reached before running the program. Test with a simple job submission first.
* **Reduce configuration size**: If uploads fail due to size, switch from lookup table decoders to QLDPC (much more compact), or use sliding window with smaller windows
* **Validate YAML locally**: Before uploading, test that ``multi_decoder_config::from_yaml_str()`` can parse the configuration file without errors
* **Check credentials**: Ensure the Quantinuum API credentials are valid and have not expired. Refresh tokens if necessary.
* **Test with emulation**: Try ``emulate=True`` first - emulation uses the same upload infrastructure but provides faster feedback if there are configuration issues

**Verification**:

After fixing configuration issues, the following log messages should appear:

.. code-block:: text

   [info] Initializing realtime decoding library with config file: config.yaml
   [info] Initializing decoders...
   [info] Creating decoder 0 of type multi_error_lut
   [info] Done initializing decoder 0 in 0.234 seconds

If errors appear instead, check the full error message - it often contains specific details about what failed (network timeout, size limit, parsing error, etc.).

See Also
--------

* :doc:`/api/qec/cpp_api` - C++ API Reference (includes Real-Time Decoding)
* :doc:`/api/qec/python_api` - Python API Reference (includes Real-Time Decoding)
* Example source code: ``libs/qec/unittests/realtime/app_examples/``

