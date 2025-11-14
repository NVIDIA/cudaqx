Real-Time Decoding
==================

Real-time decoding enables CUDA-Q QEC decoders to operate in low-latency, online environments where decoders run concurrently with quantum computations. This capability is essential for quantum error correction on real quantum hardware, where corrections must be calculated and applied within qubit coherence times.

The real-time decoding framework supports two primary deployment scenarios:

1. **Hardware Integration**: Decoders running on classical computers connected to real quantum processing units (QPUs) via low-latency networks
2. **Simulation Mode**: Decoders operating in simulated environments for testing and development on local systems

Key Features
------------

* **Low-Latency Decoding**: Syndrome processing and correction calculation within coherence time constraints
* **Streaming Syndrome Interface**: Continuous syndrome enqueueing from quantum circuits
* **Multiple Decoder Support**: Concurrent management of multiple logical qubits, each with independent decoder instances
* **Flexible Configuration**: YAML-based decoder configuration supporting various decoder types and parameters
* **Device-Agnostic API**: Unified API that works across simulation and hardware backends
* **GPU Acceleration**: Leverages CUDA for high-performance syndrome decoding

Workflow Overview
-----------------

Real-time decoding integrates seamlessly into quantum error correction pipelines through a carefully designed four-stage workflow. This workflow separates the computationally intensive characterization phase from the latency-critical runtime phase, ensuring that decoders can operate efficiently during quantum circuit execution.

The workflow consists of four stages:

1. **Detector Error Model (DEM) Generation**: Before running a quantum program, the user first characterizes how errors propagate through the quantum circuit. This involves running the circuit through a noisy simulator (like Stim) to construct a detector error model that maps error mechanisms to syndrome patterns. This step is performed once during development and produces a detailed error characterization that informs the decoder about the circuit's error structure.

2. **Decoder Configuration**: Using the DEM, the user configures decoder instances with the specific error model data. This includes converting parity check matrices to sparse format, setting decoder-specific parameters (like lookup table depth or BP iterations), and assigning unique IDs to each logical qubit's decoder. The configuration captures all the information decoders need to interpret syndrome measurements correctly.

3. **Decoder Initialization**: Just before circuit execution, the user loads the saved configuration to initialize the decoder instances. This step prepares the decoders for operation and can be done from YAML files for easy reuse across experiments. The initialization is fast and happens on the host system before any quantum operations begin.

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

The API is designed to be called from within quantum kernels (marked with ``@cudaq.kernel`` in Python or ``__qpu__``  in C++). The runtime automatically routes these calls to the appropriate backend—whether a simulation environment on your local machine or a low-latency connection to quantum hardware. The API is device-agnostic, so the same kernel code works across different deployment scenarios.

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

To generate a DEM, the user runs the circuit through a noisy simulator that tracks how errors propagate. 
The CUDA-Q QEC library uses the Memory Syndrome Matrix (MSM) representation to efficiently encode this information. 
The MSM captures all possible error chains and their syndrome signatures, which are then processed into the matrices that decoders need.

Here is how to generate a DEM for a circuit:

.. tab:: Python

   .. code-block:: python

      import cudaq
      import cudaq_qec as qec

      # Set up code and noise
      cudaq.set_target("stim")
      code = qec.get_code("surface_code", distance=5)
      noise = cudaq.NoiseModel()
      # ... configure noise ...

      # Generate DEM using MSM
      msm_strings, msm_dims, probs, err_ids = qec.compute_msm(
          my_quantum_circuit, return_full=True)

      dem = qec.DetectorErrorModel()
      dem.error_rates = probs
      dem.error_ids = err_ids
      
      # Process MSM to create detector error matrix
      mz_table = qec.construct_mz_table(msm_strings)
      # ... build detector_error_matrix from mz_table ...

.. tab:: C++

   .. code-block:: cpp

      #include "cudaq.h"
      #include "cudaq/qec/decoder.h"

      // Generate MSM
      cudaq::noise_model noise;
      cudaq::ExecutionContext ctx_msm("msm");
      ctx_msm.noiseModel = &noise;
      
      auto &platform = cudaq::get_platform();
      platform.set_exec_ctx(&ctx_msm);
      my_quantum_circuit(/*...*/);
      platform.reset_exec_ctx();
      
      // Extract DEM from MSM
      auto msm_strings = ctx_msm.result.sequential_data();
      cudaq::qec::detector_error_model dem;
      // ... build DEM from MSM ...

Step 2: Configure Decoder
^^^^^^^^^^^^^^^^^^^^^^^^^^

Once a DEM has been generated, the next step is to package this information into a decoder configuration. 
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

The generation of the configuration will depend on the decoder type and the detector error model studied in other sections of the documentation.
Here is how to create a decoder configuration:

.. tab:: Python

   .. code-block:: python

      # Create configuration
      config = qec.decoder_config()
      config.id = 0
      config.type = "multi_error_lut"
      config.block_size = dem.num_error_mechanisms()
      config.syndrome_size = dem.num_detectors()
      
      # Convert matrices to sparse format
      config.H_sparse = qec.pcm_to_sparse_vec(dem.detector_error_matrix)
      config.O_sparse = qec.pcm_to_sparse_vec(dem.observables_flips_matrix)
      config.D_sparse = qec.generate_timelike_sparse_detector_matrix(
          num_syndromes_per_round, num_rounds, False)
      
      # Set decoder parameters
      lut_config = qec.multi_error_lut_config()
      lut_config.lut_error_depth = 2
      config.set_decoder_custom_args(lut_config)

.. tab:: C++

   .. code-block:: cpp

      using namespace cudaq::qec::decoding::config;
      
      // Create configuration
      decoder_config config;
      config.id = 0;
      config.type = "multi_error_lut";
      config.block_size = dem.num_error_mechanisms();
      config.syndrome_size = dem.num_detectors();
      
      // Convert matrices to sparse format
      config.H_sparse = cudaq::qec::pcm_to_sparse_vec(dem.detector_error_matrix);
      config.O_sparse = cudaq::qec::pcm_to_sparse_vec(dem.observables_flips_matrix);
      config.D_sparse = cudaq::qec::generate_timelike_sparse_detector_matrix(
          num_syndromes_per_round, num_rounds, false);
      
      // Set decoder parameters
      multi_error_lut_config lut_config;
      lut_config.lut_error_depth = 2;
      config.decoder_custom_args = lut_config;

Step 3: Save and Load Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Decoder configurations can be substantial - containing detailed error models with thousands of entries. Rather than reconstructing these configurations for every experiment, they can be saved to YAML files for reuse. This approach offers several benefits: it separates the characterization workflow from the runtime workflow, makes configurations portable across different execution environments, and allows version control of decoder settings alongside code.

The YAML format is human-readable, making it easy to inspect, modify, and share configurations. The user can maintain a library of configurations for different code distances, noise levels, and decoder types, then simply load the appropriate one when running experiments. The configuration system handles all serialization details automatically, preserving the exact sparse matrix representations and decoder parameters.

When a configuration is loaded from file, the library validates all required fields and initializes the decoder instances in the background. This initialization happens quickly, typically only a few milliseconds at startup, so it does not add noticeable latency to the workflow.

Here is how to save and load configurations:

.. tab:: Python

   .. code-block:: python

      multi_config = qec.multi_decoder_config()
      multi_config.decoders = [config]
      
      yaml_str = multi_config.to_yaml_str(200)
      with open("decoder_config.yaml", "w") as f:
          f.write(yaml_str)
      
      # Load configuration
      status = qec.configure_decoders_from_file("decoder_config.yaml")

.. tab:: C++

   .. code-block:: cpp

      multi_decoder_config multi_config;
      multi_config.decoders.push_back(config);
      
      std::string yaml_str = multi_config.to_yaml_str(200);
      std::ofstream file("decoder_config.yaml");
      file << yaml_str;
      file.close();
      
      // Load configuration
      int status = configure_decoders_from_file("decoder_config.yaml");

Step 4: Use in Quantum Kernels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With decoders configured and initialized, they can be used within quantum kernels. The real-time decoding API provides three key functions that integrate seamlessly with CUDA-Q's quantum programming model: ``reset_decoder`` prepares a decoder for a new shot, ``enqueue_syndromes`` sends syndrome measurements to the decoder for processing, and ``get_corrections`` retrieves the decoder's recommended corrections.

These functions are designed to be called from within quantum kernels (marked with ``@cudaq.kernel`` in Python or ``__qpu__`` in C++). The runtime automatically routes these calls to the appropriate backend - whether that is a simulation environment on the local machine or a low-latency connection to quantum hardware. The API is device-agnostic, so the same kernel code works across different deployment scenarios.

The typical pattern is: reset the decoder at the start of each shot, enqueue syndromes after each stabilizer measurement round, then get corrections before measuring the logical observables. Decoders process syndromes asynchronously, so by the time ``get_corrections`` is called, the decoder has usually finished its analysis. If decoding takes longer than expected, ``get_corrections`` will block until results are available.

Here is how to use the real-time decoding API in quantum kernels:

.. tab:: Python

   .. code-block:: python

      @cudaq.kernel
      def qec_circuit(num_rounds: int, decoder_id: int):
          # Reset decoder
          qec.reset_decoder(decoder_id)
          
          # Allocate qubits
          data = cudaq.qvector(25)
          ancx = cudaq.qvector(12)
          ancz = cudaq.qvector(12)
          logical = qec.patch(data, ancx, ancz)
          
          # Prepare logical state
          prep_0(logical)
          
          # Syndrome extraction with real-time decoding
          for round_idx in range(num_rounds):
              syndromes = measure_stabilizers(logical)
              qec.enqueue_syndromes(decoder_id, syndromes)
          
          # Get and apply corrections
          corrections = qec.get_corrections(decoder_id, 1, False)
          if corrections[0]:
              x(data)  # Apply correction
          
          # Measure logical observable
          result = mz(data)

.. tab:: C++

   .. code-block:: cpp

      #include "cudaq/qec/realtime/decoding.h"
      
      __qpu__ void qec_circuit(int num_rounds, int decoder_id) {
          // Reset decoder
          cudaq::qec::decoding::reset_decoder(decoder_id);
          
          // Allocate qubits
          cudaq::qvector data(25), ancx(12), ancz(12);
          cudaq::qec::patch logical(data, ancx, ancz);
          
          // Prepare logical state
          prep0(logical);
          
          // Syndrome extraction with real-time decoding
          for (int round = 0; round < num_rounds; ++round) {
              auto syndromes = measure_stabilizers(logical);
              cudaq::qec::decoding::enqueue_syndromes(decoder_id, syndromes);
          }
          
          // Get and apply corrections
          auto corrections = cudaq::qec::decoding::get_corrections(
              decoder_id, 1, false);
          if (corrections[0]) {
              cudaq::x(data);  // Apply correction
          }
          
          // Measure logical observable
          auto result = mz(data);
      }

Decoder Types
-------------

CUDA-Q QEC provides several decoder types, each optimized for different use cases. The choice of decoder depends on the code distance, expected error rates, latency requirements, and available computational resources. This section describes each decoder type and when to use it.

single_error_lut
^^^^^^^^^^^^^^^^

The single-error lookup table decoder is the fastest option available, designed for scenarios where at most one error has occurred between correction cycles. It works by pre-computing all possible single-error syndromes and their corrections, then performing direct table lookups at runtime. This approach provides microsecond-level latency but limited error correction capability.

This decoder is well-suited for early-stage hardware with low error rates, or for very small codes where the overhead of more sophisticated decoders is not justified. 
For physical error rates below 0.1% (depending on code distance and noise model) with distance-3 or distance-5 codes, the single-error LUT offers excellent performance. However, as error rates increase or code distances grow, decoders that can handle correlated errors are needed.
Note that this decoder cannot handle correlated errors.

The configuration is straightforward with no tunable parameters:

* **Best for**: Small codes (distance ≤ 5), very low error rates (< 0.1%), ultra-low latency requirements
* **Configuration**: No additional parameters needed - just specify the decoder type

.. tab:: Python

   .. code-block:: python

      lut_config = qec.single_error_lut_config()
      config.set_decoder_custom_args(lut_config)

.. tab:: C++

   .. code-block:: cpp

      single_error_lut_config lut_config;
      config.decoder_custom_args = lut_config;

multi_error_lut
^^^^^^^^^^^^^^^

The multi-error lookup table decoder extends the single-error approach by considering combinations of multiple simultaneous errors. It pre-computes syndrome patterns for all possible combinations of up to ``N`` errors (where ``N`` is the ``lut_error_depth`` parameter), providing accurate corrections even when multiple errors occur in the same round.

The configurable parameter is ``lut_error_depth``, which controls the error correction capability. Setting this to 2 means the decoder considers all possible pairs of errors, significantly improving performance over the single-error version. Increasing to 3 provides even better correction but requires more memory and preprocessing time. 
For most practical scenarios with distance-5 to distance-9 codes and error rates around 0.1-1% (it also depends on noise model and code distance), a depth of 2 offers the sweet spot between accuracy and performance.

This decoder works well up to moderate code distances because the lookup table size scales combinatorially with the number of error locations and the error depth. Beyond distance 9, or when higher error rates need to be handled, belief propagation decoders like the NV-QLDPC decoder should be considered.

* **Best for**: Small to medium codes (distance 5-9), moderate error rates (0.1-1%), good balance of speed and accuracy
* **Parameters**:
  
  * ``lut_error_depth`` (int): Maximum number of simultaneous errors to consider (typically 2-3). Higher values improve accuracy but increase memory usage.

.. tab:: Python

   .. code-block:: python

      lut_config = qec.multi_error_lut_config()
      lut_config.lut_error_depth = 2
      config.set_decoder_custom_args(lut_config)

.. tab:: C++

   .. code-block:: cpp

      multi_error_lut_config lut_config;
      lut_config.lut_error_depth = 2;
      config.decoder_custom_args = lut_config;

nv-qldpc-decoder
^^^^^^^^^^^^^^^^

The NVIDIA QLDPC decoder is an advanced belief propagation (BP) decoder optimized for GPU execution, designed for medium to large quantum codes. Unlike lookup table approaches that pre-compute all possible syndromes, belief propagation iteratively refines probability estimates for each error location, making it scalable to larger codes. The decoder optionally includes Ordered Statistics Decoding (OSD) post-processing, which provides a fallback when BP fails to converge.

This decoder excels when working with codes beyond distance 9, where lookup tables become prohibitively large. The belief propagation algorithm scales more gracefully with code size, and GPU acceleration makes it practical for real-time use even with hundreds of syndrome bits. The OSD post-processing is particularly valuable: when BP gets stuck in local minima, OSD performs a focused search over the most likely error patterns to find the correct solution.

The decoder offers extensive tunability. The number of BP iterations can be adjusted to trade off latency for accuracy, the user can choose between sum-product and min-sum BP variants, and OSD search depth can be controlled. For real-time applications, conservative settings (50 iterations, OSD order 7) are a good starting point, with tuning based on observed error rates and latency requirements.

* **Best for**: Medium to large codes (distance ≥ 7), moderate to high error rates, scenarios where GPU acceleration is available
* **Key Parameters**:
  
  * ``error_rate_vec`` (list/vector of floats): Per-mechanism error probabilities - crucial for BP convergence. These should match the DEM's error rates.
  * ``max_iterations`` (int): Maximum BP iterations (typically 50-100). More iterations improve accuracy but increase latency.
  * ``use_osd`` (bool): Enable Ordered Statistics Decoding post-processing when BP fails to converge. Recommended for production use.
  * ``osd_order`` (int): OSD search depth (typically 7-10). Higher values are more thorough but slower.
  * ``bp_method`` (int): BP algorithm variant (0: sum-product, 1: min-sum). Min-sum is faster but slightly less accurate.

.. tab:: Python

   .. code-block:: python

      nv_config = qec.nv_qldpc_decoder_config()
      nv_config.error_rate_vec = [0.01] * config.block_size
      nv_config.max_iterations = 50
      nv_config.use_osd = True
      nv_config.osd_order = 7
      config.set_decoder_custom_args(nv_config)

.. tab:: C++

   .. code-block:: cpp

      nv_qldpc_decoder_config nv_config;
      nv_config.error_rate_vec = error_rates;
      nv_config.max_iterations = 50;
      nv_config.use_osd = true;
      nv_config.osd_order = 7;
      config.decoder_custom_args = nv_config;

sliding_window
^^^^^^^^^^^^^^
The windowed decoder is designed for larger codes where full maximum-likelihood decoding is impractical. 
It works by dividing the decoding problem into overlapping "windows" of syndrome data, 
decoding each window independently (using an inner decoder), 
and then combining the results to form a global correction. 
This approach reduces memory and computational requirements while still capturing most local error correlations.

* **Best for**: Very long circuits, memory-constrained systems
* **Parameters**:
  
  * ``window_size``: Number of rounds per window
  * ``step_size``: Window advancement (equals window_size for non-overlapping)
  * ``num_syndromes_per_round``: Syndromes per round
  * ``inner_decoder_name``: Type of decoder for the windows

.. tab:: Python

   .. code-block:: python

      sw_config = qec.sliding_window_config()
      sw_config.window_size = 10
      sw_config.step_size = 10
      sw_config.num_syndromes_per_round = 24
      sw_config.inner_decoder_name = "multi_error_lut"
      sw_config.multi_error_lut_params = qec.multi_error_lut_config()
      sw_config.multi_error_lut_params.lut_error_depth = 2
      config.set_decoder_custom_args(sw_config)

.. tab:: C++

   .. code-block:: cpp

      sliding_window_config sw_config;
      sw_config.window_size = 10;
      sw_config.step_size = 10;
      sw_config.num_syndromes_per_round = 24;
      sw_config.inner_decoder_name = "multi_error_lut";
      multi_error_lut_config inner_config;
      inner_config.lut_error_depth = 2;
      sw_config.multi_error_lut_params = inner_config;
      config.decoder_custom_args = sw_config;

Backend Selection
-----------------

CUDA-Q QEC's real-time decoding system is designed to work seamlessly across different execution environments. The backend selection determines where quantum circuits run and how decoders communicate with the quantum processor. Understanding the differences between simulation and hardware backends helps the user develop efficiently and deploy confidently.

Simulation Backend
^^^^^^^^^^^^^^^^^^

The simulation backend is the primary tool during development, testing, and algorithm validation. It runs entirely on the local machine, using quantum simulators like Stim to execute circuits while decoders process syndromes in parallel threads. This setup is ideal for rapid iteration: the user can test decoder configurations, validate circuit logic, and debug syndrome processing without waiting for hardware access or paying for compute time.

The simulation backend faithfully mimics real-time decoding's concurrent operation - decoders run in separate threads and process syndromes asynchronously, just as they would on hardware. This means code will behave the same way whether testing locally or running on a quantum computer. The main difference is that simulation does not have the same strict latency constraints, making it easier to experiment with complex decoder configurations.

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
            -lcudaq-qec-realtime-simulation
      
      ./a.out

Quantinuum Hardware Backend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Quantinuum hardware backend connects quantum circuits to real ion-trap quantum computers. Unlike the simulation backend where decoders run on the local machine, **the Quantinuum backend uploads the decoder configuration to Quantinuum's infrastructure**, where decoders run on GPU-equipped servers co-located with the quantum hardware. This architecture minimizes latency between syndrome measurements and correction application.

**Important Setup Requirements:**

1. **Configuration Upload**: When ``configure_decoders_from_file()`` or ``configure_decoders()`` is called, the decoder configuration is automatically base64-encoded and uploaded to Quantinuum's REST API (``api/gpu_decoder_configs/v1beta/``). This happens before job submission. The configuration includes all decoder parameters, error models, and sparse matrices.

2. **Extra Payload Provider**: The user **must** specify ``extra_payload_provider="decoder"`` when setting the target. This registers a payload provider that injects the decoder configuration UUID into each job request, telling Quantinuum which decoder configuration to use for the circuit.

3. **Backend Compilation**: For C++, the user must link against ``-lcudaq-qec-realtime-quantinuum`` instead of the simulation library. This library implements the Quantinuum-specific communication protocol for syndrome transmission.

4. **Configuration Lifetime**: Decoder configurations persist on Quantinuum's servers and are referenced by UUID. If the configuration is modified, it must be uploaded again - the system will generate a new UUID and use the new configuration for subsequent jobs.

**Emulation vs. Hardware Modes:**

Emulation mode (``emulate=True``) is particularly valuable for testing the deployment setup without consuming hardware credits. It uses the **exact same** communication infrastructure as hardware execution, 
the decoder config is still uploaded, decoders still run on Quantinuum's servers, but the circuit executes on their emulators instead of real quantum computers. This is crucial for verifying that:

- The decoder configuration uploads successfully
- The correct decoder is invoked during circuit execution
- Syndrome measurements are transmitted properly
- Corrections are applied at the right times
- The circuit logic handles the decoder API calls correctly

**Important Considerations:**

- **Network Dependency**: Since decoders run remotely, stable network connectivity is required. Configuration upload failures will prevent job submission.
- **Decoder Initialization Time**: The first time a decoder configuration is used, Quantinuum's servers need to initialize the decoders. This adds latency to the first job but subsequent jobs with the same configuration are faster.
- **Configuration Size Limits**: Very large decoder configurations (e.g., distance-17+ codes with lookup tables) may exceed upload limits. Use belief propagation decoders or sliding window approaches for large codes.
- **Debugging**: Set ``CUDAQ_QEC_DEBUG_DECODER=1`` to see the full decoder configuration being uploaded, which helps diagnose configuration issues.

Use the Quantinuum backend for hardware or emulation:

.. tab:: Python

   .. code-block:: python

      cudaq.set_target("quantinuum",
                       emulate=False,  # True for emulation
                       machine="H2-1",
                       extra_payload_provider="decoder")
      
      qec.configure_decoders_from_file("config.yaml")
      results = cudaq.run(my_circuit, shots_count=100)

.. tab:: C++

   .. code-block:: bash

      # Compile for Quantinuum
      nvq++ --target quantinuum --quantinuum-machine H2-1 \
            my_circuit.cpp -lcudaq-qec \
            -lcudaq-qec-realtime-quantinuum
      
      ./a.out

Compilation and Execution Examples
-----------------------------------

This section provides **complete, tested compilation and execution commands** for both simulation and hardware backends, extracted from the CUDA-Q QEC test infrastructure. The section begins with common usage patterns that guide decoder and compilation choices, then provides the specific commands needed for each backend.

Common Use Cases
^^^^^^^^^^^^^^^^^^^^^^

Before diving into compilation details, it is helpful to understand the typical scenarios and how they map to decoder choices and workflow parameters. 
A full set of common examples is provided to guide development.
These examples describe the complete workflow for developing an application that uses real-time decoding in a single file.
The relevant examples can be found at the following path: ``libs/qec/unittests/realtime/app_examples/`` (C++) and the corresponding Python examples.
The files have names like ``surface_code-1.cpp`` and ``surface_code_1.py``.

These examples provide comprehensive support for application development with real-time decoding.
The subsequent step, once the user has chosen the appropriate decoder and the appropriate backend, is to compile and execute the application.
Instructions are provided below for both the simulation and the hardware backends.

C++ Compilation
^^^^^^^^^^^^^^^

**Simulation Backend (Stim)**

Compile with the simulation backend for local testing:

.. code-block:: bash

   nvq++ --target stim my_circuit.cpp \
         -lcudaq-qec \
         -lcudaq-qec-realtime-decoding \
         -lcudaq-qec-realtime-decoding-simulation \
         -o my_circuit_sim

   # Execute
   ./my_circuit_sim --distance 3 --num_shots 1000 --save_dem config.yaml

**Key Points:**

- ``--target stim``: Use the Stim quantum simulator
- ``-lcudaq-qec``: Core QEC library with codes and experiments
- ``-lcudaq-qec-realtime-decoding``: Real-time decoding core API
- ``-lcudaq-qec-realtime-decoding-simulation``: Simulation-specific decoder backend

**Quantinuum Backend (Emulation)**

Compile for Quantinuum emulation mode:

.. code-block:: bash

   nvq++ --target quantinuum --emulate \
         --quantinuum-machine Helios-Fake \
         my_circuit.cpp \
         -lcudaq-qec \
         -lcudaq-qec-realtime-decoding \
         -lcudaq-qec-realtime-decoding-quantinuum \
         -Wl,--export-dynamic \
         -o my_circuit_quantinuum

   # Execute
   ./my_circuit_quantinuum --distance 3 --num_shots 1000 --load_dem config.yaml

**Key Points:**

- ``--target quantinuum --emulate``: Use Quantinuum emulator
- ``--quantinuum-machine Helios-Fake``: Specify machine (``Helios-Fake`` for emulation)
- ``-lcudaq-qec-realtime-decoding-quantinuum``: Quantinuum-specific decoder backend (replaces ``-simulation``)
- ``-Wl,--export-dynamic``: **Required** linker flag for dynamic symbol resolution

**Quantinuum Backend (Hardware)**

Compile for actual Quantinuum hardware:

.. code-block:: bash

   nvq++ --target quantinuum \
         --quantinuum-machine H2-1 \
         my_circuit.cpp \
         -lcudaq-qec \
         -lcudaq-qec-realtime-decoding \
         -lcudaq-qec-realtime-decoding-quantinuum \
         -Wl,--export-dynamic \
         -o my_circuit_hardware

   # Execute
   export CUDAQ_QUANTINUUM_CREDENTIALS=<credentials_file_path>
   ./my_circuit_hardware --distance 3 --num_shots 100 --load_dem config.yaml

**Key Points:**

- Remove ``--emulate`` flag for hardware execution
- Use real machine names: ``Helios``, `Helios-1SC`, etc.
- Set ``CUDAQ_QUANTINUUM_CREDENTIALS`` environment variable with the user's credentials

Check out the `Quantinuum hardware backend documentation <https://nvidia.github.io/cuda-quantum/latest/using/backends/hardware/iontrap.html#quantinuum>`_ for more information.

Python Execution
^^^^^^^^^^^^^^^^

**Simulation Backend (Stim)**

.. code-block:: python

   import os
   import cudaq
   import cudaq_qec as qec

   # IMPORTANT: Set simulator BEFORE importing cudaq
   os.environ["CUDAQ_DEFAULT_SIMULATOR"] = "stim"

   # Configure target
   cudaq.set_target("stim")
   
   # Load decoder configuration
   qec.configure_decoders_from_file("config.yaml")
   
   # Run circuit
   result = cudaq.run(my_circuit, shots_count=1000, noise_model=cudaq.NoiseModel())
   
   # Cleanup
   qec.finalize_decoders()

**Key Points:**

- ``os.environ["CUDAQ_DEFAULT_SIMULATOR"] = "stim"`` **must be set before importing cudaq**
- ``cudaq.set_target("stim")`` configures the simulator target
- No special compilation needed - Python bindings handle library loading

**Quantinuum Backend (Emulation)**

.. code-block:: python

   import cudaq
   import cudaq_qec as qec

   # Configure target with decoder support
   cudaq.set_target("quantinuum",
                    emulate=True,
                    machine="Helios-1Dummy",  # Use "Helios-1Dummy" for emulation
                    extra_payload_provider="decoder")  # REQUIRED for real-time decoding
   
   # Load decoder configuration (uploads to Quantinuum servers)
   qec.configure_decoders_from_file("config.yaml")
   
   # Run circuit
   result = cudaq.run(my_circuit, shots_count=1000)
   
   # Cleanup
   qec.finalize_decoders()

**Key Points:**

- ``emulate=True``: Use Quantinuum emulator
- ``extra_payload_provider="decoder"``: **Required** - registers decoder configuration with Quantinuum's REST API
- Decoder config is automatically uploaded to Quantinuum's servers when ``configure_decoders_from_file()`` is called

**Quantinuum Backend (Hardware)**

.. code-block:: python

   import cudaq
   import cudaq_qec as qec

   # Configure credentials (alternative: use environment variable CUDAQ_QUANTINUUM_CREDENTIALS)
   cudaq.set_credentials("quantinuum", credentials_file="path/to/credentials.json")

   # Configure target for hardware
   cudaq.set_target("quantinuum",
                    emulate=False,  # Hardware execution
                    machine="H2-1",  # Use actual hardware: H2-1, H2-2, etc.
                    extra_payload_provider="decoder")
   
   # Load decoder configuration
   qec.configure_decoders_from_file("config.yaml")
   
   # Run circuit
   result = cudaq.run(my_circuit, shots_count=100)  # Fewer shots for hardware
   
   # Cleanup
   qec.finalize_decoders()

**Key Points:**

- ``emulate=False``: Execute on real quantum hardware
- Use real machine names (check Quantinuum portal for available machines)
- Reduce shot count for hardware experiments (hardware time is expensive)

Complete Workflow Example
^^^^^^^^^^^^^^^^^^^^^^^^^^
Given that the user follows the structure of the examples provided, where each executable takes terminal arguments to configure the application, the following workflow can be used to compile and execute the application.


.. code-block:: bash

   # Phase 1: Generate Detector Error Model (DEM)
   # This is done once per code/distance/noise configuration
   
   ## C++
   ./my_circuit_sim --distance 3 --num_shots 1000 --p_spam 0.01 \
                    --save_dem config_d3.yaml --num_rounds 12 --decoder_window 6
   
   ## Python
   python my_circuit.py --distance 3 --num_shots 1000 --p_spam 0.01 \
                        --save_dem config_d3.yaml --num_rounds 12 --decoder_window 6
   
   # Phase 2: Run with Real-Time Decoding
   # Use the saved DEM configuration
   
   ## Simulation
   ./my_circuit_sim --distance 3 --num_shots 1000 --load_dem config_d3.yaml \
                    --num_rounds 12 --decoder_window 6
   
   ## Quantinuum Emulation
   ./my_circuit_quantinuum --distance 3 --num_shots 1000 --load_dem config_d3.yaml \
                           --num_rounds 12 --decoder_window 6
   
   ## Quantinuum Hardware
   export CUDAQ_QUANTINUUM_CREDENTIALS=credentials.json
   ./my_circuit_hardware --distance 3 --num_shots 100 --load_dem config_d3.yaml \
                         --num_rounds 12 --decoder_window 6

**Workflow Parameters:**

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
   
   # Set default simulator (Python only, before importing cudaq)
   export CUDAQ_DEFAULT_SIMULATOR=stim
   
   # Dump JIT IR for debugging compilation issues
   export CUDAQ_DUMP_JIT_IR=1
   
   # Keep log files after test execution
   export KEEP_LOG_FILES=true
   
   # Set Quantinuum credentials file
   export CUDAQ_QUANTINUUM_CREDENTIALS=/path/to/credentials.json

**Common Compilation Issues:**

1. **Missing libraries**: Ensure all ``-lcudaq-qec-*`` libraries are linked in correct order
2. **Wrong backend library**: Use ``-simulation`` for Stim, ``-quantinuum`` for Quantinuum
3. **Missing ``--export-dynamic``**: Required for Quantinuum targets
4. **Wrong target flags**: ``--emulate`` with ``Helios-Fake`` for emulation, remove for hardware

**Common Runtime Issues:**

1. **"Decoder X not found"**: Call ``configure_decoders_from_file()`` before circuit execution
2. **"Configuration upload failed"**: Check network connectivity and Quantinuum credentials
3. **Dimension mismatch errors**: Verify DEM dimensions match the circuit's syndrome count
4. **High error rates**: Check decoder window size matches DEM generation window

Testing Your Setup
^^^^^^^^^^^^^^^^^^^

The installation can be verified with this minimal test:

.. tab:: Python

   .. code-block:: python

      import os
      os.environ["CUDAQ_DEFAULT_SIMULATOR"] = "stim"
      
      import cudaq
      import cudaq_qec as qec
      
      # Test decoder configuration
      print("Testing real-time decoding setup...")
      
      # Create minimal decoder config
      config = qec.decoder_config()
      config.id = 0
      config.type = "multi_error_lut"
      config.block_size = 10
      config.syndrome_size = 5
      config.H_sparse = [0, 1, -1, 1, 2, -1]  # Minimal test data
      config.O_sparse = [0, -1]
      config.D_sparse = [0, -1]
      
      lut_config = qec.multi_error_lut_config()
      lut_config.lut_error_depth = 1
      config.set_decoder_custom_args(lut_config)
      
      multi_config = qec.multi_decoder_config()
      multi_config.decoders = [config]
      
      status = qec.configure_decoders(multi_config)
      print(f"Configuration status: {status}")
      
      qec.finalize_decoders()
      print("Setup verified!")

.. tab:: C++

   .. code-block:: bash

      # Compile test
      nvq++ --target stim test_setup.cpp \
            -lcudaq-qec \
            -lcudaq-qec-realtime-decoding \
            -lcudaq-qec-realtime-decoding-simulation
      
      # Run
      ./a.out

If the test completes without errors, the setup is ready for real-time decoding experiments.

Best Practices
--------------

Successfully deploying real-time decoding requires attention to several key details. These best practices emerge from real-world usage and help avoid common pitfalls while optimizing performance.

Decoder Selection
^^^^^^^^^^^^^^^^^

Choosing the right decoder is crucial for balancing accuracy, latency, and resource usage. The decision depends on multiple factors: the quantum code's distance, expected physical error rates, available computational resources, and latency requirements. This table provides initial guidance, but validation with the specific workload is always recommended:

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20

   * - Code Distance
     - Error Rate
     - Recommended Decoder
     - Speed
   * - ≤ 5
     - < 0.1%
     - single_error_lut
     - Fastest
   * - 5-9
     - 0.1-1%
     - multi_error_lut (depth=2-3)
     - Fast
   * - ≥ 7
     - Any
     - multi_error_lut or nv-qldpc
     - Medium

Configuration Validation
^^^^^^^^^^^^^^^^^^^^^^^^^

Configuration errors are one of the most common sources of problems in real-time decoding systems. A mismatch between the DEM's dimensions and the decoder configuration can lead to incorrect decoding results or runtime errors that are difficult to debug. Validating the configuration before deploying to hardware is essential - it is much easier to catch dimension mismatches or parameter errors during development than to diagnose them after they have affected experiment results.

The validation process should check that matrix dimensions are consistent, syndrome sizes match the circuit's stabilizer count, and decoder-specific parameters are within valid ranges. Pay special attention to the temporal dimension: if the circuit runs for N rounds, the D_sparse matrix must be sized appropriately. A good practice is to write a dedicated validation function that is run before every major experiment.

Always validate the configuration!

Troubleshooting
---------------

Even with careful configuration, issues may be encountered during real-time decoding. This section covers the most common problems and their solutions, organized by symptom. When troubleshooting, start by isolating whether the issue is in DEM generation, decoder configuration, or runtime execution.

High Logical Error Rate
^^^^^^^^^^^^^^^^^^^^^^^

When the logical error rate is higher than expected, the decoder is not correctly inferring the actual errors from the syndrome measurements. This typically means there is a mismatch between what the decoder expects and what is actually happening in the circuit.

The most common culprit is an inaccurate detector error model. If the DEM was generated with one noise model but the system is running with different noise characteristics, the decoder's corrections will be suboptimal. This happens frequently when moving from simulation to hardware - hardware has noise patterns that do not perfectly match simple depolarizing models. Always regenerate the DEM to match the actual error characteristics being experienced.

Another common issue is decoder capacity. If single_error_lut is being used but multiple errors are occurring within correction cycles, the decoder simply does not have the capability to handle the error patterns it is seeing. Similarly, if multi_error_lut with depth=2 is being used but three-error events are common, the user should either increase the depth or switch to a more sophisticated decoder.

**Possible Issues**:

* **DEM-noise mismatch**: The detector error model was generated with assumptions that do not match actual circuit behavior
* **Insufficient decoder capacity**: Decoder cannot handle the complexity of errors occurring in the circuit
* **Window size too small**: For sliding window decoders, small windows cannot capture temporal error correlations
* **Incorrect sparse matrices**: H_sparse or O_sparse do not accurately reflect the circuit's error structure

**Solutions**:

* **Regenerate DEM**: Create a new DEM using a noise model that matches the current hardware or simulation parameters
* **Upgrade decoder type**: Switch from single_error_lut to multi_error_lut, or from lookup tables to nv-qldpc-decoder
* **Tune decoder parameters**: Increase ``lut_error_depth`` for LUT decoders, or adjust BP parameters for QLDPC
* **Increase window size**: For windowed decoding, try larger windows to capture more temporal correlations

High Latency
^^^^^^^^^^^^

Latency problems manifest as decoders taking too long to produce corrections, potentially exceeding coherence times or slowing down experiment throughput. Real-time decoding is designed to complete within microseconds to milliseconds, but poor configuration can push latencies much higher.

For QLDPC decoders, the iteration count directly controls latency. Each belief propagation iteration requires a full pass over the parity check matrix, and if ``max_iterations`` is set too high, the decoder might spend excessive time refining estimates that were already good enough. Most problems converge within 20-50 iterations, and going beyond 100 iterations rarely improves results enough to justify the latency cost.

Window size is another latency factor. Larger windows give decoders more context, which improves accuracy, but they also mean more syndrome data to process. If latency spikes are observed, try reducing the window size - smaller windows with slightly lower accuracy may give better overall performance when latency is critical.

**Possible Issues**:

* **Excessive BP iterations**: QLDPC decoder running too many belief propagation iterations before returning results
* **Oversized windows**: Processing too many syndrome rounds at once in sliding window configurations
* **Suboptimal batch sizes**: GPU not being used efficiently due to poor batching
* **Resource contention**: Other processes competing for GPU or CPU resources

**Solutions**:

* **Reduce max_iterations**: Lower the QLDPC ``max_iterations`` parameter (try 30-50 instead of 100)
* **Smaller windows**: Reduce window size for sliding window decoders, trading some accuracy for speed
* **Optimize batching**: Adjust ``bp_batch_size`` and ``osd_batch_size`` to better utilize the GPU
* **Resource isolation**: Ensure decoders have dedicated computational resources without interference

Memory Errors
^^^^^^^^^^^^^

Out-of-memory errors occur when decoder configurations require more memory than the system has available. This is particularly common with lookup table decoders, where memory usage grows combinatorially with code distance and error depth. A distance-9 surface code with ``lut_error_depth=3`` can require gigabytes of memory for the lookup table alone.

The memory footprint also scales with the number of logical qubits being managed. Each decoder instance maintains its own state, lookup tables, and syndrome buffers. Running 10 logical qubits with multi_error_lut decoders simultaneously can easily exceed available memory, even on systems with substantial RAM.

If memory limits are being reached, several options are available. The simplest is reducing ``lut_error_depth`` - dropping from 3 to 2 can reduce memory usage by orders of magnitude while often maintaining acceptable accuracy. For longer-term solutions, consider switching to decoders with better memory scaling characteristics, like the sliding window wrapper or the nv-qldpc-decoder, which use algorithms that do not require storing exponentially large lookup tables.

**Possible Issues**:

* **LUT depth too high**: Lookup table size grows as O(n^d) where n is error locations and d is depth
* **Too many logical qubits**: Each decoder instance maintains full state and tables
* **Large window sizes**: Windowed decoders accumulate syndrome history in memory
* **Insufficient system RAM**: System does not have enough memory for the configuration

**Solutions**:

* **Reduce lut_error_depth**: Lower from 3 to 2, or from 2 to 1 if necessary - major memory savings
* **Use sliding window**: Limit memory usage by processing fixed-size syndrome windows
* **Switch to QLDPC**: Belief propagation decoders have O(n) memory scaling instead of O(n^d)
* **Reduce logical qubits**: Decrease the number of concurrent decoder instances
* **Upgrade hardware**: If the application truly needs these parameters, consider systems with more RAM or GPU memory

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

