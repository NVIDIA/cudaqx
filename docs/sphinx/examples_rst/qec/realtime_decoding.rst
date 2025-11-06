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

.. note::
   The NV-QLDPC decoder is not included in the public repository by default and requires additional installation. For most use cases, the ``multi_error_lut`` decoder provides excellent performance and is readily available.

Workflow Overview
-----------------

A typical real-time decoding workflow consists of four stages:

1. **Detector Error Model (DEM) Generation**: Characterize how errors propagate through your quantum circuit
2. **Decoder Configuration**: Configure decoders using the DEM and desired parameters
3. **Decoder Initialization**: Load configuration before circuit execution
4. **Real-Time Decoding**: Use the decoding API within quantum kernels

Complete Example
----------------

Here is a complete end-to-end example demonstrating real-time decoding:

.. tab:: Python

   .. literalinclude:: ../../libs/qec/unittests/realtime/app_examples/surface_code_1.py
      :language: python
      :start-after: # ============================================================================ #
      :end-before: if __name__ == "__main__":

.. tab:: C++

   .. literalinclude:: ../../libs/qec/unittests/realtime/app_examples/surface_code-1.cpp
      :language: cpp
      :lines: 1-100

Configuration
-------------

Step 1: Generate Detector Error Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Generate a detector error model that characterizes how errors propagate through your circuit:

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

Create decoder configuration from your DEM:

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

Save configuration to YAML file for reuse:

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

Use the decoding API within quantum kernels:

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

single_error_lut
^^^^^^^^^^^^^^^^

Fast decoder assuming at most one error occurred.

* **Best for**: Small codes (distance ≤ 5), very low error rates
* **Configuration**: No additional parameters

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

Decoder considering combinations of multiple errors. **Recommended for most use cases.**

* **Best for**: Small to medium codes (distance 5-9), moderate error rates
* **Parameters**:
  
  * ``lut_error_depth`` (int): Maximum number of simultaneous errors (typically 2-3)

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

Advanced belief propagation decoder with optional OSD post-processing.

.. note::
   Requires additional installation. Not included in the public repository.

* **Best for**: Medium to large codes (distance ≥ 7), GPU-accelerated processing
* **Key Parameters**:
  
  * ``error_rate_vec``: Per-mechanism error rates
  * ``max_iterations``: Maximum BP iterations
  * ``use_osd``: Enable OSD post-processing
  * ``osd_order``: OSD search depth (typically 7)
  * ``bp_method``: BP variant (0: standard, 1: min-sum)

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

Wrapper that applies an inner decoder to windows of syndrome data.

* **Best for**: Very long circuits, memory-constrained systems
* **Parameters**:
  
  * ``window_size``: Number of rounds per window
  * ``step_size``: Window advancement (equals window_size for non-overlapping)
  * ``num_syndromes_per_round``: Syndromes per round
  * ``inner_decoder_name``: Type of decoder for each window

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

Use Cases and Examples
----------------------

Example 1: Basic Memory Experiment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Preserve quantum information over multiple rounds of error correction.

**Application**: Quantum memory, logical qubit benchmarking

**Complete Example**: ``libs/qec/unittests/realtime/app_examples/surface_code-1.cpp`` (C++), ``surface_code_1.py`` (Python)

Key features demonstrated:

* Detector error model generation using MSM
* Multi-error LUT decoder configuration
* Syndrome extraction and correction application
* Multiple syndrome extraction rounds with windowing

Running the example:

.. tab:: Python

   .. code-block:: bash

      python surface_code_1.py --distance 5 --num_shots 100 --p_spam 0.01

.. tab:: C++

   .. code-block:: bash

      # Generate DEM
      ./surface_code-1 --distance 5 --save_dem surface_d5.yaml --p_spam 0.01
      
      # Run with DEM
      ./surface_code-1 --distance 5 --num_shots 100 --load_dem surface_d5.yaml

Example 2: Windowed Decoding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Process syndromes in fixed-size windows for long circuits.

**Application**: Long-duration quantum computations, memory-constrained systems

Key configuration changes:

.. tab:: Python

   .. code-block:: python

      decoder_window = 5  # Process 5 rounds at a time
      num_rounds = 20     # Total 20 rounds
      
      # Generate DEM for window size (not total rounds)
      msm_strings, _, _, _ = qec.compute_msm(
          lambda: circuit(decoder_window), True)

.. tab:: C++

   .. code-block:: cpp

      int decoder_window = 5;  // Process 5 rounds at a time
      int num_rounds = 20;     // Total 20 rounds
      
      // Generate DEM for window size
      cudaq::qec::qpu::demo_circuit_qpu(
          false, prep, numData, numAncx, numAncz,
          decoder_window,  // Use window size for DEM
          1, schedX, schedZ, p_spam, false, decoder_window);

Example 3: Multi-Qubit System
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Manage multiple logical qubits with independent decoders.

**Application**: Multi-qubit algorithms, quantum networking

Configuration:

.. tab:: Python

   .. code-block:: python

      num_logical_qubits = 3
      multi_config = qec.multi_decoder_config()
      
      for qubit_id in range(num_logical_qubits):
          config = qec.decoder_config()
          config.id = qubit_id
          config.type = "multi_error_lut"
          # ... configure ...
          multi_config.decoders.append(config)
      
      qec.configure_decoders(multi_config)

.. tab:: C++

   .. code-block:: cpp

      int num_logical_qubits = 3;
      multi_decoder_config multi_config;
      
      for (int i = 0; i < num_logical_qubits; ++i) {
          decoder_config config;
          config.id = i;
          config.type = "multi_error_lut";
          // ... configure ...
          multi_config.decoders.push_back(config);
      }
      
      configure_decoders(multi_config);

Usage in kernel:

.. tab:: Python

   .. code-block:: python

      @cudaq.kernel
      def multi_qubit_circuit():
          for qubit_id in range(num_logical_qubits):
              qec.reset_decoder(qubit_id)
              # ... measure and enqueue ...
              qec.enqueue_syndromes(qubit_id, syndromes)
          
          for qubit_id in range(num_logical_qubits):
              corrections = qec.get_corrections(qubit_id, 1, False)
              if corrections[0]:
                  apply_correction(qubit_id)

.. tab:: C++

   .. code-block:: cpp

      __qpu__ void multi_qubit_circuit() {
          for (int i = 0; i < num_logical_qubits; ++i) {
              cudaq::qec::decoding::reset_decoder(i);
              // ... measure and enqueue ...
              cudaq::qec::decoding::enqueue_syndromes(i, syndromes);
          }
          
          for (int i = 0; i < num_logical_qubits; ++i) {
              auto corrections = cudaq::qec::decoding::get_corrections(i, 1, false);
              if (corrections[0])
                  apply_correction(i);
          }
      }

Backend Selection
-----------------

Simulation Backend
^^^^^^^^^^^^^^^^^^

Use for local development and testing:

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

Use for hardware or emulation:

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

Best Practices
--------------

Decoder Selection
^^^^^^^^^^^^^^^^^

Choose based on code parameters and requirements:

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

Always validate your configuration:

.. tab:: Python

   .. code-block:: python

      # Check matrix dimensions
      assert config.syndrome_size == H_matrix.shape[0]
      assert config.block_size == H_matrix.shape[1]
      
      # Test configuration
      status = qec.configure_decoders(multi_config)
      if status != 0:
          print("Configuration failed!")

.. tab:: C++

   .. code-block:: cpp

      // Check matrix dimensions
      assert(config.syndrome_size == H_matrix.shape()[0]);
      assert(config.block_size == H_matrix.shape()[1]);
      
      // Test configuration
      int status = configure_decoders(multi_config);
      if (status != 0) {
          std::cerr << "Configuration failed!" << std::endl;
      }

Troubleshooting
---------------

High Logical Error Rate
^^^^^^^^^^^^^^^^^^^^^^^

**Possible Issues**:

* DEM doesn't match actual noise
* Decoder capacity too low
* Window size too small

**Solutions**:

* Regenerate DEM with accurate noise model
* Use higher-capacity decoder (multi_error_lut or QLDPC)
* Increase window size or decoder depth

High Latency
^^^^^^^^^^^^

**Possible Issues**:

* Too many BP iterations (QLDPC)
* Window size too large
* Inefficient batch sizes

**Solutions**:

* Reduce ``max_iterations`` for QLDPC
* Use smaller windows
* Tune GPU batch sizes

Memory Errors
^^^^^^^^^^^^^

**Possible Issues**:

* LUT depth too high
* Too many logical qubits

**Solutions**:

* Reduce ``lut_error_depth``
* Use sliding window decoder
* Switch to QLDPC decoder

See Also
--------

* :doc:`/api/qec/cpp_api` - C++ API Reference (includes Real-Time Decoding)
* :doc:`/api/qec/python_api` - Python API Reference (includes Real-Time Decoding)
* Example source code: ``libs/qec/unittests/realtime/app_examples/``

