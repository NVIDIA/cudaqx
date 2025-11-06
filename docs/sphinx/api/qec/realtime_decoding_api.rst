.. _realtime_decoding_api:

Real-Time Decoding API
======================

The Real-Time Decoding API enables low-latency error correction on quantum hardware by allowing CUDA-Q quantum kernels to interact with decoders during circuit execution. This API is designed for use cases where corrections must be calculated and applied within qubit coherence times.

The real-time decoding system supports both hardware integration (e.g., Quantinuum H-Series) and simulation environments for local testing.

.. note::
   The NV-QLDPC decoder is not included in the public repository by default and requires additional installation. For most applications, the ``multi_error_lut`` decoder provides excellent performance and is readily available.

Core Decoding Functions
------------------------

These functions can be called from within CUDA-Q quantum kernels (``__qpu__`` in C++, ``@cudaq.kernel`` in Python) to interact with real-time decoders.

.. function:: enqueue_syndromes(decoder_id, syndromes, tag=0)

   Enqueue syndrome measurements for decoding.

   :param decoder_id: Unique identifier for the decoder instance (matches configured decoder ID)
   :param syndromes: Vector/list of syndrome measurement results from stabilizer measurements
   :param tag: Optional tag for logging and debugging (default: 0)

   .. tab:: Python

      .. code-block:: python

         import cudaq
         import cudaq_qec as qec
         from cudaq_qec import patch

         @cudaq.kernel
         def measure_and_decode(logical: patch, decoder_id: int):
             syndromes = measure_stabilizers(logical)
             qec.enqueue_syndromes(decoder_id, syndromes, 0)

   .. tab:: C++

      .. code-block:: cpp

         #include "cudaq/qec/realtime/decoding.h"

         __qpu__ void measure_and_decode(cudaq::qec::patch logical, int decoder_id) {
             auto syndromes = measure_stabilizers(logical);
             cudaq::qec::decoding::enqueue_syndromes(decoder_id, syndromes, 0);
         }

.. function:: get_corrections(decoder_id, return_size, reset=false)

   Retrieve calculated corrections from the decoder.

   :param decoder_id: Unique identifier for the decoder instance
   :param return_size: Number of correction bits to return (typically equals number of logical observables)
   :param reset: Whether to reset accumulated corrections after retrieval (default: false)
   :returns: Vector/list of boolean values indicating detected bit flips for each logical observable

   .. tab:: Python

      .. code-block:: python

         @cudaq.kernel
         def apply_corrections(logical: patch, decoder_id: int):
             corrections = qec.get_corrections(decoder_id, 1, False)
             if corrections[0]:
                 x(logical.data)  # Apply transversal X correction

   .. tab:: C++

      .. code-block:: cpp

         __qpu__ void apply_corrections(cudaq::qec::patch logical, int decoder_id) {
             auto corrections = cudaq::qec::decoding::get_corrections(
                 decoder_id, /*return_size=*/1, /*reset=*/false);
             if (corrections[0]) {
                 cudaq::x(logical.data);  // Apply transversal X correction
             }
         }

.. function:: reset_decoder(decoder_id)

   Reset decoder state, clearing all queued syndromes and accumulated corrections.

   :param decoder_id: Unique identifier for the decoder instance to reset

   .. tab:: Python

      .. code-block:: python

         @cudaq.kernel
         def run_experiment(decoder_id: int):
             qec.reset_decoder(decoder_id)  # Reset at start of each shot
             # ... perform experiment ...

   .. tab:: C++

      .. code-block:: cpp

         __qpu__ void run_experiment(int decoder_id) {
             cudaq::qec::decoding::reset_decoder(decoder_id);  // Reset at start of each shot
             // ... perform experiment ...
         }

Configuration API
-----------------

The configuration API enables setting up decoders before circuit execution. Decoders are configured using YAML files or programmatically constructed configuration objects.

.. function:: configure_decoders(config)

   Configure decoders from a multi_decoder_config object.

   :param config: multi_decoder_config object containing decoder specifications
   :returns: 0 on success, non-zero error code on failure

.. function:: configure_decoders_from_file(config_file)

   Configure decoders from a YAML file.

   :param config_file: Path to YAML configuration file
   :returns: 0 on success, non-zero error code on failure

.. function:: configure_decoders_from_str(config_str)

   Configure decoders from a YAML string.

   :param config_str: YAML configuration as a string
   :returns: 0 on success, non-zero error code on failure

.. function:: finalize_decoders()

   Finalize and clean up decoder resources. Should be called before program exit.

Configuration Example
^^^^^^^^^^^^^^^^^^^^^

.. tab:: Python

   .. code-block:: python

      import cudaq_qec as qec

      # Create decoder configuration
      config = qec.decoder_config()
      config.id = 0
      config.type = "multi_error_lut"
      config.block_size = dem.num_error_mechanisms()
      config.syndrome_size = dem.num_detectors()
      config.H_sparse = qec.pcm_to_sparse_vec(dem.detector_error_matrix)
      config.O_sparse = qec.pcm_to_sparse_vec(dem.observables_flips_matrix)

      # Set decoder parameters
      lut_config = qec.multi_error_lut_config()
      lut_config.lut_error_depth = 2
      config.set_decoder_custom_args(lut_config)

      # Configure decoder
      multi_config = qec.multi_decoder_config()
      multi_config.decoders = [config]
      status = qec.configure_decoders(multi_config)

      # Or save to file and load
      yaml_str = multi_config.to_yaml_str()
      with open("decoder_config.yaml", "w") as f:
          f.write(yaml_str)
      
      status = qec.configure_decoders_from_file("decoder_config.yaml")

.. tab:: C++

   .. code-block:: cpp

      #include "cudaq/qec/realtime/decoding_config.h"
      #include "cudaq/qec/pcm_utils.h"

      using namespace cudaq::qec::decoding::config;

      // Create decoder configuration
      decoder_config config;
      config.id = 0;
      config.type = "multi_error_lut";
      config.block_size = dem.num_error_mechanisms();
      config.syndrome_size = dem.num_detectors();
      config.H_sparse = cudaq::qec::pcm_to_sparse_vec(dem.detector_error_matrix);
      config.O_sparse = cudaq::qec::pcm_to_sparse_vec(dem.observables_flips_matrix);

      // Set decoder parameters
      multi_error_lut_config lut_config;
      lut_config.lut_error_depth = 2;
      config.decoder_custom_args = lut_config;

      // Configure decoder
      multi_decoder_config multi_config;
      multi_config.decoders.push_back(config);
      int status = configure_decoders(multi_config);

      // Or save to file and load
      std::string yaml_str = multi_config.to_yaml_str();
      std::ofstream file("decoder_config.yaml");
      file << yaml_str;
      file.close();
      
      status = configure_decoders_from_file("decoder_config.yaml");

Decoder Types
-------------

single_error_lut
^^^^^^^^^^^^^^^^

Fast decoder assuming at most one error occurred.

**Best for:** Small codes (distance ≤ 5), very low error rates, ultra-low latency requirements.

**Configuration:**

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

**Best for:** Small to medium codes (distance 5-9), moderate error rates, excellent accuracy/speed tradeoff.

**Configuration:**

.. tab:: Python

   .. code-block:: python

      lut_config = qec.multi_error_lut_config()
      lut_config.lut_error_depth = 2  # Consider up to 2 simultaneous errors
      config.set_decoder_custom_args(lut_config)

.. tab:: C++

   .. code-block:: cpp

      multi_error_lut_config lut_config;
      lut_config.lut_error_depth = 2;  // Consider up to 2 simultaneous errors
      config.decoder_custom_args = lut_config;

**Parameters:**

- ``lut_error_depth`` (int): Maximum number of simultaneous errors to consider (typically 2-3)

nv-qldpc-decoder
^^^^^^^^^^^^^^^^

Advanced belief propagation decoder with optional OSD post-processing.

.. note::
   Requires additional installation. Not included in the public repository.

**Best for:** Medium to large codes (distance ≥ 7), tunable accuracy/performance tradeoff, GPU-accelerated processing.

**Configuration:**

.. tab:: Python

   .. code-block:: python

      nv_config = qec.nv_qldpc_decoder_config()
      nv_config.use_sparsity = True
      nv_config.error_rate_vec = [0.01] * config.block_size
      nv_config.max_iterations = 50
      nv_config.use_osd = True
      nv_config.osd_order = 7
      nv_config.bp_method = 0
      config.set_decoder_custom_args(nv_config)

.. tab:: C++

   .. code-block:: cpp

      nv_qldpc_decoder_config nv_config;
      nv_config.use_sparsity = true;
      nv_config.error_rate_vec = error_rates;
      nv_config.max_iterations = 50;
      nv_config.use_osd = true;
      nv_config.osd_order = 7;
      nv_config.bp_method = 0;
      config.decoder_custom_args = nv_config;

**Key Parameters:**

- ``error_rate_vec`` (list/vector of doubles): Per-mechanism error rates
- ``max_iterations`` (int): Maximum BP iterations
- ``use_osd`` (bool): Enable Ordered Statistics Decoding post-processing
- ``bp_method`` (int): BP algorithm variant (0: standard, 1: min-sum)
- ``proc_float`` (string): Floating-point precision ("fp32" or "fp64")

sliding_window
^^^^^^^^^^^^^^

Wrapper that applies an inner decoder to windows of syndrome data.

**Best for:** Very long circuits (many rounds), memory-constrained systems, continuous operation.

**Configuration:**

.. tab:: Python

   .. code-block:: python

      sw_config = qec.sliding_window_config()
      sw_config.window_size = 10
      sw_config.step_size = 10
      sw_config.num_syndromes_per_round = 24
      sw_config.error_rate_vec = [0.01] * config.block_size
      sw_config.inner_decoder_name = "multi_error_lut"
      
      # Configure inner decoder
      sw_config.multi_error_lut_params = qec.multi_error_lut_config()
      sw_config.multi_error_lut_params.lut_error_depth = 2
      
      config.set_decoder_custom_args(sw_config)

.. tab:: C++

   .. code-block:: cpp

      sliding_window_config sw_config;
      sw_config.window_size = 10;
      sw_config.step_size = 10;
      sw_config.num_syndromes_per_round = 24;
      sw_config.error_rate_vec = error_rates;
      sw_config.inner_decoder_name = "multi_error_lut";
      
      // Configure inner decoder
      multi_error_lut_config inner_config;
      inner_config.lut_error_depth = 2;
      sw_config.multi_error_lut_params = inner_config;
      
      config.decoder_custom_args = sw_config;

**Parameters:**

- ``window_size`` (int): Number of rounds in each decoding window
- ``step_size`` (int): Number of rounds to advance window
- ``num_syndromes_per_round`` (int): Number of syndrome measurements per round
- ``inner_decoder_name`` (string): Type of decoder for each window
- ``error_rate_vec`` (list/vector): Error rates for each mechanism

Helper Functions
----------------

.. function:: pcm_to_sparse_vec(pcm)

   Convert a parity check matrix to sparse vector representation.

   :param pcm: Dense binary matrix (tensor or numpy array)
   :returns: Sparse vector representation (list/vector of integers)

.. function:: pcm_from_sparse_vec(sparse_vec, num_rows, num_cols)

   Convert a sparse vector representation back to a dense matrix.

   :param sparse_vec: Sparse representation
   :param num_rows: Number of rows in the matrix
   :param num_cols: Number of columns in the matrix
   :returns: Dense binary matrix

.. function:: generate_timelike_sparse_detector_matrix(num_syndromes_per_round, num_rounds, include_first_round=false)
              generate_timelike_sparse_detector_matrix(num_syndromes_per_round, num_rounds, first_round)

   Generate detector correlation matrix for repeated syndrome measurements.

   :param num_syndromes_per_round: Number of syndrome measurements per round
   :param num_rounds: Total number of rounds
   :param include_first_round: Whether first round syndromes are deterministic
   :param first_round: Custom specification for first round detector relationships
   :returns: Sparse matrix encoding how detectors relate across time

Complete Example
----------------

Here is a complete workflow from DEM generation to real-time decoding:

.. tab:: Python

   .. literalinclude:: ../../libs/qec/unittests/realtime/app_examples/surface_code_1.py
      :language: python
      :start-after: # ============================================================================ #
      :end-before: if __name__ == "__main__":
      :linenos:

.. tab:: C++

   .. literalinclude:: ../../libs/qec/unittests/realtime/app_examples/surface_code-1.cpp
      :language: cpp
      :start-after: void show_help()
      :end-before: int main(int argc, char **argv)
      :linenos:

See Also
--------

- :doc:`../../examples_rst/qec/realtime_decoding` - Complete Guide with Examples and Configuration
- Example applications: ``libs/qec/unittests/realtime/app_examples/``

