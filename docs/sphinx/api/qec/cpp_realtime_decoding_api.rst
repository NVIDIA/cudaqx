.. _cpp_realtime_decoding_api:

Real-Time Decoding C++ API
===========================

The Real-Time Decoding API enables low-latency error correction on quantum hardware by allowing CUDA-Q quantum kernels to interact with decoders during circuit execution. This API is designed for use cases where corrections must be calculated and applied within qubit coherence times.

The real-time decoding system supports both hardware integration (e.g., Quantinuum H-Series) and simulation environments for local testing.

.. note::
   The NV-QLDPC decoder is not included in the public repository by default and requires additional installation. For most applications, the ``multi_error_lut`` decoder provides excellent performance and is readily available.

Core Decoding Functions
------------------------

These functions can be called from within CUDA-Q quantum kernels (``__qpu__`` functions) to interact with real-time decoders.

.. function:: void cudaq::qec::decoding::enqueue_syndromes(uint64_t decoder_id, const std::vector<cudaq::measure_result>& syndromes, uint64_t tag = 0)

   Enqueue syndrome measurements for decoding.

   :param decoder_id: Unique identifier for the decoder instance (matches configured decoder ID)
   :param syndromes: Vector of syndrome measurement results from stabilizer measurements
   :param tag: Optional tag for logging and debugging (default: 0)

   **Example:**

   .. code-block:: cpp

      #include "cudaq/qec/realtime/decoding.h"

      __qpu__ void measure_and_decode(cudaq::qec::patch logical, int decoder_id) {
          auto syndromes = measure_stabilizers(logical);
          cudaq::qec::decoding::enqueue_syndromes(decoder_id, syndromes, 0);
      }

.. function:: std::vector<bool> cudaq::qec::decoding::get_corrections(uint64_t decoder_id, uint64_t return_size, bool reset = false)

   Retrieve calculated corrections from the decoder.

   :param decoder_id: Unique identifier for the decoder instance
   :param return_size: Number of correction bits to return (typically equals number of logical observables)
   :param reset: Whether to reset accumulated corrections after retrieval (default: false)
   :returns: Vector of boolean values indicating detected bit flips for each logical observable

   **Example:**

   .. code-block:: cpp

      __qpu__ void apply_corrections(cudaq::qec::patch logical, int decoder_id) {
          auto corrections = cudaq::qec::decoding::get_corrections(
              decoder_id, /*return_size=*/1, /*reset=*/false);
          if (corrections[0]) {
              cudaq::x(logical.data);  // Apply transversal X correction
          }
      }

.. function:: void cudaq::qec::decoding::reset_decoder(uint64_t decoder_id)

   Reset decoder state, clearing all queued syndromes and accumulated corrections.

   :param decoder_id: Unique identifier for the decoder instance to reset

   **Example:**

   .. code-block:: cpp

      __qpu__ void run_experiment(int decoder_id) {
          cudaq::qec::decoding::reset_decoder(decoder_id);  // Reset at start of each shot
          // ... perform experiment ...
      }

Configuration API
-----------------

The configuration API enables setting up decoders before circuit execution. Decoders are configured using YAML files or programmatically constructed configuration objects.

.. function:: int cudaq::qec::decoding::config::configure_decoders(const multi_decoder_config& config)

   Configure decoders from a multi_decoder_config object.

   :param config: multi_decoder_config object containing decoder specifications
   :returns: 0 on success, non-zero error code on failure

.. function:: int cudaq::qec::decoding::config::configure_decoders_from_file(const std::string& config_file)

   Configure decoders from a YAML file.

   :param config_file: Path to YAML configuration file
   :returns: 0 on success, non-zero error code on failure

.. function:: int cudaq::qec::decoding::config::configure_decoders_from_str(const std::string& config_str)

   Configure decoders from a YAML string.

   :param config_str: YAML configuration as a string
   :returns: 0 on success, non-zero error code on failure

.. function:: void cudaq::qec::decoding::config::finalize_decoders()

   Finalize and clean up decoder resources. Should be called before program exit.

Configuration Structures
^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: cudaq::qec::decoding::config::decoder_config
   :members:

.. doxygenclass:: cudaq::qec::decoding::config::multi_decoder_config
   :members:

Configuration Example
^^^^^^^^^^^^^^^^^^^^^

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

Helper Functions
----------------

Real-time decoding requires converting matrices to sparse format for efficient decoder configuration. The following utility functions are essential:

.. function:: std::vector<std::int64_t> cudaq::qec::pcm_to_sparse_vec(const cudaqx::tensor<uint8_t>& pcm)

   Convert a parity check matrix (PCM) to sparse vector representation for decoder configuration.

   :param pcm: Dense binary matrix (e.g., ``dem.detector_error_matrix`` or ``dem.observables_flips_matrix``)
   :returns: Sparse vector where -1 separates rows

   **Usage in real-time decoding:**

   .. code-block:: cpp

      config.H_sparse = cudaq::qec::pcm_to_sparse_vec(dem.detector_error_matrix);
      config.O_sparse = cudaq::qec::pcm_to_sparse_vec(dem.observables_flips_matrix);

.. function:: cudaqx::tensor<uint8_t> cudaq::qec::pcm_from_sparse_vec(const std::vector<std::int64_t>& sparse_vec, std::size_t num_rows, std::size_t num_cols)

   Convert sparse vector representation back to a dense parity check matrix.

   :param sparse_vec: Sparse representation (from YAML or decoder config)
   :param num_rows: Number of rows in the output matrix
   :param num_cols: Number of columns in the output matrix
   :returns: Dense binary matrix

.. function:: std::vector<std::int64_t> cudaq::qec::generate_timelike_sparse_detector_matrix(std::uint32_t num_syndromes_per_round, std::uint32_t num_rounds, bool include_first_round = false)

   Generate the D_sparse matrix that encodes how detectors relate across syndrome measurement rounds.

   :param num_syndromes_per_round: Number of syndrome measurements per round (typically code distance squared)
   :param num_rounds: Total number of syndrome measurement rounds
   :param include_first_round: Whether first round syndromes are deterministic (default: false for standard memory experiments)
   :returns: Sparse matrix encoding detector relationships

   **Usage in real-time decoding:**

   .. code-block:: cpp

      config.D_sparse = cudaq::qec::generate_timelike_sparse_detector_matrix(
          numSyndromesPerRound, numRounds, false);

See also :ref:`Parity Check Matrix Utilities <cpp_api:Parity Check Matrix Utilities>` for additional PCM manipulation functions.
