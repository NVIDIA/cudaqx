.. _cpp_realtime_decoding_api:


The Real-Time Decoding API enables low-latency error correction on quantum hardware by allowing CUDA-Q quantum kernels to interact with decoders during circuit execution. This API is designed for use cases where corrections must be calculated and applied within qubit coherence times.

The real-time decoding system supports both hardware integration (e.g., Quantinuum H-Series) and simulation environments for local testing.

Core Decoding Functions
------------------------

These functions can be called from within CUDA-Q quantum kernels (``__qpu__`` functions) to interact with real-time decoders.

.. doxygenfunction:: cudaq::qec::decoding::enqueue_syndromes
.. doxygenfunction:: cudaq::qec::decoding::get_corrections
.. doxygenfunction:: cudaq::qec::decoding::reset_decoder


Configuration API
-----------------

The configuration API enables setting up decoders before circuit execution. Decoders are configured using YAML files or programmatically constructed configuration objects.

.. doxygenfunction:: cudaq::qec::decoding::config::configure_decoders
.. doxygenfunction:: cudaq::qec::decoding::config::configure_decoders_from_file
.. doxygenfunction:: cudaq::qec::decoding::config::configure_decoders_from_str
.. doxygenfunction:: cudaq::qec::decoding::config::finalize_decoders

Helper Functions
----------------

Real-time decoding requires converting matrices to sparse format for efficient decoder configuration. The following utility functions are essential:

.. cpp:function:: std::vector<std::int64_t> cudaq::qec::pcm_to_sparse_vec(const cudaqx::tensor<uint8_t>& pcm)

   Convert a parity check matrix (PCM) to sparse vector representation for decoder configuration.

   :param pcm: Dense binary matrix (e.g., ``dem.detector_error_matrix`` or ``dem.observables_flips_matrix``)
   :returns: Sparse vector where -1 separates rows

   **Usage in real-time decoding:**

   .. code-block:: cpp

      config.H_sparse = cudaq::qec::pcm_to_sparse_vec(dem.detector_error_matrix);
      config.O_sparse = cudaq::qec::pcm_to_sparse_vec(dem.observables_flips_matrix);

.. cpp:function:: cudaqx::tensor<uint8_t> cudaq::qec::pcm_from_sparse_vec(const std::vector<std::int64_t>& sparse_vec, std::size_t num_rows, std::size_t num_cols)

   Convert sparse vector representation back to a dense parity check matrix.

   :param sparse_vec: Sparse representation (from YAML or decoder config)
   :param num_rows: Number of rows in the output matrix
   :param num_cols: Number of columns in the output matrix
   :returns: Dense binary matrix

.. cpp:function:: std::vector<std::int64_t> cudaq::qec::generate_timelike_sparse_detector_matrix(std::uint32_t num_syndromes_per_round, std::uint32_t num_rounds, bool include_first_round = false)

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
