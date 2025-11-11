.. _python_realtime_decoding_api:


The Real-Time Decoding API enables low-latency error correction on quantum hardware by allowing CUDA-Q quantum kernels to interact with decoders during circuit execution. This API is designed for use cases where corrections must be calculated and applied within qubit coherence times.

The real-time decoding system supports both hardware integration (e.g., Quantinuum H-Series) and simulation environments for local testing.

Core Decoding Functions
------------------------

These functions can be called from within CUDA-Q quantum kernels (``@cudaq.kernel`` decorated functions) to interact with real-time decoders.

.. function:: cudaq_qec.qec.enqueue_syndromes(decoder_id, syndromes, tag=0)

   Enqueue syndrome measurements for decoding.

   :param decoder_id: Unique identifier for the decoder instance (matches configured decoder ID)
   :param syndromes: List of syndrome measurement results from stabilizer measurements
   :param tag: Optional tag for logging and debugging (default: 0)

   **Example:**

   .. code-block:: python

      import cudaq
      import cudaq_qec as qec
      from cudaq_qec import patch

      @cudaq.kernel
      def measure_and_decode(logical: patch, decoder_id: int):
          syndromes = measure_stabilizers(logical)
          qec.enqueue_syndromes(decoder_id, syndromes, 0)

.. function:: cudaq_qec.qec.get_corrections(decoder_id, return_size, reset=False)

   Retrieve calculated corrections from the decoder.

   :param decoder_id: Unique identifier for the decoder instance
   :param return_size: Number of correction bits to return (typically equals number of logical observables)
   :param reset: Whether to reset accumulated corrections after retrieval (default: False)
   :returns: List of boolean values indicating detected bit flips for each logical observable

   **Example:**

   .. code-block:: python

      @cudaq.kernel
      def apply_corrections(logical: patch, decoder_id: int):
          corrections = qec.get_corrections(decoder_id, 1, False)
          if corrections[0]:
              x(logical.data)  # Apply transversal X correction

.. function:: cudaq_qec.qec.reset_decoder(decoder_id)

   Reset decoder state, clearing all queued syndromes and accumulated corrections.

   :param decoder_id: Unique identifier for the decoder instance to reset

   **Example:**

   .. code-block:: python

      @cudaq.kernel
      def run_experiment(decoder_id: int):
          qec.reset_decoder(decoder_id)  # Reset at start of each shot
          # ... perform experiment ...

Configuration API
-----------------

The configuration API enables setting up decoders before circuit execution. Decoders are configured using YAML files or programmatically constructed configuration objects.

.. function:: cudaq_qec.configure_decoders(config)

   Configure decoders from a multi_decoder_config object.

   :param config: multi_decoder_config object containing decoder specifications
   :returns: 0 on success, non-zero error code on failure

.. function:: cudaq_qec.configure_decoders_from_file(config_file)

   Configure decoders from a YAML file.

   :param config_file: Path to YAML configuration file
   :returns: 0 on success, non-zero error code on failure

.. function:: cudaq_qec.configure_decoders_from_str(config_str)

   Configure decoders from a YAML string.

   :param config_str: YAML configuration as a string
   :returns: 0 on success, non-zero error code on failure

.. function:: cudaq_qec.finalize_decoders()

   Finalize and clean up decoder resources. Should be called before program exit.

Helper Functions
----------------

Real-time decoding requires converting matrices to sparse format for efficient decoder configuration. The following utility functions are essential:

.. function:: cudaq_qec.pcm_to_sparse_vec(pcm)

   Convert a parity check matrix (PCM) to sparse vector representation for decoder configuration.

   :param pcm: Dense binary matrix as numpy array (e.g., ``dem.detector_error_matrix`` or ``dem.observables_flips_matrix``)
   :returns: Sparse vector (list of integers) where -1 separates rows

   **Usage in real-time decoding:**

   .. code-block:: python

      config.H_sparse = qec.pcm_to_sparse_vec(dem.detector_error_matrix)
      config.O_sparse = qec.pcm_to_sparse_vec(dem.observables_flips_matrix)

.. function:: cudaq_qec.pcm_from_sparse_vec(sparse_vec, num_rows, num_cols)

   Convert sparse vector representation back to a dense parity check matrix.

   :param sparse_vec: Sparse representation (from YAML or decoder config)
   :param num_rows: Number of rows in the output matrix
   :param num_cols: Number of columns in the output matrix
   :returns: Dense binary matrix as numpy array

.. function:: cudaq_qec.generate_timelike_sparse_detector_matrix(num_syndromes_per_round, num_rounds, include_first_round)

   Generate the D_sparse matrix that encodes how detectors relate across syndrome measurement rounds.

   :param num_syndromes_per_round: Number of syndrome measurements per round (typically code distance squared)
   :param num_rounds: Total number of syndrome measurement rounds
   :param include_first_round: Boolean (False for standard memory experiments) or list for custom first round
   :returns: Sparse matrix encoding detector relationships

   **Usage in real-time decoding:**

   .. code-block:: python

      config.D_sparse = qec.generate_timelike_sparse_detector_matrix(
          numSyndromesPerRound, numRounds, False)

See also :ref:`Parity Check Matrix Utilities <python_api:Parity Check Matrix Utilities>` for additional PCM manipulation functions.
