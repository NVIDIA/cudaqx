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

- :cpp:func:`cudaq::qec::pcm_to_sparse_vec` for converting a dense PCM to a sparse PCM.
   
   **Usage in real-time decoding:**

   .. code-block:: cpp

      config.H_sparse = cudaq::qec::pcm_to_sparse_vec(dem.detector_error_matrix);
      config.O_sparse = cudaq::qec::pcm_to_sparse_vec(dem.observables_flips_matrix);
- :cpp:func:`cudaq::qec::pcm_from_sparse_vec` for converting a sparse PCM to a dense PCM.
- :cpp:func:`cudaq::qec::generate_timelike_sparse_detector_matrix` for generating a sparse detector matrix.

   **Usage in real-time decoding:**

   .. code-block:: cpp

      config.D_sparse = cudaq::qec::generate_timelike_sparse_detector_matrix(
          numSyndromesPerRound, numRounds, false);

See also :ref:`parity_check_matrix_utilities` for additional PCM manipulation functions.
