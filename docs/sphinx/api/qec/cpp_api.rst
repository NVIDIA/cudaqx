CUDA-Q QEC C++ API
******************************

Code
=============

.. doxygenclass:: cudaq::qec::code
    :members:

.. doxygenstruct:: cudaq::qec::patch
    :members:

.. doxygenclass:: cudaq::qec::repetition::repetition
    :members:

.. doxygenclass:: cudaq::qec::steane::steane
    :members:

.. _qec_stabilizer_grid_cpp:

.. doxygenclass:: cudaq::qec::surface_code::stabilizer_grid
    :members:

.. doxygenclass:: cudaq::qec::surface_code::surface_code
    :members:

Detector Error Model
====================

.. doxygenstruct:: cudaq::qec::detector_error_model
    :members:

.. doxygenstruct:: cudaq::qec::decoder_context
    :members:

.. doxygenfunction:: cudaq::qec::dem_from_memory_circuit(const code &, operation, std::size_t, cudaq::noise_model &, bool)
.. doxygenfunction:: cudaq::qec::x_dem_from_memory_circuit(const code &, operation, std::size_t, cudaq::noise_model &, bool)
.. doxygenfunction:: cudaq::qec::z_dem_from_memory_circuit(const code &, operation, std::size_t, cudaq::noise_model &, bool)
.. doxygenfunction:: cudaq::qec::decoder_context_from_memory_circuit(const code &, operation, std::size_t, cudaq::noise_model &, bool)
.. doxygenfunction:: cudaq::qec::dem_from_stim_text(const std::string &, bool)

Decoder Interfaces
==================

.. doxygenclass:: cudaq::qec::decoder
    :members:

.. doxygenstruct:: cudaq::qec::decoder_result
    :members:

Built-in Decoders
=================

.. _nv_qldpc_decoder_api_cpp:

NVIDIA QLDPC Decoder
--------------------

.. include:: nv_qldpc_decoder_api.rst

Sliding Window Decoder
----------------------

.. include:: sliding_window_api.rst

.. _trt_decoder_api_cpp:

TensorRT Decoder
----------------

.. include:: trt_decoder_api.rst

Real-Time Decoding
==================

.. include:: cpp_realtime_decoding_api.rst

.. include:: realtime_pipeline_api.rst

.. _parity_check_matrix_utilities:

Parity Check Matrix Utilities
=============================

.. doxygenfunction:: cudaq::qec::dense_to_sparse(const cudaqx::tensor<uint8_t> &)
.. doxygenfunction:: cudaq::qec::generate_random_pcm(std::size_t, std::size_t, std::size_t, int, std::mt19937_64 &&);
.. doxygenfunction:: cudaq::qec::generate_timelike_sparse_detector_matrix(std::uint32_t num_syndromes_per_round, std::uint32_t num_rounds, bool include_first_round = false)
.. doxygenfunction:: cudaq::qec::generate_timelike_sparse_detector_matrix(std::uint32_t num_syndromes_per_round, std::uint32_t num_rounds, std::vector<std::int64_t> first_round_matrix)
.. doxygenfunction:: cudaq::qec::get_pcm_for_rounds(const cudaqx::tensor<uint8_t> &, std::uint32_t, std::uint32_t, std::uint32_t, bool, bool, std::uint32_t);
.. doxygenfunction:: cudaq::qec::get_sorted_pcm_column_indices(const std::vector<std::vector<std::uint32_t>> &, std::uint32_t);
.. doxygenfunction:: cudaq::qec::get_sorted_pcm_column_indices(const std::vector<std::vector<std::uint32_t>> &, std::uint32_t, std::uint32_t);
.. doxygenfunction:: cudaq::qec::get_sorted_pcm_column_indices(const cudaqx::tensor<uint8_t> &, std::uint32_t);
.. doxygenfunction:: cudaq::qec::pcm_extend_to_n_rounds(const cudaqx::tensor<uint8_t> &, std::size_t, std::uint32_t);
.. doxygenfunction:: cudaq::qec::pcm_from_sparse_vec(const std::vector<std::int64_t>& sparse_vec, std::size_t num_rows, std::size_t num_cols)
.. doxygenfunction:: cudaq::qec::pcm_is_sorted(const cudaqx::tensor<uint8_t> &, std::uint32_t);
.. doxygenfunction:: cudaq::qec::pcm_is_sorted(const std::vector<std::vector<std::uint32_t>> &, std::uint32_t, std::uint32_t);
.. doxygenfunction:: cudaq::qec::pcm_to_sparse_vec(const cudaqx::tensor<uint8_t>& pcm)
.. doxygenfunction:: cudaq::qec::reorder_pcm_columns(const cudaqx::tensor<uint8_t> &, const std::vector<std::uint32_t> &, uint32_t, uint32_t);
.. doxygenfunction:: cudaq::qec::shuffle_pcm_columns(const cudaqx::tensor<uint8_t> &, std::mt19937_64 &&);
.. doxygenfunction:: cudaq::qec::simplify_pcm(const cudaqx::tensor<uint8_t> &, const std::vector<double> &, std::uint32_t);
.. doxygenfunction:: cudaq::qec::sort_pcm_columns(const cudaqx::tensor<uint8_t> &, std::uint32_t);

Logger
=============

The QEC logger API currently lives in ``cudaq::qec::detail`` and is used by
the ``CUDA_QEC_*`` macros exposed in ``cudaq/qec/logger.h``.

.. doxygenenum:: cudaq::qec::detail::log_level
.. doxygenenum:: cudaq::qec::detail::forward_drop_policy

.. doxygenstruct:: cudaq::qec::detail::forwarded_log_record
    :members:

.. doxygenstruct:: cudaq::qec::detail::forwarder_config
    :members:

.. doxygenstruct:: cudaq::qec::detail::forwarder_stats
    :members:

.. doxygenfunction:: cudaq::qec::detail::should_log
.. doxygenfunction:: cudaq::qec::detail::set_forwarder(forwarder_config)
.. doxygenfunction:: cudaq::qec::detail::set_forwarder()
.. doxygenfunction:: cudaq::qec::detail::clear_forwarder()
.. doxygenfunction:: cudaq::qec::detail::is_forwarder_enabled()
.. doxygenfunction:: cudaq::qec::detail::get_forwarder_message_capacity()
.. doxygenfunction:: cudaq::qec::detail::get_forwarder_stats()
.. doxygenfunction:: cudaq::qec::detail::set_log_level
.. doxygenfunction:: cudaq::qec::detail::get_log_level
.. doxygenfunction:: cudaq::qec::detail::flush_logs()
.. doxygenfunction:: cudaq::qec::detail::trace
.. doxygenfunction:: cudaq::qec::detail::info
.. doxygenfunction:: cudaq::qec::detail::debug
.. doxygenfunction:: cudaq::qec::detail::warn
.. doxygenfunction:: cudaq::qec::detail::error
.. doxygenfunction:: cudaq::qec::detail::path_to_file_name

Common
=============

.. doxygentypedef:: cudaq::qec::float_t

.. doxygenenum:: cudaq::qec::operation

.. doxygenfunction:: cudaq::qec::sample_code_capacity(const cudaqx::tensor<uint8_t> &, std::size_t, double)
.. doxygenfunction:: cudaq::qec::sample_code_capacity(const cudaqx::tensor<uint8_t> &, std::size_t, double, unsigned)
.. doxygenfunction:: cudaq::qec::sample_code_capacity(const code &, std::size_t, double)
.. doxygenfunction:: cudaq::qec::sample_code_capacity(const code &, std::size_t, double, unsigned)

.. doxygenfunction:: cudaq::qec::sample_memory_circuit(const code &, std::size_t, std::size_t)
.. doxygenfunction:: cudaq::qec::sample_memory_circuit(const code &, std::size_t, std::size_t, cudaq::noise_model &)
.. doxygenfunction:: cudaq::qec::sample_memory_circuit(const code &, operation, std::size_t, std::size_t)
.. doxygenfunction:: cudaq::qec::sample_memory_circuit(const code &, operation, std::size_t, std::size_t, cudaq::noise_model &)
.. doxygenfunction:: cudaq::qec::x_sample_memory_circuit
.. doxygenfunction:: cudaq::qec::z_sample_memory_circuit
