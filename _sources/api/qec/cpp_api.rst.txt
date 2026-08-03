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

.. doxygenclass:: cudaq::qec::surface_code::stabilizer_grid
    :members:

.. doxygenclass:: cudaq::qec::surface_code::surface_code
    :members:

Detector Error Model
====================

.. doxygenstruct:: cudaq::qec::detector_error_model
    :members:

.. doxygenfunction:: cudaq::qec::dem_from_memory_circuit(const code &, operation, std::size_t, cudaq::noise_model &)
.. doxygenfunction:: cudaq::qec::x_dem_from_memory_circuit(const code &, operation, std::size_t, cudaq::noise_model &)
.. doxygenfunction:: cudaq::qec::z_dem_from_memory_circuit(const code &, operation, std::size_t, cudaq::noise_model &)

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

.. _parity_check_matrix_utilities:

Parity Check Matrix Utilities
=============================

.. doxygenfunction:: cudaq::qec::dense_to_sparse(const cudaqx::tensor<uint8_t> &)
.. doxygenfunction:: cudaq::qec::generate_random_pcm(std::size_t, std::size_t, std::size_t, int, std::mt19937_64 &&);
.. doxygenfunction:: cudaq::qec::generate_timelike_sparse_detector_matrix(std::uint32_t num_syndromes_per_round, std::uint32_t num_rounds, bool include_first_round = false)
.. doxygenfunction:: cudaq::qec::generate_timelike_sparse_detector_matrix(std::uint32_t num_syndromes_per_round, std::uint32_t num_rounds, std::vector<std::int64_t> first_round_matrix)
.. doxygenfunction:: cudaq::qec::get_pcm_for_rounds(const cudaqx::tensor<uint8_t> &, std::uint32_t, std::uint32_t, std::uint32_t, bool, bool);
.. doxygenfunction:: cudaq::qec::get_sorted_pcm_column_indices(const std::vector<std::vector<std::uint32_t>> &, std::uint32_t);
.. doxygenfunction:: cudaq::qec::get_sorted_pcm_column_indices(const cudaqx::tensor<uint8_t> &, std::uint32_t);
.. doxygenfunction:: cudaq::qec::pcm_extend_to_n_rounds(const cudaqx::tensor<uint8_t> &, std::size_t, std::uint32_t);
.. doxygenfunction:: cudaq::qec::pcm_from_sparse_vec(const std::vector<std::int64_t>& sparse_vec, std::size_t num_rows, std::size_t num_cols)
.. doxygenfunction:: cudaq::qec::pcm_is_sorted(const cudaqx::tensor<uint8_t> &, std::uint32_t);
.. doxygenfunction:: cudaq::qec::pcm_to_sparse_vec(const cudaqx::tensor<uint8_t>& pcm)
.. doxygenfunction:: cudaq::qec::reorder_pcm_columns(const cudaqx::tensor<uint8_t> &, const std::vector<std::uint32_t> &, uint32_t, uint32_t);
.. doxygenfunction:: cudaq::qec::shuffle_pcm_columns(const cudaqx::tensor<uint8_t> &, std::mt19937_64 &&);
.. doxygenfunction:: cudaq::qec::simplify_pcm(const cudaqx::tensor<uint8_t> &, const std::vector<double> &, std::uint32_t);
.. doxygenfunction:: cudaq::qec::sort_pcm_columns(const cudaqx::tensor<uint8_t> &, std::uint32_t);

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
