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

.. _realtime_decoding_api_cpp:

Real-Time Decoding
------------------

.. include:: realtime_decoding_api.rst

Parity Check Matrix Utilities
=============================

.. doxygenfunction:: cudaq::qec::dense_to_sparse(const cudaqx::tensor<uint8_t> &)
.. doxygenfunction:: cudaq::qec::generate_random_pcm(std::size_t, std::size_t, std::size_t, int, std::mt19937_64 &&);
.. doxygenfunction:: cudaq::qec::get_pcm_for_rounds(const cudaqx::tensor<uint8_t> &, std::uint32_t, std::uint32_t, std::uint32_t, bool, bool);
.. doxygenfunction:: cudaq::qec::get_sorted_pcm_column_indices(const std::vector<std::vector<std::uint32_t>> &, std::uint32_t);
.. doxygenfunction:: cudaq::qec::get_sorted_pcm_column_indices(const cudaqx::tensor<uint8_t> &, std::uint32_t);
.. doxygenfunction:: cudaq::qec::pcm_extend_to_n_rounds(const cudaqx::tensor<uint8_t> &, std::size_t, std::uint32_t);
.. doxygenfunction:: cudaq::qec::pcm_is_sorted(const cudaqx::tensor<uint8_t> &, std::uint32_t);
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
