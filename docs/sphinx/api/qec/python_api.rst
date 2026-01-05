CUDA-Q QEC Python API
******************************

.. automodule:: cudaq_qec
    :members:

Code
=============

.. autoclass:: cudaq_qec.Code
    :members:

Detector Error Model
====================

.. autoclass:: cudaq_qec.DetectorErrorModel
    :members:

.. autofunction:: cudaq_qec.dem_from_memory_circuit
.. autofunction:: cudaq_qec.x_dem_from_memory_circuit
.. autofunction:: cudaq_qec.z_dem_from_memory_circuit

Decoder Interfaces
==================

.. autoclass:: cudaq_qec.Decoder
    :members:

.. autoclass:: cudaq_qec.DecoderResult
    :members:

Built-in Decoders
=================

.. _nv_qldpc_decoder_api_python:

NVIDIA QLDPC Decoder
--------------------

.. include:: nv_qldpc_decoder_api.rst

Sliding Window Decoder
----------------------

.. include:: sliding_window_api.rst

.. _trt_decoder_api_python:

TensorRT Decoder
----------------

.. include:: trt_decoder_api.rst

.. _tensor_network_decoder_api_python:

Tensor Network Decoder
----------------------

.. include:: tensor_network_decoder_api.rst

Real-Time Decoding
==================

.. include:: python_realtime_decoding_api.rst


Common
=============

.. autofunction:: cudaq_qec.sample_memory_circuit

.. autofunction:: cudaq_qec.sample_code_capacity

Parity Check Matrix Utilities
=============================

.. autofunction:: cudaq_qec.generate_random_pcm
.. autofunction:: cudaq_qec.generate_timelike_sparse_detector_matrix
.. autofunction:: cudaq_qec.get_pcm_for_rounds
.. autofunction:: cudaq_qec.get_sorted_pcm_column_indices
.. autofunction:: cudaq_qec.pcm_extend_to_n_rounds
.. autofunction:: cudaq_qec.pcm_is_sorted
.. autofunction:: cudaq_qec.pcm_to_sparse_vec
.. autofunction:: cudaq_qec.reorder_pcm_columns
.. autofunction:: cudaq_qec.shuffle_pcm_columns
.. autofunction:: cudaq_qec.simplify_pcm
.. autofunction:: cudaq_qec.sort_pcm_columns
