CUDA-Q QEC Python API
******************************

.. automodule:: cudaq_qec

Code
=============

.. autoclass:: cudaq_qec.Code
    :members:

Surface code layout
===================

.. _qec_stabilizer_grid_python:

The rotated surface code exposes a grid helper for stabilizer and data-qubit
indexing. In Python it is available as :class:`cudaq_qec.stabilizer_grid` (call
``cudaq_qec.stabilizer_grid(distance)``). The C++ type is
:cpp:class:`cudaq::qec::surface_code::stabilizer_grid` (:ref:`API <qec_stabilizer_grid_cpp>`).

.. autoclass:: cudaq_qec.stabilizer_grid
    :members:

Detector Error Model
====================

.. autoclass:: cudaq_qec.DetectorErrorModel
    :members:

.. autoclass:: cudaq_qec.DecoderContext
    :members:

.. note::

   The ``x_component()``, ``z_component()``, and ``full_component()`` methods each
   return a ``(dem, m2d, m2o)`` tuple:

   - ``dem`` (:class:`DetectorErrorModel`) — canonicalized detector error model
   - ``m2d`` (``list[list[int]]``) — measurement-to-detector map; ``m2d[d]`` lists
     the measurement indices whose XOR forms detector ``d``
   - ``m2o`` (``list[list[int]]``) — measurement-to-observable map

   Pass ``m2d`` to :func:`d_sparse` to produce the ``D_sparse`` vector for a
   real-time decoder config.

.. autofunction:: cudaq_qec.dem_from_memory_circuit
.. autofunction:: cudaq_qec.x_dem_from_memory_circuit
.. autofunction:: cudaq_qec.z_dem_from_memory_circuit
.. autofunction:: cudaq_qec.decoder_context_from_memory_circuit
.. autofunction:: cudaq_qec.dem_from_stim_text
.. autofunction:: cudaq_qec.d_sparse

Decoder Interfaces
==================

.. autoclass:: cudaq_qec.Decoder
    :members:

.. autoclass:: cudaq_qec.DecoderResult
    :members:

.. autoclass:: cudaq_qec.BatchDecoderResult
    :members:

.. autoclass:: cudaq_qec.AsyncDecoderResult
    :members:

.. autofunction:: cudaq_qec.get_decoder

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
.. autofunction:: cudaq_qec.x_sample_memory_circuit
.. autofunction:: cudaq_qec.z_sample_memory_circuit

.. autofunction:: cudaq_qec.sample_code_capacity

.. _dem_sampling_python_api:

Detector Error Model (DEM) Sampling
===================================

.. autofunction:: cudaq_qec.dem_sampling

.. _parity_check_matrix_utilities_python:

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
