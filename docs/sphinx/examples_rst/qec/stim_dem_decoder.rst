Decoding From Stim DEM Text
---------------------------

CUDA-Q QEC decoders can be constructed from either a parity-check matrix or raw
Stim detector error model (DEM) text. Passing the DEM text is useful when the
model is already available in Stim's ``.dem`` format, such as from a saved file,
Stim workflow, or CUDA-Q DEM generation.

For PCM-based decoders, CUDA-Q QEC parses the DEM text into a detector error
matrix and supplies DEM-derived ``O`` and ``error_rate_vec`` defaults when the
user does not provide them. C++ decoder plugins that need full Stim DEM
metadata can consume the raw DEM string from the decoder construction input.

By default, ``get_decoder(..., dem_text)`` and ``dem_from_stim_text(dem_text)``
parse with ``use_decomp_suggestions=False``. Stim ``^`` decomposition hints are
ignored and each ``error(...)`` instruction becomes one matrix column. The
example below constructs a decoder this way and uses the matching parsed matrix
for observable predictions.

``dem_from_stim_text`` also accepts ``use_decomp_suggestions=True`` to split
``^``-separated components into separate columns. That call is shown for
inspection only; it does not change how ``get_decoder`` parses the same DEM
text string.

.. tab:: Python

   .. literalinclude:: ../../examples/qec/python/stim_dem_decoder.py
      :language: python
      :start-after: [Begin Documentation]

.. tab:: C++

   .. literalinclude:: ../../examples/qec/cpp/stim_dem_decoder.cpp
      :language: cpp
      :start-after: [Begin Documentation]

   Compile and run with

   .. code-block:: bash

      nvq++ -lcudaq-qec -lcudaq-qec-decoders stim_dem_decoder.cpp -o stim_dem_decoder
      ./stim_dem_decoder
