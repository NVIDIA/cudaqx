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

      nvq++ -lcudaq-qec stim_dem_decoder.cpp -o stim_dem_decoder
      ./stim_dem_decoder
