Realtime Decoding
==================

Realtime decoding runs CUDA-Q QEC decoders concurrently with quantum execution, applying corrections within qubit coherence times. For the concepts, the four-stage workflow, and terminology, see :doc:`Realtime Decoding </components/qec/realtime_decoding>`.

The examples below cover realtime decoding end to end — start with Getting Started, then explore the specialized predecoding and decoding workloads:

.. toctree::
   :maxdepth: 2

   Getting Started with Realtime Decoding <getting_started_realtime_decoding>
   AI Predecoder with CUDA-Q Realtime <ai_predecoder>
   AI Predecoder with CUDA-Q Realtime (with FPGA Data Injection) <ai_predecoder_fpga>
   Relay BP Decoding with CUDA-Q Realtime <realtime_relay_bp>

See Also
--------

* Example source code: `libs/qec/unittests/realtime/app_examples <https://github.com/NVIDIA/cudaqx/tree/releases/v0.7.0/libs/qec/unittests/realtime/app_examples>`_
* :doc:`/api/qec/cpp_api` — C++ API Reference (includes Realtime Decoding)
* :doc:`/api/qec/python_api` — Python API Reference (includes Realtime Decoding)
