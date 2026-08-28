CUDA-QX - The CUDA-Q Libraries Collection
==========================================

CUDA-QX is a collection of libraries that build upon the CUDA-Q programming model
to enable the rapid development of hybrid quantum-classical application code leveraging
state-of-the-art CPUs, GPUs, and QPUs. It provides a collection of C++
libraries and Python packages that enable research, development, and application
creation for use cases in quantum error correction and hybrid quantum-classical
solvers.

.. note::

   **CUDA-Q QEC is actively developed and fully supported.** The deprecation
   notice below applies *only* to the CUDA-Q Solvers library; it does not
   affect CUDA-Q QEC, which continues to receive new features, performance
   improvements, and releases.

.. attention::

   **CUDA-Q Solvers is deprecated (this does not affect CUDA-Q QEC).**
   Version 0.6.0 is the final planned
   release of the CUDA-Q Solvers library. Development continues in **CUDA-Q
   Algorithms**, which supersedes CUDA-Q Solvers and expands on it. Install it
   with :code:`pip install cudaq-algorithms`
   (`cudaq-algorithms on PyPI <https://pypi.org/project/cudaq-algorithms/>`__),
   read the `CUDA-Q Algorithms documentation
   <https://nvidia.github.io/cudaq-algorithms/>`__, and find the source code at
   `NVIDIA/cudaq-algorithms on GitHub <https://github.com/NVIDIA/cudaq-algorithms>`__.
   The CUDA-Q Solvers documentation below is retained for existing 0.6.0 users;
   all new features and fixes land in CUDA-Q Algorithms.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   quickstart/installation

.. toctree::
   :maxdepth: 1
   :caption: Libraries

   components/qec/index
   components/solvers/introduction

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples_rst/qec/examples
   examples_rst/solvers/examples

.. toctree::
   :maxdepth: 1
   :caption: Performance Studies

   performance/index

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/core/cpp_api
   api/qec/cpp_api
   api/qec/python_api
   api/solvers/cpp_api
   api/solvers/python_api

Key Features
-------------

CUDA-QX is composed of two distinct libraries that build upon the CUDA-Q programming model.
The libraries provided are cudaq-qec, a library enabling performant research workflows
for quantum error correction, and cudaq-solvers, a library that provides high-level
APIs for common quantum-classical solver workflows.

* **cudaq-qec** (actively developed and supported): Quantum Error Correction Library
    * Extensible framework describing quantum error correcting codes as a collection of CUDA-Q kernels.
    * Extensible framework for describing syndrome decoders
    * State-of-the-art, performant decoder implementations on NVIDIA GPUs
    * Real-time decoding for active error correction on quantum hardware
    * Pre-built numerical experiment APIs

* **cudaq-solvers** (deprecated, superseded by `CUDA-Q Algorithms <https://nvidia.github.io/cudaq-algorithms/>`__): Performant Quantum-Classical Simulation Workflows
    * Variational Quantum Eigensolver (VQE)
    * ADAPT-VQE implementation that scales via CUDA-Q MQPU.
    * Quantum Approximate Optimization Algorithm (QAOA)
    * Version 0.6.0 is the final planned release; continue with
      `cudaq-algorithms <https://pypi.org/project/cudaq-algorithms/>`__ and its
      `documentation <https://nvidia.github.io/cudaq-algorithms/>`__.

Indices
-------

* :ref:`genindex`
* :ref:`search`
