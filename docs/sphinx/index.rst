CUDA-QX - The CUDA-Q Libraries Collection
==========================================

CUDA-QX is a collection of libraries that build upon the CUDA-Q programming model
to enable the rapid development of hybrid quantum-classical application code leveraging
state-of-the-art CPUs, GPUs, and QPUs. It provides a collection of C++
libraries and Python packages that enable research, development, and application
creation for use cases in quantum error correction and hybrid quantum-classical
solvers.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   quickstart/installation

.. toctree::
   :maxdepth: 1
   :caption: Libraries

   components/qec/introduction
   components/solvers/introduction

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples_rst/qec/examples
   examples_rst/solvers/examples

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

CUDA-QX is composed of two distinct libraries that build upon CUDA-Q programming model.
The libraries provided are cudaq-qec, a library enabling performant research workflows
for quantum error correction, and cudaq-solvers, a library that provides high-level
APIs for common quantum-classical solver workflows.

* **cudaq-qec**: Quantum Error Correction Library
    * Extensible framework describing quantum error correcting codes as a collection of CUDA-Q kernels.
    * Extensible framework for describing syndrome decoders
    * State-of-the-art, performant decoder implementations on NVIDIA GPUs (coming soon)
    * Pre-built numerical experiment APIs

* **cudaq-solvers**: Performant Quantum-Classical Simulation Workflows
    * Variational Quantum Eigensolver (VQE)
    * ADAPT-VQE implementation that scales via CUDA-Q MQPU.
    * Quantum Approximate Optimization Algorithm (QAOA)
    * More to come...

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
