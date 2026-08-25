CUDA-QX - The CUDA-Q Libraries Collection
==========================================

CUDA-QX is a collection of libraries that build upon the CUDA-Q programming model
to enable the rapid development of hybrid quantum-classical application code leveraging
state-of-the-art CPUs, GPUs, and QPUs. It provides C++ libraries and Python
packages that enable research, development, and application creation for use
cases in quantum error correction.

.. note::

   **Looking for CUDA-Q Solvers?** The CUDA-Q Solvers library has been removed
   from CUDA-QX; version 0.6.0 was its final planned release. Development
   continues in **CUDA-Q Algorithms**, which supersedes CUDA-Q Solvers and
   expands on it. Install it with :code:`pip install cudaq-algorithms`
   (`cudaq-algorithms on PyPI <https://pypi.org/project/cudaq-algorithms/>`__),
   read the `CUDA-Q Algorithms documentation
   <https://nvidia.github.io/cudaq-algorithms/>`__, and find the source code at
   `NVIDIA/cudaq-algorithms on GitHub <https://github.com/NVIDIA/cudaq-algorithms>`__.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   quickstart/installation

.. toctree::
   :maxdepth: 1
   :caption: Libraries

   components/qec/index

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples_rst/qec/examples

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

Key Features
-------------

CUDA-QX provides cudaq-qec, a library enabling performant research workflows for
quantum error correction, built upon the CUDA-Q programming model.

* **cudaq-qec**: Quantum Error Correction Library
    * Extensible framework describing quantum error correcting codes as a collection of CUDA-Q kernels.
    * Extensible framework for describing syndrome decoders
    * State-of-the-art, performant decoder implementations on NVIDIA GPUs
    * Real-time decoding for active error correction on quantum hardware
    * Pre-built numerical experiment APIs

Indices
-------

* :ref:`genindex`
* :ref:`search`
