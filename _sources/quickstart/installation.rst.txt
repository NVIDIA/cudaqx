Installation Guide
==================

Installation Methods
--------------------

CUDA-QX provides multiple installation methods to suit your needs:

pip install
^^^^^^^^^^^^

The simplest way to install CUDA-QX is via pip. (If you're on Mac, your only
option is to use the Docker container as described below.) For pip, you can
install individual components:

.. code-block:: bash

    # Install QEC library
    pip install cudaq-qec

    # Install Solvers library (deprecated - see the note below)
    pip install cudaq-solvers

    # Install both libraries
    pip install cudaq-qec cudaq-solvers

.. attention::

   **CUDA-Q Solvers is deprecated.** Version 0.6.0 is the final planned
   release of the CUDA-Q Solvers library. New users should install **CUDA-Q
   Algorithms** instead, which supersedes CUDA-Q Solvers and is where
   development continues:

   .. code-block:: bash

       pip install cudaq-algorithms

   See the `CUDA-Q Algorithms documentation
   <https://nvidia.github.io/cudaq-algorithms/>`__ for installation
   instructions, tutorials, and examples, `cudaq-algorithms on PyPI
   <https://pypi.org/project/cudaq-algorithms/>`__ for the package, and
   `NVIDIA/cudaq-algorithms on GitHub
   <https://github.com/NVIDIA/cudaq-algorithms>`__ for the source code.

CUDA-QX provides optional pip-installable components:

.. code-block:: bash

    # Install the Tensor Network Decoder from the QEC library
    pip install cudaq-qec[tensor-network-decoder]

    # Install the GQE algorithm from the Solvers library
    pip install cudaq-solvers[gqe]

.. note::

    CUDA-Q Solvers will require the presence of :code:`libgfortran`, which is
    not distributed with the Python wheel, for provided classical optimizers. If
    :code:`libgfortran` is not installed, you will need to install it via your
    distribution's package manager. On Debian based systems, you can install
    this with :code:`apt-get install gfortran`.

Docker Container
^^^^^^^^^^^^^^^^

CUDA-QX is available as a Docker container with all dependencies pre-installed:

1. Pull the container:

.. code-block:: bash

    docker pull ghcr.io/nvidia/cudaqx

2. Run the container:

.. code-block:: bash

    docker run --gpus all -it ghcr.io/nvidia/cudaqx

.. note::

    If your system does not have local GPUs (e.g., a MacBook), omit the `--gpus all`
    argument.

The container includes:
    * CUDA-Q compiler and runtime
    * CUDA-QX libraries (QEC and Solvers; note that Solvers is deprecated in
      favor of `CUDA-Q Algorithms <https://nvidia.github.io/cudaq-algorithms/>`__)
    * All required dependencies
    * Example notebooks and tutorials

Building from Source
^^^^^^^^^^^^^^^^^^^^

The instructions for building CUDA-QX from source are maintained on our GitHub
repository: `Building CUDA-QX from Source <https://github.com/NVIDIA/cudaqx/blob/main/Building.md>`__.

.. _installing-pytorch:

Installing PyTorch
------------------

PyTorch (``torch``) is required for several CUDA-QX features:

* **Tensor Network Decoder**: Used by the QEC library for tensor network-based decoding (CPU version of PyTorch is sufficient)
* **GQE Algorithm**: Used by the Solvers library for the Generative Quantum Eigensolver
* **Training AI Decoders**: Optionally used for training custom neural network decoders (see :ref:`Deploying AI Decoders with TensorRT <deploying-ai-decoders>`)

PyTorch is automatically installed when you install the optional components:

.. code-block:: bash

    # Installs PyTorch as a dependency
    pip install cudaq-qec[tensor-network-decoder]
    pip install cudaq-solvers[gqe]

Alternatively, you can install PyTorch directly. For detailed installation instructions, visit the 
`PyTorch installation page <https://pytorch.org/get-started/locally/>`_.

.. code-block:: bash

    pip install torch

.. note::
    Users with NVIDIA Blackwell architecture GPUs require PyTorch with CUDA 12.8 or later support. 
    When installing PyTorch, make sure to select the appropriate CUDA version for your system.