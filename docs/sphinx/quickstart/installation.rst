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

    # Install Solvers library
    pip install cudaq-solvers

    # Install both libraries
    pip install cudaq-qec cudaq-solvers

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

    If your system does not have local GPUs (eg. a MacBook), omit the `--gpus all`
    argument.

The container includes:
    * CUDA-Q compiler and runtime
    * CUDA-QX libraries (QEC and Solvers)
    * All required dependencies
    * Example notebooks and tutorials

Building from Source
^^^^^^^^^^^^^^^^^^^^

The instructions for building CUDA-QX from source are maintained on our GitHub
repository: `Building CUDA-QX from Source <https://github.com/NVIDIA/cudaqx/blob/main/Building.md>`__.

Known Blackwell Issues
----------------------
.. note::
    If you are attempting to use torch on Blackwell, you will need to install the nightly version of torch.
    You can do this by running:

    .. code-block:: bash

        python3 -m pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

    torch is a dependency of the tensor network decoder and the GQE algorithm.