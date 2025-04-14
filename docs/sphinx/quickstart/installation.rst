Installation Guide
==================

Installation Methods
--------------------

CUDA-QX provides multiple installation methods to suit your needs:

pip install
^^^^^^^^^^^^

The simplest way to install CUDA-QX is via pip. You can install individual components:

.. code-block:: bash

    # Install QEC library
    pip install cudaq-qec

    # Install Solvers library
    pip install cudaq-solvers

    # Install both libraries
    pip install cudaq-qec cudaq-solvers

.. note:: 

    CUDA-Q Solvers will require the presence of :code:`libgfortran`, which is
    not distributed with the Python wheel, for provided classical optimizers. If
    :code:`libgfortran` is not installed, you will need to install it via your
    distribution's package manager. On Debian based systems, you can install
    this with :code:`apt-get install gfortran`. On MacOS, you can install it by
    running :code:`brew install gcc`.

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

Prerequisites
~~~~~~~~~~~~~

Before building CUDA-QX from source, ensure your system meets the following requirements:

* **CUDA-Q**: The NVIDIA quantum-classical programming model
* **CMake**: Version 3.28 or higher (``pip install "cmake<4"``), less than 4.0
* **GCC**: Version 11 or higher
* **Python**: Version 3.10, 3.11, or 3.12
* **NVIDIA GPU**: CUDA-capable GPU with compute capability 12.0 or higher
* **Git**: For cloning the repository

Build Instructions
~~~~~~~~~~~~~~~~~~~

1. Clone the repository:

.. code-block:: bash

    git clone https://github.com/nvidia/cudaqx
    cd cudaqx

2. Create and enter build directory:

.. code-block:: bash

    mkdir build && cd build

3. Configure with CMake:

.. code-block:: bash

    cmake .. -G Ninja \
        -DCUDAQX_ENABLE_LIBS="all" \
        -DCUDAQX_INCLUDE_TESTS=ON \
        -DCUDAQX_BINDINGS_PYTHON=ON \
        -DCUDAQ_DIR=$HOME/.cudaq/lib/cmake/cudaq \
        -DCMAKE_CXX_FLAGS="-Wno-attributes" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=$HOME/.cudaqx

4. Build and install:

.. code-block:: bash

    ninja install

CMake Build Options
~~~~~~~~~~~~~~~~~~~~

* ``CUDAQX_ENABLE_LIBS``: Specify which libraries to build (``all``, ``qec``, ``solvers``)
* ``CUDAQX_INCLUDE_TESTS``: Enable building of tests
* ``CUDAQX_BINDINGS_PYTHON``: Enable Python bindings
* ``CUDAQ_DIR``: Path to CUDA-Q installation
* ``CMAKE_INSTALL_PREFIX``: Installation directory

Verifying Installation
-----------------------

To verify your installation, run the following Python code:

.. code-block:: python

    import cudaq_qec as qec 
    import cudaq_solvers as solvers


Troubleshooting (Common Issues)
--------------------------------

1. **CMake configuration fails**:
    * Ensure CUDA-Q is properly installed
    * Verify CMake version (``cmake --version``)
    * Check GCC version (``gcc --version``)

2. **CUDA device not found**:
    * Verify NVIDIA driver installation
    * Check CUDA toolkit installation
    * Ensure GPU compute capability is supported

3. **Python bindings not found**:
    * Confirm ``CUDAQX_BINDINGS_PYTHON=ON`` during build
    * Check Python environment activation
    * Verify installation path is in ``PYTHONPATH``

For additional support, please visit our `GitHub Issues <https://github.com/nvidia/cudaqx/issues>`_ page.
