# Building CUDA-QX from Source

This document is intended for anyone who wants to develop their own
modifications of, or contributions to, this code base. This document may change
over time, so be sure to always refer to the latest version of this document.

Using the latest version of CUDA-QX often requires using a recent version of
CUDA-Q. The instructions below refer to a public dev container that is made
available on this repository. It will always contain a recent version of CUDA-Q
(currently updated approximately weekly).

The instructions below provide a complete set of commands to get you up and
running. There are images available called

- `ghcr.io/nvidia/cudaqx-dev:latest-amd64-cu12.6` for AMD64 platforms with CUDA >= 12.6, < 13
- `ghcr.io/nvidia/cudaqx-dev:latest-amd64-cu13.0` for AMD64 platforms with CUDA >= 13.0
- `ghcr.io/nvidia/cudaqx-dev:latest-arm64-cu12.6` for ARM64 platforms with CUDA >= 12.6, < 13
- `ghcr.io/nvidia/cudaqx-dev:latest-arm64-cu13.0` for ARM64 platforms with CUDA >= 13.0

With the image appropriate for your system, run

```bash
docker pull <image-name>
docker run -it --gpus all --name cudaqx-dev <image-name>
```

If your system does not have local GPUs (eg. a Macbook), omit the `--gpus all`
argument.

Then inside the container...

```bash
# Then inside the container
export CUDAQ_INSTALL_PREFIX=/usr/local/cudaq
export CUDAQX_INSTALL_PREFIX=~/.cudaqx
cd /workspaces

# Get latest source code
git clone https://github.com/NVIDIA/cudaqx.git
cd cudaqx
mkdir build && cd build

# Configure your build (adjust as necessary)
cmake -G Ninja -S .. \
  -DCUDAQ_INSTALL_DIR=$CUDAQ_INSTALL_PREFIX \
  -DCMAKE_INSTALL_PREFIX=${CUDAQX_INSTALL_PREFIX} \
  -DCUDAQ_DIR=${CUDAQ_INSTALL_PREFIX}/lib/cmake/cudaq \
  -DCMAKE_BUILD_TYPE=Release

# Install your build
ninja install

# Perform tests just to prove that it is running
export PYTHONPATH=${CUDAQ_INSTALL_PREFIX}:${CUDAQX_INSTALL_PREFIX}
export PATH="${CUDAQ_INSTALL_PREFIX}/bin:${CUDAQX_INSTALL_PREFIX}/bin:${PATH}"
ctest
# Run the python tests
# The --ignore option is to bypass tests that require additional packages not contained in
# the standard docker container
cd ..
python3 -m pytest -v libs/qec/python/tests --ignore libs/qec/python/tests/test_tensor_network_decoder.py
python3 -m pytest -v libs/solvers/python/tests --ignore libs/solvers/python/tests/test_gqe.py
```

If you want to change which version of CUDA-Q that CUDA-QX is paired with, you
will need to rebuild CUDA-Q from source. This is achievable by going to the
`/workspaces/cudaq` directory in that image and using the appropriate `git`
commands to switch to whichever version you need. You can then use
[these instructions](https://github.com/NVIDIA/cuda-quantum/blob/main/Building.md)
to re-build CUDA-Q.

Additionally, the following CMake options can be configured:

- `CUDAQX_ENABLE_LIBS`: Specify which libraries to build (`all`, `qec`, `solvers`)
- `CUDAQX_INCLUDE_TESTS`: Enable building of tests
- `CUDAQX_BINDINGS_PYTHON`: Enable Python bindings

The above instructions provide a fully open-source way of building and
contributing to CUDA-QX, but it should be noted that while this environment
will have many GPU-accelerated simulators installed in it, it won't contain the
*highest* performing CUDA-Q simulators. See [this note](https://nvidia.github.io/cuda-quantum/latest/using/install/data_center_install.html)
for more details.

## Building CUDA-QX Documentation from Source

If you want to build and render our documentation from source, you can do this
with the same environment as above. In particular, after running `ninja install`,
you can run `ninja docs`. This places the documentation into the
`/workspaces/cudaqx/build/docs/build/` directory. From there, you can open
the `index.html` file in your browser, or if you are using VSCode or Cursor, you
can simply browse to the `index.html` file in the Explorer panel, right click on
the file, and select "Open with Live Server", and that will open your browser
with the main docs page loaded automatically.
