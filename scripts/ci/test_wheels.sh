#!/bin/sh

# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Exit immediately if any command returns a non-zero status
set -e

# Uncomment these lines to enable core files
#set +e
#ulimit -c unlimited

# Installing dependencies
python_version=$1
python_version_no_dot=$(echo $python_version | tr -d '.') # 3.10 --> 310
python=python${python_version}

${python} -m pip install --no-cache-dir pytest

# The following packages are needed for our tests. They are not true
# dependencies for our delivered package.
${python} -m pip install openfermion
${python} -m pip install openfermionpyscf

# If special CUDA-Q wheels have been built for this test, install them here. This will 
if [ -d /cudaq-wheels ]; then
  echo "Custom CUDA-Q wheels directory found; installing ..."
  echo "First ls /cudaq-wheels"
  ls /cudaq-wheels
  echo "Now show what will be pip installed"
  ls -1 /cudaq-wheels/cuda_quantum_*-cp${python_version_no_dot}-cp${python_version_no_dot}-*.whl
  ${python} -m pip install /cudaq-wheels/cuda_quantum_*-cp${python_version_no_dot}-cp${python_version_no_dot}-*.whl
fi

# QEC library
# ======================================

# Check platform from matrix configuration
if [ "$2" = "amd64" ]; then
  # Install the wheel with tensor network decoder optional dependencies
  ${python} -m pip install /wheels/cudaq_qec-*-cp${python_version_no_dot}-cp${python_version_no_dot}-*.whl[tn_decoder]
  ${python} -m pytest -s libs/qec/python/tests/

  # Run additional tensor network decoder tests
  ${python} -m pytest -s libs/qec/python/cudaq_qec/plugins/decoders/test_tensor_network_decoder.py
  ${python} -m pytest -s libs/qec/python/cudaq_qec/plugins/decoders/tensor_network_utils/test_tensor_network_utils.py
else
  # On ARM, install without tensor network decoder dependencies
  echo "Running on ARM architecture - skipping tensor network decoder tests"
  ${python} -m pip install /wheels/cudaq_qec-*-cp${python_version_no_dot}-cp${python_version_no_dot}-*.whl
  ${python} -m pytest -s libs/qec/python/tests/
fi

# Solvers library
# ======================================

${python} -m pip install /wheels/cudaq_solvers-*-cp${python_version_no_dot}-cp${python_version_no_dot}-*.whl
${python} -m pytest libs/solvers/python/tests/
