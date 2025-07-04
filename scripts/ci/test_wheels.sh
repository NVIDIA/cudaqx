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

${python} -m pip install quimb opt_einsum torch
# Install QEC library with tensor network decoder
wheel_file=$(ls /wheels/cudaq_qec-*-cp${python_version_no_dot}-cp${python_version_no_dot}-*.whl)
${python} -m pip install "${wheel_file}[tn_decoder]"
${python} -m pytest -v -s libs/qec/python/tests/

# Solvers library
# ======================================

${python} -m pip install /wheels/cudaq_solvers-*-cp${python_version_no_dot}-cp${python_version_no_dot}-*.whl
${python} -m pytest libs/solvers/python/tests/
