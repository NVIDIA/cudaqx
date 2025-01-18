#!/bin/sh

# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
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

#apt-get update && apt-get install -y --no-install-recommends \
#        libgfortran5 python${python_version} python$(echo ${python_version} | cut -d . -f 1)-pip

${python} -m pip install --no-cache-dir pytest

# If special CUDA-Q wheels have been built for this test, install them here. This will 
if [ -d /cudaq-wheels ]; then
  echo "Custom CUDA-Q wheels directory found; installing ..."
  ls -1 /cudaq-wheels/cuda_quantum-*-cp${python_version_no_dot}-cp${python_version_no_dot}-*.whl
  ${python} -m pip install /cudaq-wheels/cuda_quantum-*-cp${python_version_no_dot}-cp${python_version_no_dot}-*.whl
fi

# QEC library
# ======================================

${python} -m pip install /wheels/cudaq_qec-*-cp${python_version_no_dot}-cp${python_version_no_dot}-*.whl
${python} -m pytest -s libs/qec/python/tests/test_decoder.py::test_decoder_plugin_initialization # libs/qec/python/tests/

# Solvers library
# ======================================

${python} -m pip install /wheels/cudaq_solvers-*-cp${python_version_no_dot}-cp${python_version_no_dot}-*.whl
${python} -m pytest libs/solvers/python/tests/
