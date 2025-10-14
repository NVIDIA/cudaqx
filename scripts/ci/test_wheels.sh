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
python=python${python_version}
platform=$2
cuda_version=$3

# Verify the input arguments aren't empty, one at a time.
if [ -z "$python_version" ] ; then
  echo "Error: python_version is empty"
  exit 1
fi
if [ -z "$platform" ] ; then
  echo "Error: platform is empty"
  exit 1
fi
if [ -z "$cuda_version" ] ; then
  echo "Error: cuda_version is empty"
  exit 1
fi

${python} -m pip install --no-cache-dir pytest

# The following packages are needed for our tests. They are not true
# dependencies for our delivered package.
${python} -m pip install openfermion
${python} -m pip install openfermionpyscf

# TODO: Remove this once PyTorch 2.9.0 is released. That should happen before
# this PR is merged.
if [[ "$cuda_version" == "13.0" ]]; then
  ${python} -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/test/cu130
fi

FIND_LINKS="--find-links /wheels/ --find-links /cudaq-wheels/"

# If special CUDA-Q wheels have been built for this test, install them here.
if [ -d /cudaq-wheels ]; then
  echo "Custom CUDA-Q wheels directory found; installing ..."
  echo "First ls /cudaq-wheels"
  ls /cudaq-wheels
  ${python} -m pip install ${FIND_LINKS} cuda-quantum-cu${cuda_version}
fi

# QEC library
# ======================================

# Install QEC library with tensor network decoder (requires Python >=3.11)
echo "Installing QEC library with tensor network decoder"
${python} -m pip install ${FIND_LINKS} cudaq-qec-cu${cuda_version}[tensor_network_decoder]
${python} -m pytest -v -s libs/qec/python/tests/

# Solvers library
# ======================================
# Test the base solvers library without optional dependencies
echo "Installing Solvers library without GQE"
${python} -m pip install ${FIND_LINKS} cudaq-solvers-cu${cuda_version}
${python} -m pytest -v -s libs/solvers/python/tests/ --ignore=libs/solvers/python/tests/test_gqe.py

# Test the solvers library with GQE
echo "Installing Solvers library with GQE"
${python} -m pip install ${FIND_LINKS} cudaq-solvers-cu${cuda_version}[gqe]
${python} -m pytest -v -s libs/solvers/python/tests/test_gqe.py

# Test the libraries with examples
# ======================================
echo "Testing libraries with examples"

# Install stim for AMD platform for tensor network decoder examples
if echo $platform | grep -qi "amd64"; then
  echo "Installing stim and beliefmatching for AMD64 platform"
  ${python} -m pip install stim beliefmatching
fi

for domain in "solvers" "qec"; do
    echo "Testing ${domain} Python examples with Python ${python_version} ..."
    cd examples/${domain}/python
    shopt -s nullglob # don't throw errors if no Python files exist
    for f in *.py; do \
        echo Testing $f...; \
        ${python} $f 
        res=$?
        if [ $res -ne 0 ]; then
            echo "Python tests failed for ${domain} with Python ${python_version}: $res"
        fi
    done
    shopt -u nullglob  # reset setting, just for cleanliness
    cd - # back to the original directory
done
