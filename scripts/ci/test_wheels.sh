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

qec_wheel=$(ls /wheels/cudaq_qec-*-cp${python_version_no_dot}-cp${python_version_no_dot}-*.whl)
# If Python version is 3.10, then install without tensor network decoder.
# Otherwise, install with the tensor network decoder.
if [ $python_version == "3.10" ]; then
  echo "Installing QEC library without tensor network decoder"
  ${python} -m pip install "${qec_wheel}"
else
  echo "Installing QEC library with tensor network decoder"
  ${python} -m pip install "${qec_wheel}[tensor_network_decoder]"
fi
${python} -m pytest -v -s libs/qec/python/tests/


# Solvers library
# ======================================
# Test the base solvers library without optional dependencies
echo "Installing Solvers library without GQE"
solver_wheel=$(ls /wheels/cudaq_solvers-*-cp${python_version_no_dot}-cp${python_version_no_dot}-*.whl)
${python} -m pip install "${solver_wheel}"
${python} -m pytest -v -s libs/solvers/python/tests/ --ignore=libs/solvers/python/tests/test_gqe.py

# Test the solvers library with GQE
echo "Installing Solvers library with GQE"
${python} -m pip install "${solver_wheel}[gqe]"
${python} -m pytest -v -s libs/solvers/python/tests/test_gqe.py

# Test the libraries with examples
# ======================================
echo "Testing libraries with examples"
echo "Listing the directories"
ls -l $PWD
echo "Now begin testing examples"

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








