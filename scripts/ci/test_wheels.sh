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
cudaq_version=$4
cudaqx_version=$5

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
if [ -z "$cudaq_version" ] ; then
  echo "Error: cudaq_version is empty"
  exit 1
fi
if [ -z "$cudaqx_version" ] ; then
  echo "Error: cudaqx_version is empty"
  exit 1
fi

${python} -m pip install --no-cache-dir pytest

# The following packages are needed for our tests. They are not true
# dependencies for our delivered package.
${python} -m pip install openfermion
${python} -m pip install openfermionpyscf

FIND_LINKS="--find-links /wheels/ --find-links /metapackages/"

# If special CUDA-Q wheels have been built for this test, install them here.
if [ -d /cudaq-wheels ]; then
  echo "Custom CUDA-Q wheels directory found. Will add to the --find-links list."
  FIND_LINKS="${FIND_LINKS} --find-links /cudaq-wheels/"
  ${python} -m pip install $FIND_LINKS "cuda-quantum-cu${cuda_version}==${cudaq_version}"
fi

# TODO: Remove this once PyTorch 2.9.0 is released. That should happen before
# this PR is merged.
if [[ "$cuda_version" == "13" ]]; then
  ${python} -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu130
elif [[ "$cuda_version" == "12" ]]; then
  ${python} -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu126
fi

# Temporary hack until tensorrt-cu13 bumps its package version to accommodate
# for the breaking change in nvidia-cuda-runtime-cu13.
if [[ "$cuda_version" == "13" ]]; then
  ${python} -m pip install ${FIND_LINKS} "nvidia-cuda-runtime-cu13==0.0.0a0"
fi

# QEC library
# ======================================

# Install QEC library with tensor network decoder and trt_decoder (requires Python >=3.11)
echo "Installing QEC library with tensor network decoder and trt_decoder"
${python} -m pip install ${FIND_LINKS} --extra-index-url https://pypi.nvidia.com/ "cudaq-qec[all]==${cudaqx_version}"
# Check if CUDA is available
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
  # CUDA available - run all tests
  ${python} -m pytest -v -s libs/qec/python/tests/
else
  # No CUDA - skip TRT decoder tests
  ${python} -m pytest -v -s libs/qec/python/tests/ --ignore=libs/qec/python/tests/test_trt_decoder.py
fi

# Verify that the correct version of cuda-quantum and cudaq-qec were installed.
package_installed=$(python${python_version} -m pip list | grep cudaq-qec-cu | cut -d ' ' -f1)
package_expected=cudaq-qec-cu${cuda_version}
if [ "$package_installed" != "$package_expected" ]; then
  echo "::error Expected installation of $package_expected package, but got $package_installed."
  exit 1
fi

package_installed=$(python${python_version} -m pip list | grep cuda-quantum-cu | cut -d ' ' -f1)
package_expected=cuda-quantum-cu${cuda_version}
if [ "$package_installed" != "$package_expected" ]; then
  echo "::error Expected installation of $package_expected package, but got $package_installed."
  exit 1
fi

# Solvers library
# ======================================
# Test the base solvers library without optional dependencies
echo "Installing Solvers library without GQE"
${python} -m pip install ${FIND_LINKS} "cudaq-solvers==${cudaqx_version}"
${python} -m pytest -v -s libs/solvers/python/tests/ --ignore=libs/solvers/python/tests/test_gqe.py

# Verify that the correct version of cudaq-solvers was installed.
package_installed=$(python${python_version} -m pip list | grep cudaq-solvers-cu | cut -d ' ' -f1)
package_expected=cudaq-solvers-cu${cuda_version}
if [ "$package_installed" != "$package_expected" ]; then
  echo "::error Expected installation of $package_expected package, but got $package_installed."
  exit 1
fi

# Test the solvers library with GQE
echo "Installing Solvers library with GQE"
${python} -m pip install ${FIND_LINKS} "cudaq-solvers[gqe]==${cudaqx_version}"
${python} -m pytest -v -s libs/solvers/python/tests/test_gqe.py

# Test the libraries with examples
# ======================================
echo "Testing libraries with examples"

# Install stim for AMD platform for tensor network decoder examples
if echo $platform | grep -qi "amd64"; then
  echo "Installing stim and beliefmatching for AMD64 platform"
  ${python} -m pip install stim beliefmatching
fi

gpu_is_available=$(nvidia-smi &> /dev/null && echo "true" || echo "false")

for domain in "solvers" "qec"; do
    echo "Testing ${domain} Python examples with Python ${python_version} ..."
    cd examples/${domain}/python
    shopt -s nullglob # don't throw errors if no Python files exist
    for f in *.py; do
        if [ "$f" == "nv-qldpc-decoder.py" ] && [ "$gpu_is_available" == "false" ]; then
            continue
        fi
        echo Testing $f...
        ${python} $f 
        res=$?
        if [ $res -ne 0 ]; then
            echo "Python tests failed for ${domain} with Python ${python_version}: $res"
        fi
    done
    shopt -u nullglob  # reset setting, just for cleanliness
    cd - # back to the original directory
done
