#!/bin/sh

# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
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

# Validate that $cuda_version has exactly 1 dot (i.e. 12.6, not 12.6.0)
if [[ $(echo $cuda_version | grep -o '\.' | wc -l) -ne 1 ]]; then
  echo "Error: cuda_version must have exactly 1 dot"
  exit 1
fi
cuda_major=$(echo $cuda_version | cut -d '.' -f 1)

${python} -m pip install --no-cache-dir pytest

# The following packages are needed for our tests. They are not true
# dependencies for our delivered package.
${python} -m pip install openfermion
${python} -m pip install openfermionpyscf
${python} -m pip install onnxscript # for trt decoder tests

# Now install torch.
cuda_no_dot=$(echo $cuda_version | sed 's/\.//')
${python} -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu${cuda_no_dot}

FIND_LINKS="--find-links /wheels/ --find-links /metapackages/"

# If special CUDA-Q wheels have been built for this test, install them here.
if [ -d /cudaq-wheels ]; then
  echo "Custom CUDA-Q wheels directory found. Will add to the --find-links list."
  FIND_LINKS="${FIND_LINKS} --find-links /cudaq-wheels/"
  ${python} -m pip install $FIND_LINKS "cuda-quantum-cu${cuda_major}==${cudaq_version}"
fi

# Install TensorRT.
if [[ "$cuda_major" == "13" ]]; then
  ${python} -m pip install "tensorrt-cu13==10.13.*" "cuda_toolkit[cudart]==${cuda_version}.*"
elif [[ "$cuda_major" == "12" ]]; then
  # If x86_64, then install tensorrt-cu12. If arm64, then do not install tensorrt-cu12.
  if [ "$(uname -m)" == "x86_64" ]; then
    ${python} -m pip install "tensorrt-cu12==10.13.*" "cuda_toolkit[cudart]==${cuda_version}.*"
  fi
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
package_expected=cudaq-qec-cu${cuda_major}
if [ "$package_installed" != "$package_expected" ]; then
  echo "::error Expected installation of $package_expected package, but got $package_installed."
  exit 1
fi

package_installed=$(python${python_version} -m pip list | grep cuda-quantum-cu | cut -d ' ' -f1)
package_expected=cuda-quantum-cu${cuda_major}
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
package_expected=cudaq-solvers-cu${cuda_major}
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

# Skip list (PYTHON-REFACTOR): known failing examples until Python refactor is done
# - custom_repetition_code_fine_grain_noise.py: wrong number of arguments provided
# - my_steane_test.py: arity of kernel stabilizer does not match number of arguments provided
# - adapt_h2.py: wrong number of arguments provided
# - uccsd_vqe.py: unknown function call (CompilerError)
skip_python_test() {
    case "$1" in
        *custom_repetition_code_fine_grain_noise.py) return 0 ;;
        *) return 1 ;;
    esac
}

# Install stim for AMD platform for tensor network decoder examples
if echo $platform | grep -qi "amd64"; then
  echo "Installing stim and beliefmatching for AMD64 platform"
  ${python} -m pip install stim beliefmatching
fi

for domain in "solvers" "qec"; do
    echo "Testing ${domain} Python examples with Python ${python_version} ..."
    cd examples/${domain}/python
    shopt -s nullglob # don't throw errors if no Python files exist
    for f in *.py; do
        if skip_python_test "$f"; then
            echo "Skipping Python example (PYTHON-REFACTOR): $f"
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
