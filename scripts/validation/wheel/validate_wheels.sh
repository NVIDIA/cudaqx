#!/bin/bash

# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Create the test image that this script resides in by running this command. You
# may may to inspect/modify .github/workflows/build_qa_wheel_test.yaml first.
# gh workflow run "Build Image for QA Wheel Tests" -R <repo name> --ref <branch name>

# Invoke this script, which should reside in Docker test image, like this:
# docker run -it --rm --name wheels-test --gpus all -w /root <image name> bash -i -c "bash -i validate_wheels.sh"

set -e

CURRENT_ARCH=$(uname -m)
PYTHON_VERSIONS=("3.10" "3.11" "3.12")
TARGETS=("nvidia" "nvidia --option fp64", "qpp-cpu")

# OpenBLAS can get bogged down on some machines if using too many threads.
export OMP_NUM_THREADS=8

# Function to run Python tests
run_python_tests() {
    local container_name=$1
    local python_version=$2

    echo "Running Python tests for Python ${python_version} with default target..."

    python3 -m pytest libs -v

    local test_result=$?
    if [ ${test_result} -ne 0 ]; then
        echo "Python tests failed for Python ${python_version}"
        return 1
    fi

    echo "Python tests completed successfully for Python ${python_version}"
    return 0
}

# Function to test examples
test_examples() {
    echo "Testing examples ..."

    # Loop through Python versions
    for python_version in "${PYTHON_VERSIONS[@]}"; do
        echo "Testing with Python version: ${python_version}"

        conda_name=cudaqx-env-$python_version
        conda create -y -n $conda_name python=$python_version pip
        conda activate $conda_name
        pip install pypiserver
        pypi-server run -p 8080 /root/wheels &
        pip install cudaq-qec --extra-index-url http://localhost:8080
        pip install cudaq-solvers --extra-index-url http://localhost:8080
        pkill -f "pypi-server" # kill the temporary pypi-server

        # Needed for tests:
        pip install pytest
        pip install openfermion openfermionpyscf

        # Dump a version string to the logs
        python3 -c 'import cudaq; print(cudaq.__version__)'

        # Run Python tests first
        if ! run_python_tests ${container_name} ${python_version}; then
            exit 1
        fi

        # Loop through targets
        for target in "${TARGETS[@]}"; do
            echo "Testing with target: ${target}"

            # Test Python examples
            for domain in "solvers" "qec"; do
                echo "Testing ${domain} Python examples with Python ${python_version} and target ${target}..."
                cd examples/${domain}/python
                shopt -s nullglob # don't throw errors if no Python files exist
                for f in *.py; do \
                    echo Testing $f...; \
                    python3 $f --target ${target}
                    res=$?
                    if [ $res -ne 0 ]; then
                        echo "Python tests failed for ${domain} with Python ${python_version} and target ${target}: $res"
                        return 1
                    fi
                done
                shopt -u nullglob  # reset setting, just for cleanliness
                cd - # back to the original directory
            done
        done

        conda deactivate
    done
}

# Main execution
echo "Starting CUDA-Q image validation for ${CURRENT_ARCH}..."

conda init bash
source activate base

test_examples || {
    echo "Tests failed for Python on ${CURRENT_ARCH}"
    exit 1
}

echo "Validation complete successfully for ${CURRENT_ARCH}!"
