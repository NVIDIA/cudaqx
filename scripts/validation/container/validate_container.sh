#!/bin/bash

# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set -e

# Parse command line arguments
FINAL_IMAGE="ghcr.io/nvidia/private/cuda-quantum:cu12-0.13.0-cudaqx-rc1"
CUDA_VERSION="12.6"
while [[ $# -gt 0 ]]; do
    case $1 in
        --final-image)
            FINAL_IMAGE=$2
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--final-image <final image>]"
            exit 1
            ;;
    esac
done

# if FINAL_IMAGE contains cu12, then set CUDA_VERSION to 12.6
if [[ "${FINAL_IMAGE}" == *"cu12"* ]]; then
    CUDA_VERSION="12.6"
elif [[ "${FINAL_IMAGE}" == *"cu13"* ]]; then
    CUDA_VERSION="13.0"
else
    echo "Unsupported CUDA version in ${FINAL_IMAGE}"
    exit 1
fi

CURRENT_ARCH=$(uname -m)
PY_TARGETS=("nvidia" "nvidia --option fp64", "qpp-cpu")
CPP_TARGETS=("nvidia" "nvidia --target-option fp64", "qpp-cpu")
cuda_major=$(echo ${CUDA_VERSION} | cut -d '.' -f 1)
cuda_minor=$(echo ${CUDA_VERSION} | cut -d '.' -f 2)
cuda_no_dot="${cuda_major}${cuda_minor}"

# Function to run Python tests
run_python_tests() {
    local container_name=$1

    echo "Running Python tests..."

    # Install pytest and other test dependencies
    docker exec ${container_name} bash -c "\
        python3 -m pip install pytest --user"

    # Clone repository and run tests with specific target
    docker exec ${container_name} bash -c "\
        cd /home/cudaq && \
        python3 -m pytest /home/cudaq/cudaqx_pytests -v"

    local test_result=$?
    if [ ${test_result} -ne 0 ]; then
        echo "Python tests failed for target ${target}"
        return 1
    fi

    echo "Python tests completed successfully for target ${target}"
    return 0
}

# Function to test examples
test_examples() {
    local tag=$1
    local container_name="cudaqx-test-$(date +%s)"

    echo "Testing examples in ${tag}..."
    # Start container with a command that keeps it running
    docker run --net=host -d -it --name ${container_name} --gpus all ${tag}

    # Wait for container to be fully up
    sleep 2

    # Verify container is running
    if ! docker ps | grep -q ${container_name}; then
        echo "Container failed to start properly"
        docker logs ${container_name}
        return 1
    fi

    num_failures=0

    docker exec ${container_name} bash -c "pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu${cuda_no_dot}"
    # Install other required packages
    docker exec ${container_name} bash -c "pip install 'lightning>=2.0.0' 'ml_collections>=0.1.0' 'mpi4py>=3.1.0' 'transformers>=4.30.0'"
    docker exec ${container_name} bash -c "pip install 'quimb' 'opt_einsum' 'cuquantum-python-cu${cuda_major}==25.09.1'"

    # Run Python tests first
    if ! run_python_tests ${container_name} ${target}; then
        echo "Python tests failed, but continuing with other tests."
        num_failures=$((num_failures+1))
        # Note, if we want to stop tests here, uncomment these lines.
        # docker stop ${container_name}
        # docker rm ${container_name}
        # return 1
    fi

    # Run tests for each target
    for target in "${PY_TARGETS[@]}"; do
        echo "Testing with target: ${target}"

        # Test Python examples
        for domain in "solvers" "qec"; do
            if docker exec ${container_name} bash -c "[ -d /home/cudaq/cudaqx-examples/${domain}/python ] && [ -n \"\$(ls -A /home/cudaq/cudaqx-examples/${domain}/python/*.py 2>/dev/null)\" ]"; then
                echo "Testing ${domain} Python examples with target ${target}..."
                if ! docker exec ${container_name} bash -c "cd /home/cudaq/cudaqx-examples/${domain}/python && \
                    for f in *.py; do echo Testing \$f...; python3 \$f --target ${target} || exit 1; done"; then
                    echo "Python tests failed for ${domain} with target ${target}"
                    docker stop ${container_name}
                    docker rm ${container_name}
                    return 1
                fi
            else
                echo "Skipping ${domain} Python examples - directory empty or not found"
            fi
        done
    done

    for target in "${CPP_TARGETS[@]}"; do

        # Test C++ examples
        for domain in "solvers" "qec"; do
            if docker exec ${container_name} bash -c "[ -d /home/cudaq/cudaqx-examples/${domain}/cpp ] && [ -n \"\$(ls -A /home/cudaq/cudaqx-examples/${domain}/cpp/*.cpp 2>/dev/null)\" ]"; then
                echo "Testing ${domain} C++ examples with target ${target}..."
                if ! docker exec ${container_name} bash -c "cd /home/cudaq/cudaqx-examples/${domain}/cpp && \
                    for f in *.cpp; do \
                        echo Compiling and running \$f...; \
                        nvq++ --enable-mlir -lcudaq-${domain} --target ${target} \$f -o test_prog && \
                        ./test_prog || exit 1; \
                        rm test_prog; \
                    done"; then
                    echo "C++ tests failed for ${domain} with target ${target}"
                    docker stop ${container_name}
                    docker rm ${container_name}
                    return 1
                fi
            else
                echo "Skipping ${domain} C++ examples - directory empty or not found"
            fi
        done
    done

    # Cleanup
    docker stop ${container_name}
    docker rm ${container_name}

    return $num_failures
}

# Main execution
echo "Starting CUDA-Q image validation for ${CURRENT_ARCH}..."

tag="${FINAL_IMAGE}"
test_examples ${tag} || {
    echo "Tests failed on ${CURRENT_ARCH}"
    exit 1
}

echo "Validation complete successfully for ${CURRENT_ARCH}!"
