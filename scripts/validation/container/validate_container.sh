#!/bin/bash

# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set -e

CURRENT_ARCH=$(uname -m)
PY_TARGETS=("nvidia" "nvidia --option fp64", "qpp-cpu")
CPP_TARGETS=("nvidia" "nvidia --target-option fp64", "qpp-cpu")

FINAL_IMAGE="ghcr.io/nvidia/private/cuda-quantum:cu12-0.10.0-cudaqx-rc2"

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
