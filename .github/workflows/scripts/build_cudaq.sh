#!/bin/sh

# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# ==============================================================================
# Handling options
# ==============================================================================

# Exit on error
set -e

show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --python-version  Python version to build wheel for (e.g. 3.10)"
    echo "  -j                Number of parallel jobs to build CUDA-Q with"
    echo "                    (e.g. 8)"
}

parse_options() {
    while (( $# > 0 )); do
        case "$1" in
            --python-version)
                if [[ -n "$2" && "$2" != -* ]]; then
                    python_version=("$2")
                    shift 2
                else
                    echo "Error: Argument for $1 is missing" >&2
                    exit 1
                fi
                ;;
            -j)
                if [[ -n "$2" && "$2" != -* ]]; then
                    num_par_jobs=("$2")
                    cudaq_ninja_jobs_arg="-j $num_par_jobs"
                    shift 2
                else
                    echo "Error: Argument for $1 is missing" >&2
                    exit 1
                fi
                ;;
            -*)
                echo "Error: Unknown option $1" >&2
                show_help
                exit 1
                ;;
            *)
                echo "Error: Unknown argument $1" >&2
                show_help
                exit 1
                ;;
        esac
    done
}

# Defaults
python_version=3.10
cudaq_ninja_jobs_arg=""

# Parse options
parse_options "$@"


export CUDA_VERSION=12.0
export CUDAQ_INSTALL_PREFIX=/usr/local/cudaq

# We need to use a newer toolchain because CUDA-QX libraries rely on c++20
source /opt/rh/gcc-toolset-11/enable

export CC=gcc
export CXX=g++

python=python${python_version}
${python} -m pip install --no-cache-dir numpy auditwheel

echo "Building CUDA-Q."
cd cudaq

# ==============================================================================
# Building MLIR bindings
# ==============================================================================

echo "Building MLIR bindings for ${python}" && \
    rm -rf "$LLVM_INSTALL_PREFIX/src" "$LLVM_INSTALL_PREFIX/python_packages" && \
    Python3_EXECUTABLE="$(which ${python})" \
    LLVM_PROJECTS='clang;mlir;python-bindings' \
    LLVM_CMAKE_CACHE=/cmake/caches/LLVM.cmake LLVM_SOURCE=/llvm-project \
    bash /scripts/build_llvm.sh -c Release -v 

# ==============================================================================
# Building CUDA-Q
# ==============================================================================

# The patch to build_cudaq.sh is needed for manylinux builds because the
# compiler in the AlmaLinux build does not support using -fuse-ld=ld.
git apply --verbose cudaq.patch

$python -m venv --system-site-packages .venv
source .venv/bin/activate
CUDAQ_BUILD_TESTS=FALSE bash scripts/build_cudaq.sh -v ${cudaq_ninja_jobs_arg}

