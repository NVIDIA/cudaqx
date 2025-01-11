#!/bin/sh

# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #


# ==============================================================================
# Handling options
# ==============================================================================

show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --build-type      Build type (e.g., Release)"
    echo "  --cudaq-prefix    Path to CUDA-Q's install prefix"
    echo "                    (default: \$HOME/.cudaq)"
}

parse_options() {
    while (( $# > 0 )); do
        case "$1" in
            --build-type)
                if [[ -n "$2" && "$2" != -* ]]; then
                    build_type=("$2")
                    shift 2
                else
                    echo "Error: Argument for $1 is missing" >&2
                    exit 1
                fi
                ;;
            --cudaq-prefix)
                if [[ -n "$2" && "$2" != -* ]]; then
                    cudaq_prefix=("$2")
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

# Initialize an empty array to store libs names
cudaq_prefix=$HOME/.cudaq
build_type=Release

# Parse options
parse_options "$@"

# ==============================================================================
# Helpers
# ==============================================================================

python_version=3.10
python=python${python_version}

# We need to use a newer toolchain because CUDA-QX libraries rely on c++20
source /opt/rh/gcc-toolset-11/enable

export CC=gcc
export CXX=g++

# ==============================================================================
# QEC library
# ==============================================================================

cd libs/qec

SKBUILD_CMAKE_ARGS="-DCUDAQ_DIR=$cudaq_prefix/lib/cmake/cudaq;-DCMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN=/opt/rh/gcc-toolset-11/root/usr/lib/gcc/aarch64-redhat-linux/11/;-DCMAKE_BUILD_TYPE=Debug" \
$python -m build --wheel

LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/_skbuild/lib" \
$python -m auditwheel -v repair dist/*.whl \
  --exclude libcudaq-em-default.so \
  --exclude libcudaq-python-interop.so \
  --exclude libcudaq-ensmallen.so \
  --exclude libcudaq-common.so \
  --exclude libcudaq-platform-default.so \
  --exclude libnvqir-qpp.so \
  --exclude libnvqir.so \
  --exclude libcudaq.so \
  --exclude libcudaq-spin.so \
  --exclude libcudaq-nlopt.so \
  --wheel-dir /wheels

# ==============================================================================
# Solvers library
# ==============================================================================

cd ../solvers

SKBUILD_CMAKE_ARGS="-DCUDAQ_DIR=$cudaq_prefix/lib/cmake/cudaq;-DCMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN=/opt/rh/gcc-toolset-11/root/usr/lib/gcc/aarch64-redhat-linux/11/;-DCMAKE_BUILD_TYPE=Debug" \
$python -m build --wheel

LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/_skbuild/lib" \
$python -m auditwheel -v repair dist/*.whl \
  --exclude libcudaq-em-default.so \
  --exclude libcudaq-python-interop.so \
  --exclude libcudaq-ensmallen.so \
  --exclude libcudaq-common.so \
  --exclude libcudaq-platform-default.so \
  --exclude libnvqir-qpp.so \
  --exclude libnvqir.so \
  --exclude libcudaq.so \
  --exclude libcudaq-spin.so \
  --exclude libcudaq-nlopt.so \
  --exclude libgfortran.so.5 \
  --exclude libquadmath.so.0 \
  --exclude libmvec.so.1 \
  --wheel-dir /wheels

