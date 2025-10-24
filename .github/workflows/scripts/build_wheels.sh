#!/bin/sh

# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set -e  # Exit immediately if a command exits with a non-zero status

# ==============================================================================
# Handling options
# ==============================================================================

show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --build-type      Build type (e.g., Release)"
    echo "  --cuda-version    CUDA version to build wheel for (e.g. 12 or 13)"
    echo "  --cudaq-prefix    Path to CUDA-Q's install prefix"
    echo "                    (default: \$HOME/.cudaq)"
    echo "  --python-version  Python version to build wheel for (e.g. 3.11)"
    echo "  --tensorrt-path   Path to TensorRT installation directory"
    echo "                    (default: /trt_download/TensorRT-10.13.3.9)"
    echo "  --devdeps         Build wheels suitable for internal testing"
    echo "                    (not suitable for distribution but sometimes"
    echo "                    helpful for debugging)"
    echo "  --version         Specify version of wheels to produce"
    echo "                    (default: 0.0.0)"
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
            --cuda-version)
                if [[ -n "$2" && "$2" != -* ]]; then
                    cuda_version=("$2")
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
            --python-version)
                if [[ -n "$2" && "$2" != -* ]]; then
                    python_version=("$2")
                    shift 2
                else
                    echo "Error: Argument for $1 is missing" >&2
                    exit 1
                fi
                ;;
            --tensorrt-path)
                if [[ -n "$2" && "$2" != -* ]]; then
                    tensorrt_path=("$2")
                    shift 2
                else
                    echo "Error: Argument for $1 is missing" >&2
                    exit 1
                fi
                ;;
            --devdeps)
                devdeps=true
                shift 1
                ;;
            --version)
                if [[ -n "$2" && "$2" != -* ]]; then
                    wheels_version=("$2")
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
cudaq_prefix=$HOME/.cudaq
build_type=Release
python_version=3.11
tensorrt_path=/trt_download/TensorRT-10.13.3.9
devdeps=false
wheels_version=0.0.0
cuda_version=12

# Parse options
parse_options "$@"

echo "Building in $build_type mode for Python $python_version, version $wheels_version, CUDA version $cuda_version"

# ==============================================================================
# Helpers
# ==============================================================================

python=python${python_version}
ARCH=$(uname -m)
PLAT_STR=""

if $devdeps; then
  PLAT_STR="--plat manylinux_2_34_x86_64"
else
  # We need to use a newer toolchain because CUDA-QX libraries rely on c++20
  source /opt/rh/gcc-toolset-11/enable
fi

export CC=gcc
export CXX=g++
export SETUPTOOLS_SCM_PRETEND_VERSION=$wheels_version
export CUDAQX_QEC_VERSION=$wheels_version
export CUDAQX_SOLVERS_VERSION=$wheels_version

# ==============================================================================
# QEC library
# ==============================================================================

cd libs/qec
cp pyproject.toml.cu${cuda_version} pyproject.toml

SKBUILD_CMAKE_ARGS="-DCUDAQ_DIR=$cudaq_prefix/lib/cmake/cudaq;-DTENSORRT_ROOT=$tensorrt_path"
if ! $devdeps; then
  SKBUILD_CMAKE_ARGS+=";-DCMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN=/opt/rh/gcc-toolset-11/root/usr/lib/gcc/${ARCH}-redhat-linux/11/"
fi
SKBUILD_CMAKE_ARGS+=";-DCMAKE_BUILD_TYPE=$build_type"
export SKBUILD_CMAKE_ARGS
$python -m build --wheel

CUDAQ_EXCLUDE_LIST=$(for f in $(find $cudaq_prefix/lib -name "*.so" -printf "%P\n" | sort); do echo "--exclude $f"; done | tr '\n' ' ')

LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/_skbuild/lib:$tensorrt_path/lib" \
$python -m auditwheel -v repair dist/*.whl $CUDAQ_EXCLUDE_LIST \
  --exclude libcudart.so.${cuda_version} \
  --exclude libnvinfer.so.10 \
  --exclude libnvonnxparser.so.10 \
  --wheel-dir /wheels \
  --exclude libcudart.so.${cuda_version} \
  --exclude libnvinfer.so.10 \
  --exclude libnvonnxparser.so.10 \
  ${PLAT_STR}

# ==============================================================================
# Solvers library
# ==============================================================================

cd ../solvers
cp pyproject.toml.cu${cuda_version} pyproject.toml

SKBUILD_CMAKE_ARGS="-DCUDAQ_DIR=$cudaq_prefix/lib/cmake/cudaq"
if ! $devdeps; then
  SKBUILD_CMAKE_ARGS+=";-DCMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN=/opt/rh/gcc-toolset-11/root/usr/lib/gcc/${ARCH}-redhat-linux/11/;"
fi
SKBUILD_CMAKE_ARGS+=";-DCMAKE_BUILD_TYPE=$build_type" \
export SKBUILD_CMAKE_ARGS
$python -m build --wheel

LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/_skbuild/lib" \
$python -m auditwheel -v repair dist/*.whl $CUDAQ_EXCLUDE_LIST \
  --exclude libgfortran.so.5 \
  --exclude libquadmath.so.0 \
  --exclude libmvec.so.1 \
  --wheel-dir /wheels \
  ${PLAT_STR}


echo "Wheel builds are complete: "
ls -la /wheels
