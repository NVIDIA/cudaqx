#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# ==============================================================================
# Handling options
# ==============================================================================

set -eo pipefail

show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --python-version  Python version to build wheel for (e.g. 3.10)"
    echo "  --cuda-version    CUDA version to build wheel for (e.g. 12.6 or 13.0)"
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
            --cuda-version)
                if [[ -n "$2" && "$2" != -* ]]; then
                    cuda_version=("$2")
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
cuda_version=12.6

# Parse options
parse_options "$@"


export CUDA_VERSION=${cuda_version}
export CUDAQ_INSTALL_PREFIX=/usr/local/cudaq

# We need to use a newer toolchain because CUDA-QX libraries rely on c++20
source /opt/rh/gcc-toolset-12/enable

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
    bash scripts/build_llvm.sh -c Release -v

# ==============================================================================
# Building CUDA-Q
# ==============================================================================

CUDAQ_PATCH='diff --git a/CMakeLists.txt b/CMakeLists.txt
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -774,8 +774,8 @@ if(CUDAQ_BUILD_TESTS)
 endif()

 if("python" IN_LIST CUDAQ_ENABLE_PROJECTS)
-  find_package(Python 3 COMPONENTS Interpreter Development)
-  find_package(Python3 COMPONENTS Interpreter Development)
+  find_package(Python 3 COMPONENTS Interpreter Development.Module)
+  find_package(Python3 COMPONENTS Interpreter Development.Module)

   add_subdirectory(tpls/nanobind)

diff --git a/python/runtime/cudaq/domains/plugins/CMakeLists.txt b/python/runtime/cudaq/domains/plugins/CMakeLists.txt
--- a/python/runtime/cudaq/domains/plugins/CMakeLists.txt
+++ b/python/runtime/cudaq/domains/plugins/CMakeLists.txt
@@ -33,7 +33,7 @@ if (SKBUILD)
 else()
   target_link_libraries(cudaq-pyscf
     PRIVATE
-      nanobind-static Python3::Python
+      nanobind-static Python3::Module
       cudaq-chemistry cudaq-operator cudaq cudaq-py-utils cudaq-platform-default)
 endif()
'

# Apply the CMake Python-component patch (Development -> Development.Module).
# Strict apply first (matches the canonical pinned cuda-quantum tree); fall back
# to a reduced-context apply (-C1) for refs whose surrounding CMake context has
# shifted -- e.g. after upstream cuda-quantum PR #4698 restructured the
# python/nanobind block, which the shared-ring branch (NVIDIA/cuda-quantum#4712)
# carries via its upstream merge.  Both paths are non-interactive; do NOT use
# `patch`, which can hang on a "File to patch" prompt in CI.
if ! echo "$CUDAQ_PATCH" | git apply --verbose; then
  echo "build_cudaq: strict git apply failed; retrying with -C1 (reduced context)" >&2
  echo "$CUDAQ_PATCH" | git apply --verbose -C1
fi

$python -m venv --system-site-packages .venv
source .venv/bin/activate
CUDAQ_BUILD_TESTS=FALSE bash scripts/build_cudaq.sh -v ${cudaq_ninja_jobs_arg}
