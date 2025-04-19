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
CUDAQ_PATCH='diff --git a/CMakeLists.txt b/CMakeLists.txt
index 737001b56..195b2f44f 100644
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -559,7 +559,7 @@ if(CUDAQ_BUILD_TESTS)
 endif()
 
 if (CUDAQ_ENABLE_PYTHON)
-  find_package(Python 3 COMPONENTS Interpreter Development)
+  find_package(Python 3 COMPONENTS Interpreter Development.Module)
   
   # Apply specific patch to pybind11 for our documentation.
   # Only apply the patch if not already applied.
diff --git a/python/runtime/cudaq/domains/plugins/CMakeLists.txt b/python/runtime/cudaq/domains/plugins/CMakeLists.txt
index 7b7541d..2261334 100644
--- a/python/runtime/cudaq/domains/plugins/CMakeLists.txt
+++ b/python/runtime/cudaq/domains/plugins/CMakeLists.txt
@@ -17,6 +17,6 @@ if (SKBUILD)
     if (NOT Python_FOUND)
       message(FATAL_ERROR "find_package(Python) not run?")
     endif()
-    target_link_libraries(cudaq-pyscf PRIVATE Python::Python pybind11::pybind11 cudaq-chemistry cudaq-operator cudaq cudaq-py-utils)
+    target_link_libraries(cudaq-pyscf PRIVATE Python::Module pybind11::pybind11 cudaq-chemistry cudaq-operator cudaq cudaq-py-utils)
 endif()
 install(TARGETS cudaq-pyscf DESTINATION lib/plugins)
diff --git a/scripts/build_cudaq.sh b/scripts/build_cudaq.sh
index 57bdaeb..33a6677 100644
--- a/scripts/build_cudaq.sh
+++ b/scripts/build_cudaq.sh
@@ -134,10 +134,12 @@ if [ -x "$(command -v "$LLVM_INSTALL_PREFIX/bin/ld.lld")" ]; then
   NVQPP_LD_PATH="$LLVM_INSTALL_PREFIX/bin/ld.lld"
   LINKER_TO_USE="lld"
   LINKER_FLAGS="-fuse-ld=lld -B$LLVM_INSTALL_PREFIX/bin"
+  LINKER_FLAG_LIST=" \
+    -DCMAKE_LINKER='"$LINKER_TO_USE"' \
+    -DCMAKE_EXE_LINKER_FLAGS='"$LINKER_FLAGS"' \
+    -DLLVM_USE_LINKER='"$LINKER_TO_USE"'"
 else
   echo "No lld linker detected. Using the system linker."
-  LINKER_TO_USE="ld"
-  LINKER_FLAGS=""
 fi

 # Determine CUDA flags
@@ -169,9 +171,7 @@ cmake_args="-G Ninja '"$repo_root"' \
   -DCMAKE_CUDA_COMPILER='"$cuda_driver"' \
   -DCMAKE_CUDA_FLAGS='"$CUDAFLAGS"' \
   -DCMAKE_CUDA_HOST_COMPILER='"${CUDAHOSTCXX:-$CXX}"' \
-  -DCMAKE_LINKER='"$LINKER_TO_USE"' \
-  -DCMAKE_EXE_LINKER_FLAGS='"$LINKER_FLAGS"' \
-  -DLLVM_USE_LINKER='"$LINKER_TO_USE"' \
+  ${LINKER_FLAG_LIST} \
   ${OpenMP_libomp_LIBRARY:+-DOpenMP_C_LIB_NAMES=lib$OpenMP_libomp_LIBRARY} \
   ${OpenMP_libomp_LIBRARY:+-DOpenMP_CXX_LIB_NAMES=lib$OpenMP_libomp_LIBRARY} \
   ${OpenMP_libomp_LIBRARY:+-DOpenMP_libomp_LIBRARY=$OpenMP_libomp_LIBRARY} \'

echo "$CUDAQ_PATCH" | git apply --verbose

$python -m venv --system-site-packages .venv
source .venv/bin/activate
CUDAQ_BUILD_TESTS=FALSE bash scripts/build_cudaq.sh -v ${cudaq_ninja_jobs_arg}

