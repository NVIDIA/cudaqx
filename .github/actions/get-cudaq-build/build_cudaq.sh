#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

BUILD_TYPE=${1:-"Release"}
LAUNCHER=${2:-""}
CC=${3:-"gcc"}
CXX=${4:-"g++"}

LLVM_INSTALL_PREFIX=/usr/local/llvm
CUQUANTUM_INSTALL_PREFIX="$(pip show cuquantum-python-cu12 | grep "Location:" | cut -d " " -f 2)/cuquantum"
CUTENSOR_INSTALL_PREFIX="$(pip show cutensor-cu12 | grep "Location:" | cut -d " " -f 2)/cutensor"

cd cudaq

# Determine linker and linker flags
if [ -x "$(command -v "$LLVM_INSTALL_PREFIX/bin/ld.lld")" ]; then
  echo "Configuring nvq++ to use the lld linker by default."
  NVQPP_LD_PATH="$LLVM_INSTALL_PREFIX/bin/ld.lld"
fi


# Determine CUDA flags
cuda_driver=${CUDACXX:-${CUDA_HOME:-/usr/local/cuda}/bin/nvcc}

if [ -z "$CUDAHOSTCXX" ] && [ -z "$CUDAFLAGS" ]; then
  CUDAFLAGS='-allow-unsupported-compiler'
  if [ -x "$CXX" ] && [ -n "$("$CXX" --version | grep -i clang)" ]; then
    CUDAFLAGS+=" --compiler-options --stdlib=libstdc++"
  fi
  if [ -d "$GCC_TOOLCHAIN" ]; then 
    # e.g. GCC_TOOLCHAIN=/opt/rh/gcc-toolset-11/root/usr/
    CUDAFLAGS+=" --compiler-options --gcc-toolchain=\"$GCC_TOOLCHAIN\""
  fi
fi

# Determine OpenMP flags
if [ -n "$(find "$LLVM_INSTALL_PREFIX" -name 'libomp.so')" ]; then
  OMP_LIBRARY=${OMP_LIBRARY:-libomp}
  OpenMP_libomp_LIBRARY=${OMP_LIBRARY#lib}
  OpenMP_FLAGS="${OpenMP_FLAGS:-'-fopenmp'}"
fi

echo "Preparing CUDA-Q build with LLVM installation in $LLVM_INSTALL_PREFIX..."
cmake_args="-G Ninja \
  -DCMAKE_INSTALL_PREFIX='"$CUDAQ_INSTALL_PREFIX"' \
  -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
  -DCMAKE_C_COMPILER=$CC \
  -DCMAKE_CXX_COMPILER=$CXX \
  -DCMAKE_C_COMPILER_LAUNCHER=$LAUNCHER \
  -DCMAKE_CXX_COMPILER_LAUNCHER=$LAUNCHER \
  -DNVQPP_LD_PATH='"$NVQPP_LD_PATH"' \
  -DCMAKE_CUDA_COMPILER='"$cuda_driver"' \
  -DCMAKE_CUDA_FLAGS='"$CUDAFLAGS"' \
  -DCMAKE_CUDA_HOST_COMPILER='"${CUDAHOSTCXX:-$CXX}"' \
  ${OpenMP_libomp_LIBRARY:+-DOpenMP_C_LIB_NAMES=lib$OpenMP_libomp_LIBRARY} \
  ${OpenMP_libomp_LIBRARY:+-DOpenMP_CXX_LIB_NAMES=lib$OpenMP_libomp_LIBRARY} \
  ${OpenMP_libomp_LIBRARY:+-DOpenMP_libomp_LIBRARY=$OpenMP_libomp_LIBRARY} \
  ${OpenMP_FLAGS:+"-DOpenMP_C_FLAGS='"$OpenMP_FLAGS"'"} \
  ${OpenMP_FLAGS:+"-DOpenMP_CXX_FLAGS='"$OpenMP_FLAGS"'"} \
  -DCUDAQ_REQUIRE_OPENMP=TRUE \
  -DCUDAQ_ENABLE_PYTHON=TRUE \
  -DCUDAQ_BUILD_TESTS=FALSE \
  -DCUDAQ_TEST_MOCK_SERVERS=FALSE \
  -DCMAKE_COMPILE_WARNING_AS_ERROR=OFF"

echo $cmake_args | xargs cmake -S . -B "build"

cmake --build "build" --target install

