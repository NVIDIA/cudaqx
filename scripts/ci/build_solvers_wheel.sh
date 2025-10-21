#!/bin/sh

# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# /!\ This script must be run inside an docker container /!\

python_version=3.12
python=python${python_version}

# Delete previous wheels
rm wheels/cudaq_solvers*.whl

# Exit immediately if any command returns a non-zero status
set -e

git config --global --add safe.directory /cuda-qx

cd /cuda-qx/libs/solvers

# We need to use a newer toolchain because CUDA-QX libraries rely on c++20
source /opt/rh/gcc-toolset-11/enable

export CC=gcc
export CXX=g++

SKBUILD_CMAKE_ARGS="-DCUDAQ_DIR=$HOME/.cudaq/lib/cmake/cudaq;-DCMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN=/opt/rh/gcc-toolset-11/root/usr/lib/gcc/x86_64-redhat-linux/11/" \
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
  --exclude libcudaq-operator.so \
  --exclude libcudaq-nlopt.so \
  --exclude libgfortran.so.5 \
  --exclude libquadmath.so.0 \
  --exclude libmvec.so.1 \
  --wheel-dir /wheels

