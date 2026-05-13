#!/usr/bin/env bash

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
#
# Thin wrapper to (re)build a single CUDA-QX library standalone without
# rebuilding the whole repo. Optimized for the dev iteration loop.
#
# Usage:
#   scripts/build_one_lib.sh qec
#   scripts/build_one_lib.sh solvers --install --tests
#   scripts/build_one_lib.sh qec --build-type Debug
#
# Flags:
#   --install         Also run `cmake --build --target install`.
#   --tests           Build with tests enabled and run them.
#   --no-python       Skip Python bindings (faster).
#   --build-type T    Release (default) | Debug | RelWithDebInfo | MinSizeRel
#   --reconfigure     Force `cmake -S ... -B ...` even if the build dir exists.
#
# This is the fast cousin of scripts/test_libs_builds.sh: it builds *one*
# library, leaves the build dir around for incremental rebuilds, and does
# not iterate over every lib/ subdirectory.

set -euo pipefail

LIB=""
DO_INSTALL=0
DO_TESTS=0
NO_PYTHON=0
BUILD_TYPE=Release
RECONFIGURE=0

usage() {
  sed -n '11,28p' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    qec|solvers)        LIB="$1"; shift ;;
    --install)          DO_INSTALL=1; shift ;;
    --tests)            DO_TESTS=1; shift ;;
    --no-python)        NO_PYTHON=1; shift ;;
    --build-type)       BUILD_TYPE="$2"; shift 2 ;;
    --reconfigure)      RECONFIGURE=1; shift ;;
    -h|--help)          usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage; exit 2 ;;
  esac
done

if [[ -z "$LIB" ]]; then
  echo "Error: must pass a library name (qec | solvers)" >&2
  usage; exit 2
fi

CUDAQ_INSTALL_PREFIX_VAL=${CUDAQ_INSTALL_PREFIX:-$HOME/.cudaq}
CUDAQX_INSTALL_PREFIX_VAL=${CUDAQX_INSTALL_PREFIX:-$HOME/.cudaqx}
CUDAQ_DIR_VAL=${CUDAQ_DIR:-$CUDAQ_INSTALL_PREFIX_VAL/lib/cmake/cudaq}

if [[ ! -d "$CUDAQ_DIR_VAL" ]]; then
  echo "Error: CUDAQ_DIR does not exist: $CUDAQ_DIR_VAL" >&2
  echo "Set CUDAQ_INSTALL_PREFIX or CUDAQ_DIR before running." >&2
  exit 1
fi

SRC="libs/${LIB}"
BUILD="build_${LIB}"

if [[ ! -d "$SRC" ]]; then
  echo "Error: source dir $SRC not found (run from repo root)." >&2
  exit 1
fi

# Configure (only on first run, unless --reconfigure)
if [[ ! -d "$BUILD" || $RECONFIGURE -eq 1 ]]; then
  echo "Configuring $LIB ($BUILD_TYPE) ..."
  if [[ $DO_TESTS -eq 1 ]]; then
    TESTS_FLAG="-DCUDAQX_INCLUDE_TESTS=ON"
  else
    TESTS_FLAG="-DCUDAQX_INCLUDE_TESTS=OFF"
  fi
  if [[ $NO_PYTHON -eq 1 ]]; then
    PYTHON_FLAG="-DCUDAQX_BINDINGS_PYTHON=OFF"
  else
    PYTHON_FLAG="-DCUDAQX_BINDINGS_PYTHON=ON"
  fi
  cmake -G Ninja -S "$SRC" -B "$BUILD" \
    -DCUDAQ_DIR="$CUDAQ_DIR_VAL" \
    -DCMAKE_INSTALL_PREFIX="$CUDAQX_INSTALL_PREFIX_VAL" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    "$TESTS_FLAG" "$PYTHON_FLAG"
fi

echo "Building $LIB ..."
cmake --build "$BUILD" -j

if [[ $DO_TESTS -eq 1 ]]; then
  echo "Running tests for $LIB ..."
  cmake --build "$BUILD" --target run_tests
  if [[ $NO_PYTHON -eq 0 ]]; then
    cmake --build "$BUILD" --target run_python_tests
  fi
fi

if [[ $DO_INSTALL -eq 1 ]]; then
  echo "Installing $LIB to $CUDAQX_INSTALL_PREFIX_VAL ..."
  cmake --build "$BUILD" --target install
fi

echo "Done: $LIB"
