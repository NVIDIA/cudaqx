#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Thin wrapper that delegates to cudaq/scripts/build_cudaq.sh. Translates the
# positional-args contract used by .github/actions/get-cudaq-build/action.yaml
# into the env-var + flag interface expected by upstream.
BUILD_TYPE=${1:-"Release"}
LAUNCHER=${2:-""}  # accepted for backward compat; upstream auto-detects ccache via PATH
CC=${3:-"gcc"}
CXX=${4:-"g++"}

export CC
export CXX

# Keep cudaqx CI's previous defaults (these differ from upstream defaults).
export LLVM_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX:-/usr/local/llvm}
export CUDAQ_REQUIRE_OPENMP=${CUDAQ_REQUIRE_OPENMP:-TRUE}
export CUDAQ_BUILD_TESTS=${CUDAQ_BUILD_TESTS:-FALSE}
export CUDAQ_WERROR=${CUDAQ_WERROR:-OFF}
# CUDAQ_INSTALL_PREFIX / CUQUANTUM_INSTALL_PREFIX / CUTENSOR_INSTALL_PREFIX /
# CCACHE_DIR are expected to be set by the calling action (see action.yaml).

cd cudaq && bash scripts/build_cudaq.sh -v -c "$BUILD_TYPE"
