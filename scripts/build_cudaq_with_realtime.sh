#!/bin/bash
set -euo pipefail

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Build and install cudaq-realtime, then build CUDA-Q against that install so
# device_call support is enabled in the final CUDA-Q prefix.
#
# Positional arguments:
#   1. CMAKE_BUILD_TYPE
#   2. C compiler
#   3. C++ compiler
#
# Environment:
#   CUDAQ_SRC             CUDA-Q source checkout, default: ./cudaq
#   CUDAQ_INSTALL_PREFIX  CUDA-Q install prefix, default: /usr/local/cudaq
#   CUDAQ_REALTIME_ENABLE_HOLOLINK_TOOLS
#                         ON to build the hololink (HSB) bridge tools, default: OFF.
#                         When ON, HOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR and
#                         HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR must point at a built
#                         holoscan-sensor-bridge checkout (see
#                         scripts/install_cudaq_with_realtime.sh).
#   CUDAQ_REALTIME_BUILD_TESTS
#                         ON to build the realtime unit-test targets, default: OFF.
#                         (install_cudaq_with_realtime.sh may force this ON as a
#                         temporary workaround for older pinned CUDA-Q SHAs.)

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }

BUILD_TYPE=${1:-"Release"}
CC=${2:-${CC:-"gcc"}}
CXX=${3:-${CXX:-"g++"}}

export CC
export CXX

: "${CUDAQ_SRC:=cudaq}"
: "${CUDAQ_INSTALL_PREFIX:=/usr/local/cudaq}"
: "${CUDAQ_REALTIME_ENABLE_HOLOLINK_TOOLS:=OFF}"
: "${CUDAQ_REALTIME_BUILD_TESTS:=OFF}"

CUDAQ_SRC=$(cd "$CUDAQ_SRC" && pwd)

# When hololink tools are requested, the realtime CMake needs to know where the
# prebuilt holoscan-sensor-bridge lives so it can build the bridge-hololink
# wrapper library against it.
realtime_hololink_flags=()
if [ "$CUDAQ_REALTIME_ENABLE_HOLOLINK_TOOLS" = "ON" ]; then
  : "${HOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR:?set HOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR when hololink tools are enabled}"
  : "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR:?set HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR when hololink tools are enabled}"
  realtime_hololink_flags=(
    "-DHOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR=$HOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR"
    "-DHOLOSCAN_SENSOR_BRIDGE_BUILD_DIR=$HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR"
  )
fi

log "Building cudaq-realtime from $CUDAQ_SRC/realtime (hololink=$CUDAQ_REALTIME_ENABLE_HOLOLINK_TOOLS, tests=$CUDAQ_REALTIME_BUILD_TESTS)"
cmake -G Ninja -S "$CUDAQ_SRC/realtime" -B "$CUDAQ_SRC/realtime/build" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DCMAKE_INSTALL_PREFIX="$CUDAQ_INSTALL_PREFIX" \
  -DCUDAQ_REALTIME_BUILD_TESTS="$CUDAQ_REALTIME_BUILD_TESTS" \
  -DCUDAQ_REALTIME_BUILD_EXAMPLES=OFF \
  -DCUDAQ_REALTIME_ENABLE_HOLOLINK_TOOLS="$CUDAQ_REALTIME_ENABLE_HOLOLINK_TOOLS" \
  "${realtime_hololink_flags[@]}"
cmake --build "$CUDAQ_SRC/realtime/build" --target install --parallel

log "Building CUDA-Q with realtime support"
(
  cd "$CUDAQ_SRC"
  bash scripts/build_cudaq.sh -v -c "$BUILD_TYPE" -- \
    "-DCUDAQ_REALTIME_DIR=$CUDAQ_INSTALL_PREFIX"
)
