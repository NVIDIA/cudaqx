#!/bin/bash
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under    #
# the terms of the Apache License 2.0 which accompanies this distribution.    #
# ============================================================================ #
#
# In-container build for the decoding-server hardware CI.  Runs inside the
# dev image (docker/decoding-server/dev.Dockerfile) with the cudaqx checkout
# under test mounted at /workspaces/cudaqx and optional proprietary artifacts
# mounted read-only at /artifacts.
#
# This mirrors the CI recipe in .github/actions/build-lib/build_qec.sh with
# the apt/DOCA/Holoscan/HSB steps removed (baked into the image) and the
# hardware-lab deltas applied:
#   - CUDA architecture comes from the runner (GB200=100, DGX Spark=121)
#   - holoscan-sensor-bridge is the image's prebuilt /opt tree
#   - the proprietary cudevice archive and nv-qldpc plugin are picked up
#     from /artifacts when present (never required: absent pieces surface
#     later as per-lane SKIPs, not build failures)
#   - the trt_decoder plugin build is forced ON (TensorRT is in the image;
#     a detection regression should fail the configure loudly)
#   - the realtime_decoding_demo example binaries are built as well
#
# Usage (normally invoked by run_hw_ci.sh via docker exec):
#   container_build.sh --cuda-arch N
set -euo pipefail

CUDA_ARCH=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda-arch) CUDA_ARCH="$2"; shift ;;
        *) echo "ERROR: unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done
[[ -n "$CUDA_ARCH" ]] || { echo "ERROR: --cuda-arch is required" >&2; exit 1; }

CUDAQX_SRC=/workspaces/cudaqx
CUDAQ_PREFIX=${CUDAQ_INSTALL_PREFIX:-/usr/local/cudaq}
CUDAQX_INSTALL_PREFIX=/usr/local/cudaqx
CUDAQ_REALTIME_ROOT=/tmp/cudaq-realtime
HSB_ROOT=${HOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR:-/opt/holoscan-sensor-bridge}
HSB_BUILD=${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR:-$HSB_ROOT/build}
ARTIFACTS_DIR=/artifacts
NV_QLDPC_PLUGIN=$ARTIFACTS_DIR/decoder-plugins/libcudaq-qec-nv-qldpc-decoder.so
CUDEVICE_ARCHIVE=$ARTIFACTS_DIR/cudevice/libcudaq-qec-realtime-cudevice-proprietary.a

cd "$CUDAQX_SRC"
# setup_custabilizer.sh expands $CUSTABILIZER_ROOT unguarded, which is fatal
# under this script's `set -u` (the CI caller runs without -u).
export CUSTABILIZER_ROOT="${CUSTABILIZER_ROOT:-}"
. .github/actions/build-lib/setup_custabilizer.sh   # no-op: wheel is baked in
. scripts/cudaq_realtime_cmake_flags.sh             # AVX512 workaround, x86-only

export CUDA_NATIVE_ARCH="$CUDA_ARCH"

# ---------------------------------------------------------------------------
# cudaq-realtime with HSB tools, from the commit's own .cudaq_version pin
# (produces libcudaq-realtime-bridge-gpu-roce.so, the provider decoding_server
# dlopens for the gpu_roce wire).  Mirrors build_qec.sh lines 12-115.
# ---------------------------------------------------------------------------
CUDAQ_REPO=${CUDAQ_REPO:-$(jq -r '.cudaq.repository' .cudaq_version)}
CUDAQ_REF=${CUDAQ_REF:-$(jq -r '.cudaq.ref' .cudaq_version)}
echo "== cudaq-realtime source: ${CUDAQ_REPO}@${CUDAQ_REF}"

cd /tmp
rm -rf cudaq-realtime-src "$CUDAQ_REALTIME_ROOT"
git clone --filter=blob:none --no-checkout "https://github.com/${CUDAQ_REPO}.git" cudaq-realtime-src
cd cudaq-realtime-src
git sparse-checkout init --cone
git sparse-checkout set realtime cmake
git checkout "$CUDAQ_REF"

cd realtime
mkdir -p build && cd build
cmake -G Ninja -DCMAKE_INSTALL_PREFIX="$CUDAQ_REALTIME_ROOT" \
    -DCMAKE_CUDA_FLAGS="$(cudaq_realtime_cmake_cuda_flags)" \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DCUDAQ_REALTIME_ENABLE_HSB_TOOLS=ON \
    -DHOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR="$HSB_ROOT" \
    -DHOLOSCAN_SENSOR_BRIDGE_BUILD_DIR="$HSB_BUILD" \
    ..
ninja
ninja install

# ---------------------------------------------------------------------------
# Proprietary inputs (optional).
# ---------------------------------------------------------------------------
_prop_archive_flag=""
if [[ -f "$CUDEVICE_ARCHIVE" ]]; then
    _prop_archive_flag="-DCUDAQ_QEC_REALTIME_CUDEVICE_PROPRIETARY_ARCHIVE=$CUDEVICE_ARCHIVE"
    echo "== cudevice proprietary archive: $CUDEVICE_ARCHIVE"
else
    echo "== cudevice proprietary archive: absent (device_graph-dispatch tests will SKIP)"
fi
if [[ -f "$NV_QLDPC_PLUGIN" ]]; then
    # cudaqx's external-decoder install patches the plugin's rpath IN PLACE
    # and /artifacts is mounted read-only -- hand the build a writable copy
    # (the decoder-plugins symlink below then serves the patched copy too).
    mkdir -p /tmp/hwci-artifacts/decoder-plugins
    cp -f "$NV_QLDPC_PLUGIN" /tmp/hwci-artifacts/decoder-plugins/
    NV_QLDPC_PLUGIN="/tmp/hwci-artifacts/decoder-plugins/$(basename "$NV_QLDPC_PLUGIN")"
    # Configure-time gate for test_realtime_qldpc_graph_decoding and the
    # mixed-dispatch app example.
    export QEC_EXTERNAL_DECODERS="$NV_QLDPC_PLUGIN"
    echo "== nv-qldpc plugin: $NV_QLDPC_PLUGIN (writable copy from $ARTIFACTS_DIR)"
else
    echo "== nv-qldpc plugin: absent (nv-qldpc lanes will SKIP)"
fi

# ---------------------------------------------------------------------------
# cudaqx qec.  Configured from the TOP-LEVEL CMakeLists (not -S libs/qec) so
# the build tree lands at build/libs/qec/... -- the layout every in-tree
# hardware script hardcodes (hsb_fpga_decoding_server_test.sh & co. default
# to CUDAQX_DIR=/workspaces/cudaqx and resolve binaries + LD paths from
# build/libs/qec/...).  Mirrors build_qec.sh lines 120-141 plus the deltas
# listed in the header.
# ---------------------------------------------------------------------------
cd "$CUDAQX_SRC"
cmake -S . -B build \
    -DCUDAQX_ENABLE_LIBS=qec \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=gcc-12 \
    -DCMAKE_CXX_COMPILER=g++-12 \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DCUDAQ_DIR="$CUDAQ_PREFIX/lib/cmake/cudaq/" \
    -DCUDAQX_INCLUDE_TESTS=ON \
    -DCUDAQX_BINDINGS_PYTHON=ON \
    -DCMAKE_INSTALL_PREFIX="$CUDAQX_INSTALL_PREFIX" \
    -DCUDAQ_REALTIME_ROOT="$CUDAQ_REALTIME_ROOT" \
    -DCUDAQX_QEC_ENABLE_HSB_TOOLS=ON \
    -DHOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR="$HSB_ROOT" \
    -DHOLOSCAN_SENSOR_BRIDGE_BUILD_DIR="$HSB_BUILD" \
    -DCUDAQ_QEC_BUILD_TRT_DECODER=ON \
    $_prop_archive_flag
cmake --build build --target install -j "$(nproc)"

# The decoding server discovers decoder plugins in
# <prefix>/lib/decoder-plugins; expose the proprietary one when staged.
if [[ -f "$NV_QLDPC_PLUGIN" ]]; then
    mkdir -p "$CUDAQX_INSTALL_PREFIX/lib/decoder-plugins"
    ln -sfn "$NV_QLDPC_PLUGIN" \
        "$CUDAQX_INSTALL_PREFIX/lib/decoder-plugins/$(basename "$NV_QLDPC_PLUGIN")"
fi

# ---------------------------------------------------------------------------
# realtime_decoding_demo example binaries (the examples-tier lanes).  The
# arch override is mandatory: the demo's CMakeLists defaults to 80.
# ---------------------------------------------------------------------------
cmake -S docs/sphinx/examples/qec/realtime_decoding_demo -B demo-build -G Ninja \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DCUDAQ_INSTALL_DIR="$CUDAQ_PREFIX" \
    -DCUDAQX_INSTALL_DIR="$CUDAQX_INSTALL_PREFIX" \
    -DCUDAQ_REALTIME_DIR="$CUDAQ_REALTIME_ROOT"
cmake --build demo-build -j "$(nproc)"

echo "== container build complete"
echo "   cudaqx install : $CUDAQX_INSTALL_PREFIX"
echo "   realtime       : $CUDAQ_REALTIME_ROOT"
echo "   qec build tree : $CUDAQX_SRC/build"
echo "   demo binaries  : $CUDAQX_SRC/demo-build"
