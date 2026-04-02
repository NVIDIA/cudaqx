#!/bin/sh
set -e

# Build cuda-quantum realtime library + hololink tools (if CUDAQ_REALTIME_ROOT not set)
if [ -z "$CUDAQ_REALTIME_ROOT" ]; then
  CUDAQ_REALTIME_ROOT=/tmp/cudaq-realtime
  CUDAQ_REALTIME_REPO=https://github.com/NVIDIA/cuda-quantum.git
  CUDAQ_REALTIME_REF=9ce3d2e886
  _build_cwd=$(pwd)

  cd /tmp
  rm -rf cudaq-realtime-src $CUDAQ_REALTIME_ROOT
  git clone --filter=blob:none --no-checkout $CUDAQ_REALTIME_REPO cudaq-realtime-src
  cd cudaq-realtime-src
  git sparse-checkout init --cone
  git sparse-checkout set realtime
  git checkout $CUDAQ_REALTIME_REF

  # Install DOCA, Holoscan SDK, libibverbs, and nvcomp (needed for HSB)
  apt-get update && apt-get install -y --no-install-recommends ninja-build nvcomp gnupg
  bash realtime/scripts/install_dev_prerequisites.sh

  # Build holoscan-sensor-bridge (hololink) FIRST, so cuda-quantum realtime
  # can build the bridge-hololink wrapper library that links against it.
  HSB_REPO=https://github.com/nvidia-holoscan/holoscan-sensor-bridge.git
  HSB_REF=release-2.6.0-EA
  HSB_PATCHES="/tmp/cudaq-realtime-src/realtime/scripts/hololink-patches"
  HSB_ROOT=/tmp/holoscan-sensor-bridge
  HSB_BUILD=${HSB_ROOT}/build

  if [ ! -d /opt/mellanox/doca/include ]; then
    echo "ERROR: DOCA SDK installation failed" >&2
    exit 1
  fi
  if [ ! -d /opt/nvidia/holoscan ]; then
    echo "ERROR: Holoscan SDK installation failed" >&2
    exit 1
  fi

  cd /tmp
  rm -rf holoscan-sensor-bridge
  git clone --depth 1 --branch $HSB_REF $HSB_REPO holoscan-sensor-bridge
  cd holoscan-sensor-bridge
  for p in "$HSB_PATCHES"/*.patch; do
    echo "Applying patch: $(basename $p)"
    git apply "$p"
  done
  export CUDA_NATIVE_ARCH=80
  cmake -G Ninja -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DHOLOLINK_BUILD_ONLY_NATIVE=OFF \
    -DHOLOLINK_BUILD_PYTHON=OFF \
    -DHOLOLINK_BUILD_TESTS=OFF \
    -DHOLOLINK_BUILD_TOOLS=OFF \
    -DHOLOLINK_BUILD_EXAMPLES=OFF \
    -DHOLOLINK_BUILD_EMULATOR=OFF
  cmake --build build --target gpu_roce_transceiver hololink_core
  echo "holoscan-sensor-bridge built at $HSB_BUILD"

  # Build cuda-quantum realtime with hololink tools enabled,
  # which produces libcudaq-realtime-bridge-hololink.so needed by the bridge.
  cd /tmp/cudaq-realtime-src/realtime
  mkdir -p build && cd build
  cmake -G Ninja -DCMAKE_INSTALL_PREFIX="$CUDAQ_REALTIME_ROOT" \
    -DCUDAQ_REALTIME_ENABLE_HOLOLINK_TOOLS=ON \
    -DHOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR=$HSB_ROOT \
    -DHOLOSCAN_SENSOR_BRIDGE_BUILD_DIR=$HSB_BUILD \
    ..
  ninja
  ninja install

  cd "$_build_cwd"
fi

HSB_ROOT=/tmp/holoscan-sensor-bridge
HSB_BUILD=${HSB_ROOT}/build

cmake -S libs/qec -B "$1" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc-11 \
  -DCMAKE_CXX_COMPILER=g++-11 \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCUDAQ_DIR=/cudaq-install/lib/cmake/cudaq/ \
  -DCUDAQX_INCLUDE_TESTS=ON \
  -DCUDAQX_BINDINGS_PYTHON=ON \
  -DCMAKE_INSTALL_PREFIX="$2" \
  -DCUDAQ_REALTIME_ROOT=$CUDAQ_REALTIME_ROOT \
  -DCUDAQX_QEC_ENABLE_HOLOLINK_TOOLS=ON \
  -DHOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR=$HSB_ROOT \
  -DHOLOSCAN_SENSOR_BRIDGE_BUILD_DIR=$HSB_BUILD

cmake --build "$1" --target install
