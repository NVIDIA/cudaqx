#!/bin/sh

# Download realtime artifacts from GitHub release (if CUDAQ_REALTIME_ROOT not set)
# REVERT-WITH-CUDAQ-REALTIME-BUILD
if [ -z "$CUDAQ_REALTIME_ROOT" ]; then
  CUDAQ_REALTIME_ROOT=/tmp/cudaq-realtime
  mkdir -p $CUDAQ_REALTIME_ROOT
  mkdir -p $CUDAQ_REALTIME_ROOT/lib

  # Download from GitHub draft release using gh CLI
  RELEASE_TAG="cudaq-realtime-no-push2"
  ARCH=$(uname -m | sed 's/aarch64/arm64/' | sed 's/x86_64/x86_64/')
  
  if ! gh release download "$RELEASE_TAG" \
    --pattern "cudaq-realtime-headers.tar.gz" \
    --pattern "cudaq-realtime-libs-${ARCH}.tar.gz" \
    --repo NVIDIA/cudaqx \
    --dir /tmp; then
    echo "ERROR: Failed to download cudaq-realtime assets from release ${RELEASE_TAG}."
    exit 1
  fi

  if [ ! -f "/tmp/cudaq-realtime-headers.tar.gz" ] || [ ! -f "/tmp/cudaq-realtime-libs-${ARCH}.tar.gz" ]; then
    echo "ERROR: cudaq-realtime asset files missing after download."
    exit 1
  fi

  tar xzf /tmp/cudaq-realtime-headers.tar.gz -C $CUDAQ_REALTIME_ROOT
  tar xzf /tmp/cudaq-realtime-libs-${ARCH}.tar.gz -C $CUDAQ_REALTIME_ROOT/lib
fi

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
  -DCUDAQ_REALTIME_ROOT=$CUDAQ_REALTIME_ROOT

cmake --build "$1" --target install
