#!/bin/sh

# Download realtime artifacts from GitHub release (if CUDAQ_REALTIME_ROOT not set)
if [ -z "$CUDAQ_REALTIME_ROOT" ]; then
  CUDAQ_REALTIME_ROOT=/tmp/cudaq-realtime
  mkdir -p $CUDAQ_REALTIME_ROOT

  # Download from GitHub release
  RELEASE_URL="https://github.com/NVIDIA/cudaqx/releases/download/cudaq-realtime-no-push"
  wget -qO- ${RELEASE_URL}/cudaq-realtime-headers.tar.gz | tar xzf - -C $CUDAQ_REALTIME_ROOT

  ARCH=$(uname -m | sed 's/aarch64/arm64/' | sed 's/x86_64/x86_64/')
  mkdir -p $CUDAQ_REALTIME_ROOT/lib
  wget -qO- ${RELEASE_URL}/cudaq-realtime-libs-${ARCH}.tar.gz | tar xzf - -C $CUDAQ_REALTIME_ROOT/lib
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
