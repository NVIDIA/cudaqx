#!/bin/sh

# Build realtime from cuda-quantum (if CUDAQ_REALTIME_ROOT not set)
if [ -z "$CUDAQ_REALTIME_ROOT" ]; then
  CUDAQ_REALTIME_ROOT=/tmp/cudaq-realtime
  _build_cwd=$(pwd)
  cd /tmp
  git clone --filter=blob:none --no-checkout https://github.com/NVIDIA/cuda-quantum
  cd cuda-quantum
  git sparse-checkout init --cone
  git sparse-checkout set realtime
  git checkout 3a0559ec4aaa38f1ffe77f523aafc3e223c0d11c # features/cudaq.realtime
  # Install DOCA 3.3 headers (needed by dispatch kernel fence calls)
  UBUNTU_VERSION=$(. /etc/os-release && echo $VERSION_ID)
  ARCH=$(dpkg --print-architecture)
  if [ "$ARCH" = "arm64" ]; then DOCA_ARCH="arm64-sbsa"; else DOCA_ARCH="x86_64"; fi
  wget -qO - https://linux.mellanox.com/public/repo/doca/GPG-KEY-Mellanox.pub | apt-key add -
  echo "deb https://linux.mellanox.com/public/repo/doca/3.3.0/ubuntu${UBUNTU_VERSION}/${DOCA_ARCH} ./" \
      > /etc/apt/sources.list.d/doca-3.3.list
  apt-get update && apt-get install -y --no-install-recommends libdoca-sdk-gpunetio-dev
  cd realtime
  mkdir build && cd build
  cmake -G Ninja -DCMAKE_INSTALL_PREFIX="$CUDAQ_REALTIME_ROOT" ..
  ninja
  ninja install
  cd "$_build_cwd"
fi

cmake -S . -B "$1" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc-11 \
  -DCMAKE_CXX_COMPILER=g++-11 \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCUDAQ_DIR=/cudaq-install/lib/cmake/cudaq/ \
  -DCUDAQX_ENABLE_LIBS="all" \
  -DCUDAQX_INCLUDE_TESTS=ON \
  -DCUDAQX_BINDINGS_PYTHON=ON \
  -DCMAKE_INSTALL_PREFIX="$2" \
  -DCUDAQ_REALTIME_ROOT=$CUDAQ_REALTIME_ROOT

cmake --build "$1" --target install
