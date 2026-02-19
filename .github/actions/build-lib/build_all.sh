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
  git checkout b7eed833133c501a1a655905d1f58a175a0aa749 # features/cudaq.realtime
  git apply <<'PATCH'
diff --git a/realtime/lib/daemon/CMakeLists.txt b/realtime/lib/daemon/CMakeLists.txt
index 2fe4b20092..5bd0e3f22f 100644
--- a/realtime/lib/daemon/CMakeLists.txt
+++ b/realtime/lib/daemon/CMakeLists.txt
@@ -68,4 +68,9 @@ if(CUDA_FOUND)
     POSITION_INDEPENDENT_CODE ON
     ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
   )
+
+  install(TARGETS cudaq-realtime-dispatch
+    COMPONENT realtime-lib
+    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
+  )
 endif()
PATCH
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
