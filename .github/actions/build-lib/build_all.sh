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
  git checkout 3c4faa5b21068dc4e92427e76498d7eeb90272d3 # features/cudaq.realtime
  cd realtime
  mkdir build && cd build
  cmake -G Ninja -DCMAKE_INSTALL_PREFIX="$CUDAQ_REALTIME_ROOT" ..
  ninja
  ninja install
  cd "$_build_cwd"
fi

if [ -z "$CUSTABILIZER_ROOT" ] && [ -x "$(command -v python3)" ]; then
  if [ -z "$CUSTABILIZER_PIP_PACKAGE" ]; then
    CUDA_MAJOR=""
    NVCC_BIN=${CUDACXX:-$(command -v nvcc)}
    if [ -n "$NVCC_BIN" ] && [ -x "$NVCC_BIN" ]; then
      CUDA_MAJOR=$("$NVCC_BIN" --version | sed -nE 's/.*release ([0-9]+)\..*/\1/p' | head -n 1)
    fi
    if [ -n "$CUDA_MAJOR" ]; then
      CUSTABILIZER_PIP_PACKAGE="custabilizer-cu${CUDA_MAJOR}"
    else
      for candidate in custabilizer-cu13 custabilizer-cu12; do
        if python3 -m pip show "$candidate" >/dev/null 2>&1; then
          CUSTABILIZER_PIP_PACKAGE="$candidate"
          break
        fi
      done
    fi
  fi

  if [ -n "$CUSTABILIZER_PIP_PACKAGE" ] && \
     python3 -m pip show "$CUSTABILIZER_PIP_PACKAGE" >/dev/null 2>&1; then
    CUSTABILIZER_ROOT=$(python3 - "$CUSTABILIZER_PIP_PACKAGE" <<'PY'
import pathlib, subprocess, sys
package = sys.argv[1]
show = subprocess.check_output([sys.executable, "-m", "pip", "show", package], text=True)
location = ""
for line in show.splitlines():
    if line.startswith("Location:"):
        location = line.split(":", 1)[1].strip(); break
if not location:
    print(""); raise SystemExit(0)
base = pathlib.Path(location)
for root in [base / "custabilizer", base / "cuquantum", base]:
    h = root / "include" / "custabilizer.h"
    lib = root / "lib" / "libcustabilizer.so.0"
    lib64 = root / "lib64" / "libcustabilizer.so.0"
    if h.exists() and (lib.exists() or lib64.exists()):
        print(root); raise SystemExit(0)
print("")
PY
)
  fi
fi

CUSTABILIZER_CMAKE_ARG=""
if [ -n "$CUSTABILIZER_ROOT" ]; then
  echo "Resolved cuStabilizer root: $CUSTABILIZER_ROOT"
  CUSTABILIZER_CMAKE_ARG="-DCUSTABILIZER_ROOT=$CUSTABILIZER_ROOT"
else
  echo "Unable to resolve CUSTABILIZER_ROOT from pip wheel."
fi

cmake -S . -B "$1" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc-11 \
  -DCMAKE_CXX_COMPILER=g++-11 \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCUDAQ_DIR=/cudaq-install/lib/cmake/cudaq/ \
  ${CUSTABILIZER_CMAKE_ARG:+$CUSTABILIZER_CMAKE_ARG} \
  -DCUDAQX_ENABLE_LIBS="all" \
  -DCUDAQX_INCLUDE_TESTS=ON \
  -DCUDAQX_BINDINGS_PYTHON=ON \
  -DCMAKE_INSTALL_PREFIX="$2" \
  -DCUDAQ_REALTIME_ROOT=$CUDAQ_REALTIME_ROOT

cmake --build "$1" --target install
