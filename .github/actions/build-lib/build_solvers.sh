#!/bin/sh

cmake -S libs/solvers -B "$1" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc-11 \
  -DCMAKE_CXX_COMPILER=g++-11 \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCUDAQ_DIR=/cudaq-install/lib/cmake/cudaq/ \
  -DCUDAQX_INCLUDE_TESTS=ON \
  -DCUDAQX_BINDINGS_PYTHON=ON \
  -DCMAKE_INSTALL_PREFIX=$HOME/.cudaqx

cmake --build "$1" --target install

