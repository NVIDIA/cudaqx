#!/bin/sh

# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Set PATH
export PYTHONPATH="$PYTHONPATH:/cudaq-install/cudaq"
export PATH="/cudaq-install/bin:$PATH"
echo "Setting PYTHONPATH=$PYTHONPATH"

# Set CUDA-QX paths for nvq++
CUDAQX_INCLUDE="$HOME/.cudaqx/include"
CUDAQX_LIB="$HOME/.cudaqx/lib"

LIB=$1  # Accepts "qec", "solvers", or "all"

echo "Running example tests for $LIB..."

if [[ "$LIB" == "qec" || "$LIB" == "all" ]]; then
    echo "Running QEC examples..."
    for file in examples/qec/python/*.py; do
        timeout 300 python3 "$file"
    done
    for file in examples/qec/cpp/*.cpp; do
        nvq++ --enable-mlir --target=stim \
            -I"$CUDAQX_INCLUDE" -L"$CUDAQX_LIB" -Wl,-rpath,"$CUDAQX_LIB" \
            -lcudaq-qec "$file"
        if [ $? -ne 0 ]; then
            echo "Compilation failed for $file"
            exit 1
        fi
        timeout 300 ./a.out
    done
fi

if [[ "$LIB" == "solvers" || "$LIB" == "all" ]]; then
    echo "Running Solvers examples..."
    for file in examples/solvers/python/*.py; do
        timeout 300 python3 "$file"
    done
    for file in examples/solvers/cpp/*.cpp; do
        nvq++ --enable-mlir \
            -I"$CUDAQX_INCLUDE" -L"$CUDAQX_LIB" -Wl,-rpath,"$CUDAQX_LIB" \
            -lcudaq-solvers "$file"
        if [ $? -ne 0 ]; then
            echo "Compilation failed for $file"
            exit 1
        fi
        timeout 300 ./a.out
    done
fi