# ============================================================================ #
# Copyright (c) 2024 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
include_directories(${CUDAQX_SOLVERS_INCLUDE_DIR})
set(CMAKE_CXX_COMPILER "${CUDAQ_INSTALL_DIR}/bin/nvq++")
set(CMAKE_CXX_COMPILE_OBJECT "<CMAKE_CXX_COMPILER> -fPIC <DEFINES> <INCLUDES> -o <OBJECT> -c <SOURCE>")
add_library(test-kernels SHARED test_kernels.cpp)
