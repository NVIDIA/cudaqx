# ============================================================================ #
# Copyright (c) 2024 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_library(cudaqx-core STATIC
  tear_down.cpp
  graph.cpp
  tensor_impls/xtensor_impl.cpp
)

target_include_directories(cudaqx-core
  PUBLIC 
    $<BUILD_INTERFACE:${CUDAQX_CORE_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(cudaqx-core
  PUBLIC     
    cudaq::cudaq
  PRIVATE
    xtensor
    xtensor-blas
    fmt::fmt-header-only
)

install(DIRECTORY ${CUDAQX_CORE_INCLUDE_DIR}/cuda-qx
  DESTINATION "${CUDAQ_INCLUDE_DIR}"
  COMPONENT cudaqx-core
  FILES_MATCHING PATTERN "*.h"
)

