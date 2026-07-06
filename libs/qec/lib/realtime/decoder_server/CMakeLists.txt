# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_library(cudaq-qec-decoder-server STATIC
  RoundAccumulator.cpp
  DecoderSession.cpp
  SessionRegistry.cpp
  RpcDispatcher.cpp
  DecoderServer.cpp
)

target_compile_features(cudaq-qec-decoder-server PUBLIC cxx_std_20)

# Handle both standalone (libs/qec) and full-repo builds.
if(CUDAQX_QEC_STANDALONE_BUILD)
  set(_core_include_dir "${CMAKE_SOURCE_DIR}/../core/include")
else()
  set(_core_include_dir "${CMAKE_SOURCE_DIR}/libs/core/include")
endif()

target_include_directories(cudaq-qec-decoder-server
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CUDAQX_QEC_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${_core_include_dir}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(cudaq-qec-decoder-server
  PUBLIC
    cudaq-qec
    cudaqx-core
    cudaq-qec-realtime-decoding
)

set_target_properties(cudaq-qec-decoder-server PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  POSITION_INDEPENDENT_CODE ON
)

install(TARGETS cudaq-qec-decoder-server
  COMPONENT qec-lib
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
