# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_library(cudaq-qec-realtime-decoding-simulation SHARED
  simulation_host.cpp
)

target_compile_options(cudaq-qec-realtime-decoding-simulation
  PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)

cudaqx_add_device_code(cudaq-qec-realtime-decoding-simulation
  SOURCES
    simulation_device.cpp
)

target_include_directories(cudaq-qec-realtime-decoding-simulation
  PUBLIC 
    $<BUILD_INTERFACE:${CUDAQX_QEC_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${CUDAQ_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_options(cudaq-qec-realtime-decoding-simulation PUBLIC
  $<$<CXX_COMPILER_ID:GNU>:-Wl,--exclude-libs,ALL>
)

target_link_libraries(cudaq-qec-realtime-decoding-simulation
  PUBLIC 
    cudaqx-core
    cudaq-qec
    cudaq-qec-realtime-decoding
  PRIVATE
    cudaq::cudaq-common
)

set_target_properties(cudaq-qec-realtime-decoding-simulation PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)


install(TARGETS cudaq-qec-realtime-decoding-simulation
  COMPONENT qec-lib
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

