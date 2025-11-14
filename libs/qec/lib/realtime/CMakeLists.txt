# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

find_package(LLVM 16.0.6 EXACT REQUIRED CONFIG)

add_library(cudaq-qec-realtime-decoding SHARED
  realtime_decoding.cpp
  config.cpp
)

target_compile_options(cudaq-qec-realtime-decoding 
  PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)

target_include_directories(cudaq-qec-realtime-decoding
  PUBLIC 
    $<BUILD_INTERFACE:${CUDAQX_QEC_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${LLVM_INCLUDE_DIRS}>
    $<INSTALL_INTERFACE:${CUDAQ_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_options(cudaq-qec-realtime-decoding PUBLIC
  $<$<CXX_COMPILER_ID:GNU>:-Wl,--exclude-libs,ALL>
)

target_link_libraries(cudaq-qec-realtime-decoding
  PUBLIC 
    cudaqx-core
    cudaq-qec
  PRIVATE
    cudaq::cudaq-common
    LLVMSupport
)

set_target_properties(cudaq-qec-realtime-decoding PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)


install(TARGETS cudaq-qec-realtime-decoding
  COMPONENT qec-lib
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

add_subdirectory(quantinuum)
add_subdirectory(simulation)
